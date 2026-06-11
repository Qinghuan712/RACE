"""
compute_homography.py
---------------------
Compute ALL pairwise cross-camera homographies H_{i->j} for 4 cameras.
(6 pairs total: c001-c002, c001-c003, c001-c004, c002-c003, c002-c004, c003-c004)

GT format: frame_id, object_id, x, y, w, h, 1, -1, -1, -1
  bottom-center = (x + w/2,  y + h)

Protocol:
  Calibration : frames   1 -  600  (fit H via RANSAC)
  Validation  : frames 601 - 1800

Validation per pair:
  Layer 1 - Reprojection error : mean / median / p90 / p95 + CDF plot
  Layer 2 - Separability       : d_same vs d_diff histogram + F1 vs tau_d
             (analysed PER PAIR so we can spot which pairs are hard)

Outputs (in --output_dir):
  homographies.npz                   all H matrices, key = "H_c002_to_c001" etc.
  homography_report.txt
  val1_cdf_{src}_{ref}.png           Layer-1 CDF per pair
  val2_hist_{src}_{ref}.png          Layer-2 histogram per pair
  val2_f1_{src}_{ref}.png            Layer-2 F1/P/R vs tau_d per pair
  val2_summary.png                   best-F1 bar chart across all pairs
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
from itertools import combinations

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RACE.artifacts import PairCalibration, save_homography_artifact

GT_DIR   = "/home/qinghuan/Xinyan/Regenhance/dataset_preprocessing/aligned_gt_640"
CAMERAS  = ["c001", "c002", "c003", "c004"]
CALIB_FRAMES = (1,   600)
VAL_FRAMES   = (601, 1800)

# All ordered pairs (src -> ref) — 12 directed, but we only need 6 undirected.
# We store H_{src->ref} for each undirected pair {src, ref} with src < ref.
PAIRS = [(a, b) for a, b in combinations(CAMERAS, 2)]  # 6 pairs


# ─────────────────────────── I/O ─────────────────────────────────────────────

def load_gt(camera):
    """
    Returns {frame_id: {object_id: (cx, by, w)}}
      cx = x + w/2  (bottom-center x)
      by = y + h    (bottom y)
      w  = bbox width (kept for debug / scale info)
    """
    path = os.path.join(GT_DIR, f"{camera}_aligned_gt.txt")
    data = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            fid, oid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            data[fid][oid] = (x + w / 2.0, y + h, w)
    return data


def collect_correspondences(gt_src, gt_ref, frame_range):
    """
    Same-ID bottom-center point pairs within frame_range.
    """
    src_pts, ref_pts = [], []
    f_lo, f_hi = frame_range
    for fid in sorted(set(gt_src) & set(gt_ref)):
        if not (f_lo <= fid <= f_hi):
            continue
        for oid in set(gt_src[fid]) & set(gt_ref[fid]):
            cx_s, by_s, _ = gt_src[fid][oid]
            cx_r, by_r, _ = gt_ref[fid][oid]
            src_pts.append((cx_s, by_s))
            ref_pts.append((cx_r, by_r))
    return (np.array(src_pts, dtype=np.float32),
            np.array(ref_pts, dtype=np.float32))


# ─────────────────────────── Homography ──────────────────────────────────────

def fit_homography(src_pts, ref_pts, ransac_thresh=3.0):
    if len(src_pts) < 4:
        raise RuntimeError(f"Not enough correspondences: {len(src_pts)} < 4")
    H, mask = cv2.findHomography(src_pts, ref_pts,
                                  cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None")
    return H, mask


def project_points(H, pts):
    """Nx2 -> Nx2 via homogeneous projection."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1).T  # 3xN
    proj  = H @ pts_h
    return (proj[:2] / proj[2]).T  # Nx2


def reproj_errors(H, src_pts, ref_pts):
    """Per-point Euclidean reprojection error using bottom-center only."""
    if len(src_pts) == 0:
        return np.array([])
    return np.linalg.norm(project_points(H, src_pts) - ref_pts, axis=1)


def select_representative_four_points(src_pts, ref_pts):
    """Select 4 spatially spread correspondences from an inlier set.

    OpenCV RANSAC does not expose the exact minimal 4-point sample that first
    proposed the final model. For visualization we therefore pick 4
    representative inlier correspondences that are well spread in the SRC view:
    points nearest to the four corners of the SRC inlier bounding box.
    """
    if len(src_pts) < 4:
        raise RuntimeError(f"Need at least 4 inlier correspondences, got {len(src_pts)}")

    src = np.asarray(src_pts, dtype=np.float32)
    ref = np.asarray(ref_pts, dtype=np.float32)

    min_xy = src.min(axis=0)
    max_xy = src.max(axis=0)
    targets = np.array([
        [min_xy[0], min_xy[1]],  # top-left
        [max_xy[0], min_xy[1]],  # top-right
        [min_xy[0], max_xy[1]],  # bottom-left
        [max_xy[0], max_xy[1]],  # bottom-right
    ], dtype=np.float32)

    chosen = []
    used = set()
    for target in targets:
        order = np.argsort(np.sum((src - target) ** 2, axis=1))
        picked = None
        for idx in order:
            idx = int(idx)
            if idx not in used:
                picked = idx
                break
        if picked is None:
            for idx in range(len(src)):
                if idx not in used:
                    picked = idx
                    break
        used.add(picked)
        chosen.append(picked)

    return src[chosen], ref[chosen], chosen


def visualize_representative_four_points(
    src_cam,
    ref_cam,
    src_pts,
    ref_pts,
    out_dir,
    *,
    pair_tag=None,
    note=None,
):
    """Save a 2-panel figure with 4 representative corresponding points."""
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    labels = ["P1", "P2", "P3", "P4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, pts, cam in [
        (axes[0], np.asarray(src_pts, dtype=np.float32), src_cam),
        (axes[1], np.asarray(ref_pts, dtype=np.float32), ref_cam),
    ]:
        ax.set_xlim(0, 640)
        ax.set_ylim(360, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.grid(True, alpha=0.25)
        ax.set_title(cam)

        for idx, (pt, color, label) in enumerate(zip(pts, colors, labels)):
            x, y = float(pt[0]), float(pt[1])
            ax.scatter([x], [y], s=55, c=color, edgecolors="black", linewidths=0.6, zorder=3)
            ax.text(x + 8, y - 6, label, color=color, fontsize=10, weight="bold")

    title = f"Representative 4-point calibration correspondences: {src_cam} -> {ref_cam}"
    if pair_tag:
        title += f"\n{pair_tag}"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"calib4_{src_cam}_{ref_cam}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"    [Calib4] Saved -> {path}")
    return path


# ─────────────────────────── Overlap region ──────────────────────────────────

def compute_overlap_region(gt_src, gt_ref, margin=20.0):
    """
    Build per-camera empirical overlap hulls using SAME-ID co-occurring
    detections from ALL available frames (no H required).

    For camera pair (i=src, j=ref):
      S_ij^(src) = { bottom-center of src det : same OID seen in both cams, same frame }
      S_ij^(ref) = { bottom-center of ref det : same OID seen in both cams, same frame }

      Omega_src = Dilate( Hull(S_ij^(src)), margin )   -- in src image space
      Omega_ref = Dilate( Hull(S_ij^(ref)), margin )   -- in ref image space

    A detection is "in overlap" if its bottom-center falls inside the
    dilated hull of its OWN camera's overlap region.

    Args:
        margin : dilation in pixels (default 20).  Implemented via
                 pointPolygonTest distance threshold >= -margin.

    Returns:
        (hull_src, hull_ref, n_covis) — OpenCV contours (Nx1x2 float32)
        or (None, None, 0) if too few co-visible points.
    """
    src_pts, ref_pts = [], []
    for fid in sorted(set(gt_src) & set(gt_ref)):
        for oid in set(gt_src[fid]) & set(gt_ref[fid]):
            cx_s, by_s, _ = gt_src[fid][oid]
            cx_r, by_r, _ = gt_ref[fid][oid]
            src_pts.append((cx_s, by_s))
            ref_pts.append((cx_r, by_r))

    n_covis = len(src_pts)
    if n_covis < 3:
        return None, None, 0

    hull_src = cv2.convexHull(np.array(src_pts, dtype=np.float32))
    hull_ref = cv2.convexHull(np.array(ref_pts, dtype=np.float32))
    return hull_src, hull_ref, n_covis


def in_overlap(pt, hull, margin=20.0):
    """
    Return True if pt=(x,y) is inside the hull or within `margin` pixels
    of its boundary (dilated hull test).

    cv2.pointPolygonTest returns:
      +d  inside  (d = dist to nearest edge)
      -d  outside
    So >= -margin means "inside or within margin px of border".
    If hull is None, always returns True.
    """
    if hull is None:
        return True
    dist = cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), measureDist=True)
    return dist >= -margin


def visualize_overlap(hull_src, hull_ref, gt_src, gt_ref,
                       src_cam, ref_cam, out_dir, margin=20.0):
    """
    Save a two-panel figure showing the overlap region in each camera's OWN space.

    Left  panel : src  camera space — all src bottom-center pts + hull_src (Omega_src)
    Right panel : ref  camera space — all ref bottom-center pts + hull_ref (Omega_ref)

    Points inside the dilated hull (overlap zone) are highlighted.
    """
    src_all = [(cx, by) for fid in gt_src
               for cx, by, _ in gt_src[fid].values()]
    ref_all = [(cx, by) for fid in gt_ref
               for cx, by, _ in gt_ref[fid].values()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, pts_all, hull, cam_name, color in [
        (axes[0], src_all, hull_src, src_cam, "steelblue"),
        (axes[1], ref_all, hull_ref, ref_cam, "seagreen"),
    ]:
        if pts_all:
            arr = np.array(pts_all, dtype=np.float32)
            # colour each point by whether it falls in the overlap zone
            inside = np.array([in_overlap(p, hull, margin) for p in arr])
            ax.scatter(arr[~inside, 0], arr[~inside, 1],
                       s=1, c="lightgrey", alpha=0.3, label="outside overlap")
            ax.scatter(arr[inside, 0],  arr[inside, 1],
                       s=1, c=color, alpha=0.4, label="inside overlap")

        if hull is not None:
            poly = np.vstack([hull[:, 0, :], hull[0, 0, :]])
            ax.plot(poly[:, 0], poly[:, 1], color="red", linewidth=1.5,
                    label=f"hull (Omega_{cam_name})")

        ax.set_xlim(0, 640); ax.set_ylim(360, 0)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.set_title(f"{cam_name} – overlap hull\n"
                     f"(same-ID co-vis pts, all 1800 frames, margin={margin:.0f}px)")
        ax.legend(fontsize=7, markerscale=5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Empirical overlap regions: {src_cam} <-> {ref_cam}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"overlap_{src_cam}_{ref_cam}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    [Overlap] Saved -> {path}")


# ─────────────────────────── Layer-1 per pair ────────────────────────────────

def validate_layer1_pair(H, gt_src, gt_ref, src_cam, ref_cam, out_dir,
                          hull_src=None, hull_ref=None, margin=20.0):
    """
    Layer-1 reprojection error using bottom-center points.
    """
    pt_label = "bottom-center"
    errs_all     = []
    errs_overlap = []

    f_lo, f_hi = VAL_FRAMES
    for fid in sorted(set(gt_src) & set(gt_ref)):
        if not (f_lo <= fid <= f_hi):
            continue
        shared = set(gt_src[fid]) & set(gt_ref[fid])
        if not shared:
            continue
        for oid in shared:
            cx_s, by_s, _ = gt_src[fid][oid]
            cx_r, by_r, _ = gt_ref[fid][oid]
            src_pt = np.array([(cx_s, by_s)], dtype=np.float32)
            ref_pt = np.array([(cx_r, by_r)], dtype=np.float32)
            err = float(np.linalg.norm(project_points(H, src_pt) - ref_pt))
            errs_all.append(err)
            if (in_overlap((cx_s, by_s), hull_src, margin) and
                    in_overlap((cx_r, by_r), hull_ref, margin)):
                errs_overlap.append(err)

    if not errs_all:
        return {}

    def _stats(arr):
        a = np.array(arr, dtype=np.float32)
        return {"n": len(a), "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
                "p95": float(np.percentile(a, 95))}

    s_all     = _stats(errs_all)
    s_overlap = _stats(errs_overlap) if errs_overlap else None

    # ── CDF plot: all vs overlap ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, errs, title_suffix in [
        (axes[0], errs_all,     "all val pairs"),
        (axes[1], errs_overlap, f"overlap only (margin={margin:.0f}px)"),
    ]:
        if errs:
            se  = np.sort(errs)
            cdf = np.arange(1, len(se) + 1) / len(se)
            ax.plot(se, cdf, linewidth=1.5, color="steelblue",
                    label=f"center (n={len(se)})")
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{src_cam}->{ref_cam}\n{title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer-1 Reprojection Error CDF ({pt_label})", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"val1_cdf_{src_cam}_{ref_cam}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"    [L1] all  : mean={s_all['mean']:.1f}  median={s_all['median']:.1f}"
          f"  p90={s_all['p90']:.1f}  p95={s_all['p95']:.1f} px  (n={s_all['n']})")
    if s_overlap:
        print(f"    [L1] ovlp : mean={s_overlap['mean']:.1f}  "
              f"median={s_overlap['median']:.1f}"
              f"  p90={s_overlap['p90']:.1f}  p95={s_overlap['p95']:.1f} px"
              f"  (n={s_overlap['n']})")
    print(f"    [L1] CDF  -> {path}")

    result = {f"all_{k}": v for k, v in s_all.items()}
    if s_overlap:
        result.update({f"ovlp_{k}": v for k, v in s_overlap.items()})
    return result


# ─────────────────────────── Layer-2 per pair ────────────────────────────────

def compute_same_diff_pair(H, gt_src, gt_ref, frame_range,
                            hull_src=None, hull_ref=None, margin=20.0):
    """
    Compute d_same / d_diff using bottom-center points.
    """
    d_same = []
    d_diff = []
    f_lo, f_hi = frame_range

    for fid in sorted(set(gt_src) & set(gt_ref)):
        if not (f_lo <= fid <= f_hi):
            continue

        src_oids = list(gt_src[fid].keys())
        ref_oids = list(gt_ref[fid].keys())
        if not src_oids or not ref_oids:
            continue

        # Filter to overlap regions using bottom-center
        src_oids_f = [o for o in src_oids
                      if in_overlap((gt_src[fid][o][0], gt_src[fid][o][1]),
                                    hull_src, margin)]
        ref_oids_f = [o for o in ref_oids
                      if in_overlap((gt_ref[fid][o][0], gt_ref[fid][o][1]),
                                    hull_ref, margin)]
        if not src_oids_f or not ref_oids_f:
            continue

        src_ctr  = np.array([(gt_src[fid][o][0], gt_src[fid][o][1])
                              for o in src_oids_f], dtype=np.float32)
        proj_ctr = project_points(H, src_ctr)
        ref_ctr  = np.array([(gt_ref[fid][o][0], gt_ref[fid][o][1])
                              for o in ref_oids_f], dtype=np.float32)

        for i, oid_s in enumerate(src_oids_f):
            for j, oid_r in enumerate(ref_oids_f):
                d = float(np.linalg.norm(proj_ctr[i] - ref_ctr[j]))
                if oid_s == oid_r:
                    d_same.append(d)
                else:
                    d_diff.append(d)

        if len(d_diff) > 300_000:
            break

    return (np.array(d_same, dtype=np.float32),
            np.array(d_diff, dtype=np.float32))


def validate_layer2_pair(H, gt_src, gt_ref, src_cam, ref_cam,
                          out_dir, tau_max=200.0, tau_steps=200,
                          hull_src=None, hull_ref=None, margin=20.0):
    """
    Separability for one camera pair using bottom-center points.
    """
    d_same, d_diff = compute_same_diff_pair(
        H, gt_src, gt_ref, VAL_FRAMES,
        hull_src=hull_src, hull_ref=hull_ref, margin=margin)

    tag = f"{src_cam}_{ref_cam}"
    print(f"    [L2] {src_cam}->{ref_cam}: "
          f"d_same={len(d_same)}, d_diff={len(d_diff)}")

    if len(d_same) == 0:
        print(f"    [L2] No same-ID pairs in overlap region, skipping {tag}.")
        return {}

    # ── Histogram ─────────────────────────────────────────────────────────
    bins = np.linspace(0, tau_max, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(d_same):
        ax.hist(d_same[d_same <= tau_max], bins=bins, alpha=0.7,
                color="steelblue", density=True, label=f"same-ID (n={len(d_same)})")
    if len(d_diff):
        ax.hist(d_diff[d_diff <= tau_max], bins=bins, alpha=0.5,
                color="lightblue", density=True, label=f"diff-ID (n={len(d_diff)})")
    ax.set_xlabel("Projected distance (px)")
    ax.set_ylabel("Density")
    ax.set_title(f"Layer-2: d_same vs d_diff  {src_cam}->{ref_cam}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"val2_hist_{tag}.png"), dpi=120)
    plt.close(fig)

    # ── F1 vs tau_d ───────────────────────────────────────────────────────
    taus = np.linspace(0, tau_max, tau_steps)
    n_pos = len(d_same)
    f1s, precs, recs = [], [], []
    for tau in taus:
        tp   = int(np.sum(d_same <= tau))
        fp   = int(np.sum(d_diff <= tau)) if len(d_diff) else 0
        fn   = n_pos - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    best_i    = int(np.argmax(f1s))
    best_tau  = float(taus[best_i])
    best_f1   = float(f1s[best_i])
    best_prec = float(precs[best_i])
    best_rec  = float(recs[best_i])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, f1s,   linewidth=1.5, color="steelblue", label="F1")
    ax.plot(taus, precs, linewidth=1.0, linestyle="--", color="steelblue",
            alpha=0.6, label="Precision")
    ax.plot(taus, recs,  linewidth=1.0, linestyle=":",  color="steelblue",
            alpha=0.6, label="Recall")
    ax.axvline(best_tau, color="red", linestyle="--", linewidth=0.8,
               label=f"tau={best_tau:.0f}px F1={best_f1:.3f}")
    ax.set_xlabel("tau_d (px)")
    ax.set_ylabel("Score")
    ax.set_title(f"Layer-2: F1/P/R vs tau_d  {src_cam}->{ref_cam}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"val2_f1_{tag}.png"), dpi=120)
    plt.close(fig)

    print(f"    [L2] best tau={best_tau:.1f}px  F1={best_f1:.4f}"
          f"  P={best_prec:.4f}  R={best_rec:.4f}")

    return {"best_tau": best_tau, "best_f1": best_f1,
            "best_prec": best_prec, "best_rec": best_rec}


# ─────────────────────────── Summary plot ────────────────────────────────────

def plot_summary(l2_results, out_dir):
    """Bar chart of best-F1 for all pairs."""
    pairs  = list(l2_results.keys())
    f1s    = [l2_results[p]["best_f1"]  for p in pairs]
    taus   = [l2_results[p]["best_tau"] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(pairs, f1s, color="steelblue", alpha=0.8)
    for bar, tau in zip(bars, taus):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"tau={tau:.0f}px",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best F1")
    ax.set_title("Layer-2: Best F1 per Camera Pair  (val frames 601-1800)")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="F1=0.5")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(out_dir, "val2_summary.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n[Summary] Saved -> {path}")


# ─────────────────────────── Multi-camera clustering ─────────────────────────

class _UnionFind:
    """Union-find that also tracks the set of cameras in each component,
    refusing merges that would put two detections from the same camera into
    one cluster (条件 A)."""

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        # Each root keeps a frozen-ish set of cameras present in its component
        self.cams = {n: {n[0]} for n in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def try_union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 条件 A: 同一 camera 不能在同一 cluster 中重复
        if self.cams[ra] & self.cams[rb]:
            return False
        # union (smaller -> larger)
        if len(self.cams[ra]) < len(self.cams[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.cams[ra] = self.cams[ra] | self.cams[rb]
        return True

    def components(self):
        comps = defaultdict(list)
        for n in self.parent:
            comps[self.find(n)].append(n)
        return list(comps.values())


def multi_camera_clustering(gt, pair_data, frame_range, out_dir,
                            min_f1=0.5, score_floor=0.0, debug_frames=None):
    """
    Multi-camera clustering on the validation frames.

    Step 1: each detection (cam, frame, oid) is a node.
    Step 2: edges built only on overlap-filtered camera pairs that pass
            min_f1 (条件 B - 只接受强 pair).
    Step 3: greedy edge sorting + constrained union-find (条件 A).
    Output : connected components per frame -> clusters.

    pair_data : { (src, ref): {"H","hull_src","hull_ref","margin",
                                "tau","f1"} }
    """
    print(f"\n{'='*60}\n  Multi-camera clustering "
          f"(min_f1={min_f1}, score_floor={score_floor})\n{'='*60}")

    # Pairs that pass the strong-edge filter
    strong_pairs = [p for p, d in pair_data.items()
                    if d.get("f1", 0.0) >= min_f1 and d.get("tau", 0.0) > 0]
    skipped = [(p, pair_data[p].get("f1", 0.0)) for p in pair_data
               if p not in strong_pairs]
    sp_str = ", ".join(f"{a}-{b}(F1={pair_data[(a,b)]['f1']:.2f})"
                       for a, b in strong_pairs)
    print(f"  Strong pairs ({len(strong_pairs)}): {sp_str}")
    if skipped:
        sk_str = ", ".join(f"{a}-{b}(F1={f:.2f})" for (a, b), f in skipped)
        print(f"  Skipped weak pairs: {sk_str}")

    f_lo, f_hi = frame_range
    all_frames = sorted({fid for cam in CAMERAS for fid in gt[cam]
                          if f_lo <= fid <= f_hi})

    # Aggregate stats across all val frames
    cluster_size_hist = defaultdict(int)   # k cameras -> count
    n_clusters_total = 0
    # Pairwise eval against GT (same object_id => same ground-truth cluster)
    tp = fp = fn = 0
    example_frames = []          # save a few for plotting

    for fid in all_frames:
        dbg = debug_frames and fid in debug_frames
        if dbg:
            print(f"\n  --- DEBUG frame {fid} ---")
        # Build node list (only detections that fall in at least one
        # overlap region that uses this camera as src or ref).
        nodes = []
        node_pos = {}      # (cam, fid, oid) -> (x, y)
        for cam in CAMERAS:
            if fid not in gt[cam]:
                continue
            for oid, (cx, by, _) in gt[cam][fid].items():
                node = (cam, fid, oid)
                nodes.append(node)
                node_pos[node] = (cx, by)
        if len(nodes) < 2:
            continue
        if dbg:
            for n in nodes:
                print(f"    node {n}  pos={node_pos[n]}")

        # Build candidate edges from strong pairs
        edges = []  # (score, dist, u, v)
        for src, ref in strong_pairs:
            pdat   = pair_data[(src, ref)]
            H      = pdat["H"]
            hull_s = pdat["hull_src"]
            hull_r = pdat["hull_ref"]
            margin = pdat["margin"]
            tau    = pdat["tau"]
            if fid not in gt[src] or fid not in gt[ref]:
                if dbg:
                    print(f"    pair {src}->{ref}: skip (frame missing)")
                continue
            src_in = {o: in_overlap((cx, by), hull_s, margin)
                      for o, (cx, by, _) in gt[src][fid].items()}
            ref_in = {o: in_overlap((cx, by), hull_r, margin)
                      for o, (cx, by, _) in gt[ref][fid].items()}
            if dbg:
                print(f"    pair {src}->{ref}  tau={tau:.2f}  "
                      f"hull_in[{src}]={src_in}  hull_in[{ref}]={ref_in}")
            src_oids = [o for o, ok in src_in.items() if ok]
            ref_oids = [o for o, ok in ref_in.items() if ok]
            if not src_oids or not ref_oids:
                continue
            for so in src_oids:
                cx_s, by_s, _ = gt[src][fid][so]
                proj = project_points(H, np.array([[cx_s, by_s]],
                                                  dtype=np.float32))[0]
                for ro in ref_oids:
                    cx_r, by_r, _ = gt[ref][fid][ro]
                    d = float(np.linalg.norm(proj - np.array([cx_r, by_r])))
                    accepted = d < tau
                    s_uv = 1.0 - d / tau if tau > 0 else 0.0
                    if accepted and s_uv < score_floor:
                        accepted = False
                    if dbg:
                        print(f"      edge {src}:{so} -> {ref}:{ro}  "
                              f"d={d:.2f}  s={s_uv:.3f}  "
                              f"accepted={accepted}")
                    if not accepted:
                        continue
                    edges.append((s_uv, d, (src, fid, so), (ref, fid, ro)))

        # Greedy: sort edges by score descending, attempt union with constraint
        edges.sort(key=lambda x: -x[0])
        uf = _UnionFind(nodes)
        for s_uv, d, u, v in edges:
            merged = uf.try_union(u, v)
            if dbg:
                print(f"    union {u} <-> {v}  s={s_uv:.3f}  "
                      f"merged={merged}")

        comps = uf.components()
        # Restrict to multi-cam clusters for the size histogram
        for c in comps:
            cams_in = {n[0] for n in c}
            cluster_size_hist[len(cams_in)] += 1
        n_clusters_total += len(comps)

        # Pairwise eval: predicted positives = node-pairs in same predicted cluster
        # GT positives = node-pairs with same object_id (cross-cam)
        # Iterate over all node pairs in this frame
        node_to_comp = {}
        for ci, c in enumerate(comps):
            for n in c:
                node_to_comp[n] = ci
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if u[0] == v[0]:
                    continue   # ignore same-camera pairs
                same_pred = node_to_comp[u] == node_to_comp[v]
                same_gt   = u[2] == v[2]
                if same_pred and same_gt:
                    tp += 1
                elif same_pred and not same_gt:
                    fp += 1
                elif not same_pred and same_gt:
                    fn += 1

        # Save up to 3 example frames with multi-cam clusters for visualisation
        if (len(example_frames) < 3
                and any(len({n[0] for n in c}) >= 2 for c in comps)):
            example_frames.append((fid, comps, edges, node_pos))

    # ─── Stats / plots ────────────────────────────────────────────────────
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  Frames processed     : {len(all_frames)}")
    print(f"  Clusters total       : {n_clusters_total}")
    print(f"  Cluster-size hist    : "
          f"{dict(sorted(cluster_size_hist.items()))}")
    print(f"  Pairwise GT eval     : "
          f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
          f"(TP={tp}, FP={fp}, FN={fn})")

    # Bar: cluster size distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = sorted(cluster_size_hist.keys())
    counts = [cluster_size_hist[s] for s in sizes]
    bars = ax.bar([str(s) for s in sizes], counts, color="steelblue")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("# cameras per cluster")
    ax.set_ylabel("# clusters")
    ax.set_title(f"Multi-camera clustering size distribution\n"
                 f"frames {f_lo}-{f_hi}, P={prec:.3f}/R={rec:.3f}/F1={f1:.3f}")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "cluster_size_dist.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"  Saved -> {p1}")

    # Example frame visualisations (graph layout in image space, one panel per camera)
    for ex_idx, (fid, comps, edges, node_pos) in enumerate(example_frames):
        # Map cluster id -> color via tab20
        cmap = plt.cm.tab20
        node_to_comp = {}
        for ci, c in enumerate(comps):
            for n in c:
                node_to_comp[n] = ci

        fig, axes = plt.subplots(1, len(CAMERAS),
                                  figsize=(2.6 * len(CAMERAS), 2.2),
                                  sharey=True)
        for ax, cam in zip(axes, CAMERAS):
            ax.set_xlim(0, 640); ax.set_ylim(360, 0)
            ax.set_aspect("equal")
            ax.set_xlabel(cam, fontsize=10)
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            for n, (x, y) in node_pos.items():
                if n[0] != cam:
                    continue
                color = cmap(node_to_comp[n] % 20)
                ax.scatter(x, y, c=[color], s=40,
                           edgecolors="black", linewidths=0.5)
                ax.text(x + 4, y - 4, str(n[2]), fontsize=6)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.04)

        p2 = os.path.join(out_dir, f"cluster_example_frame{fid}.png")
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        print(f"  Saved -> {p2}")

    return {
        "n_frames": len(all_frames),
        "n_clusters": n_clusters_total,
        "size_hist": dict(cluster_size_hist),
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
#
#  Current baseline: Pairwise camera matching with global homography per pair
#
#  For each camera pair (src, ref):
#    1. Collect same-ID GT correspondences (bottom-center points) from frames 1-600
#    2. Fit global H_{src→ref} using RANSAC
#    3. Compute empirical overlap region (convex hull of co-visible detections)
#    4. Validate on frames 601-1800:
#         - Layer 1: Reprojection error (3 bottom-edge points: left/center/right)
#                    CDF plots for all pairs vs overlap-only pairs
#         - Layer 2: d_same vs d_diff separability (3 points separately)
#                    Histogram + F1/P/R vs tau_d curves → best tau & F1 per pair
#    5. Save per-pair plots + H matrices
#
#  Outputs:
#    - homographies.npz           : All H_{src}_to_{ref} matrices
#    - overlap_{src}_{ref}.png    : Empirical overlap hulls (both camera spaces)
#    - val1_cdf_{src}_{ref}.png   : Layer-1 reprojection CDF
#    - val2_hist_{src}_{ref}.png  : Layer-2 d_same vs d_diff histogram
#    - val2_f1_{src}_{ref}.png    : Layer-2 F1/P/R vs tau_d
#    - val2_summary.png           : Best-F1 bar chart across all pairs
#    - homography_report.txt      : Text summary of all metrics
#
#  Usage:
#    python compute_homography.py --output_dir <dir> [--cameras c002,c003,c004]
#
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────── Main ────────────────────────────────────────────

def main(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading GT files ...")
    gt = {cam: load_gt(cam) for cam in CAMERAS}

    all_H      = {}   # key: (src, ref) tuple -> 3x3 ndarray
    pair_data  = {}   # key: (src, ref) -> dict for clustering
    report_lines = []
    l2_results   = {}

    for src_cam, ref_cam in PAIRS:
        pair_tag = f"{src_cam}->{ref_cam}"
        print(f"\n{'='*60}")
        print(f"  {pair_tag}")
        print(f"{'='*60}")

        # ── Calibration ──────────────────────────────────────────────────

        # --- Debug: 输出详细对应点信息 ---
        src_cal, ref_cal = collect_correspondences(
            gt[src_cam], gt[ref_cam], CALIB_FRAMES)
        print(f"[Debug] src_cam: {src_cam}, ref_cam: {ref_cam}")
        print(f"[Debug] Total pairs found: {len(src_cal)}")

        # f_lo, f_hi = CALIB_FRAMES
        # for fid in sorted(set(gt[src_cam]) & set(gt[ref_cam])):
        #     if not (f_lo <= fid <= f_hi):
        #         continue
        #     shared = set(gt[src_cam][fid]) & set(gt[ref_cam][fid])
        #     if shared:
        #         obj_list = ', '.join(str(oid) for oid in sorted(shared))
        #         print(f"[Debug] Frame {fid}: object_ids = [{obj_list}]")

        if len(src_cal) < 4:
            msg = (f"  *** SKIP {pair_tag}: only {len(src_cal)} calib pairs "
                   f"(need >= 4) ***")
            print(msg); report_lines.append(msg)
            continue

        H, mask = fit_homography(src_cal, ref_cal,
                                  ransac_thresh=args.ransac_thresh)
        n_inliers = int(mask.sum())
        pct = 100 * n_inliers / len(src_cal)
        print(f"  RANSAC inliers: {n_inliers}/{len(src_cal)} ({pct:.1f}%)")

        errs_cal = reproj_errors(H, src_cal[mask.ravel() == 1],
                                     ref_cal[mask.ravel() == 1])
        mean_cal = float(np.mean(errs_cal))   if len(errs_cal) else float("nan")
        med_cal  = float(np.median(errs_cal)) if len(errs_cal) else float("nan")
        print(f"  Calib reproj: mean={mean_cal:.2f}  median={med_cal:.2f} px (inliers)")

        if args.viz_pair and tuple(sorted(args.viz_pair.split(","))) == tuple(sorted((src_cam, ref_cam))):
            inlier_src = src_cal[mask.ravel() == 1]
            inlier_ref = ref_cal[mask.ravel() == 1]
            rep_src, rep_ref, rep_idx = select_representative_four_points(inlier_src, inlier_ref)
            note = ("4 representative correspondences selected from final RANSAC "
                    "inliers (not the hidden OpenCV minimal sample)")
            visualize_representative_four_points(
                src_cam,
                ref_cam,
                rep_src,
                rep_ref,
                out_dir,
                pair_tag=f"inlier_idx={rep_idx}",
                note=note,
            )

        # ── Overlap region (all 1800 frames, GT-based, no H needed) ──────
        hull_src, hull_ref, n_covis = compute_overlap_region(
            gt[src_cam], gt[ref_cam], margin=args.overlap_margin)
        print(f"  Overlap region: {n_covis} co-vis pts, "
              f"hull_src={'OK' if hull_src is not None else 'None'}, "
              f"hull_ref={'OK' if hull_ref is not None else 'None'}")
        visualize_overlap(hull_src, hull_ref, gt[src_cam], gt[ref_cam],
                          src_cam, ref_cam, out_dir, margin=args.overlap_margin)

        # ── Validation Layer-1 ───────────────────────────────────────────
        src_val, ref_val = collect_correspondences(
            gt[src_cam], gt[ref_cam], VAL_FRAMES)
        print(f"  Val pairs ({VAL_FRAMES[0]}-{VAL_FRAMES[1]}): {len(src_val)}")
        l1 = validate_layer1_pair(H, gt[src_cam], gt[ref_cam],
                                   src_cam, ref_cam, out_dir,
                                   hull_src=hull_src, hull_ref=hull_ref,
                                   margin=args.overlap_margin)

        # ── Validation Layer-2 ───────────────────────────────────────────
        l2 = validate_layer2_pair(H, gt[src_cam], gt[ref_cam],
                                   src_cam, ref_cam, out_dir,
                                   tau_max=args.tau_max, tau_steps=args.tau_steps,
                                   hull_src=hull_src, hull_ref=hull_ref,
                                   margin=args.overlap_margin)
        if l2:
            l2_results[f"{src_cam}->{ref_cam}"] = l2

        all_H[(src_cam, ref_cam)] = H

        # Save everything we need for multi-camera clustering on this pair.
        # Use the per-pair center-point tau as the strong-edge threshold.
        tau_pair = float(l2["best_tau"]) if l2 else 0.0
        f1_pair  = float(l2["best_f1"])  if l2 else 0.0
        pair_data[(src_cam, ref_cam)] = {
            "H": H,
            "hull_src": hull_src,
            "hull_ref": hull_ref,
            "margin": args.overlap_margin,
            "tau": tau_pair,
            "f1": f1_pair,
        }

        report_lines += [
            f"\n{pair_tag}",
            f"  Calib pairs              : {len(src_cal)}",
            f"  RANSAC inliers           : {n_inliers}/{len(src_cal)} ({pct:.1f}%)",
            f"  Calib reproj mean/median : {mean_cal:.3f} / {med_cal:.3f} px",
            f"  Val pairs                : {len(src_val)}",
        ]
        if l1:
            report_lines += [
                f"  [L1] mean   : {l1.get('all_mean', float('nan')):.3f} px",
                f"  [L1] median : {l1.get('all_median', float('nan')):.3f} px",
                f"  [L1] p90    : {l1.get('all_p90', float('nan')):.3f} px",
                f"  [L1] p95    : {l1.get('all_p95', float('nan')):.3f} px",
            ]
            if 'ovlp_mean' in l1:
                report_lines += [
                    f"  [L1/ovlp] mean   : {l1['ovlp_mean']:.3f} px",
                    f"  [L1/ovlp] median : {l1['ovlp_median']:.3f} px",
                    f"  [L1/ovlp] p90    : {l1['ovlp_p90']:.3f} px",
                    f"  [L1/ovlp] p95    : {l1['ovlp_p95']:.3f} px",
                ]
        if l2:
            report_lines += [
                f"  [L2] best tau_d : {l2['best_tau']:.2f} px",
                f"  [L2] best F1    : {l2['best_f1']:.4f}",
                f"  [L2] Precision  : {l2['best_prec']:.4f}",
                f"  [L2] Recall     : {l2['best_rec']:.4f}",
            ]
        report_lines.append(f"  H matrix:\n{H}")

    # ── Save homographies ─────────────────────────────────────────────────
    npz_path  = os.path.join(out_dir, "homographies.npz")
    save_dict = {f"H_{s}_to_{r}": H for (s, r), H in all_H.items()}
    np.savez(npz_path, **save_dict)
    print(f"\nSaved homographies -> {npz_path}")
    print("  Keys:", list(save_dict.keys()))

    # ── Summary plot ──────────────────────────────────────────────────────
    if l2_results:
        plot_summary(l2_results, out_dir)

    # ── Multi-camera clustering on validation frames ──────────────────────
    cluster_stats = None
    if pair_data:
        cluster_stats = multi_camera_clustering(
            gt, pair_data, VAL_FRAMES, out_dir,
            min_f1=args.cluster_min_f1,
            score_floor=args.cluster_score_floor,
            debug_frames=set(args.debug_frame) if args.debug_frame else None,
        )

    if args.artifact_path:
        calibrations = []
        for (src_cam, ref_cam), pdata in pair_data.items():
            calibrations.append(
                PairCalibration(
                    src_cam=src_cam,
                    ref_cam=ref_cam,
                    H=np.asarray(pdata["H"], dtype=np.float32),
                    hull_src=pdata["hull_src"],
                    hull_ref=pdata["hull_ref"],
                    tau=float(pdata["tau"]),
                    pair_f1=float(pdata["f1"]),
                    margin=float(pdata["margin"]),
                )
            )
        save_homography_artifact(
            args.artifact_path,
            calibrations,
            cameras=CAMERAS,
            metadata={
                "gt_dir": GT_DIR,
                "calibration_frames": CALIB_FRAMES,
                "validation_frames": VAL_FRAMES,
                "ransac_thresh": args.ransac_thresh,
            },
        )
        print(f"Saved homography artifact -> {args.artifact_path}")

    # ── Report ────────────────────────────────────────────────────────────
    rpt_path = os.path.join(out_dir, "homography_report.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join(report_lines))
        if cluster_stats is not None:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("Multi-camera clustering\n")
            f.write("=" * 60 + "\n")
            for k, v in cluster_stats.items():
                f.write(f"  {k:12s} : {v}\n")
    print(f"Saved report -> {rpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairwise homography computation & validation for 4 cameras")
    parser.add_argument("--output_dir",    default="homography_output")
    parser.add_argument("--ransac_thresh", type=float, default=3.0,
                        help="RANSAC reprojection threshold (px)")
    parser.add_argument("--tau_max",        type=float, default=200.0,
                        help="Max tau_d for Layer-2 F1 sweep (px)")
    parser.add_argument("--tau_steps",      type=int,   default=200,
                        help="Number of tau steps for Layer-2 sweep")
    parser.add_argument("--overlap_margin", type=float, default=20.0,
                        help="Dilation margin for overlap hull (px, default 20)")
    parser.add_argument("--cluster_min_f1", type=float, default=0.5,
                        help="Minimum per-pair Layer-2 F1 (center) for an "
                             "edge to be considered (条件 B). Default 0.5.")
    parser.add_argument("--cluster_score_floor", type=float, default=0.0,
                        help="Drop edges whose normalised score "
                             "s = 1 - d/tau is below this floor. Default 0.")
    parser.add_argument("--cameras", default=None,
                        help="Comma-separated subset of cameras to use, e.g. "
                             "'c002,c003,c004'. Defaults to all four.")
    parser.add_argument("--debug_frame", type=int, nargs="*", default=None,
                        help="Frames for which to log clustering details.")
    parser.add_argument(
        "--artifact_path",
        default=None,
        help="Optional JSON artifact path for runtime homography loading.",
    )
    parser.add_argument(
        "--viz_pair",
        default=None,
        help="Optional pair to export representative 4-point calibration visualization, e.g. c001,c002",
    )
    args = parser.parse_args()

    if args.cameras:
        CAMERAS = [c.strip() for c in args.cameras.split(",") if c.strip()]
        PAIRS = [(a, b) for a, b in combinations(CAMERAS, 2)]
        print(f"Using camera subset: {CAMERAS}")
        print(f"Pairs: {PAIRS}")
    main(args)
