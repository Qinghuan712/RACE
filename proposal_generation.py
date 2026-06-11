"""
Multi-camera moving-object localization via MOG2 background subtraction.

Pipeline (per camera, run independently in lock-step):
    1. Multi-pass background learning:
        - Pass 1 uses a fast learning rate to bootstrap the background model.
        - Subsequent passes use a slow learning rate to refine the model.
       Detections are only emitted on the final pass.
    2. Per-frame foreground mask post-processing:
        - Drop shadow pixels (MOG2 marks them as 127).
        - Morphological CLOSE then OPEN with elliptical kernels.
    3. Connected-component / contour extraction with area + aspect-ratio filters.

For each frame the four camera views and their corresponding foreground masks
are stitched into a single hybrid panel and written to disk:

    output_dir/
        frames/   mosaic_frame_<idx>.png   (4 BGR views, 2x2)
        masks/    mosaic_mask_<idx>.png    (4 binary masks, 2x2)
        hybrid/   hybrid_<idx>.png         (frames | masks)
"""

import argparse
import os
import sys

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RACE.artifacts import save_proposal_artifact


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "bgsub_method": "mog2",
    "history": 500,
    "var_threshold": 60,
    "detect_shadows": True,
    "use_grayscale": False,
    "absdiff_threshold": 25,
    "absdiff_alpha": 0.001,
    # Single-pass schedule: the first `warmup_frames` frames build the
    # background model with `warmup_learning_rate`; remaining frames run at
    # `detection_learning_rate` and are written to disk.
    "warmup_frames": 300,
    "warmup_learning_rate": 0.01,
    "detection_learning_rate": 0.001,
    "kernel_close_size": 7,
    "kernel_open_size": 3,
    "min_area": 200,
    "min_aspect_ratio": 0.2,
    "max_aspect_ratio": 5.0,
    # Box consolidation: merge boxes that very likely belong to the same
    # object. A pair is merged when EITHER the IoU exceeds `merge_iou` OR
    # all three geometric conditions hold simultaneously:
    #   - horizontal gap (signed; negative means horizontal overlap) <= merge_gap_x
    #   - vertical overlap ratio (relative to the smaller box height) >= merge_voverlap
    #   - bottom-edge y difference <= merge_bottom_dy
    "merge_iou": 0.1,
    "merge_gap_x": 12,
    "merge_voverlap": 0.7,
    "merge_bottom_dy": 12,
    # Merged-blob split: try to break a wide blob into two side-by-side
    # vehicles using the horizontal foreground projection.
    # GT-fitted priors take precedence when available; otherwise the
    # geometric trigger thresholds below are used as a fallback.
    "gt_dir": "./dataset_preprocessing/aligned_gt_640",
    "prior_bin_size": 20,
    "prior_min_samples_per_bin": 5,
    "prior_use_q90": False,             # if True, use Q90 instead of gamma*median
    "prior_gamma_w": 1.6,               # w_b > gamma_w * median_w(y_b)
    "prior_gamma_h": 1.8,               # h_b > gamma_h * median_h(y_b)
    "prior_gamma_a": 2.0,               # a_b > gamma_a * median_a(y_b)
    "split_trigger_aspect": 1.8,        # geometric fallback: w/h or h/w
    "split_trigger_width": 100,         # geometric fallback: w or h (px)
    "split_smooth_sigma": 1.5,          # Gaussian smoothing on projection
    "split_min_peak_dist_frac": 0.25,   # min peak separation fraction
    "split_min_peak_height_ratio": 0.5, # 2nd peak must reach this * 1st peak
    "split_valley_beta": 0.6,           # valley must be < beta * min(peak1, peak2)
    # Debug visualization: dump per-step images for every triggered split.
    "split_debug": False,               # master switch
    "split_debug_max_per_frame": 4,     # cap per (frame, cam) to avoid spam
    "split_debug_paper_mode": False,    # clean horizontal 3-panel layout (no text)
    "paper_axes": True,                 # draw L-shaped axes on projection panel
    "paper_axis_ticks": False,          # short tick marks on axes
}


# ---------------------------------------------------------------------------
# Single-vehicle priors fitted from ground-truth tracks
# ---------------------------------------------------------------------------

def _load_gt_priors(gt_dir, image_height, bin_size, min_samples_per_bin=5):
    """Fit per-camera, per-y-bin single-vehicle size priors from GT files.

    GT format (each line):
        frame_id, object_id, x, y, w, h, 1, -1, -1, -1
    File names: c00<idx>_aligned_gt.txt, where <idx> is 1..N.

    For every camera we partition the image height H into bins of size
    `bin_size`. For each bin we collect the w, h, a=w*h of all GT boxes
    whose bottom-edge y2=y+h falls in that bin, then compute the median
    and the 90% quantile.

    Returns
    -------
    priors : dict[int, dict]
        Mapping cam_idx (0-based, matches the order of the camera videos)
        to a dict with numpy arrays of length n_bins:
            'bin_size'       : int
            'median_w', 'q90_w'
            'median_a', 'q90_a'
            'has_data'       : bool array (False -> bin too sparse)
        Bins without enough samples are filled in by nearest-neighbour
        interpolation along y so every bin has a usable value.
    """
    if not gt_dir or not os.path.isdir(gt_dir):
        return {}

    n_bins = int(np.ceil(image_height / bin_size))
    priors = {}

    files = sorted(
        f for f in os.listdir(gt_dir)
        if f.startswith("c00") and f.endswith("_aligned_gt.txt")
    )
    for fname in files:
        # Map file name to 0-based camera index aligned with video listing.
        try:
            cam_idx = int(fname[3]) - 1
        except ValueError:
            continue

        path = os.path.join(gt_dir, fname)
        ws_per_bin = [[] for _ in range(n_bins)]
        hs_per_bin = [[] for _ in range(n_bins)]
        as_per_bin = [[] for _ in range(n_bins)]
        with open(path) as fp:
            for line in fp:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    x = float(parts[2]); y = float(parts[3])
                    w = float(parts[4]); h = float(parts[5])
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    continue
                y2 = y + h
                bi = int(min(n_bins - 1, max(0, y2 // bin_size)))
                ws_per_bin[bi].append(w)
                hs_per_bin[bi].append(h)
                as_per_bin[bi].append(w * h)

        median_w = np.full(n_bins, np.nan, dtype=np.float32)
        q90_w    = np.full(n_bins, np.nan, dtype=np.float32)
        median_h = np.full(n_bins, np.nan, dtype=np.float32)
        q90_h    = np.full(n_bins, np.nan, dtype=np.float32)
        median_a = np.full(n_bins, np.nan, dtype=np.float32)
        q90_a    = np.full(n_bins, np.nan, dtype=np.float32)
        has_data = np.zeros(n_bins, dtype=bool)

        for bi in range(n_bins):
            if len(ws_per_bin[bi]) >= min_samples_per_bin:
                ws_arr = np.asarray(ws_per_bin[bi], dtype=np.float32)
                hs_arr = np.asarray(hs_per_bin[bi], dtype=np.float32)
                as_arr = np.asarray(as_per_bin[bi], dtype=np.float32)
                median_w[bi] = float(np.median(ws_arr))
                q90_w[bi]    = float(np.quantile(ws_arr, 0.9))
                median_h[bi] = float(np.median(hs_arr))
                q90_h[bi]    = float(np.quantile(hs_arr, 0.9))
                median_a[bi] = float(np.median(as_arr))
                q90_a[bi]    = float(np.quantile(as_arr, 0.9))
                has_data[bi] = True

        if not has_data.any():
            continue

        # Nearest-neighbour interpolation along y to fill empty bins.
        idx_full = np.where(has_data)[0]
        for arr in (median_w, q90_w, median_h, q90_h, median_a, q90_a):
            for bi in range(n_bins):
                if np.isnan(arr[bi]):
                    nearest = idx_full[np.argmin(np.abs(idx_full - bi))]
                    arr[bi] = arr[nearest]

        priors[cam_idx] = {
            "bin_size": bin_size,
            "median_w": median_w,
            "q90_w":    q90_w,
            "median_h": median_h,
            "q90_h":    q90_h,
            "median_a": median_a,
            "q90_a":    q90_a,
            "has_data": has_data,
            "n_samples": int(sum(len(b) for b in ws_per_bin)),
        }

    return priors


# ---------------------------------------------------------------------------
# Box consolidation
# ---------------------------------------------------------------------------

def _consolidate_boxes(boxes, cfg):
    """Merge boxes that likely belong to the same object.

    Two boxes are merged into their union when either condition holds:
        - IoU(b1, b2) > merge_iou
        - horizontal_gap < merge_gap_x AND
          vertical_overlap_ratio > merge_voverlap AND
          |bottom_y1 - bottom_y2| < merge_bottom_dy

    The procedure iterates until no more merges happen, so transitive
    chains (A-B-C) collapse to a single union.
    """
    if len(boxes) <= 1:
        return list(boxes)

    iou_t = cfg["merge_iou"]
    gap_t = cfg["merge_gap_x"]
    vov_t = cfg["merge_voverlap"]
    by_t = cfg["merge_bottom_dy"]

    def _pair_should_merge(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        # IoU
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        iou = inter / union if union > 0 else 0.0
        if iou > iou_t:
            return True

        # Horizontal gap (negative = overlap)
        gap_x = max(ax1, bx1) - min(ax2, bx2)
        # Vertical overlap relative to the smaller box height
        v_inter = max(0, min(ay2, by2) - max(ay1, by1))
        v_ratio = v_inter / max(1, min(ah, bh))
        bottom_dy = abs(ay2 - by2)
        if gap_x < gap_t and v_ratio > vov_t and bottom_dy < by_t:
            return True
        return False

    def _union(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        x2 = max(ax1 + aw, bx1 + bw)
        y2 = max(ay1 + ah, by1 + bh)
        return (x1, y1, x2 - x1, y2 - y1)

    cur = list(boxes)
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(cur)
        for i in range(len(cur)):
            if used[i]:
                continue
            box = cur[i]
            for j in range(i + 1, len(cur)):
                if used[j]:
                    continue
                if _pair_should_merge(box, cur[j]):
                    box = _union(box, cur[j])
                    used[j] = True
                    changed = True
            merged.append(box)
            used[i] = True
        cur = merged
    return cur


# ---------------------------------------------------------------------------
# Merged-blob bidirectional split (side-by-side OR front/back vehicles)
# ---------------------------------------------------------------------------

def _check_trigger(box, cfg, prior, axis):
    """Return (triggered, reason). axis 'h' uses w/median_w, 'v' uses h/median_h."""
    x, y, w, h = box
    if prior is not None:
        bin_size = prior["bin_size"]
        bi = int(min(prior["median_w"].shape[0] - 1,
                     max(0, (y + h) // bin_size)))
        a_b = float(w * h)
        if cfg["prior_use_q90"]:
            mw_ref = float(prior["q90_w"][bi])
            mh_ref = float(prior["q90_h"][bi])
            ma_ref = float(prior["q90_a"][bi])
        else:
            mw_ref = float(cfg["prior_gamma_w"]) * float(prior["median_w"][bi])
            mh_ref = float(cfg["prior_gamma_h"]) * float(prior["median_h"][bi])
            ma_ref = float(cfg["prior_gamma_a"]) * float(prior["median_a"][bi])
        if axis == "h":
            if w > mw_ref:
                return True, f"bi={bi} w={w}>{mw_ref:.0f}"
            if a_b > ma_ref:
                return True, f"bi={bi} a={a_b:.0f}>{ma_ref:.0f}"
        else:  # 'v'
            if h > mh_ref:
                return True, f"bi={bi} h={h}>{mh_ref:.0f}"
            if a_b > ma_ref:
                return True, f"bi={bi} a={a_b:.0f}>{ma_ref:.0f}"
        return False, ""
    # Geometric fallback.
    if axis == "h":
        ar = w / float(h) if h > 0 else 0.0
        if ar >= cfg["split_trigger_aspect"] or w >= cfg["split_trigger_width"]:
            return True, f"geom w={w} ar={ar:.2f}"
    else:
        ar = h / float(w) if w > 0 else 0.0
        if ar >= cfg["split_trigger_aspect"] or h >= cfg["split_trigger_width"]:
            return True, f"geom h={h} ar={ar:.2f}"
    return False, ""


def _try_split_axis(box, fg_mask, cfg, axis, prior=None, stats=None,
                    debug=None):
    """Try to split `box` along `axis` ('h' = vertical cut, left/right children;
    'v' = horizontal cut, top/bottom children). Returns either [box] or 2 boxes.

    `debug`, if given, is a dict that will be filled with intermediate
    artefacts (sub-mask, projection, peaks, valley) for visualization.
    """
    x, y, w, h = box
    prefix = "h_" if axis == "h" else "v_"

    def _bump(key):
        if stats is not None:
            stats[prefix + key] = stats.get(prefix + key, 0) + 1

    _bump("seen")

    # ---- Step 1: trigger ------------------------------------------------
    triggered, reason = _check_trigger(box, cfg, prior, axis)
    if not triggered:
        return [box]
    _bump("triggered")
    if stats is not None:
        stats["_last_trigger_" + axis] = reason
    if debug is not None:
        debug["axis"] = axis
        debug["box"] = box
        debug["trigger"] = reason

    # ---- Step 2: crop sub-mask -----------------------------------------
    H_img, W_img = fg_mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W_img, x + w), min(H_img, y + h)
    if x2 - x1 < 8 or y2 - y1 < 4:
        _bump("abort_too_small")
        return [box]
    sub = fg_mask[y1:y2, x1:x2]
    if debug is not None:
        debug["sub"] = sub.copy()

    # Project: axis='h' -> column sum (split column-wise);
    #          axis='v' -> row sum    (split row-wise).
    if axis == "h":
        proj = (sub > 0).sum(axis=0).astype(np.float32)
    else:
        proj = (sub > 0).sum(axis=1).astype(np.float32)

    sigma = float(cfg["split_smooth_sigma"])
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        if ksize < 3:
            ksize = 3
        proj = cv2.GaussianBlur(proj.reshape(1, -1), (ksize, 1), sigma).flatten()

    n = len(proj)
    if n < 8:
        _bump("abort_too_small")
        if debug is not None:
            debug["proj"] = proj
            debug["abort"] = "too_small"
        return [box]

    # ---- Step 3: two peaks ---------------------------------------------
    min_dist = max(4, int(cfg["split_min_peak_dist_frac"] * n))
    p1 = int(np.argmax(proj))
    if proj[p1] <= 0:
        _bump("abort_no_peak")
        if debug is not None:
            debug["proj"] = proj; debug["abort"] = "no_peak"
        return [box]
    masked = proj.copy()
    masked[max(0, p1 - min_dist): min(n, p1 + min_dist + 1)] = -1.0
    p2 = int(np.argmax(masked))
    if masked[p2] <= 0:
        _bump("abort_no_peak")
        if debug is not None:
            debug["proj"] = proj; debug["peaks"] = (p1, None); debug["abort"] = "no_peak"
        return [box]
    if proj[p2] < cfg["split_min_peak_height_ratio"] * proj[p1]:
        _bump("abort_peak_height")
        if debug is not None:
            debug["proj"] = proj; debug["peaks"] = (p1, p2); debug["abort"] = "peak_height"
        return [box]

    u1, u2 = sorted([p1, p2])
    if u2 - u1 < 4:
        _bump("abort_peaks_close")
        if debug is not None:
            debug["proj"] = proj; debug["peaks"] = (u1, u2); debug["abort"] = "peaks_close"
        return [box]

    # ---- Step 4: valley -------------------------------------------------
    valley = u1 + int(np.argmin(proj[u1:u2 + 1]))
    valley_ratio = float(proj[valley]) / max(1e-6, float(min(proj[u1], proj[u2])))
    if valley_ratio >= cfg["split_valley_beta"]:
        _bump("abort_valley_shallow")
        if debug is not None:
            debug["proj"] = proj; debug["peaks"] = (u1, u2)
            debug["valley"] = valley; debug["valley_ratio"] = valley_ratio
            debug["abort"] = "valley_shallow"
        return [box]

    # ---- Step 5: cut mask along axis -----------------------------------
    if axis == "h":
        sides = ((sub[:, :valley], 0, "x"), (sub[:, valley:], valley, "x"))
    else:
        sides = ((sub[:valley, :], 0, "y"), (sub[valley:, :], valley, "y"))

    out = []
    for sm, off, kind in sides:
        ys, xs = np.where(sm > 0)
        if len(xs) < max(20, cfg["min_area"] // 4):
            continue
        bx_min = int(xs.min()); bx_max = int(xs.max()) + 1
        by_min = int(ys.min()); by_max = int(ys.max()) + 1
        if kind == "x":
            bx_min += off; bx_max += off
        else:
            by_min += off; by_max += off
        out.append((x1 + bx_min, y1 + by_min,
                    bx_max - bx_min, by_max - by_min))

    if debug is not None:
        debug["proj"] = proj; debug["peaks"] = (u1, u2)
        debug["valley"] = valley; debug["valley_ratio"] = valley_ratio
        debug["children"] = list(out)

    if len(out) < 2:
        _bump("abort_empty_side")
        if debug is not None:
            debug["abort"] = "empty_side"
        return [box]
    _bump("success")
    if debug is not None:
        debug["abort"] = None
    return out


def _split_merged_blob(box, fg_mask, cfg, prior=None, stats=None,
                       allow=("h", "v"), debug_list=None):
    """Recursive 2-direction split: try the first allowed axis, then the other.

    The recursion depth is at most 2 because each recursive call removes the
    just-used axis from `allow`. Therefore every returned sub-box has been
    examined in BOTH directions at least once.

    Tags returned alongside boxes describe how each box was produced:
        'normal'  -> no split happened
        'h'/'hh'  -> produced by a horizontal-axis (left/right) split
        'v'/'vv'  -> produced by a vertical-axis (top/bottom) split
    Tag uppercase letters indicate the depth (first-level vs second-level).

    Returns: list of (box, tag).
    """
    if not allow:
        return [(box, "normal")]
    # Try the first allowed axis (preferred order: 'h' first if available).
    order = [a for a in ("h", "v") if a in allow]

    for axis in order:
        debug = {} if debug_list is not None else None
        children = _try_split_axis(box, fg_mask, cfg, axis,
                                   prior=prior, stats=stats, debug=debug)
        if debug is not None:
            debug_list.append(debug)
        if len(children) > 1:
            # Recurse on the OTHER axis only (avoid re-trying same axis).
            other = tuple(a for a in allow if a != axis)
            results = []
            for c in children:
                grand = _split_merged_blob(c, fg_mask, cfg, prior=prior,
                                           stats=stats, allow=other,
                                           debug_list=debug_list)
                # Re-tag: first-level produced these, propagate axis label.
                for gb, gtag in grand:
                    if gtag == "normal":
                        results.append((gb, axis))
                    else:
                        # already split again on the other axis
                        results.append((gb, axis + gtag))
            return results
    # Both axes failed (or only one allowed and it failed).
    return [(box, "normal")]


# ---------------------------------------------------------------------------
# Per-step debug visualization for the split pipeline
# ---------------------------------------------------------------------------

def _save_split_debug_paper(out_dir, frame_idx, cam_i, box_idx, debug, src_frame, cfg=None):
    """Paper-friendly variant: clean horizontal 3-panel layout, no text.

    Panels (left-to-right): before crop | projection plot | after crop with
    split children outlined. Background: white; only axis-relevant geometry
    (peaks=green, valley=red, child boxes=yellow) is drawn.
    """
    if debug is None or "axis" not in debug:
        return
    if debug.get("abort"):
        return  # only export successful splits for the paper
    os.makedirs(out_dir, exist_ok=True)

    axis = debug["axis"]
    box = debug["box"]
    bx, by, bw, bh = box
    H_img, W_img = src_frame.shape[:2]
    sx1, sy1 = max(0, bx), max(0, by)
    sx2, sy2 = min(W_img, bx + bw), min(H_img, by + bh)
    crop = src_frame[sy1:sy2, sx1:sx2].copy()
    if crop.size == 0:
        return

    # All panels share the same height; widths preserve aspect for crops, and
    # the projection panel is sized so the projection's x-axis maps directly
    # to the crop's split axis (h -> width, v -> height).
    panel_h = 220
    ch_, cw_ = crop.shape[:2]
    scale = panel_h / float(ch_)
    crop_panel_w = max(1, int(round(cw_ * scale)))

    crop_resized = cv2.resize(crop, (crop_panel_w, panel_h),
                              interpolation=cv2.INTER_CUBIC)

    # ---- Before panel: just the crop, no overlay ------------------------
    before = crop_resized.copy()

    # ---- After panel: crop + child rectangles (yellow) ------------------
    after = crop_resized.copy()
    children = debug.get("children") or []
    for cb in children:
        cx, cy, cwid, chgt = cb
        rx = int(round((cx - sx1) * scale))
        ry = int(round((cy - sy1) * scale))
        rw = max(1, int(round(cwid * scale)))
        rh = max(1, int(round(chgt * scale)))
        cv2.rectangle(after, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    # ---- Projection panel ----------------------------------------------
    proj = debug.get("proj")
    if proj is None:
        return
    n = len(proj)
    # Width of the projection panel: aligned to the corresponding image
    # dimension on a per-axis basis -> emphasizes the spatial correspondence
    # between projection valley and split column/row.
    if axis == "h":
        proj_panel_w = crop_panel_w
    else:
        proj_panel_w = max(160, int(round(panel_h * 0.9)))
    plot = np.full((panel_h, proj_panel_w, 3), 255, dtype=np.uint8)
    m = float(proj.max()) if proj.max() > 0 else 1.0

    draw_axes = bool((cfg or {}).get("paper_axes", True))
    draw_ticks = bool((cfg or {}).get("paper_axis_ticks", True))
    # Margins: leave room on the left and bottom for axes when enabled.
    if draw_axes:
        ml, mr, mt, mb = 10, 6, 6, 10  # left / right / top / bottom
    else:
        ml = mr = mt = mb = 6
    plot_w = proj_panel_w - ml - mr
    plot_h = panel_h - mt - mb
    if plot_w < 4 or plot_h < 4:
        ml = mr = mt = mb = 6
        plot_w = proj_panel_w - ml - mr
        plot_h = panel_h - mt - mb
        draw_axes = False

    def x_to_px(i):
        return ml + int(round(i * (plot_w - 1) / max(1, n - 1)))

    def y_to_px(val):
        return mt + plot_h - 1 - int(round(val / m * (plot_h - 1)))

    # Curve.
    pts = [(x_to_px(i), y_to_px(proj[i])) for i in range(n)]
    for i in range(n - 1):
        cv2.line(plot, pts[i], pts[i + 1], (60, 60, 60), 2, cv2.LINE_AA)

    # Peaks (green), valley (red): draw only inside the plot area.
    y_top = mt
    y_bot = mt + plot_h - 1
    peaks = debug.get("peaks") or []
    for p in peaks:
        if p is None:
            continue
        xp = x_to_px(p)
        cv2.line(plot, (xp, y_top), (xp, y_bot), (0, 180, 0), 2, cv2.LINE_AA)
    v = debug.get("valley")
    if v is not None:
        xv = x_to_px(v)
        cv2.line(plot, (xv, y_top), (xv, y_bot), (0, 0, 220), 2, cv2.LINE_AA)

    if draw_axes:
        ax_color = (0, 0, 0)
        # Y axis (left) and X axis (bottom)
        cv2.line(plot, (ml, mt), (ml, y_bot + 1), ax_color, 1, cv2.LINE_AA)
        cv2.line(plot, (ml, y_bot + 1), (ml + plot_w, y_bot + 1),
                 ax_color, 1, cv2.LINE_AA)
        if draw_ticks:
            tick = 4
            # X ticks: 0, mid, end
            for frac in (0.0, 0.5, 1.0):
                xt = ml + int(round(frac * (plot_w - 1)))
                cv2.line(plot, (xt, y_bot + 1), (xt, y_bot + 1 + tick),
                         ax_color, 1, cv2.LINE_AA)
            # Y ticks: 0, max
            for val_frac in (0.0, 1.0):
                yt = mt + plot_h - 1 - int(round(val_frac * (plot_h - 1)))
                cv2.line(plot, (ml - tick, yt), (ml, yt),
                         ax_color, 1, cv2.LINE_AA)

    # ---- Compose: horizontal stack with thin white separators ----------
    sep = 8
    total_w = before.shape[1] + plot.shape[1] + after.shape[1] + sep * 2
    canvas = np.full((panel_h, total_w, 3), 255, dtype=np.uint8)
    x0 = 0
    canvas[:, x0:x0 + before.shape[1]] = before
    x0 += before.shape[1] + sep
    canvas[:, x0:x0 + plot.shape[1]] = plot
    x0 += plot.shape[1] + sep
    canvas[:, x0:x0 + after.shape[1]] = after

    fname = (f"paper_f{frame_idx:05d}_cam{cam_i}_b{box_idx:02d}_"
             f"{axis}.png")
    cv2.imwrite(os.path.join(out_dir, fname), canvas)


def _save_split_debug(out_dir, frame_idx, cam_i, box_idx, debug, src_frame):
    """Render a 4-panel image showing crop / projection / peaks+valley / result."""
    if debug is None or "axis" not in debug:
        return
    os.makedirs(out_dir, exist_ok=True)
    axis = debug["axis"]
    box = debug["box"]
    bx, by, bw, bh = box
    # Panel size
    pad = 6
    panel_w = max(160, bw * 2)
    panel_h = max(120, bh * 2)
    canvas = np.full((panel_h * 2 + pad * 3, panel_w * 2 + pad * 3, 3),
                     32, dtype=np.uint8)

    def place(img, row, col):
        y0 = pad + row * (panel_h + pad)
        x0 = pad + col * (panel_w + pad)
        ih, iw = img.shape[:2]
        scale = min(panel_w / max(1, iw), panel_h / max(1, ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        canvas[y0:y0 + nh, x0:x0 + nw] = resized

    def label(text, row, col):
        y0 = pad + row * (panel_h + pad) + 14
        x0 = pad + col * (panel_w + pad) + 4
        cv2.putText(canvas, text, (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Panel 0: source crop with the candidate box overlay
    H_img, W_img = src_frame.shape[:2]
    sx1, sy1 = max(0, bx), max(0, by)
    sx2, sy2 = min(W_img, bx + bw), min(H_img, by + bh)
    crop = src_frame[sy1:sy2, sx1:sx2].copy()
    place(crop, 0, 0)
    label(f"crop ax={axis} {debug.get('trigger', '')}", 0, 0)

    # Panel 1: sub-mask
    sub = debug.get("sub")
    if sub is not None:
        place(sub, 0, 1)
        label(f"sub-mask {sub.shape[1]}x{sub.shape[0]}", 0, 1)

    # Panel 2: projection plot with peaks + valley
    proj = debug.get("proj")
    if proj is not None:
        plot = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
        n = len(proj)
        m = float(proj.max()) if proj.max() > 0 else 1.0
        for i in range(n - 1):
            x1p = int(i * (panel_w - 1) / max(1, n - 1))
            x2p = int((i + 1) * (panel_w - 1) / max(1, n - 1))
            y1p = panel_h - 1 - int(proj[i] / m * (panel_h - 4))
            y2p = panel_h - 1 - int(proj[i + 1] / m * (panel_h - 4))
            cv2.line(plot, (x1p, y1p), (x2p, y2p), (200, 200, 200), 1)
        peaks = debug.get("peaks")
        if peaks is not None:
            for p in peaks:
                if p is None:
                    continue
                xp = int(p * (panel_w - 1) / max(1, n - 1))
                cv2.line(plot, (xp, 0), (xp, panel_h - 1), (0, 200, 0), 1)
        v = debug.get("valley")
        if v is not None:
            xv = int(v * (panel_w - 1) / max(1, n - 1))
            cv2.line(plot, (xv, 0), (xv, panel_h - 1), (0, 0, 255), 1)
        canvas[pad:pad + panel_h,
               pad + panel_w + pad:pad + panel_w + pad + panel_w] = plot
        ratio = debug.get("valley_ratio")
        rtxt = f"ratio={ratio:.2f}" if ratio is not None else ""
        label(f"proj({axis}) peaks={peaks} v={v} {rtxt}", 0, 1)
        label(f"proj({axis})", 0, 1)

    # Bottom-left: result on cropped frame
    res = crop.copy()
    children = debug.get("children", []) or []
    # Re-coord into the local crop
    for cb in children:
        cx, cy, cw, ch = cb
        rx, ry = cx - sx1, cy - sy1
        cv2.rectangle(res, (rx, ry), (rx + cw, ry + ch), (0, 255, 255), 2)
    if not children:
        cv2.rectangle(res, (0, 0), (res.shape[1] - 1, res.shape[0] - 1),
                      (0, 0, 255), 2)
    place(res, 1, 0)
    abort = debug.get("abort")
    label(f"result: {'OK' if not abort else 'ABORT '+str(abort)}", 1, 0)

    # Bottom-right: text summary
    text = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
    lines = [
        f"frame={frame_idx} cam={cam_i} box#{box_idx}",
        f"box=({bx},{by},{bw},{bh})",
        f"axis={axis}  trigger={debug.get('trigger','')}",
        f"abort={debug.get('abort')}",
        f"peaks={debug.get('peaks')}  valley={debug.get('valley')}",
        f"valley_ratio={debug.get('valley_ratio')}",
        f"#children={len(children)}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(text, ln, (4, 16 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    canvas[pad + panel_h + pad: pad + panel_h + pad + panel_h,
           pad + panel_w + pad: pad + panel_w + pad + panel_w] = text

    fname = (f"f{frame_idx:05d}_cam{cam_i}_b{box_idx:02d}_"
             f"{axis}_{('ok' if not abort else abort)}.png")
    cv2.imwrite(os.path.join(out_dir, fname), canvas)


def _prepare_bgsub_input(frame, cfg):
    if cfg.get("use_grayscale", False):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _apply_absdiff_bgsub(frame, bg_model, *, alpha, threshold):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if bg_model is None:
        bg_model = gray.astype(np.float32)
        return np.zeros_like(gray, dtype=np.uint8), bg_model

    bg_u8 = cv2.convertScaleAbs(bg_model)
    diff = cv2.absdiff(gray, bg_u8)
    _, fg_mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    cv2.accumulateWeighted(gray, bg_model, float(alpha))
    return fg_mask, bg_model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_multiple_videos(video_dir, output_dir, cfg=None):
    """Run single-pass MOG2 over 4 synchronized videos and dump mosaic outputs.

    Each video is read once. The first `warmup_frames` frames train the
    background model at `warmup_learning_rate` and produce no output; the
    remaining frames run at `detection_learning_rate` and are saved to disk.

    Parameters
    ----------
    video_dir : str
        Directory containing 4 aligned camera videos (`c00*_aligned.avi`).
    output_dir : str
        Destination root; subfolders `frames/`, `masks/`, `hybrid/` are created.
    cfg : dict, optional
        Configuration overriding `DEFAULT_CONFIG`.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    proposal_records = []

    # ---- Discover the 4 aligned camera videos under `video_dir`. ---------
    prefix, suffix, limit = "c00", "_aligned.avi", 4
    files = sorted(
        f for f in os.listdir(video_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    )
    if len(files) < limit:
        print(f"[warn] found {len(files)} videos under {video_dir}, expected {limit}")
    video_paths = [os.path.join(video_dir, f) for f in files[:limit]]
    camera_ids = [os.path.basename(path).replace("_aligned.avi", "") for path in video_paths]
    n_cams = len(video_paths)
    if n_cams != 4:
        raise RuntimeError(f"expected 4 camera videos, got {n_cams}")

    frame_dir = os.path.join(output_dir, "frames")
    mask_dir = os.path.join(output_dir, "masks")
    hybrid_dir = os.path.join(output_dir, "hybrid")
    debug_dir = os.path.join(output_dir, "split_debug")
    for d in (frame_dir, mask_dir, hybrid_dir):
        os.makedirs(d, exist_ok=True)
    if cfg.get("split_debug", False):
        os.makedirs(debug_dir, exist_ok=True)

    bgsub_method = str(cfg.get("bgsub_method", "mog2")).lower()
    if bgsub_method not in {"mog2", "knn", "absdiff"}:
        raise ValueError(f"Unsupported bgsub_method={bgsub_method!r}; expected 'mog2', 'knn', or 'absdiff'")

    # ---- One background model per camera. --------------------------------
    back_subs = []
    absdiff_backgrounds = [None for _ in range(n_cams)]
    if bgsub_method == "mog2":
        back_subs = [
            cv2.createBackgroundSubtractorMOG2(
                history=cfg["history"],
                varThreshold=cfg["var_threshold"],
                detectShadows=cfg["detect_shadows"],
            )
            for _ in range(n_cams)
        ]
    elif bgsub_method == "knn":
        back_subs = [
            cv2.createBackgroundSubtractorKNN(
                history=cfg["history"],
                dist2Threshold=cfg["var_threshold"],
                detectShadows=cfg["detect_shadows"],
            )
            for _ in range(n_cams)
        ]

    # ---- Fit per-camera single-vehicle priors from GT (offline). ---------
    # Used by the merged-blob split trigger (Rule A). Falls back to the
    # geometric thresholds for any camera without GT data.
    probe = cv2.VideoCapture(video_paths[0])
    image_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    probe.release()
    priors = _load_gt_priors(
        cfg.get("gt_dir", ""),
        image_height=image_height,
        bin_size=int(cfg["prior_bin_size"]),
        min_samples_per_bin=int(cfg["prior_min_samples_per_bin"]),
    )
    if priors:
        kind = "Q90" if cfg["prior_use_q90"] else (
            f"gamma_w={cfg['prior_gamma_w']}, gamma_a={cfg['prior_gamma_a']}"
        )
        cams_with_prior = sorted(priors.keys())
        print(f"[prior] loaded GT priors for cams {cams_with_prior}, "
              f"bin={cfg['prior_bin_size']}px, trigger={kind}")
    else:
        print("[prior] no GT priors loaded; split trigger uses geometric fallback")

    # Pre-build morphological kernels (constant across frames).
    k_close = cfg["kernel_close_size"]
    k_open = cfg["kernel_open_size"]
    close_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, 
        k_close))
        if k_close > 0 else None
    )
    open_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        if k_open > 0 else None
    )

    warmup_n = int(cfg["warmup_frames"])
    lr_warmup = cfg["warmup_learning_rate"]
    lr_detect = cfg["detection_learning_rate"]
    print("-" * 60)
    print(f"Single-pass schedule: warmup_frames={warmup_n}  "
          f"warmup_lr={lr_warmup}  detection_lr={lr_detect}")
    if bgsub_method == "absdiff":
        print(
            f"Background subtraction: absdiff "
            f"threshold={cfg['absdiff_threshold']} alpha={cfg['absdiff_alpha']}"
        )
    elif bgsub_method == "knn":
        print(
            f"Background subtraction: knn "
            f"history={cfg['history']} dist2_threshold={cfg['var_threshold']}"
        )
    else:
        print(
            f"Background subtraction: mog2 "
            f"history={cfg['history']} var_threshold={cfg['var_threshold']}"
        )

    caps = [cv2.VideoCapture(p) for p in video_paths]
    if not all(c.isOpened() for c in caps):
        for c in caps:
            c.release()
        raise RuntimeError("failed to open one or more video files")

    # Per-camera diagnostic counters for the split stage.
    split_stats = [dict() for _ in range(n_cams)]
    box_counts = [0 for _ in range(n_cams)]
    log_every = int(cfg.get("log_every", 100))

    frame_idx = 0
    n_written = 0
    while True:
        frames, ok = [], True
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                ok = False
                break
            frames.append(frame)
        if not ok:
            break
        frame_idx += 1

        # Choose learning rate and decide whether to record outputs.
        if frame_idx <= warmup_n:
            lr = lr_warmup
            record = False
        else:
            lr = lr_detect
            record = True

        # Run background subtraction + post-processing on every camera in lock-step.
        fg_masks, drawn_frames = [], []
        for cam_i, frame in enumerate(frames):
            bgsub_input = _prepare_bgsub_input(frame, cfg)
            if bgsub_method == "absdiff":
                fg_mask, absdiff_backgrounds[cam_i] = _apply_absdiff_bgsub(
                    bgsub_input,
                    absdiff_backgrounds[cam_i],
                    alpha=cfg["absdiff_alpha"],
                    threshold=cfg["absdiff_threshold"],
                )
            else:
                fg_mask = back_subs[cam_i].apply(bgsub_input, learningRate=lr)

            # ---- Post-process the foreground mask. -----------------------
            # MOG2 marks shadow pixels as 127; binarize to keep only true
            # foreground, then apply CLOSE -> OPEN with elliptical kernels.
            if cfg["detect_shadows"]:
                _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
            if close_kernel is not None:
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
            if open_kernel is not None:
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)
            fg_masks.append(fg_mask)

            if record:
                # ---- Extract bounding boxes from contours. ---------------
                # Keep contours that pass area and aspect-ratio filters.
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                boxes = []
                for cnt in contours:
                    if cv2.contourArea(cnt) < cfg["min_area"]:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h <= 0:
                        continue
                    ar = w / float(h)
                    if not (cfg["min_aspect_ratio"] < ar < cfg["max_aspect_ratio"]):
                        continue
                    boxes.append((x, y, w, h))

                # Merge fragments that likely belong to the same object.
                n_before_merge = len(boxes)
                boxes = _consolidate_boxes(boxes, cfg)
                n_after_merge = len(boxes)

                # Try to split obvious merged blobs (left/right OR top/bottom).
                cam_prior = priors.get(cam_i) if priors else None
                stats = split_stats[cam_i]
                tagged = []  # list of (box, tag)
                debug_list = [] if cfg.get("split_debug", False) else None
                for b in boxes:
                    tagged.extend(_split_merged_blob(
                        b, fg_mask, cfg, prior=cam_prior, stats=stats,
                        allow=("h", "v"), debug_list=debug_list,
                    ))
                boxes = [t[0] for t in tagged]
                origin = [t[1] for t in tagged]
                box_counts[cam_i] += len(boxes)

                # Dump per-step debug panels for triggered candidates.
                if debug_list:
                    cap_n = int(cfg.get("split_debug_max_per_frame", 4))
                    triggered_dbg = [d for d in debug_list if "axis" in d]
                    paper_mode = bool(cfg.get("split_debug_paper_mode", False))
                    for k, dbg in enumerate(triggered_dbg[:cap_n]):
                        if paper_mode:
                            _save_split_debug_paper(
                                debug_dir, frame_idx, cam_i, k, dbg, frame, cfg=cfg,
                            )
                        else:
                            _save_split_debug(
                                debug_dir, frame_idx, cam_i, k, dbg, frame,
                            )

                if log_every > 0 and (frame_idx % log_every == 0):
                    last_h = stats.get("_last_trigger_h", "")
                    last_v = stats.get("_last_trigger_v", "")
                    print(
                        f"[f{frame_idx:05d} cam{cam_i}] "
                        f"raw={n_before_merge} merge={n_after_merge} "
                        f"final={len(boxes)} "
                        f"H(trig/ok)={stats.get('h_triggered',0)}/{stats.get('h_success',0)} "
                        f"V(trig/ok)={stats.get('v_triggered',0)}/{stats.get('v_success',0)} "
                        + (f"hL={last_h} " if last_h else "")
                        + (f"vL={last_v}" if last_v else "")
                    )

                # ---- Draw rectangles. Color by split origin. -------------
                # Red    = no split            Yellow  = horizontal split child
                # Cyan   = vertical split child  Magenta = both axes split child
                drawn = frame.copy()
                for (bx, by, bw, bh), tag in zip(boxes, origin):
                    if tag == "normal":
                        color = (0, 0, 255)
                    elif tag == "h":
                        color = (0, 255, 255)
                    elif tag == "v":
                        color = (255, 255, 0)
                    else:  # 'hv', 'vh'
                        color = (255, 0, 255)
                    cv2.rectangle(drawn, (bx, by), (bx + bw, by + bh), color, 2)
                    proposal_records.append(
                        {
                            "proposal_id": f"{camera_ids[cam_i]}_f{frame_idx:06d}_p{len(proposal_records):06d}",
                            "camera_id": camera_ids[cam_i],
                            "frame_id": frame_idx,
                            "bbox": [int(bx), int(by), int(bw), int(bh)],
                            "score": float(bw * bh),
                            "source_frame": frame_idx,
                            "split_tag": tag,
                        }
                    )
                drawn_frames.append(drawn)

        if not record:
            continue

        # ---- Stitch the 4 views into 2x2 mosaics. ------------------------
        assert len(drawn_frames) == 4 and len(fg_masks) == 4
        mosaic_frame = np.vstack([
            np.hstack([drawn_frames[0], drawn_frames[1]]),
            np.hstack([drawn_frames[2], drawn_frames[3]]),
        ])
        mosaic_mask = np.vstack([
            np.hstack([fg_masks[0], fg_masks[1]]),
            np.hstack([fg_masks[2], fg_masks[3]]),
        ])
        mosaic_mask_bgr = cv2.cvtColor(mosaic_mask, cv2.COLOR_GRAY2BGR)
        mosaic_hybrid = np.hstack([mosaic_frame, mosaic_mask_bgr])

        cv2.imwrite(
            os.path.join(frame_dir, f"mosaic_frame_{frame_idx:05d}.png"),
            mosaic_frame,
        )
        cv2.imwrite(
            os.path.join(mask_dir, f"mosaic_mask_{frame_idx:05d}.png"),
            mosaic_mask,
        )
        cv2.imwrite(
            os.path.join(hybrid_dir, f"hybrid_{frame_idx:05d}.png"),
            mosaic_hybrid,
        )
        n_written += 1

    for cap in caps:
        cap.release()
    print(f"  read {frame_idx} frames, wrote {n_written} outputs")

    # ---- Per-camera split summary ---------------------------------------
    print("-" * 60)
    print("Split-stage summary (per camera, H=horiz axis split, V=vert axis split):")
    keys = ("h_seen", "h_triggered", "h_success",
            "h_abort_no_peak", "h_abort_peak_height",
            "h_abort_peaks_close", "h_abort_valley_shallow",
            "h_abort_empty_side",
            "v_seen", "v_triggered", "v_success",
            "v_abort_no_peak", "v_abort_peak_height",
            "v_abort_peaks_close", "v_abort_valley_shallow",
            "v_abort_empty_side")
    for cam_i in range(n_cams):
        s = split_stats[cam_i]
        print(f"cam {cam_i}: total_boxes={box_counts[cam_i]}")
        for k in keys:
            print(f"    {k:<28} = {s.get(k, 0)}")
    artifact_path = cfg.get("artifact_path")
    if artifact_path:
        save_proposal_artifact(
            artifact_path,
            proposal_records,
            cameras=camera_ids,
            frame_size=(640, 360),
            metadata={
                "video_dir": video_dir,
                "warmup_frames": warmup_n,
                "config": cfg,
            },
        )
        print(f"Proposal artifact written to {artifact_path}")
    print(f"Done. Outputs written under {output_dir}")
    return proposal_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RACE proposal artifact")
    parser.add_argument("--video_dir", default="./dataset_preprocessing/aligned_videos_640")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "background_substraction",
            "output_localization_0530_320p",
        ),
    )
    parser.add_argument("--artifact_path", default=None)
    parser.add_argument(
        "--bgsub_method",
        choices=("mog2", "knn", "absdiff"),
        default=DEFAULT_CONFIG["bgsub_method"],
        help="Background subtraction method used to generate foreground masks.",
    )
    parser.add_argument(
        "--absdiff_threshold",
        type=int,
        default=DEFAULT_CONFIG["absdiff_threshold"],
        help="Foreground threshold for --bgsub_method absdiff.",
    )
    parser.add_argument(
        "--absdiff_alpha",
        type=float,
        default=DEFAULT_CONFIG["absdiff_alpha"],
        help="Running-average update rate for --bgsub_method absdiff.",
    )
    parser.add_argument(
        "--use_grayscale",
        action="store_true",
        help="Convert frames to grayscale before background subtraction.",
    )
    parser.add_argument("--split_debug", action="store_true")
    parser.add_argument("--split_debug_max_per_frame", type=int, default=6)
    parser.add_argument("--split_debug_paper_mode", action="store_true")
    args = parser.parse_args()

    cfg = {
        "artifact_path": args.artifact_path,
        "bgsub_method": args.bgsub_method,
        "absdiff_threshold": args.absdiff_threshold,
        "absdiff_alpha": args.absdiff_alpha,
        "use_grayscale": args.use_grayscale,
        "split_debug": args.split_debug,
        "split_debug_max_per_frame": args.split_debug_max_per_frame,
        "split_debug_paper_mode": args.split_debug_paper_mode,
    }
    process_multiple_videos(args.video_dir, args.output_dir, cfg=cfg)
