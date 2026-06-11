"""
Motivation 3: Semantic Fragmentation Experiment
MB-based vs Object-based Enhancement Comparison

对比实验：
- 方案 A (MB-based): 只增强部分宏块（模拟 NSDI 策略）
- 方案 B (Object-based): 增强整个目标 bbox（我们的策略）
- 方案 C (Full SR): 全图超分（参考上界）
- 方案 D (Bilinear): 纯双线性插值（参考下界）
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import tensorrt as trt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# CUDA context 管理说明：
# sr_infer.py 中 import pycuda.autoinit 会自动 init + 创建一个 context
# 不再导入 detect_infer.py（它的 TRT API 是旧版 8.x，与当前环境不兼容）
# 检测推理直接在本文件中用 Yolo11TRTDetector 实现
# ============================================================
import pycuda.driver as cuda
from sr_infer import SR_infer
from yolo_trt_detector import Yolo11TRTDetector
# sr_infer.py 的 import pycuda.autoinit 已经 init 了 CUDA 并创建了 context #1
# 先 pop 掉 autoinit 创建的 context #1
try:
    cuda.Context.pop()
except Exception:
    pass

MB_SIZE = 16  # 宏块大小

# Yolo11TRTDetector 已提取至 /home/qinghuan/Xinyan/Regenhance/yolo_trt_detector.py

# ============================================================
# Step 1: 筛选跨越多个 MB 的目标
# ============================================================

def get_covered_mbs(bbox, mb_size=MB_SIZE):
    """
    计算一个 bbox 覆盖了哪些宏块。
    bbox: (x1, y1, x2, y2) in pixel coordinates
    返回: list of (mb_row, mb_col)
    """
    x1, y1, x2, y2 = bbox
    mb_col_start = x1 // mb_size
    mb_col_end = (x2 - 1) // mb_size
    mb_row_start = y1 // mb_size
    mb_row_end = (y2 - 1) // mb_size
    mbs = []
    for r in range(mb_row_start, mb_row_end + 1):
        for c in range(mb_col_start, mb_col_end + 1):
            mbs.append((r, c))
    return mbs


def filter_cross_mb_objects(detections, mb_size=MB_SIZE, min_mbs=2):
    """
    筛选跨越至少 min_mbs 个宏块的目标。
    detections: list of dict, each with 'bbox': [x1, y1, x2, y2], 'class', 'confidence'
    """
    cross_mb_objects = []
    for det in detections:
        bbox = det['bbox']
        mbs = get_covered_mbs(bbox, mb_size)
        if len(mbs) >= min_mbs:
            det['covered_mbs'] = mbs
            det['num_mbs'] = len(mbs)
            cross_mb_objects.append(det)
    return cross_mb_objects


# ============================================================
# Step 2: 模拟 MB-based 和 Object-based 增强
# ============================================================

def simulate_mb_based_mask(covered_mbs, mask_shape=(22, 40), keep_ratio=0.5):
    """
    模拟 NSDI 的 MB-based 策略：只保留部分 MB（模拟轻量模型只认为一半重要）。
    keep_ratio: 保留的 MB 比例
    """
    mask = np.zeros(mask_shape, dtype=np.float32)
    num_keep = max(1, int(len(covered_mbs) * keep_ratio))
    # 只保留前 num_keep 个 MB（模拟只增强局部）
    kept_mbs = covered_mbs[:num_keep]
    for (r, c) in kept_mbs:
        if r < mask_shape[0] and c < mask_shape[1]:
            mask[r, c] = 1.0
    return mask, kept_mbs


def apply_sr_by_mask(lr_img, sr_img, mask, mb_size=MB_SIZE, scale=3):
    """
    根据 mask 将 SR 结果只贴回 mask=1 的宏块区域，其余区域用双线性插值。
    lr_img: (3, H, W) tensor, float32, 0~255
    sr_img: (3, H*scale, W*scale) tensor, float32
    mask: (num_mb_rows, num_mb_cols) numpy array
    返回: (3, H*scale, W*scale) tensor
    """
    # 先用双线性插值生成背景
    lr_4d = lr_img.unsqueeze(0)  # (1, 3, H, W)
    bg = F.interpolate(lr_4d, scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
    result = bg.clone()  # (3, H*scale, W*scale)

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] > 0:
                # 在 SR 图上对应的像素区域
                y1 = r * mb_size * scale
                y2 = (r + 1) * mb_size * scale
                x1 = c * mb_size * scale
                x2 = (c + 1) * mb_size * scale
                y2 = min(y2, result.shape[1])
                x2 = min(x2, result.shape[2])
                result[:, y1:y2, x1:x2] = sr_img[:, y1:y2, x1:x2]
    return result


def apply_sr_by_bbox(lr_img, sr_img, bbox, scale=3):
    """
    Object-based 增强：将整个 bbox 区域用 SR 结果替换。
    bbox: (x1, y1, x2, y2) in LR coordinates
    """
    lr_4d = lr_img.unsqueeze(0)
    bg = F.interpolate(lr_4d, scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
    result = bg.clone()

    x1, y1, x2, y2 = [int(v * scale) for v in bbox]
    x2 = min(x2, result.shape[2])
    y2 = min(y2, result.shape[1])
    result[:, y1:y2, x1:x2] = sr_img[:, y1:y2, x1:x2]
    return result


# ============================================================
# Step 3: 主实验逻辑
# ============================================================

def run_experiment(sr_model, detect_model, image_paths, gt_detections, save_dir, scale=3):
    """
    主实验函数
    sr_model: SR_infer 实例
    detect_model: Yolo11TRTDetector 实例
    image_paths: 低分辨率图像路径列表
    gt_detections: 每张图对应的 GT 检测框 (来自高分辨率图的检测结果)
    save_dir: 保存结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for img_idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image {img_path}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (640, 360), interpolation=cv2.INTER_CUBIC)

        # 准备 tensor
        lr_tensor = torch.from_numpy(img_resize).permute(2, 0, 1).float().cuda().contiguous()  # (3, 360, 640)
        lr_batch = lr_tensor.unsqueeze(0).contiguous()  # (1, 3, 360, 640)

        print(f"  [DEBUG] lr_batch shape: {lr_batch.shape}, contiguous: {lr_batch.is_contiguous()}")
        print(f"  [DEBUG] lr_batch range: {lr_batch.min().item():.1f} ~ {lr_batch.max().item():.1f}")

        # 全图 SR
        sr_output = sr_model.inference(lr_batch)
        if isinstance(sr_output, list):
            sr_output = sr_output[0]
        sr_tensor = sr_output.squeeze(0)  # (3, 1080, 1920)

        print(f"  [DEBUG] raw sr_tensor range: {sr_tensor.min().item():.3f} ~ {sr_tensor.max().item():.3f}")
        print(f"  [DEBUG] raw sr_tensor mean: {sr_tensor.mean().item():.3f}, std: {sr_tensor.std().item():.3f}")

        # SR 模型输出可能超出 0~255（残差学习），需要 clamp
        sr_tensor = torch.clamp(sr_tensor, 0, 255)

        # ====== DEBUG: 检查 SR 输出 ======
        print(f"  [DEBUG] sr_tensor shape: {sr_tensor.shape}")
        print(f"  [DEBUG] sr_tensor range: {sr_tensor.min().item():.3f} ~ {sr_tensor.max().item():.3f}")
        print(f"  [DEBUG] lr_tensor range: {lr_tensor.min().item():.3f} ~ {lr_tensor.max().item():.3f}")
        
        # 保存 SR 输出图片
        sr_np = sr_tensor.detach().cpu().numpy()  # (3, 1080, 1920)
        sr_np = np.transpose(sr_np, (1, 2, 0))    # (1080, 1920, 3)
        sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        debug_path = os.path.join(save_dir, f"debug_sr_output_img{img_idx}.png")
        cv2.imwrite(debug_path, sr_bgr)
        print(f"  [DEBUG] SR output saved to {debug_path}")

        # 获取该图的检测目标
        if img_idx >= len(gt_detections):
            print(f"Warning: No GT detections for image index {img_idx}, skipping.")
            continue
        dets = gt_detections[img_idx]
        cross_mb_dets = filter_cross_mb_objects(dets, mb_size=MB_SIZE, min_mbs=2)

        if len(cross_mb_dets) == 0:
            print(f"Image {img_idx}: No cross-MB objects found, skipping.")
            continue

        print(f"Image {img_idx}: Found {len(cross_mb_dets)} cross-MB objects.")

        for det_idx, det in enumerate(cross_mb_dets):
            bbox = det['bbox']  # (x1, y1, x2, y2) in LR coordinates
            covered_mbs = det['covered_mbs']

            # ============ 方案 A: MB-based (只增强部分 MB) ============
            mask_partial, kept_mbs = simulate_mb_based_mask(
                covered_mbs, mask_shape=(22, 40), keep_ratio=0.5
            )
            img_mb_based = apply_sr_by_mask(lr_tensor, sr_tensor, mask_partial, MB_SIZE, scale)

            # ============ 方案 B: Object-based (增强整个 bbox) ============
            img_obj_based = apply_sr_by_bbox(lr_tensor, sr_tensor, bbox, scale)

            # ============ 方案 C: 全图 SR (参考上界) ============
            img_full_sr = sr_tensor.clone()

            # ============ 方案 D: 纯双线性插值 (参考下界) ============
            img_bilinear = F.interpolate(
                lr_batch, scale_factor=scale, mode='bilinear', align_corners=False
            ).squeeze(0)

            # ============ 检测 ============
            print("Detecting and getting confidence.....")
            bbox_hr = [int(v * scale) for v in bbox]

            conf_mb, det_boxes_mb, det_scores_mb = detect_and_get_confidence(detect_model, img_mb_based, bbox_hr)
            conf_obj, det_boxes_obj, det_scores_obj = detect_and_get_confidence(detect_model, img_obj_based, bbox_hr)
            conf_full, det_boxes_full, det_scores_full = detect_and_get_confidence(detect_model, img_full_sr, bbox_hr)
            conf_bilinear, det_boxes_bil, det_scores_bil = detect_and_get_confidence(detect_model, img_bilinear, bbox_hr)

            result_entry = {
                'image': os.path.basename(img_path),
                'det_idx': det_idx,
                'bbox_lr': bbox,
                'num_mbs': det['num_mbs'],
                'num_kept_mbs': len(kept_mbs),
                'conf_bilinear': float(conf_bilinear),
                'conf_mb_based': float(conf_mb),
                'conf_obj_based': float(conf_obj),
                'conf_full_sr': float(conf_full),
            }
            results.append(result_entry)

            print(f"  Det {det_idx}: MBs={det['num_mbs']}, kept={len(kept_mbs)}, "
                  f"conf: bilinear={conf_bilinear:.3f}, mb={conf_mb:.3f}, "
                  f"obj={conf_obj:.3f}, full={conf_full:.3f}")

            # ============ 保存可视化 ============
            # 将 4 个方案的检测结果打包传给可视化函数
            det_results_all = {
                'bilinear': (det_boxes_bil, det_scores_bil),
                'mb_based': (det_boxes_mb, det_scores_mb),
                'obj_based': (det_boxes_obj, det_scores_obj),
                'full_sr': (det_boxes_full, det_scores_full),
            }
            save_visualization(
                img_mb_based, img_obj_based, img_full_sr, img_bilinear,
                bbox_hr, kept_mbs, covered_mbs,
                save_dir, img_idx, det_idx, scale,
                det_results=det_results_all
            )

    # 保存结果
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def detect_and_get_confidence(detect_model, img_tensor, bbox_hr):
    """
    在 HR 图上检测，返回 bbox_hr 区域内最高置信度以及所有检测框信息
    img_tensor: (3, H, W) tensor on CUDA, 0~255 范围
    bbox_hr: [x1, y1, x2, y2] in HR coordinates

    Returns:
        best_conf: float, 与 bbox_hr IoU > 0.3 的最高检测置信度
        det_boxes: list of [x1, y1, x2, y2], 所有检测框坐标 (HR 空间)
        det_scores: list of float, 对应置信度
    """
    # 调用 Yolo11TRTDetector 的推理方法，
    # 内部已正确处理：resize 到 1088x1920、/255 归一化、cfx.push/pop、execute_async_v3
    try:
        num_np, boxes_np, scores_np, labels_np = detect_model.inference_raw(img_tensor)
    except Exception as e:
        print(f"  Detection error: {e}")
        import traceback; traceback.print_exc()
        return 0.0, [], []

    num_dets = int(num_np[0]) if num_np.ndim > 0 else int(num_np)

    # 首次调用时打印调试信息
    if not hasattr(detect_and_get_confidence, '_printed_debug'):
        detect_and_get_confidence._printed_debug = True
        print(f"  [DET DEBUG] num_dets={num_dets}, "
              f"boxes[:3]={boxes_np[:3]}, scores[:3]={scores_np[:3]}, labels[:3]={labels_np[:3]}")

    det_boxes = []
    det_scores = []
    best_conf = 0.0
    bx1, by1, bx2, by2 = bbox_hr
    for i in range(min(num_dets, len(boxes_np))):
        dx1, dy1, dx2, dy2 = boxes_np[i]
        conf = float(scores_np[i])
        det_boxes.append([float(dx1), float(dy1), float(dx2), float(dy2)])
        det_scores.append(conf)
        iou = compute_iou([bx1, by1, bx2, by2], [dx1, dy1, dx2, dy2])
        if iou > 0.3 and conf > best_conf:
            best_conf = conf

    return best_conf, det_boxes, det_scores


def compute_iou(box1, box2):
    """
    计算两个 bbox 的 IoU。
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def parse_detection_confidence(det_results, bbox_hr):
    """
    解析检测结果，返回与 bbox_hr IoU 最大的检测框的置信度。
    需要根据你的检测模型输出格式调整。

    常见输出格式:
    - tensor of shape (N, 6): [x1, y1, x2, y2, confidence, class]
    - list of dicts: [{'bbox': [x1,y1,x2,y2], 'confidence': 0.9, 'class': 0}, ...]
    """
    best_conf = 0.0
    bx1, by1, bx2, by2 = bbox_hr

    if isinstance(det_results, torch.Tensor):
        det_np = det_results.detach().cpu().numpy()
        if det_np.ndim == 1:
            det_np = det_np.reshape(-1, 6) if det_np.shape[0] % 6 == 0 else det_np.reshape(1, -1)
        if det_np.ndim == 2 and det_np.shape[1] >= 5:
            for det in det_np:
                dx1, dy1, dx2, dy2, conf = det[:5]
                iou = compute_iou([bx1, by1, bx2, by2], [dx1, dy1, dx2, dy2])
                if iou > 0.3 and conf > best_conf:
                    best_conf = conf
    elif isinstance(det_results, list):
        for det in det_results:
            if isinstance(det, dict):
                dx1, dy1, dx2, dy2 = det.get('bbox', [0, 0, 0, 0])
                conf = det.get('confidence', 0.0)
            elif isinstance(det, (list, tuple, np.ndarray)):
                if len(det) >= 5:
                    dx1, dy1, dx2, dy2, conf = det[:5]
                else:
                    continue
            else:
                continue
            iou = compute_iou([bx1, by1, bx2, by2], [dx1, dy1, dx2, dy2])
            if iou > 0.3 and conf > best_conf:
                best_conf = conf
    elif isinstance(det_results, np.ndarray):
        if det_results.ndim == 2 and det_results.shape[1] >= 5:
            for det in det_results:
                dx1, dy1, dx2, dy2, conf = det[:5]
                iou = compute_iou([bx1, by1, bx2, by2], [dx1, dy1, dx2, dy2])
                if iou > 0.3 and conf > best_conf:
                    best_conf = conf

    return best_conf


# ============================================================
# Step 4: 可视化
# ============================================================

def save_visualization(img_mb, img_obj, img_full, img_bilinear,
                       bbox_hr, kept_mbs, all_mbs,
                       save_dir, img_idx, det_idx, scale,
                       det_results=None):
    """
    保存对比可视化图：
    [Bilinear | MB-based (partial) | Object-based (ours) | Full SR]

    det_results: dict with keys 'bilinear', 'mb_based', 'obj_based', 'full_sr'
                 each value is (det_boxes, det_scores) where
                 det_boxes is list of [x1,y1,x2,y2] and det_scores is list of float
    """
    x1, y1, x2, y2 = bbox_hr
    margin = 20 * scale  # 多显示一些上下文

    def tensor_to_crop(tensor, name, det_boxes=None, det_scores=None):
        """将 tensor 裁剪目标区域并保存，画 GT 框(绿) 和检测框(红)"""
        arr = tensor.detach().cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        # 裁剪目标区域 + margin
        cy1 = max(0, y1 - margin)
        cy2 = min(arr.shape[0], y2 + margin)
        cx1 = max(0, x1 - margin)
        cx2 = min(arr.shape[1], x2 + margin)
        crop = arr[cy1:cy2, cx1:cx2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # 画 GT bbox (绿色)
        cv2.rectangle(crop_bgr,
                      (x1 - cx1, y1 - cy1),
                      (x2 - cx1, y2 - cy1),
                      (0, 255, 0), 2)
        # 画检测框 (红色) 及置信度
        if det_boxes and det_scores:
            for dbox, dscore in zip(det_boxes, det_scores):
                dx1, dy1, dx2, dy2 = [int(v) for v in dbox]
                # 将检测框坐标转换到 crop 局部坐标
                local_dx1 = dx1 - cx1
                local_dy1 = dy1 - cy1
                local_dx2 = dx2 - cx1
                local_dy2 = dy2 - cy1
                # 只画与裁剪区域有交集的检测框
                if local_dx2 > 0 and local_dy2 > 0 and \
                   local_dx1 < crop_bgr.shape[1] and local_dy1 < crop_bgr.shape[0]:
                    local_dx1 = max(0, local_dx1)
                    local_dy1 = max(0, local_dy1)
                    local_dx2 = min(crop_bgr.shape[1] - 1, local_dx2)
                    local_dy2 = min(crop_bgr.shape[0] - 1, local_dy2)
                    cv2.rectangle(crop_bgr,
                                  (local_dx1, local_dy1),
                                  (local_dx2, local_dy2),
                                  (0, 0, 255), 2)  # 红色 (BGR)
                    label_text = f"{dscore:.2f}"
                    cv2.putText(crop_bgr, label_text,
                                (local_dx1, max(local_dy1 - 5, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        path = os.path.join(save_dir, f"img{img_idx}_det{det_idx}_{name}.png")
        cv2.imwrite(path, crop_bgr)
        return crop_bgr

    # 获取每个方案的检测结果
    if det_results:
        db_bil, ds_bil = det_results.get('bilinear', ([], []))
        db_mb, ds_mb = det_results.get('mb_based', ([], []))
        db_obj, ds_obj = det_results.get('obj_based', ([], []))
        db_full, ds_full = det_results.get('full_sr', ([], []))
    else:
        db_bil = ds_bil = db_mb = ds_mb = db_obj = ds_obj = db_full = ds_full = []

    crop_bilinear = tensor_to_crop(img_bilinear, "bilinear", db_bil, ds_bil)
    crop_mb = tensor_to_crop(img_mb, "mb_based", db_mb, ds_mb)
    crop_obj = tensor_to_crop(img_obj, "obj_based", db_obj, ds_obj)
    crop_full = tensor_to_crop(img_full, "full_sr", db_full, ds_full)

    # 拼接对比图
    # [Bilinear | MB-based | Object-based | Full SR]
    h = max(crop_bilinear.shape[0], crop_mb.shape[0], crop_obj.shape[0], crop_full.shape[0])

    def pad_to_height(img, target_h):
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    crops = [crop_bilinear, crop_mb, crop_obj, crop_full]
    crops = [pad_to_height(c, h) for c in crops]

    # 添加标签
    labels = ["Bilinear", "MB-based (partial)", "Object-based (ours)", "Full SR"]
    for i, (crop, label) in enumerate(zip(crops, labels)):
        cv2.putText(crop, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 添加分隔线
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 128
    parts = []
    for i, crop in enumerate(crops):
        parts.append(crop)
        if i < len(crops) - 1:
            sep = np.ones((h, 3, 3), dtype=np.uint8) * 128
            parts.append(sep)

    concat = np.hstack(parts)
    concat_path = os.path.join(save_dir, f"img{img_idx}_det{det_idx}_comparison.png")
    cv2.imwrite(concat_path, concat)
    print(f"  Visualization saved to {concat_path}")


def plot_confidence_comparison(results, save_dir):
    """
    绘制置信度对比柱状图（用于论文 Figure）
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot.")
        return

    if len(results) == 0:
        print("Warning: No results to plot.")
        return

    methods = ['Bilinear', 'MB-based', 'Object-based', 'Full SR']
    confs = {m: [] for m in methods}

    for r in results:
        confs['Bilinear'].append(r['conf_bilinear'])
        confs['MB-based'].append(r['conf_mb_based'])
        confs['Object-based'].append(r['conf_obj_based'])
        confs['Full SR'].append(r['conf_full_sr'])

    means = [np.mean(confs[m]) for m in methods]
    stds = [np.std(confs[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#cccccc', '#ff7f7f', '#7fbf7f', '#7f7fff']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

    ax.set_ylabel('Detection Confidence', fontsize=14)
    ax.set_title('Semantic Fragmentation: MB-based vs Object-based Enhancement', fontsize=12)
    ax.set_ylim(0, 1.0)

    # 添加数值标签
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, 'confidence_comparison.pdf')
    png_path = os.path.join(save_dir, 'confidence_comparison.png')
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Figure saved to {pdf_path} and {png_path}")


# ============================================================
# Step 5: 运行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Motivation: Semantic Fragmentation Experiment'
    )
    parser.add_argument('--sr_model_path', type=str, required=True,
                        help='Path to SR TensorRT engine')
    parser.add_argument('--detect_model_path', type=str, required=True,
                        help='Path to detection TensorRT engine')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing low-resolution images')
    parser.add_argument('--gt_json', type=str, required=True,
                        help='Path to GT detection JSON file')
    parser.add_argument('--save_dir', type=str, default='experiments/motivation3_results',
                        help='Directory to save results')
    parser.add_argument('--scale', type=int, default=3,
                        help='SR scale factor')
    parser.add_argument('--max_batch_size', type=int, default=4,
                        help='Max batch size for SR model')
    parser.add_argument('--keep_ratio', type=float, default=0.5,
                        help='Ratio of MBs to keep in MB-based strategy')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 初始化 CUDA — 创建唯一的 context
    # autoinit 创建的 context 已在模块导入时 pop 掉
    # 这里创建唯一一个干净的 context，传给所有模型
    cfx = cuda.Device(0).make_context()

    try:
        # 加载模型
        print("Loading SR model...")
        sr_model = SR_infer(args, cfx)
        print("SR model loaded successfully.")

        print("Loading detection model...")
        detect_model = Yolo11TRTDetector(args.detect_model_path, cfx)
        print("Detection model loaded successfully.")

        # 加载图像
        image_paths = sorted([
            os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        print(f"Found {len(image_paths)} images in {args.image_dir}")

        if len(image_paths) == 0:
            print("Error: No images found!")
            sys.exit(1)

        # 加载 GT 检测框
        # JSON 格式: [[{"bbox": [x1,y1,x2,y2], "class": "car", "confidence": 0.95}, ...], ...]
        with open(args.gt_json, 'r') as f:
            gt_detections = json.load(f)
        print(f"Loaded GT detections for {len(gt_detections)} images.")

        # 运行实验
        print("\n" + "=" * 60)
        print("Running Semantic Fragmentation Experiment...")
        print("=" * 60 + "\n")

        results = run_experiment(
            sr_model, detect_model, image_paths, gt_detections,
            args.save_dir, scale=args.scale
        )

        # 绘制图表
        if len(results) > 0:
            plot_confidence_comparison(results, args.save_dir)

            # 打印统计摘要
            print("\n" + "=" * 60)
            print("Experiment Summary")
            print("=" * 60)
            print(f"Total cross-MB objects analyzed: {len(results)}")
            print(f"Avg Bilinear confidence:    {np.mean([r['conf_bilinear'] for r in results]):.4f}")
            print(f"Avg MB-based confidence:    {np.mean([r['conf_mb_based'] for r in results]):.4f}")
            print(f"Avg Object-based confidence:{np.mean([r['conf_obj_based'] for r in results]):.4f}")
            print(f"Avg Full SR confidence:     {np.mean([r['conf_full_sr'] for r in results]):.4f}")
        else:
            print("\nNo cross-MB objects found in any image. Try adjusting min_mbs or check GT detections.")

        print("\nExperiment completed!")

    finally:
        cfx.pop()
        print("CUDA context released.")