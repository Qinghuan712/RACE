import os
import cv2
import time
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pycuda.driver as cuda
from sr_infer import SR_infer
from argparse import ArgumentParser

def parse_filename(filename):
    """
    Parse filename in the format camid_frameid_objectid.jpg
    Returns cam_id, frame_id, obj_id
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    return parts[0], int(parts[1]), int(parts[2])  # cam_id, frame_id, obj_id

def collect_targets(img_dir):
    """
    Collect all targets in different cameras.
    Returns a dict: {(frame_id, obj_id): [ (img_path, bbox_size, cam_id) ]}
    """
    targets = {}
    for fname in os.listdir(img_dir):
        if not fname.endswith('.jpg'):
            continue
        cam_id, frame_id, obj_id = parse_filename(fname)
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox_size = w * h
        key = (frame_id, obj_id)
        if key not in targets:
            targets[key] = []
        targets[key].append((img_path, cam_id, bbox_size))
    return targets

def run_sr_on_images(sr_model, img_paths, save_dir=None):
    """
    Run SR inference on a list of images, return total time used.
    """
    total_time = 0
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Convert BGR to RGB and prepare tensor for SR model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (640, 360), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite("debug_input.jpg", cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR))
        # (1, 3, H, W)
        img_tensor = torch.from_numpy(img_resize).permute(2,0,1).unsqueeze(0).float().cuda()

        # Run SR inference
        start = time.time()
        sr_result = sr_model.inference(img_tensor)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"SR time for {os.path.basename(img_path)}: {elapsed*1000:.2f} ms")
 
        # if isinstance(sr_result, list):
        #     sr_result = sr_result[0]
        # t = sr_result.detach().cpu()
        # if t.ndim == 4: 
        #     t = t[0]
        # for i in range(min(3, t.shape[0])):  # 假设 CHW
        #     ch = t[i].numpy()
        #     ch = np.clip(ch, 0, 255).astype(np.uint8)
        #     debug_path = os.path.join(save_dir, f"debug_ch{i}.png")
        #     cv2.imwrite(debug_path, ch)

        # Save SR result if needed
        # if save_dir and sr_result is not None:
        #     # Convert tensor to image and save
        #     if isinstance(sr_result, list):
        #         sr_result = sr_result[0]
        #         print("sr_img shape before transpose:", sr_result.shape, sr_result.min().item(), sr_result.max().item())
        #     sr_img = sr_result.squeeze().detach().cpu().numpy()
        #     print("SR raw shape:", sr_img.shape, "dtype:", sr_img.dtype, "min:", sr_img.min(), "max:", sr_img.max())
        #     print("Channel mean std:", [sr_img[i].mean() for i in range(sr_img.shape[0])])

        #     sr_img = np.clip(sr_img, 0, 255)
        #     # (C, H, W) to (H, W, C)
        #     if sr_img.ndim == 3 and sr_img.shape[0] in [1, 3]:
        #         sr_img = np.transpose(sr_img, (1, 2, 0))  # CHW -> HWC
            
        #     sr_img = sr_img.astype(np.uint8)
        #     sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        #     save_path = os.path.join(save_dir, os.path.basename(img_path))
        #     cv2.imwrite(save_path, sr_img)
    return total_time

def sr_latency_compare(sr_model, base_dir, base_dir_SR):
    """
    For cam_num1~4, compare two SR strategies and print results.
    """
    results = []
    for cam_num in [1,2,3,4]:
        img_dir = os.path.join(base_dir, f'cam_num{cam_num}')
        targets = collect_targets(img_dir)

        # Strategy 1: SR for all views
        img_dir_SR = os.path.join(base_dir_SR, f'cam_num{cam_num}')
        all_imgs = []
        for img_list in targets.values():
            all_imgs.extend([x[0] for x in img_list])
        all_time = run_sr_on_images(sr_model, all_imgs, img_dir_SR)
        all_count = len(all_imgs)

        # Strategy 2: SR only for the view with the largest bbox
        max_imgs = []
        max_count = 0
        max_time = 0
        if cam_num >=2:
            for img_list in targets.values():
                max_img = max(img_list, key=lambda x: x[2])
                max_imgs.append(max_img[0])
            max_time = run_sr_on_images(sr_model, max_imgs, img_dir_SR)
            max_count = len(max_imgs)
        results.append({
            'cam_num': cam_num,
            'all_count': all_count,
            'all_time': all_time,
            'max_count': all_count if cam_num == 1 else max_count,
            'max_time':  all_time if cam_num == 1 else max_time
        })
        print(f"cam_num{cam_num}: all views {all_count} SR, time={all_time*1000:.2f}ms | max view {max_count} SR, time={max_time*1000:.2f}ms")
    return results

if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--sr_model_path', type=str, required=True)
    args = parser.parse_args()

    # Initialize SR model
    cfx = cuda.Device(0).make_context()
    sr_model = SR_infer(args, cfx)

    # Set image directory
    base_dir = "/home/qinghuan/Xinyan/Redundancyenhance/AICity22_Track1_MTMC_Tracking/cropped_images"
    base_dir_SR = "/home/qinghuan/Xinyan/Redundancyenhance/AICity22_Track1_MTMC_Tracking/cropped_images_SR"
    # Run SR latency comparison experiment
    sr_latency_compare(sr_model, base_dir, base_dir_SR)