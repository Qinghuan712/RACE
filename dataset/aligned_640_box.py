import cv2
import numpy as np
import os
from collections import defaultdict

def load_gt_annotations(gt_file):
    """
    Load ground truth annotations from file
    Returns: dict {frame_id: [(object_id, x, y, w, h), ...]}
    """
    annotations = defaultdict(list)
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                object_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                w = int(parts[4])
                h = int(parts[5])
                
                annotations[frame_id].append((object_id, x, y, w, h))
    
    return annotations

def draw_boxes_on_frame(frame, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on frame
    Args:
        frame: input frame
        boxes: list of (object_id, x, y, w, h)
        color: box color in BGR
        thickness: line thickness
    """
    frame_with_boxes = frame.copy()
    
    for box in boxes:
        object_id, x, y, w, h = box
        # Draw bounding box
        cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, thickness)
        
        # Draw object ID
        # text = f"ID:{object_id}"
        # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        # text_y = y - 5 if y - 5 > text_size[1] else y + h + text_size[1] + 5
        # cv2.putText(
        #     frame_with_boxes, 
        #     text, 
        #     (x, text_y), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     0.5,
        #     color,
        #     1
        # )
    
    return frame_with_boxes

def create_aligned_video_with_gt(
    video_dir,
    gt_dir,
    output_path,
    fps=10
):
    """
    Create a mosaic video (2x2) with ground truth bounding boxes
    
    Args:
        video_dir: directory containing 4 video files
        gt_dir: directory containing 4 gt annotation files
        output_path: output video path
        fps: frames per second for output video
    """
    # Video files (sorted)
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])
    if len(video_files) != 4:
        raise ValueError(f"Expected 4 videos, found {len(video_files)}")
    
    print(f"Found {len(video_files)} videos:")
    for vf in video_files:
        print(f"  - {vf}")
    
    # GT files (sorted)
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
    if len(gt_files) != 4:
        raise ValueError(f"Expected 4 GT files, found {len(gt_files)}")
    
    print(f"\nFound {len(gt_files)} GT files:")
    for gf in gt_files:
        print(f"  - {gf}")
    
    # Load all GT annotations
    print("\nLoading GT annotations...")
    gt_annotations = []
    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        annotations = load_gt_annotations(gt_path)
        gt_annotations.append(annotations)
        print(f"  Loaded {len(annotations)} frames from {gt_file}")
    
    # Open video captures
    caps = []
    for vf in video_files:
        vpath = os.path.join(video_dir, vf)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {vpath}")
        caps.append(cap)
    
    # Get video properties
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Frame size: {frame_width}x{frame_height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output FPS: {fps}")
    
    # Mosaic dimensions (2x2)
    mosaic_width = frame_width * 2
    mosaic_height = frame_height * 2
    
    print(f"  Mosaic size: {mosaic_width}x{mosaic_height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (mosaic_width, mosaic_height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    print(f"\nProcessing frames...")
    frame_idx = 0
    
    # Define colors for each camera (BGR format)
    colors = [
        (0, 255, 0),    # Green for c001
        (255, 0, 0),    # Blue for c002
        (0, 255, 255),  # Yellow for c003
        (0, 0, 255)     # Red for c004
    ]
    
    while True:
        frames = []
        all_valid = True
        
        # Read one frame from each video
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                all_valid = False
                break
            frames.append(frame)
        
        if not all_valid:
            break
        
        frame_idx += 1
        
        # Draw bounding boxes on each frame
        processed_frames = []
        for i, (frame, annotations) in enumerate(zip(frames, gt_annotations)):
            # Get boxes for current frame
            boxes = annotations.get(frame_idx, [])
            
            # Draw boxes with camera-specific color
            frame_with_boxes = draw_boxes_on_frame(frame, boxes, color=colors[i], thickness=2)
            
            # Add camera label
            # cam_label = f"Camera {i+1} (c00{i+1})"
            # cv2.putText(
            #     frame_with_boxes,
            #     cam_label,
            #     (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1.0,
            #     colors[i],
            #     2
            # )
            
            processed_frames.append(frame_with_boxes)
        
        # Create 2x2 mosaic
        top_row = np.hstack([processed_frames[0], processed_frames[1]])
        bottom_row = np.hstack([processed_frames[2], processed_frames[3]])
        mosaic_frame = np.vstack([top_row, bottom_row])
        
        # Add frame counter to mosaic
        info_text = f"Frame: {frame_idx}/{total_frames}"
        # print(f"  Processing {info_text}")
        # cv2.putText(
        #     mosaic_frame,
        #     info_text,
        #     (mosaic_width - 300, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (255, 255, 255),
        #     2
        # )
        
        # Write frame to output video
        out.write(mosaic_frame)
        
        # Progress indicator
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
    
    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    
    print("\n" + "=" * 80)
    print("Video creation completed!")
    print(f"Total frames processed: {frame_idx}")
    print(f"Output video saved to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    # Configuration
    video_dir = "./dataset_preprocessing/aligned_videos_640"
    gt_dir = "./dataset_preprocessing/aligned_gt_640"
    output_path = "./dataset_preprocessing/aligned_videos_640/aligned_640_box.avi"
    
    print("=" * 80)
    print("Aligned Video Creation with Ground Truth Annotations")
    print("=" * 80)
    print(f"Video directory: {video_dir}")
    print(f"GT directory: {gt_dir}")
    print(f"Output path: {output_path}")
    print("=" * 80 + "\n")
    
    # Create aligned video
    create_aligned_video_with_gt(
        video_dir=video_dir,
        gt_dir=gt_dir,
        output_path=output_path,
        fps=10
    )
