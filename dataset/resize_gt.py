import os
from pathlib import Path


def resize_gt_file(input_file, output_file, scale_x, scale_y):
    """
    Resize coordinates in a single GT file
    
    Args:
        input_file: Input file path
        output_file: Output file path
        scale_x: Scaling ratio for x direction
        scale_y: Scaling ratio for y direction
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split line into fields
            fields = line.split(',')
            
            if len(fields) < 6:
                # If insufficient fields, write original line
                f_out.write(line + '\n')
                continue
            
            # Parse fields that need to be scaled
            frame_id = fields[0]
            object_id = fields[1]
            x = float(fields[2])
            y = float(fields[3])
            width = float(fields[4])
            height = float(fields[5])
            
            # Scale coordinates and dimensions
            x_new = round(x * scale_x)
            y_new = round(y * scale_y)
            width_new = round(width * scale_x)
            height_new = round(height * scale_y)
            
            # Reconstruct output line
            fields[2] = str(x_new)
            fields[3] = str(y_new)
            fields[4] = str(width_new)
            fields[5] = str(height_new)
            
            # Write to output file
            f_out.write(','.join(fields) + '\n')


def main():
    # Get script directory
    prj_dir = Path(__file__).parent.parent  # Go up one level from pre_motivation

    # Define paths relative to script directory
    input_dir = prj_dir / 'dataset_preprocessing' / 'aligned_gt_1920'
    output_dir = prj_dir / 'dataset_preprocessing' / 'aligned_gt_640'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Calculate scaling ratios
    original_width = 1920
    original_height = 1080
    target_width = 640
    target_height = 360
    
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    print(f"Prj directory: {prj_dir}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scaling ratios: x={scale_x:.4f}, y={scale_y:.4f}")
    print("-" * 50)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Process all GT files
    gt_files = sorted(input_dir.glob('*.txt'))
    
    if not gt_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return
    
    for input_file in gt_files:
        output_file = output_dir / input_file.name
        
        print(f"Processing: {input_file.name}")
        resize_gt_file(input_file, output_file, scale_x, scale_y)
        
        # Count lines
        with open(input_file, 'r') as f:
            input_lines = sum(1 for line in f if line.strip())
        with open(output_file, 'r') as f:
            output_lines = sum(1 for line in f if line.strip())
        
        print(f"  - Input lines: {input_lines}")
        print(f"  - Output lines: {output_lines}")
        print(f"  - Saved to: {output_file}")
    
    print("-" * 50)
    print(f"Completed! Processed {len(gt_files)} files in total")


if __name__ == '__main__':
    main()