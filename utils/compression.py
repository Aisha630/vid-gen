import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
import tempfile

def detect_gpu_encoder():
    """
    Detect available GPU encoders.
    
    Returns:
        list: Available GPU encoder configurations
    """
    gpu_encoders = [
        {
            'codec': 'h264_nvenc',
            'name': 'NVIDIA NVENC',
            'params': ['-gpu', 'any']
        },
        {
            'codec': 'h264_amf',
            'name': 'AMD AMF',
            'params': []
        },
        {
            'codec': 'h264_qsv',
            'name': 'Intel QuickSync',
            'params': []
        }
    ]
    
    available_encoders = []
    for encoder in gpu_encoders:
        try:
            # Check if encoder is available
            check_cmd = subprocess.run(
                f'ffmpeg -encoders | grep {encoder["codec"]}', 
                shell=True, capture_output=True, text=True
            )
            if check_cmd.stdout.strip():
                available_encoders.append(encoder)
        except Exception:
            pass
    
    return available_encoders

def compress_video(input_path, output_path, crf_value, gpu_encoder=None):
    """
    Compress video using GPU acceleration with advanced settings.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output compressed video
        crf_value (int): Constant Rate Factor for compression
        gpu_encoder (dict, optional): GPU encoder configuration
    
    Returns:
        bool: True if compression successful, False otherwise
    """
    try:
        # Base command
        command = [
            'ffmpeg', 
            '-i', input_path
        ]
        
        # Add GPU-specific parameters if available
        if gpu_encoder and gpu_encoder['codec'] == 'h264_nvenc':
            # Special handling for NVENC
            command.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'slow',
                '-gpu', 'any',
                '-cq', str(crf_value),  # NVENC uses -cq instead of -crf
                '-b:v', '0',  # Disable bitrate control
            ] + gpu_encoder['params'])
        elif gpu_encoder:
            # Other GPU encoders
            command.extend([
                '-c:v', gpu_encoder['codec'],
                '-preset', 'slow',
                '-crf', str(crf_value),
            ] + gpu_encoder['params'])
        else:
            # Fallback to CPU encoding
            command.extend([
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', str(crf_value),
            ])
        
        # Common compression parameters
        command.extend([

            '-an',  # Remove audio
            output_path
        ])
        
        # Run the compression command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Compression failed for CRF {crf_value}:")
        print(f"Error output: {e.stderr}")
        return False

def analyze_compression(input_video, output_dir):
    """
    Analyze video compression with GPU acceleration.
    
    Args:
        input_video (str): Path to input video file
        output_dir (str): Directory to save compressed videos
    
    Returns:
        dict: Compression analysis results
    """
    # Detect available GPU encoders
    gpu_encoders = detect_gpu_encoder()
    
    # Select first available GPU encoder
    selected_gpu_encoder = gpu_encoders[0] if gpu_encoders else None
    
    # Print available GPU encoders
    if gpu_encoders:
        print("Available GPU Encoders:")
        for encoder in gpu_encoders:
            print(f"- {encoder['name']} ({encoder['codec']})")
    else:
        print("No GPU encoders detected. Falling back to CPU encoding.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Original video details
    original_size = os.path.getsize(input_video)
    
    # CRF values with compression steps
    crf_values = [0, 10, 20, 30, 40, 50]
    
    # Results storage
    results = {
        'crf_values': [],
        'compressed_videos': [],
        'file_sizes': [],
        'size_reductions': []
    }
    
    # Compression process
    for crf in crf_values:
        # Generate output filename
        output_video = os.path.join(output_dir, f'compressed_crf{crf}.avi')
        
        # Compress video
        if compress_video(input_video, output_video, crf, selected_gpu_encoder):
            # Verify file exists and is not empty
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                # Get compressed video size
                compressed_size = os.path.getsize(output_video)
                
                # Calculate size reduction
                size_reduction = ((original_size - compressed_size) / original_size) * 100
                
                # Store results
                results['crf_values'].append(crf)
                results['compressed_videos'].append(output_video)
                results['file_sizes'].append(compressed_size)
                results['size_reductions'].append(size_reduction)
                
                # Print compression details
                print(f"\nCRF {crf} Compression:")
                print(f"Compressed File Size: {compressed_size} bytes")
                print(f"Size Reduction: {size_reduction:.2f}%")
            else:
                print(f"Compression failed or produced empty file for CRF {crf}")
    
    return results

def plot_compression_results(results, original_size):
    """
    Create visualization of compression results.
    
    Args:
        results (dict): Compression analysis results
        original_size (int): Original video file size
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['crf_values'], results['file_sizes'], marker='o')
    plt.axhline(y=original_size, color='r', linestyle='--', label='Original Size')
    plt.title('Video File Size vs CRF Value (GPU Accelerated)')
    plt.xlabel('CRF Value')
    plt.ylabel('File Size (bytes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('compression_analysis.png')
    plt.close()

def save_compression_details(results):
    """
    Save compression details to CSV.
    
    Args:
        results (dict): Compression analysis results
    """
    # Only proceed if we have results
    if not results['crf_values']:
        print("No compression results to save.")
        return
    
    # Prepare CSV data
    csv_data = []
    for crf, video, size, reduction in zip(
        results['crf_values'], 
        results['compressed_videos'], 
        results['file_sizes'], 
        results['size_reductions']
    ):
        csv_data.append({
            'CRF Value': crf,
            'Compressed Video Path': video,
            'File Size (bytes)': size,
            'Size Reduction (%)': reduction
        })
    
    # Write to CSV
    csv_path = 'compression_details.csv'
    keys = csv_data[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_data)
    
    print(f"\nCompression details saved to {csv_path}")

def main():
    # Path to your input video
    input_video = 'input_videos/cliffs.mp4'
    
    # Temporary directory for compressed videos
    output_dir = "compressed_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    try:
        # Analyze compression
        results = analyze_compression(
            input_video, 
            output_dir
        )
        
        # Original video size
        original_size = os.path.getsize(input_video)
        
        # Plot results
        if results['crf_values']:
            plot_compression_results(results, original_size)
            
            # Save detailed compression information
            save_compression_details(results)
            
            print(f"\nCompressed videos saved in {output_dir}")
            print("Compression analysis plot saved as compression_analysis.png")
        else:
            print("No successful compressions.")
    except:
        print("An error occurred during compression analysis.")
    

if __name__ == '__main__':
    main()