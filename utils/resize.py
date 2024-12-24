import os
import cv2
import subprocess

def resize_videos(input_dir, output_dir, resolution=(1024, 576)):
    """
    Resizes all videos in the input directory to the specified resolution and saves them in the output directory.

    Args:
        input_dir (str): Path to the input directory containing videos.
        output_dir (str): Path to the output directory for resized videos.
        resolution (tuple): Target resolution as (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        video_name = os.path.splitext(video)[0]
        output_path = os.path.join(output_dir, f"{video_name}.mp4")
        
        print(f"Processing video: {video_path}")
        print(f"Resized video output path: {output_path}")

        # Load video to get properties
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()

        # FFmpeg command to resize the video
        command = [
            'ffmpeg',
            '-i', video_path,  # Input video
            '-r', str(fps),  # Set frame rate
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Resize to target resolution
            '-c:v', 'libx264',  # Use H.264 codec
            '-pix_fmt', 'yuv420p',  # Set pixel format
            '-t', str(duration),  # Set duration
            '-y',  # Overwrite existing file
            output_path
        ]

        # Run the command
        try:
            subprocess.run(command, check=True, capture_output=True)
            print(f"Resized video saved at: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error resizing video {video_path}: {e.stderr.decode()}")

    print("All videos processed.")

if __name__ == "__main__":
    # Define the input and output directories
    input_dir = "/home/iml1/Desktop/input_videos copy"  # Change to your input directory
    output_dir = "./examples/original"  # Change to your desired output directory

    # Call the resize_videos function
    resize_videos(input_dir, output_dir)
