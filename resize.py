import os
from moviepy.editor import VideoFileClip

def resize_videos(input_folder, output_folder, width=1024, height=576):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):  # Process only .mp4 files
            video_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load video with moviepy
            with VideoFileClip(video_path) as clip:
                # Resize and write the output file
                resized_clip = clip.resize(newsize=(width, height))
                resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

            print(f"Resized video saved to {output_path}")

# Usage example
input_folder = "input"
output_folder = "output"
resize_videos(input_folder, output_folder, width=1024, height=576)
