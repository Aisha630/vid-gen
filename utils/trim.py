import os
from moviepy.editor import VideoFileClip


def trim_videos(input_folder, output_folder, max_duration=10):
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
                # Determine the end time for trimming (min of 10 seconds or video duration)
                trim_duration = min(max_duration, clip.duration)

                # Trim the video to the specified duration
                trimmed_clip = clip.subclip(0, trim_duration)

                # Write the trimmed video to the output folder
                trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

            print(f"Trimmed video saved to {output_path}")


# Usage example
input_folder = "input_old"
output_folder = "output"
trim_videos(input_folder, output_folder, max_duration=10)
