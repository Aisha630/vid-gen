import os
import subprocess

def generate_gifs(input_dir, output_dir, fps=20, resolution=(320, -1)):
    """
    Converts all videos in the input directory to GIFs with optimized palettes
    and saves them in the output directory.

    Args:
        input_dir (str): Path to the input directory containing videos.
        output_dir (str): Path to the output directory for GIFs.
        fps (int): Frames per second for the GIF.
        resolution (tuple): Target resolution as (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        video_name = os.path.splitext(video)[0]
        gif_path = os.path.join(output_dir, f"{video_name}.gif")
        palette_path = os.path.join(output_dir, f"{video_name}_palette.png")
        
        print(f"Processing video: {video_path}")
        print(f"GIF output path: {gif_path}")

        # Generate color palette
        palette_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps},scale={resolution[0]}:{resolution[1]}:flags=lanczos,palettegen',
            '-y',  # Overwrite if palette exists
            palette_path
        ]
        try:
            subprocess.run(palette_command, check=True)
            print(f"Palette created at: {palette_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating palette for {video_path}: {e.stderr.decode()}")
            continue

        # Generate GIF with palette
        gif_command = [
            'ffmpeg',
            '-i', video_path,
            '-i', palette_path,
            '-lavfi', f'fps={fps},scale={resolution[0]}:{resolution[1]}:flags=lanczos[x];[x][1:v]paletteuse',
            '-gifflags', '+transdiff',
            '-y',  # Overwrite existing file
            gif_path
        ]
        try:
            subprocess.run(gif_command, check=True)
            print(f"GIF created at: {gif_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating GIF for {video_path}: {e.stderr.decode()}")
        os.remove(palette_path)

    print("All videos processed and GIFs created!")

if __name__ == "__main__":
    # Define the input and output directories
    input_dir = "./examples/generated"  # Change to your input directory
    output_dir = "./examples/generated_gifs"  # Change to your desired output directory

    # Call the generate_gifs function
    generate_gifs(input_dir, output_dir)
