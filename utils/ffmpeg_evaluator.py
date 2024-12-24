import random
import os
import subprocess
import json
import shutil
from tabulate import tabulate


class FFmpegEvaluator:
    def get_video_info(self, video_path):
        """Get detailed video configuration, including duration and frame rate."""
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        info = json.loads(result.stdout)

        # Parse duration and frame rate
        duration = float(info['format']['duration'])
        avg_frame_rate = info['streams'][0]['avg_frame_rate']
        fps = eval(avg_frame_rate)  # Convert avg_frame_rate fraction to float

        return duration, fps

    def generate_video_from_frames(self, frame_dir, output_path, fps):
        """Generate a video from selected frames with adjusted duration."""
        num_frames = len([name for name in os.listdir(
            frame_dir) if os.path.isfile(os.path.join(frame_dir, name))])
        new_duration = num_frames / fps  # Calculate new duration
        command = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{frame_dir}/frame%04d.jpg',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-t', str(new_duration),  # Set new duration
            '-crf', '40',
            '-y',  # Overwrite existing file
            output_path
        ]
        process = subprocess.run(command, capture_output=True, text=True)

        if process.stderr:
            print("FFmpeg Errors:")
            print(process.stderr)

    def evaluate_smaller_video(self, video_path, output_path, frames_path):
        """Evaluate the process of generating a smaller video with reduced frames."""
        duration, fps = self.get_video_info(video_path)
        original_size = os.path.getsize(video_path)
        
        interim_frames_path = f"{frames_path}/interim_frames"
        
        os.makedirs(interim_frames_path, exist_ok=True)
        
        # the reason for doing this is that ffmpeg requires frames to be of the format frame0001.jpg, frame0002.jpg, etc while we are also saving the  timestamps in the filename to be able to interpolate correctly and stitch correctly
        
        for filename in os.listdir(frames_path):
            file_path = os.path.join(frames_path, filename)
            if os.path.isfile(file_path):  
               
                new_name_parts = filename.split("_")
                number_part = new_name_parts[0][5:]
                new_name = "frame" + number_part.zfill(4) + ".jpg" 
                new_file_path = os.path.join(interim_frames_path, new_name)
                shutil.copy(file_path, new_file_path)
        
        
        frame_dir = interim_frames_path
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frames_size = sum(
            os.path.getsize(os.path.join(frame_dir, f))
            for f in os.listdir(frame_dir)
            if os.path.isfile(os.path.join(frame_dir, f))
        )

        temp_output = f"{output_path}/{video_name}_reduced_frames_{len(os.listdir(frame_dir))}.avi"
        
        os.makedirs(output_path, exist_ok=True)
        directory = os.path.dirname(temp_output)
        os.makedirs(directory, exist_ok=True)

        # Check if the file exists; if not, create an empty file
        if not os.path.exists(temp_output):
            with open(temp_output, 'w') as f:
                f.write("")  # Write an empty string to create the file
        self.generate_video_from_frames(frame_dir, temp_output, fps)
        compressed_size = os.path.getsize(temp_output)
        
        savings_from_original_video = (
            (original_size - compressed_size) / original_size) * 100
        
        savings_from_keyframes_video = (
            (frames_size - compressed_size) / original_size) * 100

        table_data = [
            ["Video Property", "Value"],
            ["Original Frame Rate", f"{fps} fps"],
            ["Original Duration", f"{duration:.2f} seconds"],
            ["Original Size", f"{original_size / (1024 * 1024):.2f} MB"],
            ["Frame Directory Size", f"{frames_size / (1024 * 1024):.2f} MB"],
            ["Compressed Video Size",
                f"{compressed_size / (1024 * 1024):.2f} MB"],
            ["Size Reduction from original vidd",
                f"{savings_from_original_video:.2f}%"],
            ["Size Reduction from keyframes",
                f"{savings_from_keyframes_video:.2f}%"],
        ]

        table_output_path = f"./logs/{video_name}_keyframe_video_summary.txt"
        with open(table_output_path, "w") as file:
            file.write(
                tabulate(table_data, headers="firstrow", tablefmt="grid"))
            
        shutil.rmtree(interim_frames_path)
        
        return compressed_size, savings_from_original_video, savings_from_keyframes_video
