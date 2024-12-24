import cv2
import os
import shutil


def process_and_extract_frames(input_video_path, frames_output_dir, new_width=512, new_height=512, max_frames=24):
    output_video_path = "intermediary_video.mp4"
    # Open the video capture
    cap = cv2.VideoCapture(input_video_path)

    # Get frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height), isColor=True)

    frame_count = 0

    # Check if the frames output directory exists, if not, create it
    if not os.path.exists(frames_output_dir):
        os.makedirs(frames_output_dir)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the specified dimensions
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Write the resized frame to the output video
        out.write(resized_frame)

        # Save the frame to the frames output directory
        frame_name = os.path.join(frames_output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_name, resized_frame)

        frame_count += 1

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # delete the video now
    os.remove(output_video_path)

    print(f"Extracted frames saved in '{frames_output_dir}'.")


# Example usage:
# process_and_extract_frames(input_video_path='path/to/video.mp4',
#                            output_video_path='path/to/output_video.mp4',
#                            frames_output_dir='path/to/frames_output',
#                            new_width=512, new_height=512, max_frames=24)


def strawman_frame_extraction(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:  # Clear the folder if it exists
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = fps

    print(f"Video duration: {duration} seconds, FPS: {fps}")

    frame_count = 0
    frame_number = 0

    while frame_number < total_frames:
        # Set the frame position to capture the next frame at each interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        # Calculate the timestamp for this frame
        timestamp = frame_number / fps

        # Save the frame with count and timestamp
        output_path = os.path.join(output_folder, f"{frame_count + 1}_{timestamp:.2f}.jpg")
        cv2.imwrite(output_path, frame)
        # print(f"Saved frame {frame_count + 1} at {output_path}")

        # Move to the next frame interval
        frame_number += frame_interval * 2
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames")
