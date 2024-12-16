import os
import shutil
import cv2
import csv
import numpy as np
import time
import peakutils
from diffusers.utils import export_to_video
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics
import PIL 
import IPython.display as display

__all__ = ["keyframeDetection", "clear_directory" "keyframeDetectionByChunks", "smartKeyframeDetection"]


def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=True, fraction=1):
    clear_directory(dest)
    keyframePath = dest + "/keyFrames"
    imageGridsPath = dest + "/imageGrids"
    csvPath = dest + "/csvFile"
    path2file = csvPath + "/output.csv"
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = length / fps
    k = int(video_duration // fraction)
    print("Video duration:", video_duration)

    if not cap.isOpened():
        print("Error opening video file")
        return

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None

    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the frame could not be read

        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Current frame index
        timestamp = frame_number / fps  # Calculate the timestamp in seconds based on frame index and FPS
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        timeSpans.append(timestamp)  # Use calculated timestamp based on FPS
        lastFrame = blur_gray

    cap.release()

    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y - base, Thres, min_dist=1)

    top_k_indices = sorted(indices, key=lambda i: lstdiffMag[i], reverse=True)[:k] if len(indices) >= k else indices
    top_k_indices = sorted(top_k_indices, key=lambda i: timeSpans[i])

    if plotMetrics:
        plot_metrics(top_k_indices, lstfrm, lstdiffMag)

    cnt = 1
    with open(path2file, "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        for x in top_k_indices:
            timestamp = timeSpans[x]
            cv2.imwrite(os.path.join(keyframePath, f"{cnt}_{timestamp:.2f}.jpg"), full_color[x])
            log_message = f"keyframe {cnt} happened at {timestamp:.2f} sec."
            # if verbose:
            # print(log_message)
            writer.writerow([log_message])
            cnt += 1

    cv2.destroyAllWindows()

def find_next_best_frame(start_index, last_index, bucket_diffMag, minimum_frames_between, maximum_frames_between, threshold=None):
    """
    Given start and end indices, finds additional frame indices to insert between them such that the gaps between frames are less than or equal to maximum_frames_between and at least minimum_frames_between apart.

    Args:
        start_index (int): Starting index in bucket.
        last_index (int): Ending index in bucket.
        bucket_diffMag (list): List of difference magnitudes for frames in the bucket.
        minimum_frames_between (int): Minimum frames between selected frames.
        maximum_frames_between (int): Maximum frames between selected frames.
        threshold (float, optional): Threshold to filter frames based on difference magnitude.

    Returns:
        list: List of frame indices to insert between start_index and last_index.
    """
    indices_to_insert = []
    intervals = [(start_index, last_index)]

    while intervals:
        s, e = intervals.pop()
        if e - s <= maximum_frames_between:
            continue  # Gap is acceptable, no need to insert frames

        candidate_start = s + minimum_frames_between
        candidate_end = e - minimum_frames_between

        if candidate_start > candidate_end:
            continue  # Cannot find a frame that satisfies minimum_frames_between constraint

        # Get candidates and their diffMags
        candidates = list(range(candidate_start, candidate_end + 1))
        candidate_diffMags = [bucket_diffMag[i] for i in candidates]

        if threshold is not None:
            # Filter candidates based on threshold
            candidates_filtered = []
            candidate_diffMags_filtered = []
            for idx, diffMag in zip(candidates, candidate_diffMags):
                if diffMag >= threshold:
                    candidates_filtered.append(idx)
                    candidate_diffMags_filtered.append(diffMag)
            candidates = candidates_filtered
            candidate_diffMags = candidate_diffMags_filtered

        if not candidates:
            continue  # No candidates to insert

        # Find the candidate with the highest diffMag
        max_diffMag = max(candidate_diffMags)
        max_index = candidates[candidate_diffMags.index(max_diffMag)]

        indices_to_insert.append(max_index)

        # Add new intervals to process
        intervals.append((s, max_index))
        intervals.append((max_index, e))

    return indices_to_insert

    
def smartKeyframeDetection(source, bucket_size_in_frames, threshold=0.3, output_dir=None,minimum_frames_between = 24, maximum_frames_between=30, segment_fps=30, interim_videos_dir=None, top_k_no_interpolation=0):
    """
    Detects keyframes in a video and saves them to the specified directory. Optionally, segments with high motion can be saved as interim videos as is without interpolatoin
    Args:
        source (str): Path to the source video file.
        dest (str): Destination directory for saving keyframes.
        bucket_size_in_frames (int): Size of the frame bucket for initial keyframe detection.
        threshold (float, optional): Threshold for frame difference magnitude to consider a frame as keyframe. Default is 0.3.
        output_dir (str, optional): Directory to save the keyframes. If None, keyframes will be saved in a subdirectory of `dest`. Default is None.
        minimum_frames_between (int, optional): Minimum number of frames between consecutive keyframes. Default is 24.
        maximum_frames_between (int, optional): Maximum number of frames between consecutive keyframes. Default is 30.
        segment_fps (int, optional): Frames per second for the interim videos. Default is 30.
        interim_videos_dir (str, optional): Directory to save interim videos with high motion. If None, interim videos will not be saved. Default is None.
        top_k_no_interpolation (int, optional): Number of top motion segments to save without interpolation. Default is 0.
    Returns:
        list: List of keyframe images.
    
    """
    keyframePath = output_dir 

    selected_indices = keyframeDetectionByChunks(source, bucket_size_in_frames, 0, output_dir, minimum_frames_between)
    
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error opening video file")
        return

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None

    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the frame could not be read

        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Current frame index
        timestamp = frame_number / fps  # Calculate the timestamp in seconds based on frame index and FPS
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        timeSpans.append(timestamp)  # Use calculated timestamp based on FPS
        lastFrame = blur_gray

    cap.release()

    # if we put this here there is a chance that last frame gets eiliminated if the last frame is too close to second last
    if 0 not in selected_indices:
        selected_indices = [0] + selected_indices
    if length - 1 not in selected_indices:
        selected_indices.append(length - 1)
        
    filtered_indices = [selected_indices[0]]
    for i in range(1, len(selected_indices)):
        start_index = filtered_indices[-1]
        last_index = selected_indices[i]
        if last_index - start_index <= maximum_frames_between:
            filtered_indices.append(last_index)
            continue
        indices_to_insert = find_next_best_frame(start_index, last_index, lstdiffMag, minimum_frames_between, maximum_frames_between, threshold)
        filtered_indices.extend(indices_to_insert)
        filtered_indices.append(last_index)
    
    filtered_indices = sorted(set(filtered_indices + [length -1 ]))
    
    if interim_videos_dir:
        os.makedirs(interim_videos_dir, exist_ok=True)
    motion_percentages = [0] * (len(filtered_indices) - 1)
    for idx in range(len(filtered_indices)):
        # see if the net motion between filtered_indices[idx] and filtered_indices[idx+1] is more than a threshold
        if filtered_indices[idx] == filtered_indices[-1]:
            break
        percentage_of_motion_in_bucket = sum(lstdiffMag[filtered_indices[idx]:filtered_indices[idx+1]]) / sum(lstdiffMag)
        # print(f"Percentage of motion in bucket_{idx} is ", percentage_of_motion_in_bucket)
        motion_percentages[idx] = (percentage_of_motion_in_bucket, idx)
    not_interpolated_indices = []
    
    def save_all_frames_in_bucket(index: int):
        # Save the entire bucket as a segment_<bucket_idx> mp4
        print(f"Frames {filtered_indices[index]} to {filtered_indices[index+1]} have high motion, saving the entire bucke at {interim_videos_dir}")
        
        
        all_bucket_frames = [full_color[i] for i in range(filtered_indices[index], filtered_indices[index+1])]
        resize_specs = (1024, 576)
        
        for i in range(len(all_bucket_frames)):
            all_bucket_frames[i] = cv2.resize(all_bucket_frames[i], resize_specs)
        segment_path = os.path.join(interim_videos_dir, f"segment_{index}.mp4")
        frame_height, frame_width = all_bucket_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        video_writer = cv2.VideoWriter(
        segment_path, fourcc, segment_fps, (frame_width, frame_height))
        for image in all_bucket_frames:
            video_writer.write(image)

        video_writer.release()
        not_interpolated_indices.append(filtered_indices[index])

    #sort the motion percentages and call the fn on top_k
    motion_percentages.sort(key = lambda x: x[0], reverse=True)
    for i in range(top_k_no_interpolation):
        save_all_frames_in_bucket(motion_percentages[i][1])

    print(f"For {source}, selected {len(selected_indices)} frames, and inserted {len(filtered_indices) - len(selected_indices)} frames.")
    
    #   Save keyframe to output_dir
    keyframes = []
    for idx in filtered_indices:
        output_path = os.path.join(keyframePath, f"frame{lstfrm[idx]}_{timeSpans[idx]:.4f}.jpg")
        if idx in not_interpolated_indices:
            output_path = os.path.join(keyframePath, f"frame<s>{lstfrm[idx]}_{timeSpans[idx]:.4f}.jpg")
        cv2.imwrite(output_path, full_color[idx])
        keyframes.append(full_color[idx])
        log_message = f"keyframe at {timestamp:.2f} sec (frame {lstfrm[idx]})."
        with open(f"logs/keyframe_detection_{os.path.basename(source)}.log", "a") as log_file:
            log_file.write(log_message + "\n")
            
        # if verbose:
        #     print(log_message)
    
    cv2.destroyAllWindows()
    return keyframes
    
    
def keyframeDetectionByChunks(source, number_frames_per_bucket, threshold=0, output_dir=None, minimum_frames_between = 24):
    
    """
    Detects keyframes in a video by processing it in chunks.
    Args:
        source (str): Path to the input video file.
        dest (str): Path to the destination directory where results will be saved.
        number_frames_per_bucket (int): Number of frames to process in each chunk.
        top_k (int, optional): Number of top keyframes to select based on difference magnitude. Defaults to None.
        threshold (float, optional): Threshold for peak detection in difference magnitudes. Defaults to None.
        output_dir (str, optional): Directory to save keyframes. Defaults to None.
        verbose (bool, optional): If True, prints log messages. Defaults to True.
        minimum_frames_between (int, optional): Minimum number of frames between selected keyframes. Defaults to 24.
    Returns:
        list: List of selected keyframe indices.
    """

    keyframePath = output_dir 
    prepare_dirs(keyframePath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error opening video file")
        return []

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None

    # Process frames and compute difference magnitudes
    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the frame could not be read

        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Current frame index
        timestamp = frame_number / fps  # Calculate the timestamp in seconds based on frame index and FPS
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        timeSpans.append(timestamp)  # Use calculated timestamp based on FPS
        lastFrame = blur_gray

    cap.release()

    # Divide into chunks
    buckets = [lstfrm[i : i + number_frames_per_bucket] for i in range(0, len(lstfrm), number_frames_per_bucket)]
    all_selected_indices = []
    
    # Process each bucket
    for bucket_idx, bucket in enumerate(buckets):
        last_of_prev_bucket = sorted(buckets[bucket_idx-1])[-1] if bucket_idx > 0 else 0
        start_idx = last_of_prev_bucket if bucket_idx > 0 else bucket[0]
        end_idx = bucket[-1]  # Last frame of the bucket
        
        # Compute differences for the bucket
        bucket_diffMag = lstdiffMag[start_idx : end_idx + 1]
        bucket_frames = lstfrm[start_idx : end_idx + 1]
        bucket_timeSpans = timeSpans[start_idx : end_idx + 1]
        bucket_images = full_color[start_idx : end_idx + 1]

        # Select top-k or threshold-based frames
        y = np.array(bucket_diffMag)
        base = peakutils.baseline(y, 2)
        
        indices = peakutils.indexes(y - base, threshold, min_dist = minimum_frames_between)
        selected_indices = sorted(indices, key=lambda idx: y[idx], reverse=True)
        
        for idx in selected_indices:
            frame_number = bucket_frames[idx]
            timestamp = bucket_timeSpans[idx]
            all_selected_indices + [frame_number]
        

    cv2.destroyAllWindows()

    return sorted(list(set(all_selected_indices)))


def clear_directory(directory):
    if os.path.exists(directory):
        # Remove all files and subdirectories in the specified directory
        shutil.rmtree(directory)
        # Recreate the empty directory
        os.makedirs(directory)