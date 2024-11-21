import os
import shutil
import cv2
import csv
import numpy as np
import time
import peakutils
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

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

    
def smartKeyframeDetection(source, dest, bucket_size_in_frames, threshold=0.3, output_dir=None,  minimum_frames_between = 24, maximum_frames_between=30):
    keyframePath = output_dir if output_dir else os.path.join(dest, "keyFrames")

    selected_indices = keyframeDetectionByChunks(source, dest, bucket_size_in_frames, 0, output_dir, minimum_frames_between)
    
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
    
    print(f"For {source}, selected {len(selected_indices)} frames, and inserted {len(filtered_indices) - len(selected_indices)} frames.")
    
    #   Save keyframe to output_dir
    keyframes = []
    for idx in filtered_indices:
        output_path = os.path.join(keyframePath, f"frame{lstfrm[idx]:04d}.jpg")
        # output_path = os.path.join(keyframePath, f"bucket{bucket_idx}_frame{frame_number}_{timestamp:.2f}.jpg")
        cv2.imwrite(output_path, full_color[idx])
        keyframes.append(full_color[idx])
        log_message = f"keyframe at {timestamp:.2f} sec (frame {lstfrm[idx]})."
        with open(f"logs/keyframe_detection_{os.path.basename(source)}.log", "a") as log_file:
            log_file.write(log_message + "\n")
            
        if verbose:
            print(log_message)
    
    cv2.destroyAllWindows()
    return keyframes
    
    
def keyframeDetectionByChunks(source, dest, number_frames_per_bucket, threshold=0, output_dir=None, minimum_frames_between = 24):
    
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

    # Prepare output directories
    clear_directory(dest)
    keyframePath = output_dir if output_dir else os.path.join(dest, "keyFrames")
    imageGridsPath = os.path.join(dest, "imageGrids")
    csvPath = os.path.join(dest, "csvFile")
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    # Open video file
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print('Video length (frames):', length)
    # print('FPS:', fps)

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
        start_idx = bucket[0]  # First frame of the bucket
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
    
        # if len(indices) > top_k:
        #     selected_indices = sorted(set(indices[:top_k]))
        # else:
        #     selected_indices = sorted(set(indices))

        # # Save keyframes and log
        for idx in selected_indices:
            frame_number = bucket_frames[idx]
            timestamp = bucket_timeSpans[idx]
        all_selected_indices = all_selected_indices + selected_indices
        #     # Save keyframe to output_dir
        #     output_path = os.path.join(keyframePath, f"frame{frame_number:04d}.jpg")
        #     # output_path = os.path.join(keyframePath, f"bucket{bucket_idx}_frame{frame_number}_{timestamp:.2f}.jpg")
        #     cv2.imwrite(output_path, bucket_images[idx])
        #     log_message = f"Bucket {bucket_idx}, keyframe at {timestamp:.2f} sec (frame {frame_number})."
        #     with open(f"logs/keyframe_detection_{os.path.basename(source)}.log", "w+") as log_file:
        #         log_file.write(log_message + "\n")
                
        #     if verbose:
        #         print(log_message)

    cv2.destroyAllWindows()

    return all_selected_indices


def clear_directory(directory):
    if os.path.exists(directory):
        # Remove all files and subdirectories in the specified directory
        shutil.rmtree(directory)
        # Recreate the empty directory
        os.makedirs(directory)
