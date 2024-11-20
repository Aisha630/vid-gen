import os
import shutil
import cv2
import csv
import numpy as np
import time
import peakutils
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

__all__ = ["keyframeDetection", "clear_directory" "keyframeDetectionByChunks"]


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


def keyframeDetectionByChunks(source, dest, x, k=None, Thres=None, output_dir=None, verbose=True):
    """
    Detects keyframes in a video by processing it in chunks.

    Parameters:
    source (str): Path to the input video file.
    dest (str): Path to the destination directory where results will be saved.
    x (int): Number of frames per chunk.
    k (int, optional): Number of top keyframes to select per chunk. If None, all frames are considered.
    Thres (float, optional): Threshold for peak detection. If None, no thresholding is applied.
    output_dir (str, optional): Directory to save keyframes. If None, a default directory is used.
    verbose (bool, optional): If True, prints log messages. Default is True.

    Returns:
    list: List of selected keyframe indices.

    Notes:
    - The function processes the video in chunks of 'x' frames.
    - It computes the difference magnitude between consecutive frames to detect keyframes.
    - Keyframes are selected based on either the top 'k' difference magnitudes or a threshold 'Thres'.
    - The first and last frames of each chunk are always included as keyframes.
    - Keyframes are saved as images in the specified output directory.
    """
    # Prepare output directories
    clear_directory(dest)
    keyframePath = output_dir if output_dir else os.path.join(dest, "keyFrames")
    imageGridsPath = os.path.join(dest, "imageGrids")
    csvPath = os.path.join(dest, "csvFile")
    path2file = os.path.join(csvPath, "output.csv")
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
    buckets = [lstfrm[i : i + x] for i in range(0, len(lstfrm), x)]
    all_selected_frames = []

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
        if Thres:
            y = np.array(bucket_diffMag)
            base = peakutils.baseline(y, 2)
            indices = peakutils.indexes(y - base, Thres, min_dist=1)
        else:
            indices = range(len(bucket_diffMag))

        if k:
            top_k_indices = sorted(indices, key=lambda i: bucket_diffMag[i], reverse=True)[:k]
        else:
            top_k_indices = indices

        # Ensure first and last frames are always included
        selected_indices = set(top_k_indices)
        selected_indices.add(0)  # First frame of the bucket
        selected_indices.add(len(bucket_diffMag) - 1)  # Last frame of the bucket
        selected_indices = sorted(selected_indices)

        # Save keyframes and log
        for idx in selected_indices:
            frame_number = bucket_frames[idx]
            timestamp = bucket_timeSpans[idx]
            all_selected_frames.append(frame_number)

            # Save keyframe to output_dir
            output_path = os.path.join(keyframePath, f"frame{frame_number:04d}.jpg")
            # output_path = os.path.join(keyframePath, f"bucket{bucket_idx}_frame{frame_number}_{timestamp:.2f}.jpg")
            cv2.imwrite(output_path, bucket_images[idx])
            log_message = f"Bucket {bucket_idx}, keyframe at {timestamp:.2f} sec (frame {frame_number})."
            if verbose:
                print(log_message)

    cv2.destroyAllWindows()

    return all_selected_frames


def clear_directory(directory):
    if os.path.exists(directory):
        # Remove all files and subdirectories in the specified directory
        shutil.rmtree(directory)
        # Recreate the empty directory
        os.makedirs(directory)
