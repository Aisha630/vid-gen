import os
import shutil
import cv2
import csv
import numpy as np
import time
import peakutils
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

# def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=True, fraction=2):
#     clear_directory(dest)
#     keyframePath = dest+'/keyFrames'
#     imageGridsPath = dest+'/imageGrids'
#     csvPath = dest+'/csvFile'
#     path2file = csvPath + '/output.csv'
#     prepare_dirs(keyframePath, imageGridsPath, csvPath)

#     cap = cv2.VideoCapture(source)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     video_duration = length / fps  
#     k = int(video_duration // fraction)   
#     print('video duration: ', video_duration)
#     if (cap.isOpened()== False):
#         print("Error opening video file")

#     lstfrm = []
#     lstdiffMag = []
#     timeSpans = []
#     images = []
#     full_color = []
#     lastFrame = None
#     Start_time = time.process_time()
    
#     # Read until video is completed
#     for i in range(length):
#         ret, frame = cap.read()
#         grayframe, blur_gray = convert_frame_to_grayscale(frame)

#         frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
#         lstfrm.append(frame_number)
#         images.append(grayframe)
#         full_color.append(frame)
#         if frame_number == 0:
#             lastFrame = blur_gray

#         diff = cv2.subtract(blur_gray, lastFrame)
#         diffMag = cv2.countNonZero(diff)
#         lstdiffMag.append(diffMag)
#         stop_time = time.process_time()
#         time_Span = stop_time-Start_time
#         timeSpans.append(time_Span)
#         lastFrame = blur_gray

#     cap.release()
#     y = np.array(lstdiffMag)
#     base = peakutils.baseline(y, 2)
#     indices = peakutils.indexes(y-base, Thres, min_dist=1)
    
#     top_k_indices = indices
#     top_k_indices = sorted(indices, key=lambda i: lstdiffMag[i], reverse=True)[:k] # get the top k keyframes
#     if len(indices) < k:
#         top_k_indices = indices
#     else:
#         step = len(indices) // k
#         top_k_indices = indices[::step][:k]
#     top_k_indices = sorted(top_k_indices, key=lambda i: timeSpans[i]) # sort the top k keyframes by time
    
#     ##plot to monitor the selected keyframe
#     if (plotMetrics):
#         # plot_metrics(indices, lstfrm, lstdiffMag)
#         plot_metrics(top_k_indices, lstfrm, lstdiffMag)

#     cnt = 1
#     # for x in indices:
#     for x in top_k_indices:
#         cv2.imwrite(os.path.join(keyframePath, str(cnt) + '_' + str(timeSpans[x]) + '.jpg'), full_color[x])
#         cnt +=1
#         log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
#         if(verbose):
#             print(log_message)
#         with open(path2file, 'w') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerows(log_message)
#             csvFile.close()

#     cv2.destroyAllWindows()
def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=True, fraction=1):
    clear_directory(dest)
    keyframePath = dest + '/keyFrames'
    imageGridsPath = dest + '/imageGrids'
    csvPath = dest + '/csvFile'
    path2file = csvPath + '/output.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = length / fps  
    k = int(video_duration // fraction)
    print('Video duration:', video_duration)

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
    with open(path2file, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for x in top_k_indices:
            timestamp = timeSpans[x]
            cv2.imwrite(os.path.join(keyframePath, f'{cnt}_{timestamp:.2f}.jpg'), full_color[x])
            log_message = f'keyframe {cnt} happened at {timestamp:.2f} sec.'
            # if verbose:
                # print(log_message)
            writer.writerow([log_message])
            cnt += 1

    cv2.destroyAllWindows()

    
def clear_directory(directory):
    if os.path.exists(directory):
        # Remove all files and subdirectories in the specified directory
        shutil.rmtree(directory)
        # Recreate the empty directory
        os.makedirs(directory)
