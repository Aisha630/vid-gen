from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os


# For windows, the below if condition is must.
def num_frames_keyframe(input_video_path, output_path, num_frames):
    # initialize video module
    vd = Video()

    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=output_path)

    # extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(no_of_frames=num_frames, file_path=input_video_path, writer=diskwriter)
