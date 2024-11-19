from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os

# For windows, the below if condition is must.
if __name__ == "__main__":

  # initialize video module
  vd = Video()
  horses="/VideoReconstruction/input/horses.mp4"
  results = "/VideoReconstruction/keyframe-detection/katna/results"
  # number of images to be returned
  no_of_frames_to_returned = 12

  # initialize diskwriter to save data at desired location
  diskwriter = KeyFrameDiskWriter(location=results)

  # Video file path
  video_file_path = horses
  print(f"Input video file path = {video_file_path}")

  # extract keyframes and process data with diskwriter
  vd.extract_video_keyframes(
       no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
       writer=diskwriter
  )