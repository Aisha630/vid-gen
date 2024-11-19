import argparse

from KeyFrameDetector.key_frame_detector import keyframeDetection

def main():
    parser = argparse.ArgumentParser()
    # horses="/home/iml2/inputs/horses.mp4"
    horses="/VideoReconstruction/input/horses.mp4"
    results = "/VideoReconstruction/keyframe-detection/results"
    # parser.add_argument('-s','--source', help='source file', default=horses)
    # parser.add_argument(
    #     '-d', '--dest', help='destination folder', default="/home/iml2/inputs")
    # parser.add_argument('-t','--Thres', help='Threshold of the image difference', default=0.3)

    # args = parser.parse_args()


    # keyframeDetection(args.source, args.dest, float(args.Thres))
    keyframeDetection(horses, results, 0.3)

if __name__ == '__main__':
    main()
