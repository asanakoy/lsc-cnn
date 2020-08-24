import os
import cv2
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video", help="video to process")
parser.add_argument("--num", help="num of frames to sample from video", type=int, default=10)
args = parser.parse_args()

assert os.path.exists(args.video), "Video path does not exists"

basename = os.path.basename(args.video).split(".")[0]
out_dir = os.path.join("outputs", basename + "_sample_frames")
if not os.path.exists(out_dir) and not os.path.isdir(out_dir):
    os.makedirs(out_dir)

assert os.path.isdir(out_dir)

cap = cv2.VideoCapture(args.video)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

samples = random.sample(list(range(int(total_frames))), k=args.num)

count = 0
write_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count in samples:
        write_count += 1
        print("Writing: {}/{}".format(write_count, args.num))
        cv2.imwrite(os.path.join(out_dir, "{}.png".format(count)), frame)
