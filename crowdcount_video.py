import argparse
import os
import cv2
import time
import configparser
import numpy as np

from crowdcount_lsccnn import CrowdCounter
from drawer import draw_count

parser = argparse.ArgumentParser()
parser.add_argument("video", help="video to process")
parser.add_argument("--cfg", help="Config file to specify no count/deadzones", default="ndp.cfg")
parser.add_argument("--checkpoly", help="Flag to check polygons only", action="store_true")
parser.add_argument("--displaypoly", help="Flag to display polygons", action="store_true")
args = parser.parse_args()

assert os.path.exists(args.video), "Video does not exist!"
basename = os.path.basename(args.video).split(".")[0]

cap = cv2.VideoCapture(args.video)
frame_w = cap.get(3)
frame_h = cap.get(4)
vid_fps = cap.get(5)
assert os.path.exists(args.cfg), "Config file given does not exist!"
config = configparser.ConfigParser()
config.read(args.cfg)

skyline = int(config["SKYLINE"]["Y"])
skybuffer = 10
skyline_polygon = [
    (0 - skybuffer, 0 - skybuffer),
    (0 - skybuffer, skyline),
    (frame_w - 1 + skybuffer, skyline),
    (frame_w - 1 + skybuffer, 0 - skybuffer),
]
dead_polygons = []
for poly in config["POLYGONS"]:
    polystring = [int(x) for x in config["POLYGONS"][poly].split(",")]
    dead_polygons.append(list(zip(polystring[::2], polystring[1::2])))
print("Num of custom dead polygons: {}".format(len(dead_polygons)))
dead_polygons.append(skyline_polygon)

compress_ratio = float(config["VIDEO"]["CompressRatio"])
assert compress_ratio > 0, "compress ratio given is negative."
if "OmitScales" in config["VIDEO"]:
    omit_scales = [int(x) for x in config["VIDEO"]["OmitScales"].split(",")]
else:
    omit_scales = []

sample_rate_sec = float(config["VIDEO"]["SampleRate"])
if sample_rate_sec > 0:
    sample_rate_frame = int(vid_fps * sample_rate_sec)
else:
    sample_rate_frame = 1

display = config["VIDEO"].getboolean("Display")
outputVideo = config["VIDEO"].getboolean("OutputVideo")
outputCSV = config["VIDEO"].getboolean("OutputCSV")

if args.checkpoly:
    display = True
    outputVideo = False
    outputCSV = False

if display:
    print("Displaying outputs")
    cv2.namedWindow("LSC-CNN", cv2.WINDOW_NORMAL)
else:
    print("Hiding outputs")

if outputCSV:
    print("Outputing to CSV")
    csv_dir = os.path.join("outputs", basename + "_output_CSVs")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    assert os.path.isdir(csv_dir), "CSV dir path is not a dir!"
    csv_path = os.path.join(csv_dir, basename + "_intermediate.csv")
    out_csv = open(csv_path, "w")
    out_csv.write("TimeStamp(sec), FrameCount, PeopleCount\n")
else:
    print("Not outputing to CSV.")
cc = CrowdCounter(
    frame_w,
    frame_h,
    compress_ratio=compress_ratio,
    omit_scales=omit_scales,
    ignore_polys=dead_polygons,
)
resized_dead_polygons = cc.ignore_polys_raw

if outputVideo:
    print("Outputing to video file")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out_vid = cv2.VideoWriter(os.path.join('outputs',basename+'_cc.avi'),fourcc, vid_fps, (int(frame_w), int(frame_h)))
    out_vid = cv2.VideoWriter(
        os.path.join("outputs", basename + "_crowdcount.mp4"),
        fourcc,
        vid_fps,
        (int(cc.new_img_w), int(cc.new_img_h)),
    )
else:
    print("Not outputing to video file")

total_dur = 0
frame_count = 0
total_dur_count = 0
# frame_skip = 1
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(frame_count)
            break
        frame_count += 1
        # if frame_count%frame_skip != 0:
        #     continue
        if (frame_count - 1) % sample_rate_frame != 0:
            continue
        tic = time.time()
        show_img, count = cc.visualise_count(frame,)
        # print('Current crowd count @ {}s: {}'.format(round((frame_count-1)/vid_fps,2), count))
        if display or outputVideo:
            if args.displaypoly:
                draw_count(show_img, count, ignore_polys=resized_dead_polygons)
            else:
                draw_count(show_img, count, ignore_polys=[])
        # cv2.imwrite(os.path.join(out_dir,'{}.png'.format(frame_count)),frame)
        if outputCSV:
            out_text = "{},{},{}\n".format(
                round((frame_count - 1) / vid_fps, 2), frame_count, count
            )
            # print(out_text)
            out_csv.write(out_text)
        if outputVideo:
            out_vid.write(show_img)
        if display:
            cv2.imshow("LSC-CNN", show_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # cv2.waitKey(5)
        toc = time.time()
        total_dur += toc - tic
        total_dur_count += 1

except Exception as e:
    print(e)
finally:
    cap.release()
    if outputVideo:
        out_vid.release()
    if outputCSV:
        out_csv.close()
    if total_dur_count:
        print("Avrg inference time:{}".format(total_dur / total_dur_count))
