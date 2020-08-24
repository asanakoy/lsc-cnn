import argparse
import os
import cv2
import time
import configparser

# import numpy as np

from crowdcount_lsccnn import CrowdCounter
from drawer import draw_count

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to process")
parser.add_argument("--cfg", help="Config file", default="image.cfg")
args = parser.parse_args()

assert os.path.exists(args.image), "Image does not exist!"

assert os.path.exists(args.cfg), "Config file given does not exist!"
config = configparser.ConfigParser()
config.read(args.cfg)

# compress_ratio = float(config['IMAGE']['CompressRatio'])
# assert compress_ratio > 0,'compress ratio given is negative.'
max_dim = float(config["IMAGE"]["MaxDim"])
if "OmitScales" in config["IMAGE"]:
    omit_scales = [int(x) for x in config["IMAGE"]["OmitScales"].split(",")]
else:
    omit_scales = []

outputCount = config["IMAGE"].getboolean("OutputCount")

im = cv2.imread(args.image)
# find ratio to compress
im_h = im.shape[0]
im_w = im.shape[1]
max_size = max(im_h, im_w)
compress_ratio = max_size / max_dim
# print('height: {}'.format(im_h))
# print('width: {}'.format(im_w))
# print('ratio: {}'.format(compress_ratio))

cc = CrowdCounter(im_w, im_h, compress_ratio=compress_ratio, omit_scales=omit_scales)

tic = time.time()
show_img, count = cc.visualise_count(im, omit_scales=omit_scales)

if outputCount:
    draw_count(show_img, count)

toc = time.time()
total_dur = toc - tic

print("Inference time: {}".format(total_dur))

# cv2.imwrte('./data/demo.jpg', show_img)
cv2.imshow("image", show_img)

while True:
    k = cv2.waitKey(1)
    if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
