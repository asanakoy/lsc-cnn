import argparse
import glob
import cv2
from pathlib import Path
import os
from os.path import join
from tqdm import tqdm
from model import LSCCNN

from utils_lsccnn import draw_on_image


def is_image_file(p):
    return Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]


def maybe_resize(img):
    h, w = img.shape[:2]
    area = h * w
    factor = (area / 5.5e6) ** 0.5
    new_h = int(h / factor)
    new_w = int(w / factor)
    if factor > 1:
        print(f"resize {(w,h)} -> {(new_w,new_h)}")
        img = cv2.resize(img, (new_w, new_h))
    return img


if __name__ == "__main__":
    default_ckpt_path = "./models/ucfqnrf/train2/snapshots/scale_4_epoch_46.pth"
    parser = argparse.ArgumentParser(description="Dataset_setting")
    parser.add_argument("--ckpt-path", default=default_ckpt_path, help="choose chekpoint to load")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("-o", "--output_dir", default="./output/tmp", help="choose output dir")
    parser.add_argument("-i", "--input_dir", default="./images/", help="choose images dir")
    args = parser.parse_args()

    image_save_dir = args.output_dir
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    network = LSCCNN(checkpoint_path=args.ckpt_path, output_downscale=2)
    network.cuda()
    network.eval()

    img_paths = list(glob.glob(join(args.input_dir, "*")))
    img_paths = list(filter(is_image_file, img_paths))
    for path in tqdm(img_paths, desc="Predict"):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = maybe_resize(image)
        print("img shape:", image.shape)

        pred_dot_map, pred_box_map, boxes, img_out = network.predict_single_image(
            image, nms_thresh=args.threshold
        )
        predicted_count = len(boxes)

        img_out = draw_on_image(img_out, predicted_count)

        image_name = Path(path).stem
        print(f"\n- {image_name}: cnt = {predicted_count}")
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(image_save_dir, image_name + f"_nmsthr{args.threshold}.jpg"), img_out
        )
