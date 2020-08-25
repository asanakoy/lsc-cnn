import argparse
import glob
import cv2
from pathlib import Path
import os
import sys
from os.path import join
from tqdm import tqdm
from wrappa import WrappaObject, WrappaImage, WrappaText

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
print('FILE_DIR:', FILE_DIR)
print('CUR_DIR:', os.getcwd())
sys.path.append(FILE_DIR)

from model import LSCCNN
from utils_lsccnn import draw_on_image


def maybe_resize(img):
    h, w = img.shape[:2]
    area = h * w
    factor = (area / 5.0e6) ** 0.5
    new_h = int(h / factor)
    new_w = int(w / factor)
    if factor > 1:
        print(f"resize {(w,h)} -> {(new_w,new_h)}")
        img = cv2.resize(img, (new_w, new_h))
    return img


class DSModel:

    def __init__(self, **kwargs):
        ckpt_path = kwargs['ckpt_path']
        self.threshold = kwargs['nms_threshold']
        self.config = kwargs
        print(self.config)
        os.chdir(FILE_DIR)
        print('CUR_DIR (__init__):', os.getcwd())
        self.net = LSCCNN(checkpoint_path=ckpt_path, output_downscale=2)
        self.net.cuda()
        self.net.eval()
        self.cnt_images = 0

    def _predict_single_image(self, image, image_save_dir):
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = maybe_resize(image)
        print("img shape:", image.shape)

        pred_dot_map, pred_box_map, boxes, img_out = self.net.predict_single_image(
            image, nms_thresh=self.threshold
        )
        predicted_count = len(boxes)

        img_out = draw_on_image(img_out, predicted_count)

        image_name = f'{self.cnt_images:05d}'
        print(f"\n- {image_name}: cnt = {predicted_count}")
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(
        #    os.path.join(image_save_dir, image_name + f"_nmsthr{self.threshold}.jpg"), img_out
        #)
        self.cnt_image += 1
        return img_out, predicted_count

    def predict(self, data, **kwargs):
        _ = kwargs

        print(f'predict(payload={data})')
        os.chdir(FILE_DIR)
        print('CUR_DIR:', os.getcwd())
        result_dir = Path('/app/output/').resolve()
        result_dir.mkdir(parents=True, exist_ok=True)

        responses = []
        try:
            # Data is always an array of WrappaObjects
            for obj in data:
                img = obj.image.as_ndarray
                res_img, count = self._predict_single_image(img, str(result_dir))
                wt = WrappaText(f"{count}")
                wi = WrappaImage.init_from_ndarray(
                    payload=res_img,
                    ext=obj.image.ext,
                )
                resp = WrappaObject(wi, wt)
                responses.append(resp)
        except Exception as e:
            print(f'Failed to predict! Exception: {e}')
            print('=================================')
        return responses

