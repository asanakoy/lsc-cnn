import cv2
import numpy as np
from random import randint 
from copy import deepcopy

color = (255,255,0)

font = cv2.FONT_HERSHEY_DUPLEX
fontColor = (255,255,0)

def draw_count(frame, crowd_count, ignore_polys=[], gt_count=None, alpha=0.5):
    '''
    :param ignore_polys: list of polygons, each polygon being a list of tuples of (x,y) containing at least 3 points in one polygon. 
    '''
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    fontScale = max(int(frame_w/525), 1)
    fontThickness = int(fontScale*2)

    if gt_count:
        text = 'NDPeeps Counted: {} (GT:{})'.format(crowd_count, gt_count)
    else:
        text = 'NDPeeps Counted: {}'.format(crowd_count)


    overlay = deepcopy(frame)
    for poly in ignore_polys:
        cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], (0,0,0))
    # cv2.fillConvexPoly(overlay, np.array([[0,0], [0,200], [frame_w-1,200], [frame_w-1, 0],[0,0]]), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    # print('Drawing {}'.format(text))
    # print(fontScale)
    cv2.putText(frame, text, (20,frame_h-20), font, fontScale, (255,255,255), fontThickness)
    # cv2.putText(frame, text, (20,150), font, fontScale, fontColor, fontThickness)