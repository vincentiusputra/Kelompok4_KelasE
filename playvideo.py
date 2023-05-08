import copy
import numpy as np
import cv2
from glob import glob


from src import util
from src.body import Body
from src.hand import Hand

import time

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

video_file = "ASLLiterature.mp4"

cap = cv2.VideoCapture(video_file)




while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    a = time.time()
    posed_frame = process_frame(frame, body=not False,
                                       hands=not False)
    print(f"{1 / (time.time()-a)} fps")

    cv2.imshow('frame', posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
