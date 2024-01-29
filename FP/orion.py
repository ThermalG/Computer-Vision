import sys
import cv2
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt5.QtWidgets import QApplication, QFileDialog

PATH = './Data/'
WRITE_VIDEO = True  # choose if you need a processed video (saving to PATH) or just statistics
CALC_SPEED = True  # choose if you need speed estimation
app = QApplication.instance() or QApplication([])  # QApp should only be created once


# REF[4][5][8]
class OFBuffer:  #! SEMI-FINISHED
    def __init__(self, size = 5):
        self.size = size
        self.frames = []
        self.flow_data = []

    def add(self, frame, flow):
        if len(self.frames) >= self.size:
            self.frames.pop(0)
            self.flow_data.pop(0)
        self.frames.append(frame)
        self.flow_data.append(flow)

    def getter(self):  # get the latest frame and its flow data
        if self.frames:
            return self.frames[-1], self.flow_data[-1]
        return None, None


# buffer = OF_Buffer(size = 10)   # depending on available memory. adjust along with S & CONTIGUITY
def optic_flow(frame_prev, frame, gpu = False, roi = .5):   # only supporting NVIDIA GPUs
    if gpu and cv2.cuda.getCudaEnabledDeviceCount():    # build your own OpenCV using CMake
        g1 = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame_prev), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame), cv2.COLOR_BGR2GRAY)
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(.5, False, 15, 3, 5, 1.2, 0)
        flow = gpu_flow.calc(g1, g2, None).download()
    else:
        g1, g2 = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # noinspection PyTypeChecker
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, .5, 3, 15, 3, 5, 1.2, 0)  ##
    # buffer.add(frame, flow)
    return cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]  # magnitude


def radar(height, f):  #! UNUSED. intend to help differentiate between parked & mobile vehicles. REF[9]
    """
    Estimate the distance of a vehicle from a monocular camera system
        :param f: Focal length in mm.
        :param height: Real height of the viewpoint vehicle in mm.
        :return: Estimated distance in meters.
    """
    return (f * height) / (y2 - y1) / 1000  # D = (F * W) / P, converted to meters


def marker(img, lbl = '', status = '', clr = (255, 255, 0)):  # REF[6]
    W, H = x2 - x1, y2 - y1
    # dynamic text relevant to box size (alternatively, use radar())
    size = max(.5, min(1.5, .005 * (H * W) ** .5)) * C
    L = min(W, H) // 8  # L-shaped corner size
    for (x, y) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.line(img, (x, y), (x + (L if x == x1 else -L), y), clr, int(4 * C))
        cv2.line(img, (x, y), (x, y + (L if y == y1 else -L)), clr, int(4 * C))
    if lbl:
        w, h = cv2.getTextSize(lbl, 0, size, 1)[0]
        coord = y1 - h - 1 if y1 - h - 1 >= h else y1 + h + 1
        cv2.putText(img, lbl, (x1, coord), 0, size, (0, 155, 255), int(2 * C))
    if status:
        w, h = cv2.getTextSize(status, 0, .7 * size, 1)[0]
        x = max(0, min(x2 - w, img.shape[1] - w))
        y = max(h, min(y2 + h + 10, img.shape[0] - h))
        cv2.putText(img, status, (x, y), 0, .7 * size, c, int(2 * C))


# CONSTANTS & THRESHOLDS
C = 1.0             # image sacling factor
S = 1               # frame extraction freq. processing always starts from the first frame
assert 0 < C <= 1, "COMPRESSION COEFFICIENT MUST LIE WITHIN (0, 1]"
assert S > 0 and isinstance(S, int), "FRAME SAMPLING INTERVAL MUST BE A POSITIVE INTEGER"
PROXIMITY = 100 * C # proximity threshold between a person and a bike
CONTIGUITY = 4 // S # consecutive frames for a car/person to be considered parked/cyclist
DELTA = 3.6 * C / S # correction factor for speed estimation in km/h. REF[5][8]
ROI = .5            # % of which most movements are expected in pictures. primarily varied with cam angle

# INITIALIZATION
detector = YOLO(PATH + 'yolov8n.pt')  # REF[2]
# tracker = DeepSort(PATH + 'mars-small128.ckpt-68577', nms_max_overlap = .5, max_age = 60, nn_budget = 100)
path, _ = QFileDialog.getOpenFileName(None, "Orion by ThermalG", PATH, "File types (*.mp4 *.avi *.mov)")
if not path:  # REF[7]
    print("NO FILE SELECTED. PROGRAM TERMINATED.")
    sys.exit()
source = cv2.VideoCapture(path)
size = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH) * C), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) * C))
if WRITE_VIDEO:
    fps = int(source.get(cv2.CAP_PROP_FPS))
    #! X264 can take double storage with compromised output
    # noinspection PyUnresolvedReferences
    output = cv2.VideoWriter('orion_' + path, cv2.VideoWriter_fourcc(*'MP4V'), fps / S, size)
v_stack, p_stack, b_stack, cyclist = (defaultdict(list) for _ in range(4))  # tracking history init
parked, v_lastSeen, p_lastSeen = (defaultdict(int) for _ in range(3))
v_ids, p_ids = {}, {}
summary, speed_trap = [], []
frame_prev = None
ctr = 0  # frame counter
s = 0  # current speed
lbl = {2: 'CAR', 5: 'BUS', 7: 'TRUCK'}

# frame-wise Main Loop
with tqdm(total = math.ceil(source.get(cv2.CAP_PROP_FRAME_COUNT) / S), desc = "PROGRESS", unit = "frame") as pbar:
    while source.isOpened():  # REF[1][3]
        read, frame = source.read()
        if not read:  # read failure (file corrupted) or finished
            break
        if ctr % S == 0:
            frame = cv2.resize(frame, size)
            results = detector.track(frame, conf = .5, persist = True, verbose = False)  ##
            if frame_prev is not None:  # REF[4][5]
                mag = optic_flow(frame_prev, frame)
                ## weight-average for smoothing at the cost of latency showing speed change (not affecting max)
                speed_trap.append((s := .9 * s + .1 * DELTA * np.mean(mag[int(ROI * mag.shape[0]):])))
                cv2.putText(frame, f'{s:.1f} KPH', (int(.05 * size[0]), int(.05 * size[1])), 2, 2 * C, (255, 0, 127), int(2 * C)) if WRITE_VIDEO else None
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for id, box in zip(results[0].boxes.id.int().cpu().tolist(), results[0].boxes.data):
                    x1, y1, x2, y2 = map(int, box[:4])
                    idx = int(box[-1])
                    if idx == 2:  # car category
                        v_lastSeen[id] = ctr
                        if id not in v_ids:
                            v_ids[id] = len(v_ids)
                        pos = (box[0], box[1])
                        pos_prev = v_stack[id][-1] if id in v_stack else pos
                        v_stack[id].append(pos)
                        # monitor vehicle movement
                        dist = math.hypot(abs(pos_prev[0] - pos[0]) + abs(pos_prev[1] - pos[1]))
                        diff = 7.5 * C * S * s / DELTA  ##
                        if dist < diff:
                            if dist < .03 * diff:
                                # mobile in the same lane or parked far ahead. not passed nor interested
                                status = 'INDECISIVE'
                            else:
                                parked[id] += 1
                                status = 'PARKED' if parked[id] > CONTIGUITY else 'INDECISIVE'
                        else:
                            parked[id] = 0
                            status = 'MOBILE'
                        c = (0, 0, 255) if status == 'MOBILE' else (0, 255, 0) if status == 'PARKED' else (0, 0, 0)
                        marker(frame, f'CAR{[idx]} {v_ids[id] + 1}', status, c) if WRITE_VIDEO else None
                    elif idx == 0 and frame_prev is not None:  # person category
                        p_lastSeen[id] = ctr
                        pos = (box[0], box[1])
                        p_stack[id].append(pos)
                        if id not in p_ids and id not in cyclist:
                            p_ids[id] = len(p_ids)
                        marker(frame, f'PEDESTRIAN {p_ids[id] + 1}') if WRITE_VIDEO else None
                    elif idx in [1, 3]:  # bicycle and motorcycle categories
                        b_stack[id].append((box[0], box[1]))
                for b_id, p_pos in p_stack.items():  # rule out cyclists
                    for b_pos in b_stack.items():  # REF[6]
                        if all(len(x) >= CONTIGUITY for x in [b_pos, p_pos]) and all(abs(p_pos[-i][0] - b_pos[-i][0].item()) + abs(p_pos[-i][1] - b_pos[-i][1].item()) < PROXIMITY for i in range(1, CONTIGUITY + 1)):
                            cyclist.add(b_id)
                            break
            summary = [  # update counts
                f'MOBILE: {max(0, len([v for v, f in v_lastSeen.items() if f < ctr]) - sum(parked[v] > CONTIGUITY for v in parked))}',
                f'PARKED: {sum(parked[v] > CONTIGUITY for v in parked)}',
                f'PEDESTRIANS: {len([p for p, f in p_lastSeen.items() if f < ctr]) - len(cyclist)}']
            if WRITE_VIDEO:
                for i, info in enumerate(summary, start = 1):
                    cv2.putText(frame, info, (int(.8 * size[0]), int(.03 * i * size[1])), 0, C, (255, 0, 0), int(2 * C))
                # noinspection PyUnboundLocalVariable
                output.write(frame)
            pbar.update(1)
            frame_prev = frame.copy()  # previous frame now becomes the current
        ctr += 1

print('\n'.join(summary))
print(f'Interval Speed Maximum: {max(speed_trap):.1f} KPH')
source.release()
if WRITE_VIDEO:
    output.release()
    cv2.destroyAllWindows()
