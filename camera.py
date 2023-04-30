try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    dnn = None
from typing import List, Union, Tuple, Dict
from numpy import ndarray

import time
import cv2
import random
import math
import numpy as np

random.seed(0)
MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4

CLASS_NAMES: Tuple = ('ok','rock','like','stop','chs')

COLORS: Dict[str, List] = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASS_NAMES)
}


def get_hw(pro):
    if pro.layout == 'NCHW':
        return pro.shape[2], pro.shape[3]
    elif pro.layout == 'NHWC':
        return pro.shape[1], pro.shape[2]
    else:
        raise NotImplementedError


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def bgr2nv12_opencv(image: ndarray) -> ndarray:
    height, width = image.shape[:2]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:area] = y
    nv12[area:] = uv_packed
    return nv12


def det_postprocess(output: List[ndarray],
                    score_thres: float,
                    iou_thres: float,
                    orin_h: int,
                    orin_w: int,
                    ratio_h: float,
                    ratio_w: float,
                    reg_max: int = 16):
    dfl = np.arange(0, reg_max, dtype=np.float32)
    proposal_boxes: Union[ndarray, List[ndarray]] = []
    proposal_scores: Union[ndarray, List[ndarray]] = []
    proposal_labels: Union[ndarray, List[ndarray]] = []
    for i in range(len(output) // 2):
        stride = 8 << i
        scores_feat = sigmoid(output[i * 2])
        boxes_feat = output[i * 2 + 1]
        max_scores = scores_feat.max(-1)
        label_ids = scores_feat.argmax(-1)
        indices = np.where(max_scores > score_thres)
        hIdx, wIdx = indices
        if not hIdx.size:
            continue
        scores = max_scores[hIdx, wIdx]
        labels = label_ids[hIdx, wIdx]
        boxes = boxes_feat[hIdx, wIdx].reshape(-1, 4, reg_max)
        boxes = softmax(boxes, -1) @ dfl

        shift_boxes = np.concatenate([-boxes[:, :2], boxes[:, 2:]], -1)
        grid_points = np.stack([wIdx, hIdx, wIdx, hIdx], 1)

        boxes = (grid_points + 0.5 + shift_boxes) * stride
        boxes[:, 2:] -= boxes[:, :2]

        proposal_boxes.append(boxes)
        proposal_scores.append(scores)
        proposal_labels.append(labels)

    if not len(proposal_boxes):
        return [], [], []

    proposal_boxes = np.concatenate(proposal_boxes, 0)
    proposal_scores = np.concatenate(proposal_scores, 0)
    proposal_labels = np.concatenate(proposal_labels, 0)
    if MINOR == 7:
        indices = cv2.dnn.NMSBoxesBatched(proposal_boxes, proposal_scores, proposal_labels, score_thres, iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, score_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, score_thres, iou_thres).flatten()

    if not len(indices):
        return [], [], []

    nmsd_boxes: List[List] = []
    nmsd_scores: List[float] = []
    nmsd_labels: List[int] = []
    for idx in indices:
        x1, y1, w, h = proposal_boxes[idx]
        x2, y2 = x1 + w, y1 + h
        x1 /= ratio_w
        y1 /= ratio_h
        x2 /= ratio_w
        y2 /= ratio_h

        x1 = math.floor(min(max(x1, 1), orin_w - 1))
        y1 = math.floor(min(max(y1, 1), orin_h - 1))
        x2 = math.ceil(min(max(x2, 1), orin_w - 1))
        y2 = math.ceil(min(max(y2, 1), orin_h - 1))

        nmsd_boxes.append([x1, y1, x2, y2])
        nmsd_scores.append(float(proposal_scores[idx]))
        nmsd_labels.append(int(proposal_labels[idx]))
    return nmsd_boxes, nmsd_scores, nmsd_labels


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


if __name__ == '__main__':
    score_thres = 0.4
    iou_thres = 0.65
    model_path = './yolov8_horizon.bin'

    # load model
    try:
        model = dnn.load(model_path)[0]
        print_properties(model.inputs[0].properties)
        model_h, model_w = get_hw(model.inputs[0].properties)
    except Exception as e:
        print(f'Load model error.\n{e}')
        exit()
    else:
        try:
            for _ in range(10):
                model.forward(
                    np.random.randint(0, 255, (model_h * model_w * 3,), dtype=np.uint8)
                )
        except Exception as e:
            print(f'Warm up model error.\n{e}')

    
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ratio_h = model_h / height
    ratio_w = model_w / width

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        t0 = time.perf_counter()
        image = cv2.resize(frame, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        outputs = [o.buffer[0] for o in model.forward(image)]
        t2 = time.perf_counter()
        nmsd_boxes, nmsd_scores, nmsd_labels = det_postprocess(
            outputs, score_thres, iou_thres, height, width, ratio_h, ratio_w)
        t3 = time.perf_counter()
        print(f'Resize: {(t1 - t0) * 1000:5.2f} ms '
              f'Infer : {(t2 - t1) * 1000:5.2f} ms '
              f'Post  : {(t3 - t2) * 1000:5.2f} ms '
              f'Total : {(t3 - t0) * 1000:5.2f} ms')
        for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
            x0, y0, x1, y1 = box
            name = CLASS_NAMES[label]
            box_color = COLORS[name]
            cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, 2)
            cv2.putText(frame, f'{name}: {score:.2f}',
                        (x0, max(y0 - 5, 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        cv2.imshow('res', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
         
