"""Yolo v3 detection script.

Saves the detections in the `detection` folder.

Usage:
    python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>

Example:
    python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg
    python detect.py video 0.5 0.5 data/video/shinjuku.mp4

Note that only one video can be processed at one run.
"""

import sys
import cv2
import tensorflow as tf
import outputWindow
import threading
import requests

from utils import load_class_names, draw_frame, draw_frame_2
from yolo_v3 import Yolo_v3

tf.compat.v1.disable_eager_execution()

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20
_LAST_DETECTION = ''
_NEW_DETECTION = ''
_DETECTION_COUNTER = 0;
_ADD_THRESHOLD = 3


def main(iou_threshold, confidence_threshold):
    global _LAST_DETECTION, _NEW_DETECTION, _DETECTION_COUNTER, _ADD_THRESHOLD
    saver, detections, inputs, class_names = setup(iou_threshold, confidence_threshold)
    cap, frame_size, win_name = setupWindow()
    requests.get(url="http://192.168.4.1/?Command=setspeed+800")
    requests.get(url="http://192.168.4.1/?Command=backwards")
    try:
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')
            while True:
                result = detect(sess, detections, inputs, class_names, cap, frame_size, win_name)
                for obj in result:
                    if not obj == _LAST_DETECTION:
                        if obj == _NEW_DETECTION:
                            _DETECTION_COUNTER += 1
                            print(_NEW_DETECTION + " " + str(_DETECTION_COUNTER) + " mal erkannt")
                            if _DETECTION_COUNTER >= _ADD_THRESHOLD:
                                _DETECTION_COUNTER = 0
                                _LAST_DETECTION = obj
                                outputWindow.addLabel(obj)
                                print(obj)
                        else:
                            _NEW_DETECTION = obj
                            _DETECTION_COUNTER = 0
    finally:
        cv2.destroyAllWindows()
        cap.release()


def setup(iou_threshold, confidence_threshold):
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
    detections = model(inputs, training=False)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))
    return saver, detections, inputs, class_names


def setupWindow():
    win_name = 'Webcam detection'
    cv2.namedWindow(win_name)
    cap = cv2.VideoCapture(1)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, frame_size, win_name


def detect(sess, detections, inputs, class_names, cap, frame_size, win_name):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                   interpolation=cv2.INTER_NEAREST)
        detection_result = sess.run(detections,
                                    feed_dict={inputs: [resized_frame]})
        result = draw_frame_2(frame, frame_size, detection_result,
                              class_names, _MODEL_SIZE)

        draw_frame(frame, frame_size, detection_result,
                   class_names, _MODEL_SIZE)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if len(result) > 0:
            return result


if __name__ == '__main__':
    tensorflowThread = threading.Thread(target=main, args=(float(sys.argv[1]), float(sys.argv[2])))
    tensorflowThread.start()
    outputWindow.init()
