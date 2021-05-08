import os
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import load_yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names, write_csv
from yolov3.configs import *
from face_detection.FaceDetector import FaceDetector
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def track_object(Yolo, video_path, vid_output_path, text_output_path, input_size=416, show=False,
                 CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45,
                 rectangle_colors='', tracking=True, track_only=[], tracker_max_age=30,
                 passenger_det=False, face_score_threshold=0.3, color="bincount"):
    """
    Do detection on video
    :param Yolo: <model_obj> YOLO model for vehicle detection
    :param video_path: <str> Path to video file. Leave empty to use camera
    :param vid_output_path: <str> Path to save processed video. Leave empty to not save
    :param input_size: <int> YOLO model input size
    :param show: <bool> True if you want to see processing live
    :param CLASSES: <obj> YOLO model classed. By default they are taken from the config file
    :param score_threshold: <float> minimum confidence for vehicle detection
    :param iou_threshold: <float> minimum bounding box overlap for them to be counted as same object
    :param rectangle_colors: bounding box colors. Currently does nothing
    :param tracking: whether to use vehicle tracking
    :param track_only: <list> List of objects to track if detector detects more
    :param tracker_max_age: <int> number of missed before track is deleted
    :param face_det: <bool> whether to initialize face detection
    :param face_score_threshold: <float> minimum confidence for face detection
    :param color: <str> Color detection method to use. None if neither one
    :return:
    """
    if not Yolo:
        Yolo = load_yolo_model()

    if passenger_det:
        passenger_det = FaceDetector()
    else:
        passenger_det = None

    if text_output_path:
        write_csv([["x1", "y1", "x2", "y2", "id", "class", "probability", "color" if color else None,
                    "passengers" if passenger_det else None]], text_output_path)

    # Definition of the deep sort parameters
    max_cosine_distance = 0.7
    nn_budget = None
    # initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=tracker_max_age)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video
    else:
        vid = cv2.VideoCapture(0)  # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_output_path, codec, fps, (width, height))  # vid_output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())
    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        # image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(track_only) != 0 and NUM_CLASS[int(bbox[5])] in track_only or len(track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                              bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]  # if score >= confidence_threshold]

        # Pass detections to the deep sort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index, track.class_confidence])  # Structure data, that we could use it with our draw_bbox function

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True, color=color,
                          text_output_path=text_output_path, passenger_detector=passenger_det,
                          passenger_threshold=face_score_threshold)

        t3 = time.time()
        times.append(t2 - t1)
        times_2.append(t3 - t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 2)

        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if vid_output_path != '':
            out.write(image)
        if show:
            cv2.imshow('output', image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
