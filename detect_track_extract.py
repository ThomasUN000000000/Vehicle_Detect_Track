import numpy as np
import cv2
from ultralytics import YOLO
from supervision.video.sink import VideoSink
from supervision.video.dataclasses import VideoInfo
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import subprocess


# Configuration and global constants
VIDEO_PATH = '/path/to/video/file'
MODEL_WEIGHTS_PATH = 'yolov8x.pt'
FRAME_NAME = 'Your frame name'
URL_LINK = 'Real-Time traffic live link'
TIME_DURATION = 'Time to record'
VIDEO_FILE = 'Original Video File'
curr_dt = datetime.now()
OUTPUT_FILE = "trajectory-{}.mp4".format(str(int(curr_dt.timestamp())))
class BYTETrackerArgs:
    track_thresh: float = 0.05
    track_buffer: int = 50
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
    
model = YOLO(MODEL_WEIGHTS_PATH)
model.fuse()
CLASS_NAMES_DICT = model.model.names
byte_tracker = BYTETracker(BYTETrackerArgs())
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_padding=1, text_scale=0.4)
track_history = defaultdict(list)

# Utility functions
def download_video(url_link, time_duration, output):
    """
    Downloads a video from a given URL using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-i', url_link,
        '-t', str(time_duration),
        '-c', 'copy',
        output
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode == 0:
        print("Video downloaded successfully.")
    else:
        print("Error downloading video:", process.stderr)
        
def tracks2boxes(tracks):
    """Convert track objects to bounding box format."""
    return np.array([track.tlbr for track in tracks], dtype=float)

def detections2boxes(detections):
    """Convert detections to format for matching with tracks."""
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

def match_detections_with_tracks(detections, tracks):
    """Match detections with track predictions using IOU."""
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))
    
    tracks_boxes = tracks2boxes(tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids

def setup_video_capture(path):
    """Set up video capture and validate its opening."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    return cap

def detect_objects(frame, model):
    """Detect vehicles for each frame"""
    results = model(frame)[0]
    return Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )

def annotate_frame(frame, detections, track_history):
    """Annotate frame with ID, vehicle type, confidence, and trajectories plot"""
    labels = [
        f"ID:{tracker_id},{CLASS_NAMES_DICT[class_id]},{format(confidence, '.2f')}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    for box, track_id, vehicle_class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        track_history[track_id].append((float(bbox_center[0]), float(bbox_center[1])))
        points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=colors(vehicle_class_id, True), thickness=2)
    return frame

# Main processing function
def process_video():
    """Process video for vehicle tracking and trajectory plotting."""
    cap = setup_video_capture(VIDEO_PATH)
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (960, 720))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960,720))
        detections = detect_objects(frame, model)
        formatted_detections = detections2boxes(detections)
        tracks = byte_tracker.update(output_results=formatted_detections, img_info=frame.shape, img_size=frame.shape)
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        annotated_frame = annotate_frame(frame, detections, track_history)
        # real-time visualize annotated effect
        cv2.imshow(FRAME_NAME,annotated_frame)
        output.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry point of the script
if __name__ == "__main__":
    download_video(URL_LINK, TIME_DURATION, VIDEO_FILE)
    process_video()