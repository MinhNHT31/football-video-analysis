from ultralytics import YOLO
from ViewTransformer.ViewTransformer import ViewTransformation
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from collections import defaultdict, deque

import os
import supervision as sv
import numpy as np
import cv2

load_dotenv

api_key = os.getenv("API_KEY")

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.keypoint_tracker =InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key= api_key,
                            )
        
        self.pitch_all_points = np.array([
                                        (0.0, 0.0),
                                        (0.0, 14.5),
                                        (0.0, 25.84),
                                        (0.0, 44.16),
                                        (0.0, 55.5),
                                        (0.0, 70.0),
                                        (5.5, 25.84),
                                        (5.5, 44.16),
                                        (11.0, 35.0),
                                        (20.15, 14.5),
                                        (20.15, 25.84),
                                        (20.15, 44.16),
                                        (20.15, 55.5),
                                        (60.0, 0.0),
                                        (60.0, 25.85),
                                        (60.0, 44.15),
                                        (60.0, 70.0),
                                        (99.85, 14.5),
                                        (99.85, 25.84),
                                        (99.85, 44.16),
                                        (99.85, 55.5),
                                        (109.0, 35.0),
                                        (114.5, 25.84),
                                        (114.5, 44.16),
                                        (120.0, 0.0),
                                        (120.0, 14.5),
                                        (120.0, 25.84),
                                        (120.0, 44.16),
                                        (120.0, 55.5),
                                        (120.0, 70.0),
                                        (50.85, 35.0),
                                        (69.15, 35.0)
                                        ])

    def detect_frame(self, frames):
        batch_size = 120
        detections = []
        # just for testing
        length = len(frames) 
        for i in range(0, length, batch_size):
            frame = frames[i:i+batch_size]
            detection = self.model.predict(frame, conf = 0.1)
            detections += detection
            #break just for testing
            break
        return detections    

    def get_object_tracker(self, frames):
        
        detections = self.detect_frame(frames)
        
        tracks=[]
        ball_tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)


            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Ball Only
            if (detection_supervision.class_id == 0).any():
                ball_tracks.append(detection_supervision[detection_supervision.class_id == 0])
            else:
                ball_tracks.append(None)

            # Track objects
            detection_with_tracker = self.tracker.update_with_detections(detection_supervision)
            tracks.append(detection_with_tracker)

        return tracks, ball_tracks

    def get_key_points(self, frames, tracks):

        project_points = []

        for frame, track in zip(frames, tracks):
            # Get key points from the frames
            result = self.keypoint_tracker.infer(frame, model_id="football-field-detection-f07vi/15")
            keypoints = sv.KeyPoints.from_inference(result)

            filter = keypoints.confidence > 0.5
            # Get the keypoints of the pitch in the frame
            frame_points = keypoints.xy[filter]
            # frame_key_points = sv.KeyPoints(
            #     xy= frame_points[np.newaxis, ...]
            # )
            # transform the key points to the pitch coordinate system
            transform = ViewTransformation(
                source = self.pitch_all_points[np.newaxis][filter],
                target = frame_points,
            )
            # transform the key points of player and referee to the frame coordinate system
            track_points = track.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            transformed_keypoints = transform.transform_points(points = track_points).astype(np.int32)
            project_points.append(transformed_keypoints)

        return project_points


    def draw_annotator(self, frames, tracks, ball_tracks, project_points):
        # For Ball
        triangle = sv.TriangleAnnotator(
                    color = sv.Color.from_hex('#3357FF'),
                    base = 20, height=10
        )
        # For Players
        ellipsis = sv.EllipseAnnotator(
                    color = sv.ColorPalette.from_hex(['#FF5733','#FFC300']),
                    thickness= 2
        )
        label_annotator = sv.LabelAnnotator(
                    color = sv.ColorPalette.from_hex(['#FF5733','#FFC300']),
                    text_color= sv.Color.from_hex('#000000'),
                    text_position= sv.Position.BOTTOM_CENTER
        )

        annotated_frames = []
        # distance measurement
        coordinates = defaultdict( lambda: deque(maxlen=30))
        
        for frame, track, ball_track, points in zip(frames, tracks, ball_tracks, project_points):
            player = track

            IDs= [tracker_id for tracker_id in track.tracker_id]
            
            labels = []

            for tracker_id, point in zip(IDs, points):
                coordinates[tracker_id].append(point)
                if len(coordinates[tracker_id]) < 30//2:
                    labels.append(f"#{tracker_id}")
                else:
                    distances = np.linalg.norm(np.diff(coordinates[tracker_id], axis=0), axis=1)
                    speed = np.mean(distances) * 30/100
                    labels.append(f"#{tracker_id} {speed:.2f} m/s")

            annotated_frame = frame.copy()
            annotated_frame = ellipsis.annotate(annotated_frame, player)

            if ball_track is not None:
                annotated_frame = triangle.annotate(annotated_frame, ball_track)

            annotated_frame = label_annotator.annotate(annotated_frame, track, labels)
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
    
    