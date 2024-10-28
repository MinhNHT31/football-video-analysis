from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frame(self, frames):
        batch_size = 30
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



    def draw_annotator(self, frames, tracks, ball_tracks):
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
        i = 0
        for frame, track in zip(frames, tracks):
            player = track

            labels= [f"{tracker_id}" for tracker_id in track.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ellipsis.annotate(annotated_frame, player)
            if ball_tracks[i] is not None:
                annotated_frame = triangle.annotate(annotated_frame, ball_tracks[i])
            annotated_frame = label_annotator.annotate(annotated_frame, track, labels)
            annotated_frames.append(annotated_frame)
            i += 1


        return annotated_frames
    
    