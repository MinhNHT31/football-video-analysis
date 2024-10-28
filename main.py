from utils import read_video, save_video
from tracker import Tracker

def main():
    # read video
    frames = read_video("input_videos/121364_0.mp4")

    #initialize tracker
    tracker = Tracker("models/best.pt")

    tracked,ball_tracked = tracker.get_object_tracker(frames)
    
    # Draw bounding boxes
    annotated_frames = tracker.draw_annotator(frames, tracked, ball_tracked)
    frames = annotated_frames

    # save video
    
    save_video(frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()