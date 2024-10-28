from ultralytics import YOLO

mode = YOLO('models/best.pt')

results = mode.predict("input_videos/0bfacc_0.mp4",save = True)
print(results[0])
for box in results[0].boxes:
    print(box)