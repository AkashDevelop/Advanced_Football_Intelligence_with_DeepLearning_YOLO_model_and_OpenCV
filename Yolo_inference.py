from ultralytics import YOLO

model= YOLO("Models/best.pt")

results=model.predict("input_video/08fd33_4.mp4",save=True)

print(results[0])

print('___________________________________________')

for box in results[0].boxes:
    print(box)