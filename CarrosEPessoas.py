import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")
#model = YOLO("yolov8x.pt")
#model = YOLO("yolo11x.pt")
#model = YOLO("yolo11m.pt")


# Open the video file
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

paused = False  # Flag de pausa

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            break

        # Timestamp e frame
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        total_ms = int(timestamp_ms)
        minutes = (total_ms // 1000) // 60
        seconds = (total_ms // 1000) % 60
        milliseconds = total_ms % 1000
        timestamp_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # YOLO inference
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        names = model.names
        target_classes = ["person", "car"]
        count = {"person": 0, "car": 0}

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            if class_name in target_classes:
                count[class_name] += 1
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"
                color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"[{timestamp_str} | Frame {frame_id}] Pessoas: {count['person']}, Carros: {count['car']}")

    # Mostrar o frame (mesmo durante a pausa)
    cv2.imshow("YOLO - Pessoas (Verde) e Carros (Azul)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Tecla espaço
        paused = not paused

cap.release()
cv2.destroyAllWindows()
