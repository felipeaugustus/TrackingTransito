import cv2
from ultralytics import YOLO
import numpy as np
import time

model = YOLO("Modelos/yolo11n.pt")
video_path = "test1.mp4"
cap = cv2.VideoCapture(video_path)

paused = False

# Parâmetros
MIN_PERSIST_TIME_MS = 1000
DUPLICATE_DISTANCE_THRESHOLD = 50  # pixels
DUPLICATE_TIME_THRESHOLD_MS = 1000

# Armazenamento dos IDs detectados
track_history = {}
counted_ids = {"person": set(), "car": set()}
last_positions = {"person": [], "car": []}  # cada item: (track_id, centro, timestamp_ms)

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) // 2, (y1 + y2) // 2)

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            break

        # Timestamp e frame info
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        minutes = (timestamp_ms // 1000) // 60
        seconds = (timestamp_ms // 1000) % 60
        milliseconds = timestamp_ms % 1000
        timestamp_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Inference com tracking
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes
        names = model.names
        target_classes = ["person", "car"]
        current_count = {"person": 0, "car": 0}

        if boxes.id is not None:
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i])
                track_id = int(boxes.id[i])
                class_name = names[cls_id]

                if class_name not in target_classes:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                center = get_center(xyxy)

                # Atualizar histórico
                if track_id not in track_history:
                    track_history[track_id] = {
                        "class": class_name,
                        "first_seen": timestamp_ms,
                        "last_seen": timestamp_ms,
                        "center": center,
                        "counted": False,
                    }
                else:
                    track_history[track_id]["last_seen"] = timestamp_ms
                    track_history[track_id]["center"] = center

                # Verificar se deve contar
                track = track_history[track_id]
                duration = track["last_seen"] - track["first_seen"]

                if not track["counted"] and duration >= MIN_PERSIST_TIME_MS:
                    # Verificar duplicação na mesma região
                    too_close = False
                    for prev_id, prev_center, prev_time in last_positions[class_name]:
                        dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                        time_diff = timestamp_ms - prev_time
                        if dist < DUPLICATE_DISTANCE_THRESHOLD and time_diff < DUPLICATE_TIME_THRESHOLD_MS:
                            too_close = True
                            break

                    if not too_close:
                        counted_ids[class_name].add(track_id)
                        last_positions[class_name].append((track_id, center, timestamp_ms))
                        track["counted"] = True

                if track["counted"]:
                    current_count[class_name] += 1
                    label = f"{class_name} ID:{track_id}"
                    color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        total_persons = len(counted_ids["person"])
        total_cars = len(counted_ids["car"])

        print(f"[{timestamp_str} | Frame {frame_id}] Pessoas ativas: {current_count['person']} (total únicas: {total_persons}), Carros ativos: {current_count['car']} (total únicos: {total_cars})")

    # Mostrar o frame
    cv2.imshow("YOLO + Tracking + Contagem Robusta", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
