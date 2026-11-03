from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # pehli baar auto-download hoga

cap = cv2.VideoCapture("data/sample_videos/test.mp4")

# Output video setup
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("outputs/result.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detection + ByteTrack tracking
    results = model.track(frame, persist=True,
                          tracker="bytetrack.yaml",
                          classes=[0, 2],  # 0=person, 2=car
                          verbose=False)

    annotated = results[0].plot()  # boxes + IDs draw ho jayenge
    out.write(annotated)

    cv2.imshow("Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()