import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
from ultralytics import YOLO

VIDEO_IN = "data/sample_videos/test.mp4"
VIDEO_OUT = "outputs/result.mp4"

model = YOLO("yolov8s.pt")

# Video info
info = sv.VideoInfo.from_video_path(VIDEO_IN)
w, h = info.width, info.height

# Line: vertical, video ke beech mein
START = sv.Point(w // 2, 0)
END = sv.Point(w // 2, h)
line_zone = sv.LineZone(start=START, end=END)

# Dwell zone: video ka left half
polygon = np.array([[0, 0], [w // 2, 0], [w // 2, h], [0, h]])
zone = sv.PolygonZone(polygon=polygon)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)

# Har ID ke kitne frames zone mein rahe
frames_in_zone = defaultdict(int)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=40)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=1)
heat_map_annotator = sv.HeatMapAnnotator(opacity=0.5, radius=25)

out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                      info.fps, (w, h))

for frame in sv.get_video_frames_generator(VIDEO_IN):
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          classes=[0, 2], verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)

    # Agar kisi frame mein tracking ID na mile to skip
    if detections.tracker_id is None:
        out.write(frame)
        continue

    line_zone.trigger(detections)

    # Dwell time: kaun se IDs zone ke andar hain
    inside = zone.trigger(detections)
    for tid, is_in in zip(detections.tracker_id, inside):
        if is_in:
            frames_in_zone[int(tid)] += 1

    labels = [f"#{tid} {model.names[c]}"
              for tid, c in zip(detections.tracker_id, detections.class_id)]

    annotated = frame.copy()
    annotated = heat_map_annotator.annotate(annotated, detections)
    annotated = trace_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    annotated = line_annotator.annotate(annotated, line_zone)
    annotated = zone_annotator.annotate(annotated)

    out.write(annotated)
    cv2.imshow("Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()

print(f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}")
print("\nDwell time (seconds) in left zone:")
for tid, f in sorted(frames_in_zone.items()):
    print(f"  ID {tid}: {f / info.fps:.1f} s")