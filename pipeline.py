import subprocess
from collections import defaultdict

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO


def process_video(video_in, video_out, model_name="yolov8s.pt",
                  line_pos=0.5, zone_end=0.5, classes=(0, 2),
                  tracker_cfg="bytetrack.yaml",          # CHANGE 1 (naya)
                  progress_cb=None):
    model = YOLO(model_name)
    info = sv.VideoInfo.from_video_path(video_in)
    w, h = info.width, info.height

    # Vertical line
    line_x = int(w * line_pos)
    line_zone = sv.LineZone(start=sv.Point(line_x, 0), end=sv.Point(line_x, h))

    # Dwell zone: left se zone_end tak
    zone_x = int(w * zone_end)
    polygon = np.array([[0, 0], [zone_x, 0], [zone_x, h], [0, h]])
    zone = sv.PolygonZone(polygon=polygon)

    frames_in_zone = defaultdict(int)
    all_ids = set()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=40)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=1)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)
    heat_annotator = sv.HeatMapAnnotator(opacity=0.5, radius=25)

    tmp_path = video_out + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps, (w, h))

    for i, frame in enumerate(sv.get_video_frames_generator(video_in)):
        results = model.track(frame, persist=True,
                              tracker=tracker_cfg,       # CHANGE 2 (pehle "bytetrack.yaml" tha)
                              classes=list(classes), verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections.tracker_id is None:
            writer.write(frame)
        else:
            line_zone.trigger(detections)
            inside = zone.trigger(detections)
            for tid, is_in in zip(detections.tracker_id, inside):
                all_ids.add(int(tid))
                if is_in:
                    frames_in_zone[int(tid)] += 1

            labels = [f"#{t} {model.names[c]}"
                      for t, c in zip(detections.tracker_id, detections.class_id)]

            annotated = frame.copy()
            annotated = heat_annotator.annotate(annotated, detections)
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            annotated = line_annotator.annotate(annotated, line_zone)
            annotated = zone_annotator.annotate(annotated)
            writer.write(annotated)

        if progress_cb and info.total_frames:
            progress_cb(min((i + 1) / info.total_frames, 1.0))

    writer.release()

    # Browser-friendly H.264 mein convert
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", tmp_path, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", video_out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    df = pd.DataFrame(
        [{"track_id": t, "dwell_seconds": round(f / info.fps, 2)}
         for t, f in sorted(frames_in_zone.items())]
    )

    return {
        "video_path": video_out,
        "in_count": line_zone.in_count,
        "out_count": line_zone.out_count,
        "unique_ids": len(all_ids),
        "dwell_df": df,
    }