import os
import tempfile

import pandas as pd
import streamlit as st

from pipeline import process_video

st.set_page_config(page_title="Tracking Analytics", layout="wide")
st.title("Object Tracking and Analytics Dashboard")
st.caption("YOLOv8 + ByteTrack / BoT-SORT | Line counter, dwell time, heatmap")

# Sidebar settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=1)
class_map = {"Person": 0, "Bicycle": 1, "Car": 2, "Motorcycle": 3, "Bus": 5}
chosen = st.sidebar.multiselect("Classes", list(class_map), default=["Person", "Car"])
line_pos = st.sidebar.slider("Line position (0 = left, 1 = right)", 0.0, 1.0, 0.5, 0.05)
zone_end = st.sidebar.slider("Dwell zone (left se kitna hissa)", 0.1, 1.0, 0.5, 0.05)

# NAYA: tracker selection
tracker_choice = st.sidebar.selectbox("Tracker", ["ByteTrack", "BoT-SORT + ReID"])
tracker_cfg = "bytetrack.yaml" if tracker_choice == "ByteTrack" else "botsort_reid.yaml"

uploaded = st.file_uploader("Video upload karein", type=["mp4", "mov", "avi"])

if uploaded and st.button("Process"):
    tmp_dir = tempfile.mkdtemp()
    in_path = os.path.join(tmp_dir, "input.mp4")
    out_path = os.path.join(tmp_dir, "output.mp4")
    with open(in_path, "wb") as f:
        f.write(uploaded.read())

    bar = st.progress(0.0, text="Processing...")
    result = process_video(
        in_path, out_path,
        model_name=model_name,
        line_pos=line_pos,
        zone_end=zone_end,
        classes=[class_map[c] for c in chosen],
        tracker_cfg=tracker_cfg,  # NAYA
        progress_cb=lambda p: bar.progress(p, text=f"Processing... {int(p * 100)}%"),
    )
    bar.empty()
    with open(out_path, "rb") as f:
        result["video_bytes"] = f.read()
    result["tracker"] = tracker_choice  # NAYA
    st.session_state["result"] = result

if "result" in st.session_state:
    r = st.session_state["result"]
    df = r["dwell_df"]

    st.info(f"Tracker used: **{r.get('tracker', 'ByteTrack')}**")  # NAYA

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IN", r["in_count"])
    c2.metric("OUT", r["out_count"])
    c3.metric("Total IDs", r["unique_ids"])
    c4.metric("Avg dwell (s)", round(df["dwell_seconds"].mean(), 2) if len(df) else 0)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Processed video")
        st.video(r["video_bytes"])
    with right:
        st.subheader("Dwell time per ID")
        if len(df):
            st.bar_chart(df.set_index("track_id")["dwell_seconds"])
        else:
            st.info("Zone mein koi object nahi aaya.")

    if len(df):
        st.subheader("Dwell time distribution")
        bins = pd.cut(df["dwell_seconds"], bins=[0, 1, 2, 3, 5, 10, 1000],
                      labels=["<1s", "1-2s", "2-3s", "3-5s", "5-10s", "10s+"])
        st.bar_chart(bins.value_counts().sort_index())

        st.download_button("Download CSV", df.to_csv(index=False),
                           file_name="dwell_time.csv", mime="text/csv")