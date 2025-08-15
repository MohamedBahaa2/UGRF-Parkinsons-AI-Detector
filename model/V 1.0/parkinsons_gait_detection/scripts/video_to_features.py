# scripts/video_to_features.py
# Extract timing-based gait features from a video using MediaPipe + OpenCV.
# Includes a debug plot to visualize heel trajectories and detected steps.
# Outputs a one-row CSV compatible with scripts/predict_video.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import math

def _mp_pose():
    import mediapipe as mp
    return mp.solutions.pose

def moving_avg(x, w=7):
    x = np.asarray(x, dtype=float)
    if x.size < 3 or w <= 1:
        return x
    kernel = np.ones(int(w), dtype=float) / float(w)
    # pad to be less phase-distorting at edges
    pad = min(int(w)//2, len(x)-1 if len(x) > 1 else 0)
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="same")[pad:-pad]
    return y

def zscore_norm(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)
    return (x - m) / s

def detect_local_minima(z, thresh=-0.7, min_dist=0):
    """
    Very simple local-minima detector on a z-scored signal.
    Returns indices of minima below 'thresh' separated by >= min_dist.
    """
    idxs = []
    last = -10**9
    for i in range(1, len(z)-1):
        a, b, c = z[i-1], z[i], z[i+1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        if b < a and b < c and b <= thresh:
            if i - last >= min_dist:
                idxs.append(i)
                last = i
    return np.array(idxs, dtype=int)

def segments_from_binary(b, fps):
    """Return durations (s) of consecutive 1s (moving) and 0s (not moving)."""
    if len(b) == 0:
        return [], []
    ones, zeros = [], []
    curr, count = b[0], 1
    for x in b[1:]:
        if x == curr:
            count += 1
        else:
            (ones if curr == 1 else zeros).append(count / fps)
            curr, count = x, 1
    (ones if curr == 1 else zeros).append(count / fps)
    return ones, zeros

def coef_var(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    m = np.mean(arr)
    if m == 0:
        return np.nan
    return (np.std(arr, ddof=1) / abs(m)) * 100.0

def safe_mean(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan

def extract_heel_traces(video_path, model_complexity=1,
                        min_det=0.5, min_track=0.5):
    mp_pose = _mp_pose()
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    heel_left_y, heel_right_y = [], []

    L_HEEL, R_HEEL = 29, 30  # MediaPipe Pose indices

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            heel_left_y.append(np.nan)
            heel_right_y.append(np.nan)
            continue
        lm = res.pose_landmarks.landmark
        def ny(i):
            return lm[i].y if 0 <= i < len(lm) else np.nan  # normalized y (0..1)
        heel_left_y.append(ny(L_HEEL))
        heel_right_y.append(ny(R_HEEL))

    cap.release()
    pose.close()
    return fps, np.array(heel_left_y, float), np.array(heel_right_y, float)

def main():
    ap = argparse.ArgumentParser(description="Extract gait features from a video (with debug plot).")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="results/video_features.csv", help="Output CSV path")
    ap.add_argument("--smooth", type=int, default=9, help="Smoothing window for heel traces (odd, >=3)")
    ap.add_argument("--z_thresh", type=float, default=0.4, help="Z-score threshold for minima detection (lower = more sensitive)")
    ap.add_argument("--min_interval_s", type=float, default=0.30, help="Minimum time between detected steps (seconds)")
    ap.add_argument("--show", action="store_true", help="Show the debug plot interactively")
    ap.add_argument("--save_plot", default="results/video_debug_plot.png", help="Where to save the debug plot image")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 1) Extract heel traces
    fps, heel_left_y, heel_right_y = extract_heel_traces(video_path)

    # Interpolate & smooth
    heel_left_y  = pd.Series(heel_left_y).interpolate(limit_direction="both").to_numpy()
    heel_right_y = pd.Series(heel_right_y).interpolate(limit_direction="both").to_numpy()

    win = max(3, args.smooth if args.smooth % 2 == 1 else args.smooth + 1)  # enforce odd
    heel_left_y_s  = moving_avg(heel_left_y,  w=win)
    heel_right_y_s = moving_avg(heel_right_y, w=win)

    # 2) Step detection on z-scored, smoothed signals (minima)
    min_dist = max(1, int(fps * args.min_interval_s))
    zL = zscore_norm(heel_left_y_s)
    zR = zscore_norm(heel_right_y_s)
    mL = detect_local_minima(zL, thresh=-abs(args.z_thresh), min_dist=min_dist)
    mR = detect_local_minima(zR, thresh=-abs(args.z_thresh), min_dist=min_dist)

    t = np.arange(len(heel_left_y_s)) / float(fps)
    step_t_L = t[mL] if mL.size else np.array([])
    step_t_R = t[mR] if mR.size else np.array([])

    # 3) Durations & variability metrics
    step_int_L   = np.diff(step_t_L) if step_t_L.size >= 2 else np.array([])
    step_int_R   = np.diff(step_t_R) if step_t_R.size >= 2 else np.array([])
    step_int_all = np.sort(np.concatenate([step_int_L, step_int_R])) if (step_int_L.size or step_int_R.size) else np.array([])

    stride_L = step_int_L[1:] + step_int_L[:-1] if step_int_L.size >= 2 else np.array([])
    stride_R = step_int_R[1:] + step_int_R[:-1] if step_int_R.size >= 2 else np.array([])
    stride_all = np.sort(np.concatenate([stride_L, stride_R])) if (stride_L.size or stride_R.size) else np.array([])

    # Simple swing/stance proxy using velocity thresholding
    vel_L = np.gradient(heel_left_y_s)  * fps
    vel_R = np.gradient(heel_right_y_s) * fps
    # threshold proportional to signal std to be scale-invariant
    th = (np.nanstd(vel_L + vel_R) or 1.0) * 0.5
    moving_L = (np.abs(vel_L) > th).astype(int)
    moving_R = (np.abs(vel_R) > th).astype(int)
    swing_L, stance_L = segments_from_binary(moving_L, fps)
    swing_R, stance_R = segments_from_binary(moving_R, fps)

    # 4) Build features (length/speed need calibration → NaN)
    row = {
        "stride_length_left_m":        np.nan,
        "double_support_time_s":       safe_mean(stance_L + stance_R),
        "stride_length_cv":            np.nan,
        "stride_time_cv":              coef_var(stride_all),
        "step_length_cv":              np.nan,
        "step_time_cv":                coef_var(step_int_all),
        "swing_time_cv":               coef_var(swing_L + swing_R),
        "stance_time_cv":              coef_var(stance_L + stance_R),
        "gait_speed_cv":               np.nan,
        "stride_length_asymmetry":     abs(coef_var(step_int_L) - coef_var(step_int_R)) if (step_int_L.size and step_int_R.size) else np.nan,
        # optional extras (kept for pipeline compatibility)
        "age": np.nan, "height_cm": np.nan, "weight_kg": np.nan,
        "label": np.nan, "disease_duration_years": np.nan,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False)

    # Console summary
    print("==============================================")
    print(f"[OK] Wrote features to {out_path} (fps={fps:.2f})")
    print(f"Detected steps: L={len(step_t_L)}  R={len(step_t_R)}")
    if step_int_all.size:
        print(f"Mean step time: {np.mean(step_int_all):.3f}s  CV: {coef_var(step_int_all):.2f}%")
    if stride_all.size:
        print(f"Mean stride time: {np.mean(stride_all):.3f}s  CV: {coef_var(stride_all):.2f}%")
    print(f"Swing_time_CV: {row['swing_time_cv'] if np.isfinite(row['swing_time_cv']) else 'NaN'}")
    print(f"Stance_time_CV: {row['stance_time_cv'] if np.isfinite(row['stance_time_cv']) else 'NaN'}")
    print("Tip: If steps are 0, lower --z_thresh (e.g., 0.3 → 0.2) or increase --smooth (e.g., 11).")
    print("==============================================")

    # 5) Debug plot (saved; optionally shown)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 6))
        plt.plot(t, zL, label="Left heel (z)")
        plt.plot(t, zR, label="Right heel (z)")
        if mL.size:
            plt.scatter(t[mL], zL[mL], marker="v", label="Left steps", zorder=5)
        if mR.size:
            plt.scatter(t[mR], zR[mR], marker="v", label="Right steps", zorder=5)
        plt.axhline(-abs(args.z_thresh), linestyle="--", label=f"threshold {-abs(args.z_thresh):.2f}")
        plt.title("Heel vertical trajectory (z-score) & detected step events")
        plt.xlabel("Time (s)")
        plt.ylabel("z-score (normalized)")
        plt.legend()
        fig.tight_layout()

        save_path = Path(args.save_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Saved debug plot to {save_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print(f"[WARN] Could not create debug plot: {e}")

if __name__ == "__main__":
    main()
