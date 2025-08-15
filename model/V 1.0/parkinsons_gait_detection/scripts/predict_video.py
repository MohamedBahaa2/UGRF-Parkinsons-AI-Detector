# scripts/predict_video.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

# Fallback feature list (from your training printout "Selected 10 features: [...]")
DEFAULT_SELECTED_10 = [
    "stride_length_left_m",
    "double_support_time_s",
    "stride_length_cv",
    "stride_time_cv",
    "step_length_cv",
    "step_time_cv",
    "swing_time_cv",
    "stance_time_cv",
    "gait_speed_cv",
    "stride_length_asymmetry",
]

def load_bundle(path: Path):
    obj = load(path)
    return obj

def pick_model_from_bundle(obj):
    """
    Prefer a full pipeline if available; else fall back to the bare estimator.
    Returns (model_like, selected_features_or_None)
    """
    # If it's already a model/pipeline
    if hasattr(obj, "predict"):
        return obj, getattr(obj, "feature_names_in_", None)

    sel_feats = None
    if isinstance(obj, dict):
        # try to read selected features if provided
        for k in ("selected_features", "feature_names", "input_features"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                sel_feats = list(obj[k])

        # prefer a pipeline-like object
        for key in ("pipeline", "full_pipeline", "best_pipeline", "clf_pipeline"):
            if key in obj and hasattr(obj[key], "predict"):
                print(f"[INFO] Using pipeline from bundle key '{key}'.")
                return obj[key], sel_feats

        # else take the bare model/estimator
        for key in ("model", "best_model", "estimator", "clf"):
            if key in obj and hasattr(obj[key], "predict"):
                print(f"[INFO] Using model from bundle key '{key}'.")
                return obj[key], sel_feats

        # fallback: first value that has predict
        for k, v in obj.items():
            if hasattr(v, "predict"):
                print(f"[INFO] Using model from bundle key '{k}'.")
                return v, sel_feats

        raise ValueError(f"Bundle has no usable model. Keys: {list(obj.keys())}")

    raise TypeError(f"Unsupported model object type: {type(obj)}")

def main():
    ap = argparse.ArgumentParser(description="Predict Parkinson's from video features CSV")
    ap.add_argument("--features", required=True, help="Path to video_features.csv")
    ap.add_argument("--model", default="results/best_model_random_forest.joblib",
                    help="Path to saved model (.joblib)")
    args = ap.parse_args()

    feats_path = Path(args.features)
    model_path = Path(args.model)
    if not feats_path.exists():
        raise FileNotFoundError(f"Features file not found: {feats_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("[1/4] Loading features...")
    df = pd.read_csv(feats_path)

    print("[2/4] Loading model/pipeline bundle...")
    bundle = load_bundle(model_path)

    print("[3/4] Selecting model & aligning inputs...")
    model, sel_feats = pick_model_from_bundle(bundle)

    # If we have a full pipeline, just predict with df (it should handle columns itself)
    if hasattr(model, "named_steps") or "Pipeline" in type(model).__name__:
        # Align by names if available; otherwise pass as-is
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            # add missing cols as NaN and order
            for c in expected:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[expected]
        # Let the pipeline handle imputation/scaling/selection
        X = df
    else:
        # Bare estimator (e.g., RandomForest) trained on 10 features
        # Determine which 10 to use
        feat_list = sel_feats if sel_feats else DEFAULT_SELECTED_10
        # build aligned frame with just those 10
        for c in feat_list:
            if c not in df.columns:
                df[c] = np.nan
        df = df[feat_list]
        # simple imputation (since we don't have the training imputer here)
        # use 0.0 as a safe constant; you can switch to df.fillna(df.median()) if you prefer
        if df.isna().any().any():
            print("[WARN] Missing values found; imputing NaN with 0.0 for estimator compatibility.")
            df = df.fillna(0.0)
        # pass numpy array so sklearn doesn't warn about feature names
        X = df.to_numpy(dtype=float)

    print(f"    Using {X.shape[1]} features.")

    print("[4/4] Predicting...")
    y_pred = int(model.predict(X)[0])
    y_proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    label = "Parkinson's" if y_pred == 1 else "Control"
    print("\n=== Parkinson's Video Prediction ===")
    print(f"Prediction : {label}")
    if y_proba is not None:
        print(f"Probability: {y_proba:.3f}")

if __name__ == "__main__":
    main()
