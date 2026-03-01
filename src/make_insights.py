# src/make_insights.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RISK_MAP = {"LOW":0, "MEDIUM":1, "HIGH":2}
INV_MAP = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}

def smooth_risk(labels: pd.Series, window: int) -> pd.Series:
    s = labels.map(RISK_MAP).astype(float)
    sm = s.rolling(window, min_periods=1).median().round().astype(int)
    return sm.map(INV_MAP)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/navienta_train_clean.csv")
    parser.add_argument("--model", default="model/navienta_rf.joblib")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--max_points", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    rf = joblib.load(args.model)

    FEATURES = [
        "d_front_m","v_ego_ms","acc_ms2",
        "closing_speed",
        "min_d_front_1s","med_d_front_1s","max_close_1s","time_to_reach_1s",
        "Lane_ID"
    ]

    # pilih vehicle demo (yang datanya paling banyak)
    demo_vid = df["Vehicle_ID"].value_counts().index[0]
    demo = df[df["Vehicle_ID"] == demo_vid].sort_values("timestamp").copy()

    demo["pred_class"] = rf.predict(demo[FEATURES])
    demo["t_rel"] = demo["timestamp"] - demo["timestamp"].min()

    # clip untuk visual (biar angka ekstrem tidak mengganggu)
    demo["d_front_m_clip"] = demo["d_front_m"].clip(lower=0.5, upper=200)
    demo["min_d_front_1s_clip"] = demo["min_d_front_1s"].clip(lower=0.5, upper=200)

    # smoothing supaya tidak kedip-kedip
    demo["pred_smooth"] = smooth_risk(demo["pred_class"], args.smooth_window)

    # simpan demo CSV untuk Streamlit
    demo_out = os.path.join(args.outdir, "navienta_demo_with_pred.csv")
    demo.to_csv(demo_out, index=False)

    # plot jarak
    plot_df = demo.head(args.max_points)
    fig = plt.figure(figsize=(10,4))
    plt.plot(plot_df["t_rel"], plot_df["d_front_m_clip"])
    plt.xlabel("Time (s) from start")
    plt.ylabel("Front distance (m)")
    plt.title("Front Distance (Demo)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plot_distance.png"), dpi=200)
    plt.close(fig)

    # plot risk timeline (smoothed)
    fig = plt.figure(figsize=(10,3))
    plt.plot(plot_df["t_rel"], plot_df["pred_smooth"].map(RISK_MAP))
    plt.yticks([0,1,2], ["LOW","MEDIUM","HIGH"])
    plt.xlabel("Time (s) from start")
    plt.ylabel("Risk level")
    plt.title("Risk Timeline (Smoothed)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pred_risk_timeline_clean.png"), dpi=200)
    plt.close(fig)

    # segmen HIGH
    is_high = (demo["pred_smooth"] == "HIGH").astype(int)
    seg_id = (is_high.diff().fillna(0) == 1).cumsum()
    demo["seg_id"] = seg_id.where(is_high == 1, 0)

    segments = (demo[demo["seg_id"] != 0]
                .groupby("seg_id")
                .agg(
                    start_s=("t_rel","min"),
                    end_s=("t_rel","max"),
                    duration_s=("t_rel", lambda s: float(s.max()-s.min())),
                    min_d_m=("min_d_front_1s_clip","min"),
                    max_close=("max_close_1s","max")
                )
                .reset_index(drop=True)
                .sort_values("duration_s", ascending=False))

    seg_out = os.path.join(args.outdir, "high_segments_clean.csv")
    segments.to_csv(seg_out, index=False)

    # feature importance (Step C)
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fig = plt.figure(figsize=(8,4))
    importances.head(10).plot(kind="bar")
    plt.title("Top Feature Importances (RandomForest)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "feature_importance.png"), dpi=200)
    plt.close(fig)

    print("[OK] Saved demo CSV:", demo_out)
    print("[OK] Saved plots: plot_distance.png, pred_risk_timeline_clean.png, feature_importance.png")
    print("[OK] Saved segments:", seg_out)
    print("Demo Vehicle_ID:", int(demo_vid))

if __name__ == "__main__":
    main()