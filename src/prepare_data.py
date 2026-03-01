# src/prepare_data.py
import os
import argparse
import numpy as np
import pandas as pd

FEET_TO_M = 0.3048

def label_3class(headway_s: float) -> str:
    # baseline sederhana: aman/hati-hati/bahaya
    if headway_s < 1.0:
        return "HIGH"
    if headway_s < 2.0:
        return "MEDIUM"
    return "LOW"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ngsim_trajectory.csv")
    parser.add_argument("--output", default="outputs/navienta_train_clean.csv")
    parser.add_argument("--window", type=int, default=10, help="~10 rows = ~1 second (dt≈0.1s)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1) LOAD (SAMA dengan notebook kamu)
    usecols = [
        "Vehicle_ID","Global_Time","Lane_ID",
        "Preceding_Vehicle","Spacing","Headway",
        "Vehicle_Velocity","Vehicle_Acceleration"
    ]
    dtypes = {
        "Vehicle_ID": "int32",
        "Global_Time": "int64",
        "Lane_ID": "int16",
        "Preceding_Vehicle": "int32",
        "Spacing": "float32",
        "Headway": "float32",
        "Vehicle_Velocity": "float32",
        "Vehicle_Acceleration": "float32"
    }

    df = pd.read_csv(args.input, usecols=usecols, dtype=dtypes)
    print("Loaded:", df.shape)

    # 2) EDA (SAMA)
    print("Preceding==0:", (df["Preceding_Vehicle"]==0).mean())
    print("Spacing<=0:", (df["Spacing"]<=0).mean())
    print("Headway<=0:", (df["Headway"]<=0).mean())
    print("Headway>=10:", (df["Headway"]>=10).mean())
    print("Velocity<=0:", (df["Vehicle_Velocity"]<=0).mean())

    # 3) CLEANING (SAMA)
    dfc = df[
        (df["Preceding_Vehicle"] != 0) &
        (df["Spacing"] > 0) &
        (df["Headway"] > 0) &
        (df["Headway"] < 10) &
        (df["Vehicle_Velocity"] > 0)
    ].copy()
    print("Before:", df.shape, "After cleaning:", dfc.shape)

    # 4) KONVERSI (SAMA)
    dfc["timestamp"] = dfc["Global_Time"] / 1000.0
    dfc["d_front_m"] = dfc["Spacing"] * FEET_TO_M
    dfc["v_ego_ms"]  = dfc["Vehicle_Velocity"] * FEET_TO_M
    dfc["acc_ms2"]   = dfc["Vehicle_Acceleration"] * FEET_TO_M
    dfc["headway_s"] = dfc["Headway"]

    dfc = dfc.sort_values(["Vehicle_ID","timestamp"]).reset_index(drop=True)

    # ---- Tambahan kecil yang "terbaik" (feature engineering sederhana, masih mudah dipahami)

    # dt dan closing_speed
    dt = dfc.groupby("Vehicle_ID")["timestamp"].diff()
    dd = dfc.groupby("Vehicle_ID")["d_front_m"].diff()

    dfc["dt"] = dt
    dfc["closing_speed"] = (-dd / dt).replace([np.inf, -np.inf], np.nan)

    # filter dt yang wajar (dataset kamu dt≈0.1s)
    dfc = dfc[(dfc["dt"] > 0) & (dfc["dt"] < 1)].copy()
    dfc = dfc.dropna(subset=["closing_speed"])
    dfc["closing_speed"] = dfc["closing_speed"].clip(lower=-50, upper=50)

    # window features 1 detik (W=10)
    W = args.window
    g = dfc.groupby("Vehicle_ID", group_keys=False)
    dfc["min_d_front_1s"] = g["d_front_m"].rolling(W, min_periods=1).min().reset_index(level=0, drop=True)
    dfc["med_d_front_1s"] = g["d_front_m"].rolling(W, min_periods=1).median().reset_index(level=0, drop=True)
    dfc["max_close_1s"]   = g["closing_speed"].rolling(W, min_periods=1).max().reset_index(level=0, drop=True)

    eps = 0.1
    dfc["time_to_reach_1s"] = dfc["min_d_front_1s"] / np.maximum(dfc["v_ego_ms"], eps)

    # label baseline (two-second rule)
    dfc["risk_class"] = dfc["headway_s"].apply(label_3class)

    # simpan
    out_cols = [
        "Vehicle_ID","timestamp","Lane_ID",
        "d_front_m","v_ego_ms","acc_ms2",
        "dt","closing_speed",
        "min_d_front_1s","med_d_front_1s","max_close_1s","time_to_reach_1s",
        "headway_s","risk_class"
    ]
    dfc[out_cols].to_csv(args.output, index=False)
    print("[OK] Saved:", args.output)
    print(dfc["risk_class"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
