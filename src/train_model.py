# src/train_model.py
import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/navienta_train_clean.csv")
    parser.add_argument("--model_out", default="model/navienta_rf.joblib")
    parser.add_argument("--metrics_out", default="outputs/metrics.txt")
    parser.add_argument("--cm_out", default="outputs/confusion_matrix.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    df = pd.read_csv(args.input).dropna()

    # IMPORTANT: headway_s tidak dipakai sebagai fitur (karena sumber label)
    FEATURES = [
        "d_front_m","v_ego_ms","acc_ms2",
        "closing_speed",
        "min_d_front_1s","med_d_front_1s","max_close_1s","time_to_reach_1s",
        "Lane_ID"
    ]
    TARGET = "risk_class"

    # split by Vehicle_ID (biar test kendaraan berbeda)
    vids = df["Vehicle_ID"].unique()
    train_vids, test_vids = train_test_split(vids, test_size=0.2, random_state=42)

    tr = df[df["Vehicle_ID"].isin(train_vids)]
    te = df[df["Vehicle_ID"].isin(test_vids)]

    X_train, y_train = tr[FEATURES], tr[TARGET]
    X_test, y_test = te[FEATURES], te[TARGET]

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    report = classification_report(y_test, pred, digits=4)
    print(report)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write(report)

    labels = ["LOW","MEDIUM","HIGH"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax, values_format="d")
    plt.title("NAVIENTA Confusion Matrix (3-class)")
    plt.tight_layout()
    plt.savefig(args.cm_out, dpi=200)
    plt.close(fig)

    joblib.dump(rf, args.model_out)
    print("[OK] Saved model:", args.model_out)
    print("[OK] Saved metrics:", args.metrics_out)
    print("[OK] Saved CM:", args.cm_out)

if __name__ == "__main__":
    main()