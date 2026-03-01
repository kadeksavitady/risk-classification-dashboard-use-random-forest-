# NAVIENTA - Real-time Driving Risk Classification 

# NAVIENTA - Klasifikasi Risiko Berkendara Real-time 

---

## Overview (English)

NAVIENTA is a driving safety support prototype that classifies risk into three levels:  
**LOW (safe)**, **MEDIUM (be careful)**, and **HIGH (danger)**.

This prototype uses **Random Forest** and a **real-time dashboard simulation** to show risk changes over time.  
The current focus is **safe distance risk** as a first development stage.

**Why Random Forest (instead of rule-based logic)?**
- **Data-driven**: learns patterns from data, not only hand-made rules  
- **Flexible**: easier to extend when adding new input features  
- **Easier to evaluate**: performance can be measured with clear metrics, and feature importance helps explain model behavior

**Baseline reference**
- Uses the **two-second rule** as an initial guideline for safe following distance (NY State DMV Driver’s Manual, p. 47)

**Next scope (planned)**
- Integrate **LiDAR TF350** data
- Expand to **overtaking risk** by considering distance to the front vehicle and the oncoming vehicle

---

## Gambaran Umum (Bahasa Indonesia)

NAVIENTA adalah prototipe sistem pendukung keselamatan berkendara yang membagi risiko menjadi 3 level:  
**LOW (aman)**, **MEDIUM (hati-hati)**, dan **HIGH (bahaya)**.

Prototipe ini menggunakan **Random Forest** dan **dashboard real-time (simulasi)** untuk menampilkan perubahan risiko dari waktu ke waktu.  
Fokus saat ini adalah **risiko jarak aman** sebagai tahap pengembangan awal.

**Mengapa Random Forest (dibanding sistem berbasis aturan)?**
- **Berbasis data**: belajar pola dari data, tidak hanya aturan manual  
- **Fleksibel**: lebih mudah dikembangkan saat menambah fitur input  
- **Mudah dievaluasi**: performa dapat diukur dengan metrik yang jelas, dan feature importance membantu menjelaskan perilaku model

**Acuan awal**
- Menggunakan **two-second rule** sebagai pedoman awal jarak aman (NY State DMV Driver’s Manual, hlm. 47)

**Rencana pengembangan**
- Integrasi data **LiDAR TF350**
- Perluasan ke **risiko menyalip** dengan mempertimbangkan jarak kendaraan depan dan kendaraan lawan arah

---
## Dataset Source (English)
The dataset used in this project is publicly available on Kaggle and can
be accessed via the following link:
[https://www.kaggle.com/datasets/nigelwilliams/ngsim-vehicle-trajectory-data-us-101]

Please note that the dataset is not included in this repository in
accordance with Kaggle’s data usage policy.

## Dataset Source (Bahasa Indonesia)
Dataset yang digunakan dalam proyek ini tersedia secara publik di Kaggle
dan dapat diakses melalui tautan berikut:
[https://www.kaggle.com/datasets/nigelwilliams/ngsim-vehicle-trajectory-data-us-101]

Dataset tidak disertakan langsung dalam repository ini sesuai dengan
kebijakan penggunaan data Kaggle.

---

## Repository Structure

- `src/prepare_data.py` — data loading, cleaning, unit conversion, feature engineering, baseline labels  
- `src/train_model.py` — Random Forest training, metrics, confusion matrix, model saving  
- `src/make_insights.py` — demo CSV generation, smoothed risk timeline, high-risk segments, feature importance  
- `notebooks/train_navienta.ipynb` — notebook version (EDA + results)  
- `app_realtime.py` — Streamlit simulated real-time dashboard

> Note: `data/`, `outputs/`, and `model/` are not tracked by Git (see `.gitignore`).

---

## Data Preview (English)

**Raw columns used (proxy dataset):**
- `Vehicle_ID`, `Global_Time` → `timestamp`
- `Spacing` → `d_front_m` (meters)
- `Vehicle_Velocity` → `v_ego_ms` (m/s)
- `Vehicle_Acceleration` → `acc_ms2` (m/s²)
- `Headway` → `headway_s` (seconds) for baseline labeling
- (optional) `Lane_ID`, `Frame_ID`

**Derived features (feature engineering):**
- `dt`
- `closing_speed` (distance closing rate)
- ~1-second rolling features (≈10 rows at 10Hz):  
  `min_d_front_1s`, `med_d_front_1s`, `max_close_1s`, `time_to_reach_1s`

**Prototype labels:**
- `risk_class` ∈ {LOW, MEDIUM, HIGH} created from headway rule (baseline)

> Important: `headway_s` is not used as a model feature to avoid leakage.

---

## Preview Data (Bahasa Indonesia)

**Kolom mentah yang digunakan (dataset proxy):**
- `Vehicle_ID`, `Global_Time` → `timestamp`
- `Spacing` → `d_front_m` (meter)
- `Vehicle_Velocity` → `v_ego_ms` (m/s)
- `Vehicle_Acceleration` → `acc_ms2` (m/s²)
- `Headway` → `headway_s` (detik) untuk baseline labeling
- (opsional) `Lane_ID`, `Frame_ID`

**Fitur turunan (feature engineering):**
- `dt`
- `closing_speed` (laju mendekat / jarak mengecil)
- fitur rolling ~1 detik (≈10 baris pada 10Hz):  
  `min_d_front_1s`, `med_d_front_1s`, `max_close_1s`, `time_to_reach_1s`

**Label prototipe:**
- `risk_class` ∈ {LOW, MEDIUM, HIGH} dibentuk dari aturan headway (baseline)

> Penting: `headway_s` tidak dipakai sebagai fitur model untuk menghindari leakage.

---

## How to Run (English)

Install dependencies:
```bash
pip install -r requirements.txt


