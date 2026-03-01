# train_global_model.py — run ONCE before starting the API
#
# Trains the global RF on synthetic ENS data.
# Features: 7 patient-reported clinical symptoms only.
#
# Class distributions:
#   Green  (50%) — mild/no symptoms, good sleep, low anxiety
#   Yellow (30%) — moderate symptoms
#   Red    (20%) — severe symptoms, poor sleep, high anxiety

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Air_Sensation",
    "Nasal_Dryness",
    "Nasal_Burning",
    "Suffocation",
    "Anxiety_Score",
    "Humidity_Level_Pct",
    "Sleep_Quality_Hrs",
]


def generate_ens_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic ENS dataset — 7 clinical features, 3 classes.

    Signal_Clean: noise-free label for unbiased evaluation.
    Signal:       ~10% label noise for realistic training.
    """
    np.random.seed(seed)
    data = []

    for _ in range(n_samples):
        state        = np.random.choice([0, 1, 2], p=[0.50, 0.30, 0.20])
        shared_noise = np.random.normal(0, 1.0)  # within-day symptom correlation

        if state == 2:   # Red — severe
            indoor_hum    = np.clip(np.random.normal(22,  8) + shared_noise * -2,    5,  65)
            air_sensation = np.clip(np.random.normal(8.2, 1.2) + shared_noise * 0.5, 1,  10)
            dryness       = np.clip(np.random.normal(7.8, 1.3) + shared_noise * 0.5, 1,  10)
            burning       = np.clip(np.random.normal(8.0, 1.2) + shared_noise * 0.5, 1,  10)
            suffocation   = np.clip(np.random.normal(8.2, 1.1) + shared_noise * 0.5, 1,  10)
            anxiety       = np.clip(np.random.normal(8.0, 1.3) + shared_noise * 0.4, 1,  10)
            sleep         = np.clip(np.random.normal(3.5, 1.0) + shared_noise * -0.3, 1, 12)

        elif state == 1:  # Yellow — moderate
            indoor_hum    = np.clip(np.random.normal(42,  9) + shared_noise * -1.5,  10, 80)
            air_sensation = np.clip(np.random.normal(5.3, 1.5) + shared_noise * 0.5,  1,  10)
            dryness       = np.clip(np.random.normal(5.0, 1.5) + shared_noise * 0.5,  1,  10)
            burning       = np.clip(np.random.normal(5.1, 1.5) + shared_noise * 0.5,  1,  10)
            suffocation   = np.clip(np.random.normal(5.3, 1.4) + shared_noise * 0.5,  1,  10)
            anxiety       = np.clip(np.random.normal(5.2, 1.5) + shared_noise * 0.4,  1,  10)
            sleep         = np.clip(np.random.normal(6.0, 1.0) + shared_noise * -0.3,  1, 12)

        else:             # Green — mild
            indoor_hum    = np.clip(np.random.normal(63, 10) + shared_noise * -1,    20, 100)
            air_sensation = np.clip(np.random.normal(2.3, 1.2) + shared_noise * 0.4,  1,  10)
            dryness       = np.clip(np.random.normal(2.0, 1.1) + shared_noise * 0.4,  1,  10)
            burning       = np.clip(np.random.normal(2.1, 1.1) + shared_noise * 0.4,  1,  10)
            suffocation   = np.clip(np.random.normal(2.2, 1.1) + shared_noise * 0.4,  1,  10)
            anxiety       = np.clip(np.random.normal(2.5, 1.2) + shared_noise * 0.3,  1,  10)
            sleep         = np.clip(np.random.normal(7.8, 0.9) + shared_noise * -0.2,  4, 12)

        signal_map  = {0: "Green", 1: "Yellow", 2: "Red"}
        clean_label = signal_map[state]

        noisy_label = clean_label
        if np.random.rand() < 0.10:
            noisy_label = signal_map[np.random.choice([s for s in [0, 1, 2] if s != state])]

        data.append({
            "Air_Sensation":      round(float(air_sensation), 1),
            "Nasal_Dryness":      round(float(dryness),       1),
            "Nasal_Burning":      round(float(burning),       1),
            "Suffocation":        round(float(suffocation),   1),
            "Anxiety_Score":      round(float(anxiety),       1),
            "Humidity_Level_Pct": round(float(indoor_hum),    1),  # indoor/personal humidity
            "Sleep_Quality_Hrs":  round(float(sleep),         1),
            "Signal":             noisy_label,
            "Signal_Clean":       clean_label,
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Generating ENS training data (7 clinical features)...")
    df = generate_ens_dataset(n_samples=2000)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['Signal'].value_counts()}\n")

    X       = df[FEATURES]
    le      = LabelEncoder()
    y_noisy = le.fit_transform(df["Signal"])
    y_clean = le.transform(df["Signal_Clean"])
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    X_train, X_test, y_train_noisy, y_test_noisy, _, y_test_clean = train_test_split(
        X, y_noisy, y_clean, test_size=0.2, random_state=42, stratify=y_noisy
    )

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_noisy, cv=cv, scoring="f1_weighted")
    print(f"CV F1 (noisy labels): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X_train_scaled, y_train_noisy)

    train_acc = model.score(X_train_scaled, y_train_noisy)
    test_acc  = model.score(X_test_scaled,  y_test_noisy)
    print(f"Train acc: {train_acc:.3f}  Test acc: {test_acc:.3f}  Gap: {train_acc - test_acc:.3f}")

    clean_acc = model.score(X_test_scaled, y_test_clean)
    print(f"Test acc (clean labels, unbiased): {clean_acc:.3f}\n")
    print("Classification report (clean labels):")
    print(classification_report(y_test_clean, model.predict(X_test_scaled), target_names=le.classes_))

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for feat, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    joblib.dump(model,  MODEL_DIR / "global_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")
    print(f"\nGlobal model  -> {MODEL_DIR}/global_model.pkl")
    print(f"Scaler        -> {MODEL_DIR}/scaler.pkl")
    print(f"Label encoder -> {MODEL_DIR}/label_encoder.pkl")
