import pandas as pd
import os
import joblib
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


df = pd.read_csv("data/landmarks_dataset_clean.csv")

X = df.drop("label", axis=1)
y = df["label"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("=" * 60)
print("CLASSES")
print("=" * 60)

for i, cls in enumerate(label_encoder.classes_):
    print(i, "->", cls)


print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv_pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    cv_pipeline,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print("\nMean Accuracy : {:.2f}%".format(cv_scores.mean() * 100))
print("Std Deviation : {:.2f}%".format(cv_scores.std() * 100))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain Samples:", len(X_train))
print("Test Samples :", len(X_test))


print("\nApplying SMOTE on training data...")

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE:")
print("Balanced Train Samples:", len(X_train_balanced))


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None,
    n_jobs=-1
)

model.fit(X_train_balanced, y_train_balanced)


train_pred = model.predict(X_train_balanced)
test_pred = model.predict(X_test)

train_acc = accuracy_score(
    y_train_balanced,
    train_pred
)

test_acc = accuracy_score(
    y_test,
    test_pred
)


print("\n" + "=" * 60)
print("FINAL TRAINING RESULTS")
print("=" * 60)

print(f"Train Accuracy (Balanced Set): {train_acc * 100:.2f}%")
print(f"Test Accuracy                : {test_acc * 100:.2f}%")
print(f"Overfitting Gap              : {(train_acc - test_acc) * 100:.2f}%")


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (TEST)")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_pred,
        target_names=label_encoder.classes_
    )
)


cm = confusion_matrix(y_test, test_pred)

print("\nConfusion Matrix Shape:", cm.shape)


os.makedirs("models", exist_ok=True)
os.makedirs("models/rf_model", exist_ok=True)
os.makedirs("models/encoders", exist_ok=True)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_name = f"rf_smote_best_{timestamp}.pkl"


encoder_name = "label_encoder.pkl"

model_path = os.path.join(
    "models",
    "rf_model",
    model_name
)

encoder_path = os.path.join(
    "models",
    "encoders",
    encoder_name
)

joblib.dump(model, model_path)
joblib.dump(label_encoder, encoder_path)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Model Path  :", model_path)
print("Encoder Path:", encoder_path)
