import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Flatten,
    Dense,
    Input,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Project root config
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart_disease_uci.csv"
MODEL_DIR = ROOT_DIR / "models"
GRAPH_DIR = MODEL_DIR / "model_cnn_graph"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print("\nDataset shape:")
print(df.shape)
print("\nDataset columns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Create binary disease target
# 0 = no disease
# 1 = disease
# Original 'num' column uses 0..4 where 0 indicates no disease and 1-4 indicate disease severity
# We convert to binary classification for the model.
df["target"] = (df["num"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

# Select features for model input
X = df.drop(columns=["id", "dataset", "num", "target"])
y = df["target"]

print("\nFeatures used:")
print(X.columns.tolist())

# Identify feature types
categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumerical features:")
print(numeric_columns)
print("\nCategorical features:")
print(categorical_columns)

# Train / validation / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Numeric preprocessing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
)

print("\nPreprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Save preprocessor
preprocessor_path = MODEL_DIR / "preprocessor_stage2_cnn.pkl"
joblib.dump(preprocessor, preprocessor_path)
print("\nPreprocessor saved to:", preprocessor_path)

# Handle class imbalance
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("\nClass weights:")
print(class_weights)

# Reshape to 1D CNN input: (samples, timesteps, channels)
input_features = X_train_processed.shape[1]
X_train_cnn = X_train_processed.reshape(-1, input_features, 1)
X_test_cnn = X_test_processed.reshape(-1, input_features, 1)

# Build a 1D CNN model for tabular clinical data
model = Sequential(
    [
        Input(shape=(input_features, 1)),
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.20),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid"),
    ]
)

print("\nModel architecture:\n")
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.000001,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
)

print("\nStarting CNN training...\n")
history = model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=60,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
)

# Plot model performance curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CorVeritas CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
accuracy_plot_path = GRAPH_DIR / "cnn_accuracy_curve.png"
plt.savefig(accuracy_plot_path, dpi=300)
plt.close()
print("Accuracy graph saved to:", accuracy_plot_path)

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CorVeritas CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = GRAPH_DIR / "cnn_loss_curve.png"
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print("Loss graph saved to:", loss_plot_path)

plt.figure(figsize=(8, 5))
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.title("CorVeritas CNN ROC-AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()
auc_plot_path = GRAPH_DIR / "cnn_auc_curve.png"
plt.savefig(auc_plot_path, dpi=300)
plt.close()
print("AUC graph saved to:", auc_plot_path)

plt.figure(figsize=(8, 5))
plt.plot(history.history["precision"], label="Training Precision")
plt.plot(history.history["val_precision"], label="Validation Precision")
plt.title("CorVeritas CNN Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
precision_plot_path = GRAPH_DIR / "cnn_precision_curve.png"
plt.savefig(precision_plot_path, dpi=300)
plt.close()
print("Precision graph saved to:", precision_plot_path)

plt.figure(figsize=(8, 5))
plt.plot(history.history["recall"], label="Training Recall")
plt.plot(history.history["val_recall"], label="Validation Recall")
plt.title("CorVeritas CNN Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
recall_plot_path = GRAPH_DIR / "cnn_recall_curve.png"
plt.savefig(recall_plot_path, dpi=300)
plt.close()
print("Recall graph saved to:", recall_plot_path)

# Save trained model
model_path = MODEL_DIR / "corveritas_stage2_cnn.keras"
model.save(model_path)
print("\nModel saved to:", model_path)

# Predictions
print("\nGenerating predictions...")
probabilities = model.predict(X_test_cnn, verbose=0).ravel()
predictions = (probabilities >= 0.50).astype(int)

# Evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
auc = roc_auc_score(y_test, probabilities)

print("\n")
print("=" * 60)
print("CORVERITAS CNN MODEL EVALUATION")
print("=" * 60)
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Disease", "Disease"],
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

print("\n")
print("=" * 60)
print("CNN TRAINING COMPLETED")
print("=" * 60)
print("\nSaved files:")
print("1.", model_path)
print("2.", preprocessor_path)
