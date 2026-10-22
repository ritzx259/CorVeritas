import os
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
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Cardiovascular Disease Detection
DATA_PATH = "data/heart_disease_uci.csv"
MODEL_DIR = "/Users/ritzmk/Documents/CorVeritas/models"
GRAPH_DIR = os.path.join(MODEL_DIR, "model_mlp_graph")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print("\nDataset shape:")
print(df.shape)
print("\nDataset columns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
 
# CHECK MISSING VALUES
print("\nMissing values:")
print(df.isnull().sum())

# 4. CREATE BINARY TARGET
# Original 'num':
# 0 = no disease
# 1-4 = disease
#
# Convert to:
# 0 = no disease
# 1 = disease
df["target"] = (df["num"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

# 5. SELECT FEATURES
# Remove:
# id      -> identifier, not a clinical feature
# dataset -> source of the patient record
# num     -> original target
# target  -> newly created target
X = df.drop(
    columns=[
        "id",
        "dataset",
        "num",
        "target"
    ]
)

y = df["target"]

print("\nFeatures used:")
print(X.columns.tolist())

# 6. IDENTIFY FEATURE TYPES
categorical_columns = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_columns)

print("\nCategorical features:")
print(categorical_columns)

# 7. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. NUMERICAL PREPROCESSING
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# 9. CATEGORICAL PREPROCESSING
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

# 10. COMBINE PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

# 11. FIT PREPROCESSOR
print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)

# 12. SAVE PREPROCESSOR
preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

print(
    "\nPreprocessor saved to:",
    preprocessor_path
)

# 13. HANDLE CLASS IMBALANCE
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")
print(class_weights)

# 14. BUILD DEEP LEARNING MODEL
input_features = X_train_processed.shape[1]

model = Sequential([
    Input(shape=(input_features,)),
    Dense(
        128,
        activation="relu"
    ),
    BatchNormalization(),
    Dropout(0.30),
    Dense(
        64,
        activation="relu"
    ),
    BatchNormalization(),
    Dropout(0.25),
    Dense(
        32,
        activation="relu"
    ),
    Dropout(0.20),
    Dense(
        1,
        activation="sigmoid"
    )
])

# 15. DISPLAY MODEL
print("\nModel architecture:\n")
model.summary()

# 16. COMPILE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

# 17. CALLBACKS
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.000001
)

# 18. TRAIN MODEL
print("\nStarting training...\n")

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        reduce_lr
    ],
    verbose=1
)

# Plotting Metrics Graphs
# Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CorVeritas Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
accuracy_plot_path = os.path.join(GRAPH_DIR, "accuracy_curve.png")
plt.savefig(accuracy_plot_path, dpi=300)
plt.close()
print("Accuracy graph saved to:", accuracy_plot_path)

# Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CorVeritas Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(GRAPH_DIR, "loss_curve.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print("Loss graph saved to:", loss_plot_path)

# AUC
plt.figure(figsize=(8, 5))
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.title("CorVeritas Model ROC-AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()
auc_plot_path = os.path.join(GRAPH_DIR, "auc_curve.png")
plt.savefig(auc_plot_path, dpi=300)
plt.close()
print("AUC graph saved to:", auc_plot_path)

# Precision
plt.figure(figsize=(8, 5))
plt.plot(history.history["precision"], label="Training Precision")
plt.plot(history.history["val_precision"], label="Validation Precision")
plt.title("CorVeritas Model Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
precision_plot_path = os.path.join(GRAPH_DIR, "precision_curve.png")
plt.savefig(precision_plot_path, dpi=300)
plt.close()
print("Precision graph saved to:", precision_plot_path)

# Recall
plt.figure(figsize=(8, 5))
plt.plot(history.history["recall"], label="Training Recall")
plt.plot(history.history["val_recall"], label="Validation Recall")
plt.title("CorVeritas Model Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
recall_plot_path = os.path.join(GRAPH_DIR, "recall_curve.png")
plt.savefig(recall_plot_path, dpi=300)
plt.close()
print("Recall graph saved to:", recall_plot_path)

# 19. SAVE MODEL
model_path = os.path.join(
    MODEL_DIR,
    "corveritas_stage1.keras"
)

model.save(model_path)

print(
    "\nModel saved to:",
    model_path
)

# 20. PREDICTIONS
print("\nGenerating predictions...")

probabilities = model.predict(
    X_test_processed,
    verbose=0
).ravel()

# Convert probability to class
predictions = (
    probabilities >= 0.50
).astype(int)

# 21. EVALUATION
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n")
print("=" * 60)
print("CORVERITAS MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)

# 22. CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Disease",
            "Disease"
        ],
        zero_division=0
    )
)

# 23. CONFUSION MATRIX 
cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    "\nSaved files:"
)

print(
    "1.",
    model_path
)

print(
    "2.",
    preprocessor_path
)