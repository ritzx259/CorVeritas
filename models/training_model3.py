from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart_disease_uci.csv"
MODEL_DIR = ROOT_DIR / "models"
GRAPH_DIR = MODEL_DIR / "model_random_forest_graph"
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

# The source target uses 0 for no disease and 1-4 for disease.
df["target"] = (df["num"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

# Exclude identifiers, source site, and the original target to prevent leakage.
X = df.drop(columns=["id", "dataset", "num", "target"])
y = df["target"]

print("\nFeatures used:")
print(X.columns.tolist())

categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

print("\nNumerical features:")
print(numeric_columns)
print("\nCategorical features:")
print(categorical_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Fit all imputation and encoding steps inside each CV fold to avoid leakage.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
)

parameter_distributions = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 8, 12, 16],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.75],
    "model__class_weight": [None, "balanced_subsample"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_distributions,
    n_iter=15,
    scoring="average_precision",
    cv=cv,
    refit=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    error_score="raise",
    verbose=1,
)

print("\nTuning Random Forest with stratified 5-fold cross-validation...")
search.fit(X_train, y_train)
model = search.best_estimator_

print("\nBest cross-validation average precision: {:.4f}".format(search.best_score_))
print("Best model parameters:")
print(search.best_params_)

model_path = MODEL_DIR / "corveritas_stage3_random_forest.joblib"
preprocessor_path = MODEL_DIR / "preprocessor_stage3_random_forest.pkl"
joblib.dump(model, model_path)
joblib.dump(model.named_steps["preprocessor"], preprocessor_path)

print("\nModel saved to:", model_path)
print("Preprocessor saved to:", preprocessor_path)

print("\nGenerating predictions...")
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
auc = roc_auc_score(y_test, probabilities)
average_precision = average_precision_score(y_test, probabilities)

print("\n")
print("=" * 60)
print("CORVERITAS RANDOM FOREST MODEL EVALUATION")
print("=" * 60)
print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"ROC-AUC           : {auc:.4f}")
print(f"Average Precision : {average_precision:.4f}")

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

# Tree ensembles do not train in epochs, so report held-out metrics and curves.
metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Precision"]
metric_values = [accuracy, precision, recall, f1, auc, average_precision]
plt.figure(figsize=(9, 5))
bars = plt.bar(metric_names, metric_values, color="#4C78A8")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("CorVeritas Random Forest Test Metrics")
plt.grid(axis="y", alpha=0.3)
for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )
plt.tight_layout()
metrics_plot_path = GRAPH_DIR / "test_metrics.png"
plt.savefig(metrics_plot_path, dpi=300)
plt.close()
print("\nTest metrics graph saved to:", metrics_plot_path)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"],
).plot(cmap="Blues", values_format="d", ax=plt.gca(), colorbar=False)
plt.title("CorVeritas Random Forest Confusion Matrix")
plt.tight_layout()
confusion_plot_path = GRAPH_DIR / "confusion_matrix.png"
plt.savefig(confusion_plot_path, dpi=300)
plt.close()
print("Confusion matrix graph saved to:", confusion_plot_path)

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
ax.set_title("CorVeritas Random Forest ROC Curve")
ax.grid(True, alpha=0.3)
fig.tight_layout()
roc_plot_path = GRAPH_DIR / "roc_curve.png"
fig.savefig(roc_plot_path, dpi=300)
plt.close(fig)
print("ROC curve saved to:", roc_plot_path)

precision_values, recall_values, _ = precision_recall_curve(y_test, probabilities)
plt.figure(figsize=(7, 5))
plt.plot(recall_values, precision_values, label=f"AP = {average_precision:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("CorVeritas Random Forest Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
pr_plot_path = GRAPH_DIR / "precision_recall_curve.png"
plt.savefig(pr_plot_path, dpi=300)
plt.close()
print("Precision-recall curve saved to:", pr_plot_path)

print("\n")
print("=" * 60)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 60)
print("\nSaved files:")
print("1.", model_path)
print("2.", preprocessor_path)
print("3.", metrics_plot_path)
print("4.", confusion_plot_path)
print("5.", roc_plot_path)
print("6.", pr_plot_path)