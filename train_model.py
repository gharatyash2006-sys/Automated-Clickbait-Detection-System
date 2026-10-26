import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

from preprocessing import clean_text



# PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "clickbait_data.csv"
)

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = (
    MODEL_DIR
    / "clickbait_svm_model.joblib"
)

METRICS_PATH = (
    MODEL_DIR
    / "metrics.json"
)

CONFUSION_MATRIX_PATH = (
    MODEL_DIR
    / "svm_confusion_matrix.png"
)

MISCLASSIFIED_PATH = (
    MODEL_DIR
    / "svm_misclassified_examples.csv"
)

COMPARISON_PATH = (
    MODEL_DIR
    / "model_comparison.csv"
)



# HEADER


print("===================================")
print("     CLICKBAIT MODEL COMPARISON")
print("===================================")



# LOAD DATASET


print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total rows:", len(df))



# CHECK COLUMNS


required_columns = [
    "headline",
    "clickbait"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"was not found in dataset."
        )

    
# CLEAN DATA


df = df[
    ["headline", "clickbait"]
].copy()

df = df.dropna(
    subset=[
        "headline",
        "clickbait"
    ]
)

df["headline"] = (
    df["headline"]
    .astype(str)
)

df["clickbait"] = pd.to_numeric(
    df["clickbait"],
    errors="coerce"
)

df = df.dropna(
    subset=["clickbait"]
)

df["clickbait"] = (
    df["clickbait"]
    .astype(int)
)

df = df[
    df["clickbait"].isin([0, 1])
]


print("\nAfter cleaning:")
print(
    "Total usable rows:",
    len(df)
)



# LABEL DISTRIBUTION


print("\nLabel distribution:")

print(
    df["clickbait"]
    .value_counts()
    .sort_index()
)

print("\n0 = Non-clickbait")
print("1 = Clickbait")



# INPUT AND TARGET


X = df["headline"]

y = df["clickbait"]



# TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)



# MODEL DEFINITIONS


models = {

    "Logistic Regression": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=1000
            )
        )

    ]),


    "Naive Bayes": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            MultinomialNB()
        )

    ]),


    
    # CALIBRATED LINEAR SVM
    

    "Linear SVM": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            CalibratedClassifierCV(
                estimator=LinearSVC(),
                method="sigmoid",
                cv=3
            )
        )

    ])
}



# TRAIN MODELS


results = []

trained_models = {}


for model_name, model in models.items():

    print("\n===================================")
    print(
        f"Training: {model_name}"
    )
    print("===================================")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )


    
    # PREDICTION
    

    y_pred = model.predict(
        X_test
    )


    
    # METRICS
    

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    
    # PRINT RESULTS
    

    print(
        f"\nAccuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )


    # STORE RESULTS

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1

    })


    trained_models[
        model_name
    ] = model



# MODEL COMPARISON


comparison_df = pd.DataFrame(
    results
)


print("\n===================================")
print("        MODEL COMPARISON")
print("===================================")

print(
    comparison_df.to_string(
        index=False
    )
)



# SAVE COMPARISON

MODEL_DIR.mkdir(
    exist_ok=True
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print(
    "\nModel comparison saved to:"
)

print(
    COMPARISON_PATH
)



# SELECT LINEAR SVM


final_model = trained_models[
    "Linear SVM"
]

y_pred = final_model.predict(
    X_test
)



# FINAL MODEL REPORT

print("\n===================================")
print("    Linear SVM REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-clickbait",
            "Clickbait"
        ],
        zero_division=0
    )
)



# CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n===================================")
print("        CONFUSION MATRIX")
print("===================================")

print(
    cm
)



# SAVE CONFUSION MATRIX


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,

    display_labels=[
        "Non-clickbait",
        "Clickbait"
    ]
)

disp.plot(
    values_format="d"
)

plt.title(
    "Clickbait Detection - Linear SVM"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved to:"
)

print(
    CONFUSION_MATRIX_PATH
)



# MISCLASSIFIED HEADLINES


test_results = pd.DataFrame({

    "headline": X_test.values,

    "actual": y_test.values,

    "predicted": y_pred

})


misclassified = test_results[
    test_results["actual"]
    != test_results["predicted"]
].copy()


misclassified["actual"] = (
    misclassified["actual"]
    .map({
        0: "Non-clickbait",
        1: "Clickbait"
    })
)


misclassified["predicted"] = (
    misclassified["predicted"]
    .map({
        0: "Non-clickbait",
        1: "Clickbait"
    })
)


print(
    "\nMisclassified headlines:",
    len(misclassified)
)


misclassified.to_csv(
    MISCLASSIFIED_PATH,
    index=False,
    encoding="utf-8"
)


print(
    "Misclassified examples saved to:"
)

print(
    MISCLASSIFIED_PATH
)



# SAVE FINAL MODEL


joblib.dump(
    final_model,
    MODEL_PATH
)


print(
    "\nFinal Linear SVM model saved to:"
)

print(
    MODEL_PATH
)



# SAVE METRICS


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


metrics = {

    "dataset_size": int(len(df)),

    "training_samples": int(
        len(X_train)
    ),

    "testing_samples": int(
        len(X_test)
    ),

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "confusion_matrix": cm.tolist(),

    "misclassified_samples": int(
        len(misclassified)
    )

}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    "\nMetrics saved to:"
)

print(
    METRICS_PATH
)



# COMPLETE


print("\n===================================")
print("       TRAINING COMPLETE")
print("===================================")