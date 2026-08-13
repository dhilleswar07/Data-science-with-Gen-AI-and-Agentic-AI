# ============================================================
# MLOps with MLflow
# Iris Classification
# Logistic Regression + Random Forest
# ============================================================

import os
import time

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models import infer_signature


# ============================================================
# 1. MLFLOW TRACKING URI
# ============================================================

# IMPORTANT:
# Use the SAME mlflow.db that your MLflow UI/server uses.

mlflow.set_tracking_uri("http://localhost:5000")

print("=" * 60)
print("MLflow Tracking URI:")
print(mlflow.get_tracking_uri())
print("=" * 60)


# ============================================================
# 2. CREATE / SET EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "Iris_Classification_Experiment"

client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is not None:

    if experiment.lifecycle_stage == "deleted":

        print(
            "\nExperiment is deleted."
        )

        print(
            "Restoring experiment..."
        )

        client.restore_experiment(
            experiment.experiment_id
        )

        print(
            "Experiment restored successfully."
        )

    else:

        print(
            "\nExperiment already exists."
        )

else:

    print(
        "\nExperiment does not exist."
    )

    print(
        "Creating new experiment..."
    )

    mlflow.create_experiment(
        EXPERIMENT_NAME
    )


# Set active experiment
mlflow.set_experiment(
    EXPERIMENT_NAME
)

print(
    f"\nUsing experiment: {EXPERIMENT_NAME}"
)

print("=" * 60)


# ============================================================
# 3. CREATE ARTIFACT DIRECTORY
# ============================================================

os.makedirs("artifacts", exist_ok=True)


# ============================================================
# 4. LOAD IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

print("\nDataset loaded successfully.")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Classes:", iris.target_names)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

print("\n")
print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)


with mlflow.start_run(
    run_name="Logistic_Regression_Run"
):

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    lr_model = LogisticRegression(
        max_iter=1000,
        C=1000,
        solver="lbfgs",
        random_state=42
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    lr_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    lr_predictions = lr_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    lr_accuracy = accuracy_score(
        y_test,
        lr_predictions
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    lr_conf_matrix = confusion_matrix(
        y_test,
        lr_predictions
    )

    # --------------------------------------------------------
    # Log Parameters
    # --------------------------------------------------------

    mlflow.log_params({

        "model": "Logistic Regression",

        "max_iter": 1000,

        "C": 1000,

        "solver": "lbfgs",

        "random_state": 42

    })

    # --------------------------------------------------------
    # Log Metric
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        lr_accuracy
    )

    # --------------------------------------------------------
    # Create Model Signature
    # --------------------------------------------------------

    lr_signature = infer_signature(
        X_train,
        lr_model.predict(X_train)
    )

    # --------------------------------------------------------
    # Create Confusion Matrix Image
    # --------------------------------------------------------

    timestamp = time.strftime(
        "%Y%m%d-%H%M%S"
    )

    lr_cm_path = (
        f"artifacts/"
        f"lr_conf_matrix_{timestamp}.png"
    )

    plt.figure(
        figsize=(6, 6)
    )

    sns.heatmap(
        lr_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title(
        "Confusion Matrix - Logistic Regression"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        lr_cm_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Log Confusion Matrix
    # --------------------------------------------------------

    mlflow.log_artifact(
        lr_cm_path
    )

    # --------------------------------------------------------
    # LOG + REGISTER MODEL
    #
    # This replaces:
    # mlflow.register_model(...)
    #
    # MLflow will automatically create a new
    # model version.
    # --------------------------------------------------------

    lr_model_info = mlflow.sklearn.log_model(

        sk_model=lr_model,

        name="logistic_regression_model",

        signature=lr_signature,

        registered_model_name=(
            "Logistic_Regression_Model"
        )
    )

    # --------------------------------------------------------
    # Get Run ID
    # --------------------------------------------------------

    lr_run_id = mlflow.active_run().info.run_id

    print("\nLogistic Regression Run ID:")
    print(lr_run_id)

    # --------------------------------------------------------
    # Get Registered Version
    # --------------------------------------------------------

    lr_registered_version = (
        lr_model_info.registered_model_version
    )

    print("\nRegistered Model:")
    print("Logistic_Regression_Model")

    print("Model Version:")
    print(lr_registered_version)


# ============================================================
# 7. LOGISTIC REGRESSION RESULTS
# ============================================================

print("\n")
print("=" * 60)

print(
    "Logistic Regression Accuracy:"
)

print(
    f"{lr_accuracy:.4f}"
)

print(
    f"Percentage: {lr_accuracy * 100:.2f}%"
)

print("\nLogistic Regression Confusion Matrix:")

print(lr_conf_matrix)

print("=" * 60)


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)


with mlflow.start_run(
    run_name="Random_Forest_Run"
):

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(

        n_estimators=10,

        max_depth=3,

        min_samples_split=10,

        criterion="entropy",

        random_state=0
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    rf_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    rf_predictions = rf_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    rf_accuracy = accuracy_score(
        y_test,
        rf_predictions
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    rf_conf_matrix = confusion_matrix(
        y_test,
        rf_predictions
    )

    # --------------------------------------------------------
    # Log Parameters
    # --------------------------------------------------------

    mlflow.log_params({

        "model": "Random Forest",

        "n_estimators": 10,

        "max_depth": 3,

        "min_samples_split": 10,

        "criterion": "entropy",

        "random_state": 0

    })

    # --------------------------------------------------------
    # Log Metric
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        rf_accuracy
    )

    # --------------------------------------------------------
    # Create Model Signature
    # --------------------------------------------------------

    rf_signature = infer_signature(
        X_train,
        rf_model.predict(X_train)
    )

    # --------------------------------------------------------
    # Create Confusion Matrix Image
    # --------------------------------------------------------

    timestamp = time.strftime(
        "%Y%m%d-%H%M%S"
    )

    rf_cm_path = (
        f"artifacts/"
        f"rf_conf_matrix_{timestamp}.png"
    )

    plt.figure(
        figsize=(6, 6)
    )

    sns.heatmap(
        rf_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title(
        "Confusion Matrix - Random Forest"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        rf_cm_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Log Confusion Matrix
    # --------------------------------------------------------

    mlflow.log_artifact(
        rf_cm_path
    )

    # --------------------------------------------------------
    # LOG + REGISTER MODEL
    # --------------------------------------------------------

    rf_model_info = mlflow.sklearn.log_model(

        sk_model=rf_model,

        name="random_forest_model",

        signature=rf_signature,

        registered_model_name=(
            "Random_Forest_Model"
        )
    )

    # --------------------------------------------------------
    # Get Run ID
    # --------------------------------------------------------

    rf_run_id = mlflow.active_run().info.run_id

    print("\nRandom Forest Run ID:")
    print(rf_run_id)

    # --------------------------------------------------------
    # Get Registered Version
    # --------------------------------------------------------

    rf_registered_version = (
        rf_model_info.registered_model_version
    )

    print("\nRegistered Model:")
    print("Random_Forest_Model")

    print("Model Version:")
    print(rf_registered_version)


# ============================================================
# 9. RANDOM FOREST RESULTS
# ============================================================

print("\n")
print("=" * 60)

print(
    "Random Forest Accuracy:"
)

print(
    f"{rf_accuracy:.4f}"
)

print(
    f"Percentage: {rf_accuracy * 100:.2f}%"
)

print("\nRandom Forest Confusion Matrix:")

print(rf_conf_matrix)

print("=" * 60)


# ============================================================
# 10. LOAD LATEST LOGISTIC REGRESSION MODEL
# ============================================================

print("\n")
print("=" * 60)
print("LOADING REGISTERED MODELS")
print("=" * 60)


try:

    loaded_lr_model = mlflow.sklearn.load_model(
        "models:/Logistic_Regression_Model/latest"
    )

    print(
        "\nLogistic Regression model "
        "loaded successfully."
    )

except Exception as e:

    print(
        "\nError loading Logistic Regression:"
    )

    print(e)


# ============================================================
# 11. LOAD LATEST RANDOM FOREST MODEL
# ============================================================

try:

    loaded_rf_model = mlflow.sklearn.load_model(
        "models:/Random_Forest_Model/latest"
    )

    print(
        "\nRandom Forest model "
        "loaded successfully."
    )

except Exception as e:

    print(
        "\nError loading Random Forest:"
    )

    print(e)


# ============================================================
# 12. TEST LOADED MODELS
# ============================================================

print("\n")
print("=" * 60)
print("TESTING LOADED MODELS")
print("=" * 60)


# Test one sample
sample = X_test[0].reshape(1, -1)


try:

    lr_loaded_prediction = (
        loaded_lr_model.predict(sample)
    )

    print(
        "\nLogistic Regression prediction:"
    )

    print(
        iris.target_names[
            lr_loaded_prediction[0]
        ]
    )

except Exception as e:

    print(
        "Could not test Logistic Regression:"
    )

    print(e)


try:

    rf_loaded_prediction = (
        loaded_rf_model.predict(sample)
    )

    print(
        "\nRandom Forest prediction:"
    )

    print(
        iris.target_names[
            rf_loaded_prediction[0]
        ]
    )

except Exception as e:

    print(
        "Could not test Random Forest:"
    )

    print(e)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MLFLOW EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nLogistic Regression Accuracy : "
    f"{lr_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy       : "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    "\nRegistered Models:"
)

print(
    "1. Logistic_Regression_Model"
)

print(
    "2. Random_Forest_Model"
)

print(
    "\nMLflow UI:"
)

print(
    "http://localhost:5000"
)

print("=" * 60)