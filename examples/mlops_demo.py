#!/usr/bin/env python3
"""Example MLOps integration with the personality classification pipeline."""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mlops import MLOpsPipeline
from modules.config import Paths, setup_logging


def run_mlops_example():
    """Run a complete MLOps pipeline example."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting MLOps pipeline example")

    # Check if data exists
    if not Paths.TRAIN_CSV.exists():
        logger.error(f"Training data not found at {Paths.TRAIN_CSV}")
        logger.info("Please ensure you have the training data in the correct location")
        return

    # Initialize MLOps pipeline
    mlops = MLOpsPipeline(
        experiment_name="personality_classification_mlops_demo",
        model_name="demo_personality_model",
    )

    logger.info("Loading and preparing data...")

    # Load data
    train_data = pd.read_csv(Paths.TRAIN_CSV)
    logger.info(f"Loaded training data: {train_data.shape}")

    # For demo purposes, let's prepare the data
    # Assuming we have target columns (adjust based on actual data structure)
    feature_cols = [col for col in train_data.columns if col not in ["Id", "target"]]

    # If we don't have a target column, create a dummy one for demo
    if "target" not in train_data.columns:
        # Create a simple binary target based on a feature for demo
        if len(feature_cols) > 0:
            train_data["target"] = (train_data[feature_cols[0]] > train_data[feature_cols[0]].median()).astype(int)
        else:
            logger.error("No suitable columns found for creating target variable")
            return

    # Prepare features and target
    X = train_data[feature_cols]
    y = train_data["target"]

    # Fill missing values for demo
    X = X.fillna(X.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 1: Data Validation
    logger.info("Step 1: Data Validation")
    validation_results = mlops.validate_and_track_data(
        train_data=train_data,
        test_data=pd.concat([X_test, y_test], axis=1),
    )

    # Log data quality scores
    for dataset, score in validation_results.get("quality_scores", {}).items():
        logger.info(f"Data quality score for {dataset}: {score:.2f}")

    # Step 2: Model Training with MLflow Tracking
    logger.info("Step 2: Model Training with MLflow Tracking")

    # Simple model for demo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=1,  # Conservative for demo
    )

    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "algorithm": "RandomForest",
    }

    model_tags = {
        "model_type": "demo",
        "data_version": "v1.0",
        "experiment_type": "mlops_demo",
    }

    training_results = mlops.train_and_track_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_params=model_params,
        model_tags=model_tags,
        register_model=True,
    )

    logger.info("Model training completed")
    for metric, value in training_results["metrics"].items():
        logger.info(f"{metric}: {value:.4f}")

    # Step 3: Model Promotion
    logger.info("Step 3: Model Promotion")

    # In practice, you'd check if the model meets quality criteria
    test_accuracy = training_results["metrics"].get("test_accuracy", 0)
    if test_accuracy > 0.6:  # Simple threshold for demo
        try:
            # Get the latest model version
            latest_version = mlops.model_registry.get_latest_model_version("demo_personality_model")
            if latest_version:
                mlops.promote_model(
                    model_version=latest_version.version,
                    stage="Production",
                    description="Demo model promoted to production after meeting quality criteria",
                )
                logger.info(f"Model version {latest_version.version} promoted to Production")
        except Exception as e:
            logger.warning(f"Could not promote model: {e}")
    else:
        logger.warning(f"Model accuracy {test_accuracy:.4f} below threshold, not promoting")

    # Step 4: Model Monitoring Setup
    logger.info("Step 4: Model Monitoring Setup")

    # Simulate some predictions for monitoring
    sample_predictions = []
    for i in range(10):
        # Get a sample from test set
        sample_idx = i % len(X_test)
        features = X_test.iloc[sample_idx].to_dict()

        # Make prediction
        pred = model.predict(X_test.iloc[[sample_idx]])[0]
        proba = model.predict_proba(X_test.iloc[[sample_idx]])[0].max()
        actual = y_test.iloc[sample_idx]

        sample_predictions.append({
            "prediction": int(pred),
            "features": features,
            "confidence": float(proba),
            "actual": int(actual),
            "request_id": f"demo_{i}",
        })

    # Log predictions for monitoring
    mlops.monitor_production_model(
        prediction_data=sample_predictions,
        reference_data=X_train,
        baseline_metrics={"accuracy": test_accuracy, "f1_weighted": training_results["metrics"].get("test_f1_weighted", 0)},
    )

    logger.info("Monitoring setup completed")

    # Step 5: Generate MLOps Report
    logger.info("Step 5: Generate MLOps Report")

    mlops_report = mlops.generate_mlops_report()
    logger.info(f"MLOps Report generated with {len(mlops_report.keys())} sections")

    # Print key metrics
    if "monitoring" in mlops_report and "dashboard_data" in mlops_report["monitoring"]:
        dashboard = mlops_report["monitoring"]["dashboard_data"]
        logger.info(f"Total predictions logged: {dashboard.get('total_predictions', 0)}")
        logger.info(f"Average confidence: {dashboard.get('average_confidence', 0):.4f}")

    # Step 6: Model Comparison (if we had multiple versions)
    logger.info("Step 6: Best Model Retrieval")

    best_run = mlops.get_best_model("test_accuracy")
    if best_run:
        logger.info(f"Best model run ID: {best_run.info.run_id}")
        logger.info(f"Best accuracy: {best_run.data.metrics.get('test_accuracy', 'N/A')}")

    logger.info("MLOps pipeline example completed successfully!")

    # Print next steps
    print("\n" + "="*80)
    print("🚀 MLOps Pipeline Demo Completed Successfully!")
    print("="*80)
    print("\nWhat was demonstrated:")
    print("✅ Data validation and quality scoring")
    print("✅ Experiment tracking with MLflow")
    print("✅ Model registration and versioning")
    print("✅ Model promotion workflows")
    print("✅ Production monitoring setup")
    print("✅ Comprehensive MLOps reporting")

    print("\nNext steps:")
    print("1. View MLflow UI: mlflow ui")
    print("2. Check monitoring data in: monitoring/")
    print("3. Integrate with CI/CD for automated deployments")
    print("4. Set up alerts for model drift and performance degradation")
    print("5. Scale serving with container orchestration")

    print("\nMLOps capabilities implemented:")
    print("• Experiment Tracking (MLflow)")
    print("• Model Registry & Versioning")
    print("• Data Validation & Quality Monitoring")
    print("• Model Performance Monitoring")
    print("• Data Drift Detection")
    print("• Model Serving Infrastructure")
    print("• Automated Model Promotion")
    print("• Comprehensive Reporting")


if __name__ == "__main__":
    try:
        run_mlops_example()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
