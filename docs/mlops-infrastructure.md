# MLOps Infrastructure Documentation

## Overview

This document describes the comprehensive MLOps (Machine Learning Operations) infrastructure implemented for the personality classification project. The MLOps system provides end-to-end lifecycle management for machine learning models, from development to production deployment and monitoring.

## Architecture

### Components

1. **Experiment Tracking** (`ExperimentTracker`)
   - MLflow-based experiment tracking
   - Parameter and metric logging
   - Model artifacts management
   - Experiment comparison and analysis

2. **Model Registry** (`ModelRegistry`)
   - Centralized model versioning
   - Model stage management (Development, Staging, Production)
   - Model lineage tracking
   - Automated model promotion workflows

3. **Data Validation** (`DataValidator`)
   - Comprehensive data quality checks
   - Data drift detection
   - Schema validation
   - Statistical profiling

4. **Model Monitoring** (`ModelMonitor`)
   - Real-time performance tracking
   - Data drift detection
   - Performance degradation alerts
   - Prediction logging and analysis

5. **Model Serving** (`ModelServer`)
   - HTTP API for model inference
   - Batch prediction support
   - Model versioning in production
   - Health checks and monitoring

6. **MLOps Pipeline** (`MLOpsPipeline`)
   - Integrated workflow orchestration
   - End-to-end pipeline automation
   - Cross-component coordination

## Getting Started

### Prerequisites

```bash
# Install MLOps dependencies
pip install mlflow flask joblib

# Or install with all dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.mlops import MLOpsPipeline

# Initialize MLOps pipeline
mlops = MLOpsPipeline(
    experiment_name="personality_classification",
    model_name="personality_model"
)

# Validate data
validation_results = mlops.validate_and_track_data(train_data, test_data)

# Train and track model
training_results = mlops.train_and_track_model(
    model=your_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_params={"param1": "value1"},
    register_model=True
)

# Promote model to production
mlops.promote_model(model_version="1", stage="Production")

# Monitor production model
monitoring_results = mlops.monitor_production_model(
    prediction_data=recent_predictions,
    reference_data=reference_dataset
)
```

## Detailed Component Guide

### 1. Experiment Tracking

The `ExperimentTracker` provides comprehensive experiment management using MLflow.

#### Key Features:
- **Parameter Logging**: Hyperparameters, model configurations
- **Metric Tracking**: Performance metrics, custom metrics
- **Artifact Storage**: Models, plots, datasets
- **Run Comparison**: Side-by-side experiment comparison

#### Example Usage:
```python
tracker = ExperimentTracker("my_experiment")

with tracker.start_run("model_training"):
    # Log parameters
    tracker.log_params({"learning_rate": 0.01, "batch_size": 32})

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.93})

    # Log model
    tracker.log_model(model, "model")

    # Log confusion matrix
    tracker.log_confusion_matrix(y_true, y_pred)
```

### 2. Model Registry

The `ModelRegistry` manages model versions and deployment stages.

#### Model Stages:
- **None**: Initial registration
- **Staging**: Testing and validation
- **Production**: Live deployment
- **Archived**: Deprecated models

#### Example Usage:
```python
registry = ModelRegistry()

# Register model
model_version = registry.register_model(
    model_uri="runs:/run_id/model",
    name="personality_model",
    description="Random Forest classifier"
)

# Promote to production
registry.promote_model("personality_model", "1", "Production")

# Load production model
model = registry.load_model("personality_model", stage="Production")
```

### 3. Data Validation

The `DataValidator` ensures data quality and consistency.

#### Validation Checks:
- **Missing Data**: Null values, completeness
- **Data Types**: Schema consistency
- **Duplicates**: Row-level duplicates
- **Outliers**: Statistical outlier detection
- **Distributions**: Class balance, feature distributions
- **Data Drift**: Distribution changes over time

#### Example Usage:
```python
validator = DataValidator()

# Validate dataset
results = validator.validate_dataset(df, "train_data")

# Check data quality score
score = validator.get_data_quality_score("train_data")

# Validate train/test split
split_results = validator.validate_train_test_split(
    X_train, X_test, y_train, y_test
)
```

### 4. Model Monitoring

The `ModelMonitor` tracks model performance in production.

#### Monitoring Capabilities:
- **Performance Metrics**: Accuracy, F1-score, precision, recall
- **Data Drift Detection**: Feature distribution changes
- **Prediction Logging**: Request/response tracking
- **Alerting**: Automatic issue detection
- **Dashboard Data**: Real-time monitoring metrics

#### Example Usage:
```python
monitor = ModelMonitor("personality_model")

# Log predictions
monitor.log_prediction(
    prediction=pred,
    features=input_features,
    confidence=confidence_score,
    actual=actual_value
)

# Calculate performance metrics
metrics = monitor.calculate_performance_metrics(window_hours=24)

# Detect data drift
drift_results = monitor.detect_data_drift(reference_data)
```

### 5. Model Serving

The `ModelServer` provides an interactive Dash-based dashboard for model inference and monitoring.

#### Dashboard Features:
- **📊 Interactive Dashboard**: Modern web-based interface
- **🔮 Multiple Input Methods**: Manual forms, JSON input, file upload
- **📈 Real-time Monitoring**: Live prediction history and statistics
- **🎨 Beautiful UI**: Professional styling with confidence visualization
- **🔄 Auto-refresh**: Live updates of prediction history

#### Example Usage:
```python
# Create interactive dashboard server
server = ModelServer(
    model_name="personality_model",
    model_stage="Production",
    port=8050
)

# Run dashboard server
server.run()
# Access at http://localhost:8050
```

#### Dashboard Components:
- **Model Status Cards**: Real-time model health and statistics
- **Prediction Interface**: Multiple input methods with validation
- **Results Visualization**: Confidence scores and probability distributions
- **History Table**: Searchable prediction history with timestamps

#### API Examples:
```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": 2.0}}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feature1": 1.0}, {"feature1": 2.0}]}'
```

## Deployment Patterns

### 1. Local Development
```python
# Run MLOps demo
python examples/mlops_demo.py

# Start MLflow UI
mlflow ui

# Start model server
python -m src.mlops.serving --model-name personality_model
```

### 2. Docker Deployment
```dockerfile
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
WORKDIR /app

EXPOSE 5000
CMD ["python", "-m", "src.mlops.serving", "--model-name", "personality_model"]
```

### 3. Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: personality-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: personality-model-server
  template:
    metadata:
      labels:
        app: personality-model-server
    spec:
      containers:
      - name: model-server
        image: personality-model:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: pip install -e ".[dev]"
    - name: Validate data
      run: python scripts/validate_data.py

  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Train model
      run: python scripts/train_model.py
    - name: Register model
      run: python scripts/register_model.py

  model-deployment:
    needs: model-training
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to staging
      run: python scripts/deploy_model.py --stage staging
    - name: Run integration tests
      run: python scripts/test_model_api.py
    - name: Promote to production
      run: python scripts/promote_model.py --stage production
```

## Monitoring and Alerting

### Setting Up Alerts
```python
# Configure monitoring thresholds
monitor = ModelMonitor("personality_model")

# Set up performance degradation alerts
baseline_metrics = {"accuracy": 0.85, "f1_score": 0.83}
degradation_results = monitor.detect_performance_degradation(
    baseline_metrics,
    degradation_threshold=0.05  # 5% degradation threshold
)

# Set up data drift alerts
drift_results = monitor.detect_data_drift(
    reference_data,
    drift_threshold=0.1  # 10% drift threshold
)
```

### Dashboard Integration
```python
# Get dashboard data
dashboard_data = monitor.get_monitoring_dashboard_data(hours=24)

# Generate monitoring report
report = monitor.generate_monitoring_report()
```

## Best Practices

### 1. Experiment Organization
- Use descriptive experiment names
- Tag experiments with metadata
- Document parameter choices
- Compare similar experiments

### 2. Model Versioning
- Semantic versioning for models
- Clear version descriptions
- Tag models with deployment info
- Maintain model lineage

### 3. Data Quality
- Validate all data inputs
- Monitor for drift continuously
- Set quality thresholds
- Automate data checks

### 4. Monitoring
- Log all predictions
- Track performance metrics
- Set up alerting thresholds
- Regular monitoring reviews

### 5. Security
- Secure MLflow tracking server
- API authentication/authorization
- Data privacy compliance
- Audit trail maintenance

## Troubleshooting

### Common Issues

1. **MLflow Connection Errors**
   ```python
   # Check MLflow server status
   import mlflow
   print(mlflow.get_tracking_uri())
   ```

2. **Model Loading Issues**
   ```python
   # Verify model exists
   registry = ModelRegistry()
   models = registry.list_models()
   print([m.name for m in models])
   ```

3. **Data Validation Failures**
   ```python
   # Check validation details
   validator = DataValidator()
   results = validator.validate_dataset(df)
   print(results['missing_data'])
   ```

4. **Monitoring Data Issues**
   ```python
   # Check monitoring logs
   monitor = ModelMonitor("model_name")
   dashboard = monitor.get_monitoring_dashboard_data()
   print(f"Total predictions: {dashboard['total_predictions']}")
   ```

## Performance Optimization

### 1. MLflow Optimization
- Use artifact stores (S3, Azure Blob)
- Configure database backend
- Enable model caching

### 2. Serving Optimization
- Use model serialization (joblib, pickle)
- Implement request batching
- Add response caching

### 3. Monitoring Optimization
- Aggregate metrics efficiently
- Use sampling for large volumes
- Implement data retention policies

## Future Enhancements

1. **Advanced Monitoring**
   - A/B testing framework
   - Feature importance tracking
   - Bias detection and mitigation

2. **Automated Workflows**
   - Auto-retaining on drift
   - Automated model selection
   - Self-healing deployments

3. **Integration Enhancements**
   - Kubernetes operators
   - Stream processing integration
   - Multi-cloud deployment

4. **Observability**
   - Distributed tracing
   - Custom metrics collection
   - Performance profiling

## Support and Resources

- **Documentation**: See `/docs` directory
- **Examples**: See `/examples` directory
- **Issues**: GitHub Issues
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Flask Docs**: https://flask.palletsprojects.com/
