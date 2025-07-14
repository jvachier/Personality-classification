# Technical Guide - Six-Stack Personality Classification Pipeline

## Overview

This document provides a deep technical dive into the Six-Stack Personality Classification Pipeline, covering architecture decisions, algorithm implementations, and advanced features.

## Architecture Philosophy

### Modular Design Principles

The pipeline follows **SOLID principles** and **separation of concerns**:

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Easy to extend without modifying existing code
- **Dependency Inversion**: High-level modules don't depend on low-level details
- **Interface Segregation**: Clean, focused interfaces between modules

### Core Architecture Pattern

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│  Processing     │───▶│   Model Layer   │
│                 │    │     Layer       │    │                 │
│ • data_loader   │    │ • preprocessing │    │ • model_builders│
│ • external_data │    │ • augmentation  │    │ • optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Config Layer   │    │  Ensemble Layer │    │  Utils Layer    │
│                 │    │                 │    │                 │
│ • configuration │    │ • ensemble.py   │    │ • utils.py      │
│ • logging       │    │ • blending      │    │ • helpers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Algorithm Deep Dive

### Stack Specialization Strategy

Each stack is designed to capture different aspects of the data:

#### Stack A: Gradient Boosting Core (Narrow)

- **Purpose**: Stable baseline with conservative hyperparameters
- **Models**: XGBoost, LightGBM, CatBoost
- **Meta-learner**: Adaptive (Logistic Regression, Ridge, or XGBoost)
- **Search Space**: Conservative ranges (500-1000 estimators)

```python
# Example hyperparameter ranges for Stack A
xgb_params = {
    'n_estimators': (500, 1000),
    'learning_rate': (0.01, 0.25),
    'max_depth': (5, 12),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}
```

#### Stack B: Gradient Boosting Core (Wide)

- **Purpose**: Broader exploration of hyperparameter space
- **Models**: XGBoost, LightGBM, CatBoost
- **Meta-learner**: Adaptive (Logistic Regression, Ridge, or XGBoost)
- **Search Space**: Extended ranges (600-1200 estimators)

#### Stack C: Dual Boosting Specialists

- **Purpose**: Focus on XGBoost + CatBoost combination
- **Models**: XGBoost, CatBoost (specialized dual configuration)
- **Meta-learner**: Adaptive (Logistic Regression, Ridge, or XGBoost)
- **Features**: Advanced tree-specific parameters, categorical handling

#### Stack D: Sklearn Ensemble

- **Purpose**: Leverage sklearn's diverse algorithms
- **Models**: Random Forest, Extra Trees, Hist Gradient Boosting
- **Meta-learner**: Adaptive (Logistic, XGBoost, LightGBM, or Ridge)
- **Advantage**: Different algorithmic foundations with preprocessing

#### Stack E: Neural Networks & Classical ML

- **Purpose**: Capture non-linear patterns and classical methods
- **Models**: MLPClassifier (2 architectures), SVM, Gaussian Naive Bayes
- **Meta-learner**: Adaptive (Logistic Regression or Ridge)
- **Features**: Deep/wide neural networks, probability-enabled SVM

#### Stack F: Noise-Robust Training

- **Purpose**: Improve generalization through label noise
- **Models**: XGBoost, LightGBM, CatBoost (same as Stack A)
- **Meta-learner**: Logistic Regression (fixed for noise robustness)
- **Innovation**: Deliberate label noise injection (2% rate)

### Detailed Stack Composition

#### Model Distribution Summary

The pipeline uses a total of **6 stacks** with carefully selected algorithms:

| Stack | Base Models                                        | Meta-Learner Options                | Preprocessing            |
| ----- | -------------------------------------------------- | ----------------------------------- | ------------------------ |
| **A** | XGBoost, LightGBM, CatBoost                        | Logistic Regression, Ridge, XGBoost | None (uses raw features) |
| **B** | XGBoost, LightGBM, CatBoost                        | Logistic Regression, Ridge, XGBoost | None (uses raw features) |
| **C** | XGBoost, CatBoost                                  | Logistic Regression, Ridge, XGBoost | None (uses raw features) |
| **D** | Random Forest, Extra Trees, Hist Gradient Boosting | Logistic, XGBoost, LightGBM, Ridge  | RobustScaler             |
| **E** | MLP (Deep), MLP (Wide), SVM, Gaussian NB           | Logistic Regression, Ridge          | RobustScaler             |
| **F** | XGBoost, LightGBM, CatBoost + Label Noise          | Logistic Regression (fixed)         | None (uses raw features) |

#### Meta-Learner Implementation Details

**Important**: The "Ridge" meta-learner option is implemented as `LogisticRegression` with L2 penalty, not `Ridge` directly. This design choice ensures:

1. **Probability Support**: LogisticRegression naturally outputs calibrated probabilities
2. **Classification Compatibility**: Native support for binary classification
3. **Ridge Regularization**: L2 penalty provides Ridge-like regularization effects
4. **Ensemble Consistency**: All meta-learners output probabilities for consistent blending

```python
# Ridge meta-learner implementation
meta = LogisticRegression(
    C=1.0 / alpha,  # C = 1/alpha converts Ridge alpha to LogReg C
    penalty="l2",   # Ridge regularization
    solver="lbfgs", # Suitable for L2 penalty
    max_iter=2000
)
```

### Ensemble Strategy

#### Out-of-Fold (OOF) Prediction Generation

```python
def oof_probs(builder, X, y, X_test, sample_weights=None):
    """Generate unbiased out-of-fold predictions."""
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        # Train on fold data
        model = builder()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Predict on validation fold (unbiased)
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        # Accumulate test predictions
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof_preds, test_preds
```

#### Optimized Blending

The ensemble uses Optuna to find optimal weights:

```python
def improved_blend_obj(trial, *oof_predictions, y_true):
    """Objective function for blend optimization."""
    # Generate weights that sum to 1
    weights = []
    remaining = 1.0

    for i in range(len(oof_predictions) - 1):
        w = trial.suggest_float(f'weight_{i}', 0.0, remaining)
        weights.append(w)
        remaining -= w
    weights.append(remaining)

    # Weighted ensemble prediction
    ensemble_pred = sum(w * pred for w, pred in zip(weights, oof_predictions))

    # Optimize accuracy
    binary_pred = (ensemble_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, binary_pred)

    # Store weights for retrieval
    trial.set_user_attr('weights', weights)

    return accuracy
```

## Data Processing Pipeline

### External Data Integration

The pipeline implements the **advanced external data merge strategy**:

1. **Deduplication**: Remove duplicate rows from external dataset
2. **Feature Matching**: Match samples based on feature similarity
3. **Strategic Merging**: Add external features without label conflicts
4. **Validation**: Ensure merge quality and distribution preservation

### Advanced Preprocessing

#### Correlation-Based Imputation

```python
def correlation_imputation(df, target_col, n_corr=3):
    """Impute missing values using most correlated features."""
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    top_corr_features = correlations.iloc[1:n_corr+1].index.tolist()

    # Use top correlated features for imputation
    for feature in top_corr_features:
        if df[feature].notna().sum() > 0:
            median_val = df[feature].median()
            df[target_col].fillna(median_val, inplace=True)
            break
```

#### Smart Feature Engineering

- **One-hot encoding** for categorical variables
- **Robust scaling** for numerical stability
- **Feature interaction** discovery
- **Dimensionality** optimization

### Data Augmentation Deep Dive

#### Adaptive Strategy Selection

```python
def select_augmentation_method(data_characteristics):
    """Intelligent method selection based on data properties."""
    n_samples = data_characteristics['n_samples']
    class_balance = data_characteristics['class_balance_ratio']
    categorical_ratio = data_characteristics['categorical_ratio']

    if n_samples < 1000:
        return "smote"  # SMOTE for small datasets
    elif class_balance < 0.3:
        return "adasyn"  # ADASYN for severe imbalance
    elif categorical_ratio > 0.5:
        return "basic"  # Basic for high categorical
    else:
        return "sdv_copula"  # SDV for complex distributions
```

#### Quality Control Framework

```python
def enhanced_quality_filtering(synthetic_samples, original_samples):
    """Multi-dimensional quality assessment."""
    quality_scores = []

    for sample in synthetic_samples:
        # Feature distribution similarity
        distribution_score = calculate_distribution_similarity(sample, original_samples)

        # Correlation preservation
        correlation_score = calculate_correlation_preservation(sample, original_samples)

        # Anomaly detection
        anomaly_score = 1 - isolation_forest.decision_function([sample])[0]

        # Combined quality score
        quality = 0.4 * distribution_score + 0.3 * correlation_score + 0.3 * anomaly_score
        quality_scores.append(quality)

    return quality_scores
```

## Performance Optimization

### Threading Configuration

```python
class ThreadConfig(Enum):
    N_JOBS = 4          # sklearn parallel jobs
    THREAD_COUNT = 4    # XGBoost/LightGBM threads

    @classmethod
    def optimize_for_system(cls):
        """Auto-detect optimal threading."""
        cpu_count = multiprocessing.cpu_count()
        return min(cpu_count, 8)  # Cap at 8 for memory efficiency
```

### Memory Management

- **Lazy loading** of large datasets
- **Chunked processing** for memory efficiency
- **Garbage collection** at strategic points
- **Memory monitoring** and warnings

### Computational Efficiency

- **Early stopping** in hyperparameter optimization
- **Warm starting** with saved parameters
- **Incremental learning** where applicable
- **Parallel processing** optimization

## Error Handling & Robustness

### Graceful Degradation

```python
def robust_model_training(builder, X, y, max_retries=3):
    """Robust training with fallback strategies."""
    for attempt in range(max_retries):
        try:
            model = builder()
            model.fit(X, y)
            return model
        except MemoryError:
            # Reduce complexity and retry
            logger.warning(f"Memory error on attempt {attempt + 1}, reducing complexity")
            builder = create_simpler_builder(builder)
        except Exception as e:
            logger.error(f"Training failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise

    return None
```

### Timeout Protection

```python
@timeout_decorator(seconds=300)  # 5-minute timeout
def train_with_timeout(builder, X, y):
    """Training with automatic timeout."""
    return builder().fit(X, y)
```

## Monitoring & Logging

### Structured Logging

```python
def setup_structured_logging():
    """Configure comprehensive logging."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'personality_classifier.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler with colors
    console_handler = ColoredConsoleHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(LOG_LEVEL)
```

### Performance Metrics

- **Training time** per stack
- **Memory usage** monitoring
- **CPU utilization** tracking
- **Model size** reporting
- **Prediction latency** measurement

## Reproducibility

### Deterministic Behavior

```python
def ensure_reproducibility(seed=42):
    """Guarantee reproducible results."""
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Sklearn random state
    set_global_random_state(seed)

    # XGBoost determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Thread safety
    os.environ['OMP_NUM_THREADS'] = '1'
```

### Parameter Persistence

```python
def save_experiment_state(study, stack_name, metadata):
    """Save complete experiment state."""
    state = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'optimization_history': [trial.value for trial in study.trials],
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }

    with open(f'experiments/{stack_name}_state.json', 'w') as f:
        json.dump(state, f, indent=2)
```

## Extension Points

### Adding New Model Stacks

```python
def create_custom_stack(name, models, meta_learner='logistic'):
    """Template for creating new stacks."""
    def custom_objective(trial):
        # Define hyperparameter search space
        params = suggest_hyperparameters(trial, models)

        # Build ensemble
        ensemble = create_stacking_classifier(models, params, meta_learner)

        # Evaluate with cross-validation
        scores = cross_val_score(ensemble, X, y, cv=N_SPLITS, scoring='accuracy')

        return scores.mean()

    return custom_objective
```

### Custom Augmentation Methods

```python
def register_augmentation_method(name, method_class):
    """Register new augmentation strategies."""
    AUGMENTATION_REGISTRY[name] = method_class

    # Update configuration validation
    update_config_validation(name)

    logger.info(f"Registered new augmentation method: {name}")
```

## Dash Application Architecture

### Overview

The Dash application provides an interactive web interface for personality classification using the trained ensemble models. Built with Plotly Dash, it offers real-time predictions with confidence scores and probability visualizations.

### Application Structure

```
dash_app/
├── main.py              # Entry point and CLI argument handling
├── src/
│   ├── __init__.py
│   ├── app.py           # Core Dash application setup
│   ├── layout.py        # UI components and layout definition
│   ├── callbacks.py     # Interactive callbacks and prediction logic
│   └── model_loader.py  # Model management and prediction interface
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-container orchestration
└── .dockerignore        # Docker build exclusions
```

### Core Components

#### 1. Application Bootstrap (`main.py`)

```python
def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Personality Classification Dashboard")
    parser.add_argument("--model-name", required=True, help="Model name to load")
    parser.add_argument("--model-version", help="Specific model version")
    parser.add_argument("--model-stage", default="Production", help="Model stage")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Initialize and run application
    app = create_dash_app(args.model_name, args.model_version, args.model_stage)
    app.run_server(host=args.host, port=args.port, debug=args.debug)
```

#### 2. Model Loader (`model_loader.py`)

The model loader handles multiple model sources and provides a unified prediction interface:

```python
class ModelLoader:
    """Handles loading and managing ML models from various sources."""

    def __init__(self, model_name: str, model_version: str | None = None,
                 model_stage: str = "Production"):
        self.model_name = model_name
        self.model = None
        self.model_metadata = {}
        self._load_model()

    def _load_model(self):
        """Load model with fallback strategies."""
        # Priority order:
        # 1. Local models directory (ensemble_model.pkl, stack_X_model.pkl)
        # 2. Best params directory (saved optimization results)
        # 3. Dummy model for demonstration

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make prediction with metadata-driven label mapping."""
        # Feature engineering and validation
        # Prediction with confidence scores
        # Label mapping using metadata
        return {
            "prediction": personality_type,
            "confidence": confidence,
            "probability_extrovert": prob_extrovert,
            "probability_introvert": prob_introvert,
            "model_name": self.model_name
        }
```

**Key Features:**
- **Multi-source loading**: Supports ensemble and individual stack models
- **Metadata-driven mapping**: Uses saved label mapping for consistent predictions
- **Feature validation**: Ensures proper feature ordering and defaults
- **Graceful fallbacks**: Creates dummy model if no trained model available

#### 3. User Interface (`layout.py`)

The UI is designed with a modern, responsive layout using Dash Bootstrap Components:

```python
def create_personality_input_form():
    """Create the main personality assessment form."""
    return dbc.Card([
        dbc.CardHeader("Personality Assessment"),
        dbc.CardBody([
            # Time spent alone (0-10 scale)
            create_slider_input("time-spent-alone", "Time Spent Alone", 0, 10, 5),

            # Social event attendance (0-10 scale)
            create_slider_input("social-event-attendance", "Social Event Attendance", 0, 10, 5),

            # Categorical inputs with dropdowns
            create_dropdown_input("stage-fear", "Stage Fear", ["No", "Yes", "Unknown"]),
            create_dropdown_input("drained-after-socializing", "Drained After Socializing",
                                ["No", "Yes", "Unknown"]),

            # Prediction button and results
            dbc.Button("Predict Personality", id="predict-button", color="primary"),
            html.Div(id="prediction-results")
        ])
    ])
```

**UI Features:**
- **Interactive sliders**: For numerical personality traits
- **Dropdown selectors**: For categorical responses
- **Real-time validation**: Input validation and feedback
- **Responsive design**: Mobile-friendly layout
- **Accessibility**: ARIA labels and keyboard navigation

#### 4. Prediction Display (`layout.py`)

```python
def format_prediction_result(result: dict[str, Any]) -> html.Div:
    """Format prediction result with visual enhancements."""
    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0)
    prob_extrovert = result.get("probability_extrovert", 0)
    prob_introvert = result.get("probability_introvert", 0)

    # Dynamic styling based on prediction
    personality_color = "#e74c3c" if prediction == "Extrovert" else "#3498db"
    confidence_color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"

    return html.Div([
        # Main prediction with personality-specific styling
        html.H2(f"🧠 You are classified as: {prediction}",
                style={"color": personality_color, "textAlign": "center"}),

        # Confidence score with color coding
        html.P(f"Confidence Score: {confidence:.1%}",
               style={"color": confidence_color, "textAlign": "center"}),

        # Probability bars for both classes
        create_probability_bars(prob_extrovert, prob_introvert),

        # Personality description
        create_personality_description(prediction)
    ])
```

#### 5. Interactive Callbacks (`callbacks.py`)

The callback system handles user interactions and real-time predictions:

```python
@app.callback(
    Output("prediction-results", "children"),
    Input("predict-button", "n_clicks"),
    [State("time-spent-alone", "value"),
     State("social-event-attendance", "value"),
     State("going-outside", "value"),
     State("friends-circle-size", "value"),
     State("post-frequency", "value"),
     State("stage-fear", "value"),
     State("drained-after-socializing", "value")],
    prevent_initial_call=True
)
def make_prediction(n_clicks, time_alone, social_events, going_outside,
                   friends_size, post_freq, stage_fear, drained_social):
    """Handle prediction requests with comprehensive feature engineering."""
    if not n_clicks:
        return ""

    try:
        # Build feature dictionary with proper encoding
        data = {
            "Time_spent_Alone": time_alone if time_alone is not None else 2.0,
            "Social_event_attendance": social_events if social_events is not None else 4.0,
            "Going_outside": going_outside if going_outside is not None else 3.0,
            "Friends_circle_size": friends_size if friends_size is not None else 8.0,
            "Post_frequency": post_freq if post_freq is not None else 3.0,

            # One-hot encode categorical features
            "Stage_fear_No": 1 if stage_fear == "No" else 0,
            "Stage_fear_Unknown": 1 if stage_fear == "Unknown" else 0,
            "Stage_fear_Yes": 1 if stage_fear == "Yes" else 0,

            "Drained_after_socializing_No": 1 if drained_social == "No" else 0,
            "Drained_after_socializing_Unknown": 1 if drained_social == "Unknown" else 0,
            "Drained_after_socializing_Yes": 1 if drained_social == "Yes" else 0,

            # External match features (set to Unknown as default)
            "match_p_Extrovert": 0,
            "match_p_Introvert": 0,
            "match_p_Unknown": 1
        }

        # Make prediction and format results
        result = model_loader.predict(data)
        return format_prediction_result(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return html.Div(f"Error: {e!s}", style={"color": "red"})
```

### Prediction History & Monitoring

```python
@app.callback(
    Output("prediction-history", "children"),
    [Input("interval-component", "n_intervals"),
     Input("predict-button", "n_clicks")]
)
def update_prediction_history(n_intervals, n_clicks):
    """Update prediction history display with recent predictions."""
    if not prediction_history:
        return html.Div("No predictions yet", style={"color": "#7f8c8d"})

    # Create interactive history table
    table_data = []
    for i, pred in enumerate(reversed(prediction_history[-10:])):
        table_data.append({
            "ID": f"#{len(prediction_history) - i}",
            "Timestamp": pred["timestamp"][:19],
            "Prediction": pred["result"].get("prediction", "N/A"),
            "Confidence": f"{pred['result'].get('confidence', 0):.3f}"
        })

    return dash_table.DataTable(
        data=table_data,
        columns=[
            {"name": "ID", "id": "ID"},
            {"name": "Timestamp", "id": "Timestamp"},
            {"name": "Prediction", "id": "Prediction"},
            {"name": "Confidence", "id": "Confidence"}
        ],
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "#3498db", "color": "white"}
    )
```

### Deployment Options

#### 1. Local Development

```bash
# Start with default ensemble model
make dash

# Start with specific model
uv run python dash_app/main.py --model-name ensemble --debug

# Start with stack model
uv run python dash_app/main.py --model-name A --port 8051
```

#### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8050

CMD ["python", "dash_app/main.py", "--model-name", "ensemble", "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  personality-dashboard:
    build: .
    ports:
      - "8050:8050"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    environment:
      - MODEL_NAME=ensemble
      - DEBUG=false
    restart: unless-stopped
```

#### 3. Production Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale for high availability
docker-compose up --scale personality-dashboard=3 -d

# Behind reverse proxy (nginx/traefik)
# Configure load balancing and SSL termination
```

### Performance Optimization

#### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_prediction(feature_hash: str):
    """Cache predictions for identical feature combinations."""
    # Convert hash back to features and predict
    # Useful for repeated identical inputs
    pass

# Memory-efficient model loading
class LazyModelLoader:
    """Load models only when needed."""
    def __init__(self):
        self._model = None
        self._model_path = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

#### Resource Management

```python
# Graceful shutdown handling
import signal
import sys

def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    logger.info("Shutting down Dash application...")
    # Cleanup resources
    if hasattr(app, 'cleanup'):
        app.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### Security Considerations

#### Input Validation

```python
def validate_input_data(data: dict) -> dict:
    """Comprehensive input validation and sanitization."""
    validated = {}

    # Numerical range validation
    for key, value in data.items():
        if key in NUMERICAL_FEATURES:
            validated[key] = max(0, min(10, float(value)))  # Clamp to valid range
        elif key in CATEGORICAL_FEATURES:
            validated[key] = value if value in VALID_CATEGORIES[key] else "Unknown"
        else:
            validated[key] = value

    return validated
```

#### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app.server,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

@limiter.limit("5 per minute")
def prediction_endpoint():
    """Rate-limited prediction endpoint."""
    pass
```

### Monitoring & Analytics

#### Application Metrics

```python
import time
from collections import defaultdict

class DashboardMetrics:
    """Track application performance and usage."""

    def __init__(self):
        self.prediction_count = 0
        self.prediction_times = []
        self.error_count = 0
        self.user_sessions = defaultdict(int)

    def record_prediction(self, duration: float, user_id: str = None):
        """Record prediction metrics."""
        self.prediction_count += 1
        self.prediction_times.append(duration)
        if user_id:
            self.user_sessions[user_id] += 1

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "total_predictions": self.prediction_count,
            "avg_prediction_time": sum(self.prediction_times) / len(self.prediction_times),
            "error_rate": self.error_count / max(1, self.prediction_count),
            "active_sessions": len(self.user_sessions)
        }
```

### Testing Strategy

#### Unit Tests

```python
# tests/dash_app/test_model_loader.py
def test_model_loader_prediction():
    """Test model loader prediction functionality."""
    loader = ModelLoader("ensemble")

    test_data = {
        "Time_spent_Alone": 5.0,
        "Social_event_attendance": 7.0,
        "Stage_fear": "No",
        "Drained_after_socializing": "Yes"
    }

    result = loader.predict(test_data)

    assert "prediction" in result
    assert result["prediction"] in ["Extrovert", "Introvert"]
    assert 0 <= result["confidence"] <= 1
```

#### Integration Tests

```python
# tests/dash_app/test_app_integration.py
def test_full_prediction_workflow():
    """Test complete prediction workflow."""
    from dash.testing.application_runners import import_app

    app = import_app("dash_app.main")
    dash_duo.start_server(app)

    # Simulate user input
    dash_duo.find_element("#time-spent-alone").send_keys("5")
    dash_duo.find_element("#predict-button").click()

    # Verify prediction result
    dash_duo.wait_for_element("#prediction-results", timeout=10)
    result_text = dash_duo.find_element("#prediction-results").text
    assert "You are classified as:" in result_text
```

### Usage Guidelines

#### Starting the Application

```bash
# Method 1: Using Makefile (recommended)
make dash

# Method 2: Direct Python execution
uv run python dash_app/main.py --model-name ensemble

# Method 3: Docker deployment
docker-compose up -d

# Stop the application
make stop-dash
# or
Ctrl+C (for local development)
```

#### Model Selection

- **ensemble**: Recommended for production use (balanced performance)
- **A-F**: Individual stack models for specialized analysis
- **Auto-detection**: Falls back to dummy model if no trained model available

#### Interpreting Results

- **Prediction**: Primary personality classification (Extrovert/Introvert)
- **Confidence**: Model certainty (0-100%, higher is better)
- **Probabilities**: Individual class probabilities (sum to 100%)
- **Personality Description**: Detailed trait explanations

### Future Enhancements

### Planned Features

- **AutoML integration** for automatic architecture search
- **Distributed training** support
- **Model interpretability** tools
- **A/B testing** framework
- **Real-time inference** API
- **Model versioning** system

### Research Directions

- **Meta-learning** for stack selection
- **Neural architecture search** for Stack E
- **Federated learning** capabilities
- **Continual learning** for model updates
- **Uncertainty quantification** methods

---

## Document Revision Notes

**Last Updated**: July 14, 2025

### Recent Updates

- **Dash Application Documentation**: Added comprehensive documentation for the interactive web dashboard
  - Application architecture and component structure
  - Model loader with metadata-driven predictions
  - User interface design and responsive layout
  - Interactive callbacks and real-time predictions
  - Deployment options (local, Docker, production)
  - Performance optimization and caching strategies
  - Security considerations and input validation
  - Monitoring, testing, and usage guidelines

### Previous Corrections

- **Stack A/B Models**: Corrected from "Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost" to "XGBoost, LightGBM, CatBoost"
- **Stack C Models**: Clarified as "XGBoost, CatBoost" (dual boosting specialists)
- **Stack D Models**: Confirmed as "Random Forest, Extra Trees, Hist Gradient Boosting"
- **Stack E Models**: Clarified as "MLPClassifier (2 architectures), SVM, Gaussian NB"
- **Meta-learners**: Updated to show adaptive selection (Logistic, Ridge, XGBoost) for most stacks
- **Ridge Implementation**: Added technical note explaining Ridge is implemented as LogisticRegression with L2 penalty
- **Parameter Ranges**: Updated Stack A (500-1000) and Stack B (600-1200) estimator ranges
- **Stack Summary Table**: Added comprehensive table showing exact model compositions

All descriptions now accurately reflect the actual code implementation in `src/modules/model_builders.py` and `dash_app/` directory.
