# Testing Framework for Personality Classification Pipeline

This comprehensive testing framework covers all components of the personality classification pipeline including data processing, model building, and MLOps infrastructure.

## 📁 Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest configuration and fixtures
├── modules/                       # Tests for core modules
│   ├── __init__.py
│   ├── test_data_loader.py       # Data loading and processing tests
│   └── test_model_builders.py    # Model building tests
├── dash_app/                      # Tests for Dashboard components
│   ├── __init__.py
│   └── test_dash_app.py          # Dash application tests
└── fixtures/                      # Test data and fixtures
```

## 🧪 Test Categories

### **Unit Tests** (`@pytest.mark.unit`)
- Test individual functions and classes in isolation
- Fast execution, no external dependencies
- Mock external services and file systems

### **Integration Tests** (`@pytest.mark.integration`)
- Test interactions between components
- Test end-to-end workflows
- May use real file systems and external services

### **Slow Tests** (`@pytest.mark.slow`)
- Tests involving hyperparameter tuning
- Large dataset processing
- Model training with multiple iterations

## 🚀 Running Tests

### **Quick Start**
```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --type all

# Run only fast tests (exclude slow tests)
python run_tests.py --type fast
```

### **Test Categories**
```bash
# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run module tests
python run_tests.py --type modules

# Run MLOps tests
python run_tests.py --type mlops
```

### **Specific Test Execution**
```bash
# Run specific test file
python run_tests.py --test tests/modules/test_data_loader.py

# Run specific test class
python run_tests.py --test tests/modules/test_data_loader.py::TestDataLoader

# Run specific test method
python run_tests.py --test tests/modules/test_data_loader.py::TestDataLoader::test_init
```

### **Coverage Options**
```bash
# Run without coverage (faster)
python run_tests.py --no-coverage

# Run with verbose output
python run_tests.py --verbose
```

### **Direct Pytest Usage**
```bash
# Run with pytest directly
pytest tests/

# Run with specific markers
pytest -m "unit and not slow" tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/
```

## 🔧 Configuration

### **pytest.ini**
```ini
[tool.pytest.ini_options]
minversion = "6.0"
addopts = """
    -ra
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
"""
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
    "mlops: MLOps related tests",
]
```

## 🎯 Test Fixtures

The test suite includes comprehensive fixtures for different testing scenarios:

### **Data Fixtures**
- `sample_data`: Synthetic personality classification dataset
- `sample_features`: Feature data without target variable
- `sample_model`: Pre-trained RandomForest model

### **Environment Fixtures**
- `temp_dir`: Temporary directory for test files
- `config_dict`: Sample configuration dictionary

### **Custom Assertions**
- `assert_model_performance()`: Validate model accuracy
- `assert_data_shape()`: Check DataFrame dimensions
- `assert_no_missing_values()`: Verify data quality

## 📊 Coverage Reports

Test coverage reports are generated in multiple formats:

### **Terminal Output**
```bash
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/modules/__init__.py           0      0   100%
src/modules/data_loader.py       45      2    96%   23, 67
src/modules/model_builders.py    78      5    94%   45-49
src/mlops/__init__.py             0      0   100%
src/mlops/experiment_tracking.py 92      8    91%   134-142
-----------------------------------------------------------
TOTAL                           215     15    93%
```

### **HTML Report**
Open `htmlcov/index.html` in your browser for detailed coverage analysis.

### **XML Report**
`coverage.xml` for CI/CD integration.

## 🔍 Test Examples

### **Data Processing Tests**
```python
def test_handle_missing_values_drop(self, sample_data):
    """Test dropping missing values."""
    # Introduce missing values
    data_with_missing = sample_data.copy()
    data_with_missing.iloc[0, 0] = None

    processor = DataProcessor()
    cleaned_data = processor.handle_missing_values(data_with_missing, strategy="drop")

    assert len(cleaned_data) == len(sample_data) - 1
    assert not cleaned_data.isnull().any().any()
```

### **Model Building Tests**
```python
def test_train_model(self, sample_data):
    """Test training a model."""
    builder = ModelBuilder()
    model = builder.create_random_forest(n_estimators=10, random_state=42)

    X = sample_data.drop("personality_type", axis=1)
    y = sample_data["personality_type"]

    trained_model = builder.train_model(model, X, y)
    assert hasattr(trained_model, "predict")
```

## 🐛 Debugging Tests

### **Running in Debug Mode**
```bash
# Run with verbose output and show local variables
pytest -vvv --tb=long tests/

# Run specific test with debugging
pytest -s tests/modules/test_data_loader.py::TestDataLoader::test_init
```

### **Using Print Statements**
```python
def test_debug_example(self, sample_data):
    print(f"Data shape: {sample_data.shape}")
    print(f"Columns: {sample_data.columns.tolist()}")
    # ... test logic
```

### **Using Debugger**
```python
def test_with_debugger(self, sample_data):
    import pdb; pdb.set_trace()
    # ... test logic
```

## 🔄 Continuous Integration

### **GitHub Actions Example**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: python run_tests.py --type all

    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📝 Best Practices

### **Writing Tests**
1. **Descriptive Names**: Use clear, descriptive test names
2. **Single Responsibility**: Each test should verify one specific behavior
3. **Independent Tests**: Tests should not depend on each other
4. **Use Fixtures**: Leverage pytest fixtures for setup and teardown
5. **Mock External Dependencies**: Use mocks for external services

### **Test Organization**
1. **Group Related Tests**: Use test classes to group related functionality
2. **Use Markers**: Tag tests appropriately for selective execution
3. **Parametrize Tests**: Use `@pytest.mark.parametrize` for multiple scenarios
4. **Document Complex Tests**: Add docstrings explaining test purpose

### **Performance**
1. **Fast Unit Tests**: Keep unit tests fast and focused
2. **Mark Slow Tests**: Use `@pytest.mark.slow` for time-consuming tests
3. **Use Smaller Datasets**: Create minimal datasets for testing
4. **Parallel Execution**: Consider pytest-xdist for parallel test execution

## 🛠️ Dependencies

The testing framework requires:

```bash
# Core testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Data science testing
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Dashboard testing  
dash>=2.14.0

# Optional: for parallel execution
pytest-xdist>=3.0.0
```

## 📈 Metrics and Reporting

### **Test Metrics**
- **Test Count**: Total number of tests
- **Pass Rate**: Percentage of passing tests
- **Coverage**: Code coverage percentage
- **Execution Time**: Test suite runtime

### **Quality Gates**
- Minimum 90% code coverage
- All tests must pass
- No critical security vulnerabilities
- Performance benchmarks met

---

## 🚀 Quick Commands Reference

```bash
# Essential commands
python run_tests.py                    # Run all tests
python run_tests.py --type fast        # Fast tests only
python run_tests.py --type modules     # Module tests
python run_tests.py --type mlops       # MLOps tests
python run_tests.py --no-coverage      # Skip coverage
python run_tests.py --verbose          # Verbose output

# Pytest direct usage
pytest tests/                          # All tests
pytest -m "not slow"                   # Exclude slow tests
pytest --cov=src tests/                # With coverage
pytest -x tests/                       # Stop on first failure
```

This testing framework ensures code quality, reliability, and maintainability of the personality classification pipeline through comprehensive test coverage of all components.
