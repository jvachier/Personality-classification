# Documentation Index

Welcome! This documentation covers all aspects of the Six-Stack Personality Classification Pipeline.

## Main Guides

- [Technical Guide](technical-guide.md): Architecture, algorithms, and stacks
- [API Reference](api-reference.md): Modules, functions, and usage
- [Configuration Guide](configuration.md): All config options
- [Performance Tuning](performance-tuning.md): Speed, memory, accuracy
- [Data Augmentation](data-augmentation.md): Synthetic data strategies
- [Deployment Guide](deployment.md): Docker, Compose, production

## Quick Start

1. See [README](../README.md) for setup
2. Try examples in `examples/`
3. Read [Configuration Guide](configuration.md) for customization
4. Explore [Technical Guide](technical-guide.md) for details

## Quick Reference

**Config:**
```python
TESTING_MODE = True  # Fast dev
N_TRIALS_STACK = 5
ENABLE_DATA_AUGMENTATION = False
```
**Production:**
```python
TESTING_MODE = False
N_TRIALS_STACK = 100
AUGMENTATION_METHOD = "sdv_copula"
```
**Docker:**
```bash
docker-compose up --build
# or
docker build -t personality-classifier .
docker run -p 8080:8080 personality-classifier
```

## ️ Resources

- Code: `src/main_modular.py`, `examples/`
- Config templates: [Configuration Guide](configuration.md)
- Monitoring: [Performance Tuning](performance-tuning.md)
- Deployment: [Deployment Guide](deployment.md)

## Latest Features

- Advanced SDV Copula augmentation
- Centralized config system
- Modular architecture
- Dockerized deployment
- Comprehensive documentation

## Help & Contributing

- Review guides and examples
- Search or create issues in the repo
- See [README](../README.md#contributing) for contribution steps
