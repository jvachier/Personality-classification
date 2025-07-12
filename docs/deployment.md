# Deployment Guide

## Overview

This guide covers deploying the Six-Stack Personality Classification Pipeline using modern containerization and orchestration technologies. The focus is on **Docker containerization** and **Kubernetes orchestration** with a **Dash web application** for interactive model serving.

## Deployment Strategy

### Core Technologies
- **🐳 Docker**: Containerization for consistent environments
- **☸️ Kubernetes**: Container orchestration and scaling
- **📊 Dash**: Interactive web application for model inference
- **📈 Monitoring**: Prometheus and Grafana integration

### Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dash Web App  │───▶│  ML Pipeline    │───▶│   Data Store    │
│   (Port 8050)   │    │  (Containers)   │    │   (Volumes)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Kubernetes    │    │   Monitoring    │
│   (Ingress)     │    │   Cluster       │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🐳 Docker Deployment

### Prerequisites
```bash
# System requirements
- Docker 20.10+ 
- Docker Compose 2.0+
- 8GB+ RAM available for containers
- 4+ CPU cores
- 20GB+ disk space

# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin
```

### Dockerfile
```dockerfile
# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash personality

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ src/
COPY data/ data/
COPY examples/ examples/

# Set ownership
RUN chown -R personality:personality /app

# Switch to non-root user
USER personality

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.modules.config; print('OK')" || exit 1

# Default command
CMD ["python", "src/main_modular.py"]
```

### Pipeline Dockerfile
```dockerfile
# Dockerfile.pipeline - ML Training Pipeline
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash personality

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ src/
COPY data/ data/

# Create model artifacts directory
RUN mkdir -p models best_params submissions logs

# Set ownership
RUN chown -R personality:personality /app

# Switch to non-root user
USER personality

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.modules.config; print('OK')" || exit 1

# Default command
CMD ["python", "src/main_modular.py"]
```

### Dash Application Dockerfile
```dockerfile
# Dockerfile.dash - Interactive Dash Application
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files (include dash dependencies)
COPY pyproject.toml uv.lock ./
COPY requirements-dash.txt ./

# Install dependencies
RUN uv sync --no-dev --frozen
RUN uv pip install -r requirements-dash.txt

# Copy application code
COPY src/ src/
COPY dash_app/ dash_app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash dashuser
RUN chown -R dashuser:dashuser /app

# Switch to non-root user
USER dashuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Expose Dash port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Run Dash application
CMD ["python", "dash_app/app.py"]
```

### Dash Requirements
```txt
# requirements-dash.txt
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.17.0
pandas>=2.1.0
numpy>=1.24.0
gunicorn>=21.2.0
```

### Multi-Service Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  # ML Pipeline Service
  ml-pipeline:
    build: 
      context: .
      dockerfile: Dockerfile.pipeline
    container_name: personality-ml-pipeline
    restart: unless-stopped
    
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 2G
          cpus: '1'
    
    environment:
      - PERSONALITY_LOG_LEVEL=INFO
      - PERSONALITY_TESTING_MODE=false
      - RUNNING_IN_DOCKER=true
    
    volumes:
      - ./data:/app/data:ro
      - ./best_params:/app/best_params
      - ./submissions:/app/submissions
      - ./logs:/app/logs
      - model-artifacts:/app/models
    
    networks:
      - personality-net

  # Dash Web Application
  dash-app:
    build:
      context: .
      dockerfile: Dockerfile.dash
    container_name: personality-dash-app
    restart: unless-stopped
    ports:
      - "8050:8050"
    
    depends_on:
      - ml-pipeline
    
    environment:
      - DASH_HOST=0.0.0.0
      - DASH_PORT=8050
      - MODEL_PATH=/app/models
    
    volumes:
      - model-artifacts:/app/models:ro
      - ./data:/app/data:ro
    
    networks:
      - personality-net

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - personality-net

  # Visualization with Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - personality-net

volumes:
  model-artifacts:
  prometheus-data:
  grafana-storage:

networks:
  personality-net:
    driver: bridge
```

### Build and Deploy with Docker Compose
```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f dash-app
docker-compose logs -f ml-pipeline

# Scale pipeline instances
docker-compose up --scale ml-pipeline=3 -d

# Stop all services
docker-compose down

# Clean up (removes containers, networks, and volumes)
docker-compose down -v
```

## ☸️ Kubernetes Deployment

### ML Pipeline Deployment
```yaml
# k8s/ml-pipeline-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline
  labels:
    app: ml-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-pipeline
  template:
    metadata:
      labels:
        app: ml-pipeline
    spec:
      containers:
      - name: ml-pipeline
        image: personality-ml-pipeline:latest
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        
        env:
        - name: PERSONALITY_LOG_LEVEL
          value: "INFO"
        - name: RUNNING_IN_KUBERNETES
          value: "true"
        
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
          readOnly: true
        - name: model-artifacts
          mountPath: /app/models
        - name: logs-volume
          mountPath: /app/logs
        
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import src.modules.config; print('OK')"
          initialDelaySeconds: 60
          periodSeconds: 30
        
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import src.modules.config; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 10
      
      volumes:
      - name: data-volume
        configMap:
          name: training-data
      - name: model-artifacts
        persistentVolumeClaim:
          claimName: model-artifacts-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc

---
### Dash Application Deployment
```yaml
# k8s/dash-app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dash-app
  labels:
    app: dash-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dash-app
  template:
    metadata:
      labels:
        app: dash-app
    spec:
      containers:
      - name: dash-app
        image: personality-dash-app:latest
        ports:
        - containerPort: 8050
        
        resources:
          requests:
            memory: "1Gi"
            cpu: "200m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        
        env:
        - name: DASH_HOST
          value: "0.0.0.0"
        - name: DASH_PORT
          value: "8050"
        - name: MODEL_PATH
          value: "/app/models"
        
        volumeMounts:
        - name: model-artifacts
          mountPath: /app/models
          readOnly: true
        - name: data-volume
          mountPath: /app/data
          readOnly: true
        
        livenessProbe:
          httpGet:
            path: /
            port: 8050
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /
            port: 8050
          initialDelaySeconds: 15
          periodSeconds: 5
      
      volumes:
      - name: model-artifacts
        persistentVolumeClaim:
          claimName: model-artifacts-pvc
      - name: data-volume
        configMap:
          name: training-data
```

### Services and Ingress
```yaml
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: dash-app-service
spec:
  selector:
    app: dash-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8050
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: ml-pipeline-service
spec:
  selector:
    app: ml-pipeline
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: personality-classifier-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - personality.yourdomain.com
    secretName: personality-tls
  rules:
  - host: personality.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dash-app-service
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ml-pipeline-service
            port:
              number: 80
```

### Persistent Storage
```yaml
# k8s/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-artifacts-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-data
data:
  # Add your training data files here
  # or mount from external storage
```

### Deploy to Kubernetes
```bash
# Build and push images to registry
docker build -f Dockerfile.pipeline -t your-registry/personality-ml-pipeline:latest .
docker build -f Dockerfile.dash -t your-registry/personality-dash-app:latest .

docker push your-registry/personality-ml-pipeline:latest
docker push your-registry/personality-dash-app:latest

# Create namespace
kubectl create namespace personality-classifier

# Apply storage resources
kubectl apply -f k8s/storage.yaml -n personality-classifier

# Apply deployments
kubectl apply -f k8s/ml-pipeline-deployment.yaml -n personality-classifier
kubectl apply -f k8s/dash-app-deployment.yaml -n personality-classifier

# Apply services and ingress
kubectl apply -f k8s/services.yaml -n personality-classifier
kubectl apply -f k8s/ingress.yaml -n personality-classifier

# Check deployment status
kubectl get all -n personality-classifier
kubectl get pvc -n personality-classifier

# View logs
kubectl logs -f deployment/ml-pipeline -n personality-classifier
kubectl logs -f deployment/dash-app -n personality-classifier

# Scale deployments
kubectl scale deployment dash-app --replicas=5 -n personality-classifier
kubectl scale deployment ml-pipeline --replicas=3 -n personality-classifier

# Port forward for local access (development)
kubectl port-forward service/dash-app-service 8050:80 -n personality-classifier
```

## 🔧 Production Best Practices

### Security Considerations
```bash
# Use secrets for sensitive configuration
kubectl create secret generic model-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=db-password=your-password \
  -n personality-classifier

# Apply security contexts
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  capabilities:
    drop:
    - ALL
```

### Backup Strategy
```bash
#!/bin/bash
# backup.sh - Automated backup script

# Backup model artifacts
kubectl exec deployment/ml-pipeline -n personality-classifier -- \
  tar -czf /tmp/models-backup-$(date +%Y%m%d).tar.gz /app/models

# Copy to persistent storage
kubectl cp personality-classifier/ml-pipeline-pod:/tmp/models-backup-$(date +%Y%m%d).tar.gz \
  ./backups/models-backup-$(date +%Y%m%d).tar.gz

# Upload to cloud storage (optional)
aws s3 cp ./backups/models-backup-$(date +%Y%m%d).tar.gz \
  s3://your-backup-bucket/models/

# Rotate old backups (keep last 30 days)
find ./backups -name "models-backup-*.tar.gz" -mtime +30 -delete
```

### Health Checks and Monitoring
```yaml
# k8s/monitoring.yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
```

## 🚀 Quick Deployment Commands

### Local Development
```bash
# Quick start with Docker Compose
git clone <repository-url>
cd Personality-classification

# Build and start all services
docker-compose up --build -d

# Access Dash application
open http://localhost:8050

# View logs
docker-compose logs -f dash-app
```

### Production Deployment
```bash
# Build and push images
docker build -f Dockerfile.pipeline -t your-registry/ml-pipeline:v1.0 .
docker build -f Dockerfile.dash -t your-registry/dash-app:v1.0 .
docker push your-registry/ml-pipeline:v1.0
docker push your-registry/dash-app:v1.0

# Deploy to Kubernetes
kubectl create namespace personality-classifier
kubectl apply -f k8s/ -n personality-classifier

# Verify deployment
kubectl get all -n personality-classifier
kubectl logs -f deployment/dash-app -n personality-classifier
```

## 📋 Troubleshooting

### Common Issues

#### Container Memory Issues
```bash
# Check memory usage
kubectl top pods -n personality-classifier

# Increase memory limits in deployment
resources:
  limits:
    memory: "16Gi"  # Increase from 8Gi
```

#### Model Loading Problems
```bash
# Check persistent volumes
kubectl get pvc -n personality-classifier

# Verify model artifacts
kubectl exec -it deployment/ml-pipeline -n personality-classifier -- ls -la /app/models
```

#### Dash Application Not Starting
```bash
# Check logs
kubectl logs deployment/dash-app -n personality-classifier

# Test local connectivity
kubectl port-forward service/dash-app-service 8050:80 -n personality-classifier
```

#### Network Connectivity Issues
```bash
# Test service connectivity
kubectl exec -it deployment/dash-app -n personality-classifier -- \
  curl http://ml-pipeline-service

# Check ingress status
kubectl get ingress -n personality-classifier
kubectl describe ingress personality-classifier-ingress -n personality-classifier
```

---

## 📚 Additional Resources

- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)
- **Kubernetes Documentation**: [kubernetes.io/docs](https://kubernetes.io/docs)
- **Dash Documentation**: [dash.plotly.com](https://dash.plotly.com)
- **Prometheus Monitoring**: [prometheus.io/docs](https://prometheus.io/docs)

---

*This deployment guide focuses on containerized deployment with Docker and Kubernetes orchestration. For specific platform requirements or custom deployments, consult the platform documentation or create an issue in the repository.*
