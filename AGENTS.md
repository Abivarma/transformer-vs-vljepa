# Agents & CI/CD Automation
## Complete Automation Setup Guide

**Last Updated**: 2026-01-31

---

## Overview

This document covers all automation agents used in the project:
- Development automation (pre-commit hooks)
- CI/CD pipelines (GitHub Actions)
- Testing automation (pytest)
- Deployment automation (Docker, Kubernetes)
- Monitoring agents (Prometheus, Grafana)

---

## Development Agents

### Pre-commit Hooks

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Setup**:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Test
```

---

## CI/CD Pipelines

### GitHub Actions: Test Pipeline

**File**: `.github/workflows/test.yml`

```yaml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .

    - name: Lint with flake8
      run: |
        flake8 src/ tests/

    - name: Type check with mypy
      run: |
        mypy src/

    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### GitHub Actions: Build & Deploy

**File**: `.github/workflows/deploy.yml`

```yaml
name: Build and Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          yourusername/transformer-vljepa:latest
          yourusername/transformer-vljepa:${{ github.ref_name }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Your deployment commands
        echo "Deploying version ${{ github.ref_name }}"
```

---

## Testing Automation

### pytest Configuration

**File**: `pyproject.toml` (testing section)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = """
    --cov=src
    --cov-report=html
    --cov-report=term
    --cov-report=xml
    --strict-markers
    -v
"""
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

### Automated Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run only fast tests
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

---

## Deployment Automation

### Docker Multi-Stage Build

**File**: `Dockerfile`

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@db:5432/transformerdb
    depends_on:
      - redis
      - db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: transformerdb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Monitoring Agents

### Prometheus Configuration

**File**: `07-deployment/monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'transformer-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Create dashboards for:
- Request rate
- Latency (p50, p95, p99)
- Error rate
- Model inference time
- GPU/CPU usage

---

## Continuous Integration Best Practices

1. **Fast Feedback**: Tests should complete in < 5 minutes
2. **Parallel Execution**: Run tests in parallel
3. **Caching**: Cache dependencies
4. **Matrix Testing**: Test multiple Python versions
5. **Code Coverage**: Require 80%+ coverage

---

## Deployment Strategies

### Blue-Green Deployment
```bash
# Deploy new version (green)
kubectl apply -f k8s/deployment-green.yaml

# Test green deployment
curl https://green.your-app.com/health

# Switch traffic
kubectl patch service app-service -p '{"spec":{"selector":{"version":"green"}}}'
```

### Canary Deployment
```yaml
# Gradually shift traffic: 10% → 50% → 100%
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: transformer-vljepa
  # Traffic split handled by ingress controller
```

---

## Monitoring & Alerts

### Key Metrics to Monitor
- **Request Rate**: requests/second
- **Latency**: p50, p95, p99
- **Error Rate**: 4xx, 5xx errors
- **Model Performance**: inference time
- **Resources**: CPU, memory, GPU usage

### Alert Rules
```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "95th percentile latency > 1s"
```

---

## Security Automation

### Dependabot Configuration

**File**: `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

### Security Scanning
```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

---

For complete automation setup, see SPRINT_STORIES.md Phase 0.

Last Updated: 2026-01-31
