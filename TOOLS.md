# Tools & Setup Guide
## Complete Reference for Transformer vs VL-JEPA Project

**Last Updated**: 2026-01-31
**Platform**: macOS (with Linux/Windows notes where different)

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Development Tools](#development-tools)
3. [Core Libraries](#core-libraries)
4. [Visualization Tools](#visualization-tools)
5. [Production Tools](#production-tools)
6. [Deployment Tools](#deployment-tools)
7. [Documentation Tools](#documentation-tools)
8. [Hardware Requirements](#hardware-requirements)
9. [Setup Scripts](#setup-scripts)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Python 3.10+
**Why**: Modern features, better type hints, performance improvements

**Installation (macOS)**:
```bash
# Using Homebrew
brew install python@3.11

# Verify
python3.11 --version  # Should show 3.11.x
```

**Installation (Linux)**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

**Installation (Windows)**:
- Download from python.org/downloads
- Ensure "Add Python to PATH" is checked
- Verify in PowerShell: `python --version`

---

### Git
**Why**: Version control, collaboration

**Installation**:
```bash
# macOS
brew install git

# Linux
sudo apt install git

# Verify
git --version  # Should show 2.x+
```

**Configuration**:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

---

## Development Tools

### 1. Virtual Environment
**Why**: Isolate project dependencies

**Setup**:
```bash
cd /Users/abivarma/ML\ Learnings/transformer-vs-vljepa

# Create virtual environment
python3.11 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify
which python  # Should point to venv/bin/python
pip --version
```

**Deactivate**: `deactivate`

---

### 2. IDE / Code Editor

#### VS Code (Recommended)
**Why**: Excellent Python support, extensions, integrated terminal

**Installation**:
```bash
# macOS
brew install --cask visual-studio-code

# Or download from code.visualstudio.com
```

**Essential Extensions**:
- Python (Microsoft)
- Pylance
- Jupyter
- Docker
- GitLens
- Better Comments
- Error Lens

**Install Extensions**:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-azuretools.vscode-docker
```

#### Alternative: PyCharm
- Download from jetbrains.com/pycharm
- Professional (paid) or Community (free)

---

### 3. Jupyter Lab
**Why**: Interactive development, visualization, experimentation

**Installation**:
```bash
pip install jupyterlab

# Launch
jupyter lab

# Or use VS Code Jupyter extension (recommended)
```

---

## Core Libraries

### Essential Dependencies

Create `requirements.txt`:
```txt
# Deep Learning
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.38.0

# Numerical & Data
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# ML Utilities
scikit-learn>=1.3.0
datasets>=2.16.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0

# Web Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-asyncio>=0.23.0

# Code Quality
black>=24.1.0
isort>=5.13.0
flake8>=7.0.0
mypy>=1.8.0

# Monitoring
prometheus-client>=0.19.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
pyyaml>=6.0
```

**Installation**:
```bash
source venv/bin/activate
pip install -r requirements.txt

# Or install individually
pip install torch torchvision transformers
pip install numpy pandas scikit-learn
pip install fastapi uvicorn pytest black
```

---

### PyTorch Setup

#### CPU Only
```bash
pip install torch torchvision torchaudio
```

#### GPU (CUDA)
**Check CUDA version**:
```bash
nvidia-smi  # Shows CUDA version
```

**Install PyTorch with CUDA** (example for CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

### Transformers Library
**Why**: Pre-trained models, tokenizers, utilities

```bash
pip install transformers

# With optional dependencies
pip install transformers[torch,sentencepiece,tokenizers]
```

**Verify**:
```python
from transformers import AutoModel, AutoTokenizer
print("Transformers installed successfully")
```

---

## Visualization Tools

### Matplotlib & Seaborn
```bash
pip install matplotlib seaborn
```

**Usage Example**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

---

### Plotly (Interactive)
```bash
pip install plotly
```

---

### Streamlit (Dashboards)
```bash
pip install streamlit

# Test
streamlit hello

# Run app
streamlit run app.py
```

---

### Weights & Biases (Experiment Tracking)
```bash
pip install wandb

# Login
wandb login  # Enter API key from wandb.ai
```

**Alternative: TensorBoard**:
```bash
pip install tensorboard

# Launch
tensorboard --logdir=runs
```

---

## Production Tools

### FastAPI & Uvicorn
```bash
pip install fastapi uvicorn[standard]

# Create test app (test_api.py)
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

# Run
uvicorn test_api:app --reload
```

Visit: http://localhost:8000

---

### Testing: pytest
```bash
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest

# With coverage
pytest --cov=src tests/

# Generate HTML report
pytest --cov=src --cov-report=html tests/
```

---

### Code Quality

#### Black (Formatting)
```bash
pip install black

# Format files
black src/

# Check only (no changes)
black --check src/
```

#### isort (Import Sorting)
```bash
pip install isort

# Sort imports
isort src/

# Check only
isort --check src/
```

#### flake8 (Linting)
```bash
pip install flake8

# Run linter
flake8 src/
```

#### mypy (Type Checking)
```bash
pip install mypy

# Type check
mypy src/
```

---

### Pre-commit Hooks
```bash
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Deployment Tools

### Docker
**Why**: Containerization, consistent environments

**Installation (macOS)**:
```bash
brew install --cask docker

# Or download Docker Desktop from docker.com
```

**Verify**:
```bash
docker --version
docker ps
```

**Basic Commands**:
```bash
# Build image
docker build -t transformer-vljepa:latest .

# Run container
docker run -p 8000:8000 transformer-vljepa:latest

# List containers
docker ps -a

# Stop container
docker stop <container_id>
```

---

### Docker Compose
**Why**: Multi-container applications

Comes with Docker Desktop. **Verify**:
```bash
docker-compose --version
```

---

### Kubernetes (kubectl)
**Why**: Container orchestration

**Installation (macOS)**:
```bash
brew install kubectl

# Verify
kubectl version --client
```

**Minikube (local Kubernetes)**:
```bash
brew install minikube

# Start cluster
minikube start

# Verify
kubectl get nodes
```

---

### Terraform (Infrastructure as Code)
**Why**: Provision cloud resources

**Installation (macOS)**:
```bash
brew install terraform

# Verify
terraform --version
```

---

### AWS CLI (if using AWS)
```bash
brew install awscli

# Configure
aws configure
# Enter: Access Key, Secret Key, Region, Output format
```

---

### Google Cloud SDK (if using GCP)
```bash
brew install --cask google-cloud-sdk

# Initialize
gcloud init
```

---

## Documentation Tools

### MkDocs (Documentation Site)
```bash
pip install mkdocs mkdocs-material

# Create project
mkdocs new docs

# Serve locally
mkdocs serve

# Build
mkdocs build
```

---

### Sphinx (Alternative)
```bash
pip install sphinx sphinx-rtd-theme

# Quickstart
sphinx-quickstart docs
```

---

## Hardware Requirements

### Minimum (For Development)
- **CPU**: Intel i5 / Apple M1 or equivalent
- **RAM**: 16GB
- **Storage**: 20GB free space
- **GPU**: None required (CPU training for small demos)

### Recommended (For Training)
- **CPU**: Intel i7 / Apple M1 Pro or equivalent
- **RAM**: 32GB
- **Storage**: 50GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 2060, etc.)

### Production (For Deployment)
- **Cloud**: AWS EC2 t3.medium or equivalent
- **RAM**: 8GB+ for inference
- **GPU**: T4 or better for VL-JEPA

---

## Setup Scripts

### Complete Setup (macOS)
```bash
#!/bin/bash
# setup.sh - Run this to setup everything

set -e  # Exit on error

echo "🚀 Setting up Transformer vs VL-JEPA project..."

# 1. Check Python
echo "📦 Checking Python..."
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
fi

# 2. Create virtual environment
echo "🔨 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# 5. Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# 6. Verify installations
echo "✅ Verifying installations..."
python --version
pip --version
pytest --version
black --version

echo "🎉 Setup complete! Activate environment with: source venv/bin/activate"
```

**Run**:
```bash
chmod +x setup.sh
./setup.sh
```

---

### Quick Start (After Setup)
```bash
#!/bin/bash
# start.sh - Quick start script

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Set environment variables if needed
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH  # macOS OpenMP

echo "✅ Environment activated"
echo "📊 Python: $(python --version)"
echo "📦 Virtual env: $VIRTUAL_ENV"
```

---

## Troubleshooting

### Common Issues

#### 1. "libomp not found" (macOS)
**Problem**: OpenMP library missing for PyTorch/scikit-learn

**Solution**:
```bash
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Add to ~/.zshrc or ~/.bashrc
echo 'export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
```

---

#### 2. "ModuleNotFoundError"
**Problem**: Package not installed or wrong environment

**Solution**:
```bash
# Verify environment
which python  # Should be in venv/

# Reinstall
pip install <package_name>

# Or reinstall all
pip install -r requirements.txt
```

---

#### 3. CUDA not found
**Problem**: PyTorch can't find CUDA

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

#### 4. Port already in use
**Problem**: Port 8000 (or other) is occupied

**Solution**:
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app:app --port 8001
```

---

#### 5. Permission denied
**Problem**: Can't execute script

**Solution**:
```bash
chmod +x script.sh
```

---

## Environment Variables

### Create `.env` file
```bash
# .env (DO NOT commit to git!)

# Weights & Biases
WANDB_API_KEY=your_key_here

# AWS (if using)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-west-2

# API Keys
HUGGINGFACE_TOKEN=your_token

# Production
PRODUCTION=false
LOG_LEVEL=INFO
```

### Load in Python
```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
```

---

## pyproject.toml Configuration

Create comprehensive project configuration:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "transformer-vljepa"
version = "0.1.0"
description = "Comprehensive comparison of Transformer and VL-JEPA architectures"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "torch>=2.2.0",
    "transformers>=4.38.0",
    "numpy>=1.24.0",
    "fastapi>=0.109.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.1.0",
    "mypy>=1.8.0",
    "flake8>=7.0.0",
]

[tool.black]
line-length = 100
target-version = ['py311']
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=src --cov-report=html --cov-report=term"
```

---

## Verification Checklist

After setup, verify everything works:

```bash
# Activate environment
source venv/bin/activate

# Run verification
python -c "
import sys
import torch
import transformers
import fastapi
import pytest

print(f'✅ Python: {sys.version}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ FastAPI: {fastapi.__version__}')
print(f'✅ PyTest: {pytest.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print('\\n🎉 All tools installed successfully!')
"
```

---

## Next Steps

1. ✅ Complete this tools setup
2. ⏳ Configure Git & GitHub (see SPRINT_STORIES.md TVLJ-004)
3. ⏳ Setup CI/CD pipeline (see AGENTS.md)
4. ⏳ Start Phase 1 (Foundations)

---

## Quick Reference

### Daily Workflow
```bash
# Start work
cd /Users/abivarma/ML\ Learnings/transformer-vs-vljepa
source venv/bin/activate

# Update dependencies (if needed)
pip install -r requirements.txt

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Before commit (automatic with pre-commit hooks)
pre-commit run --all-files
```

### Package Management
```bash
# Add new package
pip install package_name
pip freeze > requirements.txt

# Update all packages
pip list --outdated
pip install --upgrade package_name
```

---

**This tools guide will be updated as new tools are added to the project.**

Last Updated: 2026-01-31
