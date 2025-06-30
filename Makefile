# Makefile for 3D Segmentation Project

# Configuration
PROJECT_NAME = Semantic3DSegmantation
VERSION = 1.0.0
DOCKER_IMAGE = $(PROJECT_NAME):$(VERSION)
DATA_DIR = $(shell pwd)/data
RESULTS_DIR = $(shell pwd)/results
MODELS_DIR = $(shell pwd)/outputs
CONFIG_DIR = $(shell pwd)/configs

# Docker targets
.PHONY: docker-build docker-run-train docker-run-infer

docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run-train:
	@echo "Running training in container..."
	docker run --gpus all -it --rm \
		$(DOCKER_IMAGE) train


docker-run-infer:
	@echo "Running inference in container..."
	docker run --gpus all -it --rm \
		$(DOCKER_IMAGE) infer input_path=static/scene0022_01_vh_clean_2.ply output_path=/workspace/results/segmented.ply

# Local execution targets
.PHONY: train infer

train:
	@echo "Starting local training..."
	python train.py

infer:
	@echo "Running local inference..."
	python inference.py input_path=$(input_path) output_path=$(output_path)

streamlit-run:
	@echo "Starting Streamlit application..."
	streamlit run s2app.py

# Development targets
.PHONY: setup format lint clean data-download

setup:
	@echo "Setting up development environment..."
	python -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
    source .venv/bin/activate && pip install --extra-index-url https://download.pytorch.org/whl/cu118
	source .venv/bin/activate && pip install -r requirements.txt

PYTHON_FILES := $(shell find . -name '*.py' \
    ! -path "*/.ipynb_checkpoints/*" \
    ! -path "*/venv/*" \
    ! -path "*/build/*" \
    ! -path "*/models/external/*")

format:
	@echo "Formatting all Python files..."
	pip install black isort
	black $(PYTHON_FILES)
	isort $(PYTHON_FILES)

clean:
	@echo "Cleaning cache files..."
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@echo "Cache cleaned!"

data-download:
	@echo "Downloading dataset..."
	@mkdir -p downloaded
	curl -L -o downloaded/scannet.zip "https://www.kaggle.com/api/v1/datasets/download/dngminhli/scannet"
	unzip downloaded/scannet.zip -d downloaded/
	@echo "Dataset downloaded and extracted to downloaded/ folder"