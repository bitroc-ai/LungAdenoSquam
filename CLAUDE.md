# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a containerized machine learning project for lung cancer subtype classification (adenocarcinoma vs squamous cell carcinoma) using the Slideflow framework. The project processes whole slide images (WSI) of lung tissue to train deep learning models for pathological classification.

## Docker Workflow

### Building the Container
```bash
docker build -t lung-adeno-squam .
```

### Running Training
The container expects data to be mounted from the host system:

```bash
# Basic training run
docker run -v ./data:/data -v ./output:/output lung-adeno-squam

# With GPU support
docker run --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam

# With custom parameters
docker run --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam \
  --epochs 20 --batch-size 32 --learning-rate 0.001 --model resnet50

# Interactive mode for debugging
docker run -it --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam bash
```

**Important**: The `/data` directory should be mapped from host - it's not created in the container. Only `/output` is created internally.

## Training Architecture

### Slideflow Pipeline
The training follows this sequence:
1. **Project Setup**: Uses `sf.project.LungAdenoSquam()` configuration 
2. **Dataset Creation**: Processes WSI files into tile datasets
3. **Tile Extraction**: Extracts image tiles with Otsu QC filtering
4. **Dataset Split**: Splits into training/validation sets by `val_fraction`
5. **Model Training**: Trains classification model with early stopping

### Key Components
- **Target**: Binary classification for `subtype` labels (adenocarcinoma vs squamous)
- **Tile Processing**: Configurable tile size (`tile_px`) and magnification (`tile_um`)
- **Model Architectures**: Supports multiple architectures (default: `xception`)
- **Training Features**: Multi-GPU support, early stopping (patience=3), configurable hyperparameters

### Data Organization
```
/data/
├── slides/           # Raw WSI files
└── tiles/           # Extracted tiles (auto-created)
    └── {tile_px}px_{tile_um}/

/output/
├── models/          # Trained model checkpoints
├── logs/            # Training logs and metrics  
└── results/         # Evaluation results
```

## Training Parameters

### Core Parameters
- `--tile-px` (299): Tile size in pixels - affects memory usage and model input
- `--tile-um` (10x): Magnification level - impacts feature resolution  
- `--model` (xception): Architecture choice - affects model capacity and speed
- `--batch-size` (64): Training batch size - balance memory and convergence
- `--epochs` (10): Maximum training epochs with early stopping

### Tuning Guidelines
- **Memory Issues**: Reduce `batch-size` or `tile-px`
- **Training Speed**: Use smaller models (mobilenet) or reduce `tile-px`
- **Accuracy**: Increase `epochs`, try different architectures (resnet50, efficientnet)
- **Overfitting**: Increase `val-fraction`, reduce model complexity

## Development Notes

### Local Development
- The training script can be run directly: `python train_model.py --help`
- Requires Slideflow installation and dependencies
- For development, mount the script: `-v ./train_model.py:/app/train_model.py`

### Model Evaluation
- Training outputs include validation metrics and model checkpoints
- Use Slideflow's evaluation tools to assess model performance on test data
- Monitor training logs for convergence and potential overfitting

### Common Issues
- **Tile extraction takes time**: First run extracts tiles, subsequent runs reuse them
- **GPU memory**: Reduce batch size if encountering OOM errors
- **Data path errors**: Ensure slide files are in mounted `/data` directory
- **Permission issues**: Check file permissions in mounted volumes