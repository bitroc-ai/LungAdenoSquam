# Lung Adenocarcinoma vs Squamous Cell Carcinoma Classification

A containerized deep learning pipeline for classifying lung cancer subtypes from whole slide images using the Slideflow framework.

## Overview

This project trains deep learning models to distinguish between lung adenocarcinoma and squamous cell carcinoma from histopathological whole slide images (WSI). The entire training pipeline is containerized for reproducible execution across different environments.

## Quick Start

### Prerequisites
- Docker with GPU support (recommended)
- Whole slide image data in a local directory

### Training a Model

1. **Build the container:**
   ```bash
   docker build -t lung-adeno-squam .
   ```

2. **Run training:**
   ```bash
   docker run --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam
   ```

   This will:
   - Extract tiles from slides in `./data/`
   - Train an Xception model for 10 epochs
   - Save results to `./output/`

### Custom Training Parameters

```bash
docker run --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model resnet50 \
  --tile-px 512 \
  --val-fraction 0.2
```

## Data Requirements

Your data directory should contain whole slide images organized for the Slideflow LungAdenoSquam project configuration:

```
data/
├── slides/
│   ├── adenocarcinoma_slide1.svs
│   ├── adenocarcinoma_slide2.svs
│   ├── squamous_slide1.svs
│   └── ...
└── (tiles will be auto-generated)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `/data` | Directory containing slide data |
| `--output-dir` | `/output` | Directory for model outputs |
| `--tile-px` | `299` | Tile size in pixels |
| `--tile-um` | `10x` | Tile magnification level |
| `--model` | `xception` | Model architecture |
| `--batch-size` | `64` | Training batch size |
| `--learning-rate` | `0.0001` | Learning rate |
| `--epochs` | `10` | Maximum training epochs |
| `--val-fraction` | `0.3` | Validation split fraction |

## Supported Model Architectures

- `xception` (default)
- `resnet50`
- `efficientnet`
- `mobilenet`
- And other Slideflow-supported architectures

## Output Structure

After training, your output directory will contain:

```
output/
├── models/          # Trained model checkpoints
├── logs/            # Training logs and metrics
└── results/         # Evaluation results and plots
```

## Development

### Running Locally
```bash
python train_model.py --data-dir ./data --output-dir ./output
```

### Interactive Container
```bash
docker run -it --gpus all -v ./data:/data -v ./output:/output lung-adeno-squam bash
```

## Technical Details

- **Framework**: [Slideflow](https://slideflow.dev) for computational pathology
- **Base Image**: `jamesdolezal/slideflow:latest-tf`
- **GPU Support**: Automatic GPU detection and utilization
- **Early Stopping**: Prevents overfitting with patience=3
- **Multi-GPU**: Automatic multi-GPU training when available

## Performance Notes

- **First Run**: Tile extraction can take significant time depending on slide size and number
- **Subsequent Runs**: Tiles are cached and reused automatically
- **Memory Usage**: Reduce `--batch-size` if encountering GPU memory issues
- **Training Time**: Varies by dataset size, model architecture, and hardware

## License

This project uses the Slideflow framework. Please refer to Slideflow's licensing terms for usage guidelines.