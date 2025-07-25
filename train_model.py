#!/usr/bin/env python3

import argparse
import slideflow as sf
from pathlib import Path

def train_slideflow_model(
    data_dir: str,
    output_dir: str,
    tile_px: int = 299,
    tile_um: str = '10x',
    model_arch: str = 'xception',
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    epochs: int = 10,
    val_fraction: float = 0.3
):
    """Train a Slideflow model for lung adenocarcinoma vs squamous classification."""

    # Create project
    print("Setting up Slideflow project...")
    project = sf.load_project(
        root=data_dir,
        cfg=sf.project.LungAdenoSquam(),
        download=False  # Data already fetched
    )

    # Create dataset
    print(f"Creating dataset with {tile_px}px tiles at {tile_um} magnification...")
    dataset = project.dataset(
        tile_px=tile_px,
        tile_um=tile_um
    )

    # Extract tiles if not already done
    tile_dir = Path(data_dir) / f"tiles/{tile_px}px_{tile_um}"
    if not tile_dir.exists() or not any(tile_dir.iterdir()):
        print("Extracting tiles from slides...")
        dataset.extract_tiles(qc='otsu')
    else:
        print("Tiles already extracted, skipping extraction...")

    # Split dataset
    print(f"Splitting dataset with {val_fraction:.0%} validation...")
    train_dataset, val_dataset = dataset.split(
        model_type='classification',
        labels='subtype',
        val_fraction=val_fraction
    )

    # Define model parameters
    print(f"Configuring {model_arch} model...")
    params = sf.ModelParams(
        tile_px=tile_px,
        tile_um=tile_um,
        model=model_arch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stop=True,
        early_stop_patience=3
    )

    # Train model
    print("Starting training...")
    results = project.train(
        'subtype',
        dataset=train_dataset,
        params=params,
        val_dataset=val_dataset,
        multi_gpu=True,
        outdir=output_dir
    )

    print(f"Training complete! Results saved to: {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train Slideflow model for lung cancer classification')
    parser.add_argument('--data-dir', type=str, default='/data', help='Directory containing slide data')
    parser.add_argument('--output-dir', type=str, default='/output', help='Directory for model outputs')
    parser.add_argument('--tile-px', type=int, default=299, help='Tile size in pixels')
    parser.add_argument('--tile-um', type=str, default='10x', help='Tile magnification')
    parser.add_argument('--model', type=str, default='xception', help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--val-fraction', type=float, default=0.3, help='Validation split fraction')

    args = parser.parse_args()

    train_slideflow_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tile_px=args.tile_px,
        tile_um=args.tile_um,
        model_arch=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        val_fraction=args.val_fraction
    )

if __name__ == '__main__':
    main()
