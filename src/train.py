import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model configuration file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training.")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory for saving results.")
    parser.add_argument("--name", type=str, default="exp", help="Name of the experiment.")
    return parser.parse_args()