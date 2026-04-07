import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from dataset import ISLVideoDataset
from model import VideoTransformerModel   # <-- corrected import

def evaluate(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = ISLVideoDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    num_classes = len(dataset.label_to_idx)  # should be 11
    model = VideoTransformerModel(num_classes=num_classes).to(device)  # <-- corrected class name
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("Detailed classification report:")
    target_names = list(dataset.idx_to_label.values())  # class names
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    evaluate(args)