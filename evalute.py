import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

def calculate_accuracy(dataset_dir, checkpoint_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # 1. Prepare Data Loader (Points to your test/validation subset)
    img_dir = f"{dataset_dir}/images"
    label_dir = f"{dataset_dir}/labels"

    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    dataset = VOC2012Dataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model and Load Trained Weights
    model = VOC2012Model(num_classes=20, max_objects=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with historical best acc: {checkpoint.get('best_acc', 0):.2f}%")

    # 3. Evaluation Metrics Tracking
    correct_labels = 0
    total_labels = 0
    total_loss_box = 0.0
    criterion_box = nn.MSELoss()

    print("\nEvaluating model performance across the dataset...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluation", colour="cyan"):
            images = images.to(device)
            true_boxes = targets['boxes'].to(device)
            true_labels = targets['labels'].to(device)

            # Model Forward Pass
            outputs = model(images)
            pred_boxes = outputs['pred_boxes']
            pred_classes = outputs['pred_classes']

            # Calculate Bounding Box Coordinate Error (MSE)
            loss_box = criterion_box(pred_boxes, true_boxes)
            total_loss_box += loss_box.item()

            # Filter out padding slots (-1) using target mask
            mask = true_labels >= 0
            if mask.sum() > 0:
                # Get the highest probability class index
                preds = torch.argmax(pred_classes[mask], dim=1)
                correct_labels += (preds == true_labels[mask]).sum().item()
                total_labels += mask.sum().item()

    # 4. Final Metrics Compilation
    final_accuracy = (correct_labels / total_labels) * 100 if total_labels > 0 else 0.0
    average_box_mse = total_loss_box / len(dataloader)

    print("\n" + "="*40)
    print("🎯 EVALUATION REPORT")
    print("="*40)
    print(f"📊 Total Objects Evaluated : {total_labels}")
    print(f"✅ Correctly Classified    : {correct_labels}")
    print(f"📈 Classification Accuracy : {final_accuracy:.2f}%")
    print(f"📉 Average Box Coordinate MSE: {average_box_mse:.6f}")
    print("="*40)

if __name__ == "__main__":
    # Configure your paths here
    DATASET_PATH = "D:\\football\\pascal-voc-2012\\val" # Or point to validation subset if available
    CHECKPOINT = "D:\\football\\voc_checkpoint_best.pt" # Path to your best checkpoint file
    
    calculate_accuracy(DATASET_PATH, CHECKPOINT)