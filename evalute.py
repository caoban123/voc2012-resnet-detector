import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

def cxcywh_to_xyxy(boxes, img_size=224):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=1)

def calculate_accuracy(dataset_dir, checkpoint_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # 1. Prepare Dataset
    img_dir = f"{dataset_dir}/images"
    label_dir = f"{dataset_dir}/labels"

    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    dataset = VOC2012Dataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Load Model
    model = VOC2012Model(num_classes=20, max_objects=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # 3. Setup Torchmetrics mAP
    metric = MeanAveragePrecision(iou_type="bbox")

    print("\nCalculating mAP metrics across the dataset...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluation", colour="cyan"):
            images = images.to(device)
            true_boxes = targets['boxes'].to(device)
            true_labels = targets['labels'].to(device)

            outputs = model(images)
            pred_boxes = outputs['pred_boxes']
            pred_classes = outputs['pred_classes']

            mask = true_labels >= 0

            preds_fmt = []
            targets_fmt = []
            
            p_boxes_cpu = pred_boxes.cpu()
            p_classes_cpu = pred_classes.cpu()
            t_boxes_cpu = true_boxes.cpu()
            t_labels_cpu = true_labels.cpu()
            mask_cpu = mask.cpu()

            for i in range(len(images)):
                m = mask_cpu[i]
                if m.sum() > 0:
                    valid_classes = p_classes_cpu[i][m]
                    scores = torch.softmax(valid_classes, dim=1)

                    pred_boxes_scaled = cxcywh_to_xyxy(p_boxes_cpu[i][m])
                    true_boxes_scaled = cxcywh_to_xyxy(t_boxes_cpu[i][m])

                    preds_fmt.append({
                        "boxes": pred_boxes_scaled,
                        "scores": scores.max(dim=1).values,
                        "labels": valid_classes.argmax(dim=1)
                    })
                    targets_fmt.append({
                        "boxes": true_boxes_scaled,
                        "labels": t_labels_cpu[i][m]
                    })
                else:
                    preds_fmt.append({
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "scores": torch.empty((0,), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64)
                    })
                    targets_fmt.append({
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64)
                    })

            metric.update(preds_fmt, targets_fmt)

    # 4. Final report
    result = metric.compute()
    print("\n" + "="*40)
    print("🎯 LOCAL EVALUATION REPORT (mAP)")
    print("="*40)
    print(f"📈 Mean Average Precision (mAP)   : {result['map'].item():.4f}")
    print(f"📊 mAP @ IoU=0.50 (mAP@50)        : {result['map_50'].item():.4f}")
    print(f"📉 mAP @ IoU=0.75 (mAP@75)        : {result['map_75'].item():.4f}")
    print("="*40)

if __name__ == "__main__":
    DATASET_PATH = "D:\\football\\pascal-voc-2012\\train" 
    CHECKPOINT = "voc_checkpoint_best.pt"
    
    calculate_accuracy(DATASET_PATH, CHECKPOINT)