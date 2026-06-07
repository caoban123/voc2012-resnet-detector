import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.ops import nms  # 🌟 Bổ sung NMS để dọn dẹp dự đoán khi đánh giá
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

NUM_CLASSES = 20
NUM_ANCHORS = 3
GRID_SIZE = 7
IMG_SIZE = 224

# Bộ 3 Anchor mẫu bắt buộc phải đồng nhất với file train
ANCHORS = torch.tensor([
    [0.15, 0.22],
    [0.45, 0.50],
    [0.78, 0.82]
])

def decode_boxes(pred_boxes, anchors, grid_size=GRID_SIZE):
    """Giải mã ma trận đặc trưng từ dạng offset về tọa độ góc pixel [x1, y1, x2, y2]"""
    device = pred_boxes.device
    batch_size, grid_h, grid_w, _, _ = pred_boxes.shape

    c_y, c_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    c_x = c_x.view(1, grid_h, grid_w, 1).to(device)
    c_y = c_y.view(1, grid_h, grid_w, 1).to(device)
    anchors = anchors.view(1, 1, 1, NUM_ANCHORS, 2).to(device)

    bx = (torch.sigmoid(pred_boxes[..., 0]) + c_x) / grid_w
    by = (torch.sigmoid(pred_boxes[..., 1]) + c_y) / grid_h
    bw = anchors[..., 0] * torch.exp(pred_boxes[..., 2].clamp(-4, 4))
    bh = anchors[..., 1] * torch.exp(pred_boxes[..., 3].clamp(-4, 4))

    x1 = (bx - bw / 2) * IMG_SIZE
    y1 = (by - bh / 2) * IMG_SIZE
    x2 = (bx + bw / 2) * IMG_SIZE
    y2 = (by + bh / 2) * IMG_SIZE

    return torch.stack([x1, y1, x2, y2], dim=-1)

def calculate_accuracy(dataset_dir, checkpoint_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # 1. Prepare Dataset (Đồng bộ khâu preprocessing giống file train)
    img_dir = f"{dataset_dir}/images"
    label_dir = f"{dataset_dir}/labels"

    transform = Compose([
        ToTensor(),                 # Đưa ToTensor lên trước nếu chạy local từ OpenCV/PIL
        Resize((IMG_SIZE, IMG_SIZE))
    ])

    dataset = VOC2012Dataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Load Model với cấu hình Anchor mới
    model = VOC2012Model(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Lỗi: Không tìm thấy file checkpoint tại {checkpoint_path}")
        return

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
            pred_conf = outputs['pred_conf']
            pred_classes = outputs['pred_classes']

            # Giải mã toàn bộ ma trận dự đoán sang pixel thực trên CPU
            decoded_boxes = decode_boxes(pred_boxes, ANCHORS).cpu()
            p_conf = torch.sigmoid(pred_conf).cpu()
            p_cls = pred_classes.cpu()
            
            t_boxes_cpu = true_boxes.cpu()
            t_labels_cpu = true_labels.cpu()

            preds_fmt = []
            targets_fmt = []
            
            for i in range(len(images)):
                # Làm phẳng lưới 7x7x3 thành danh sách 147 hộp đơn lẻ
                b_boxes = decoded_boxes[i].view(-1, 4)
                b_confs = p_conf[i].view(-1)
                b_cls = p_cls[i].view(-1, NUM_CLASSES)

                # Tính điểm class_score và nhân chéo tính điểm tự tin final_score
                box_scores, box_labels = torch.softmax(b_cls, dim=-1).max(dim=-1)
                final_scores = b_confs * box_scores

                # Bộ lọc thô lọc bỏ hộp rác độ tự tin thấp (> 0.1)
                mask = final_scores > 0.1
                if mask.sum() > 0:
                    fb = b_boxes[mask]
                    fs = final_scores[mask]
                    fl = box_labels[mask]
                    
                    # 🌟 ÁP DỤNG NMS ĐỂ LỌC TRÙNG KHI ĐÁNH GIÁ MTRIC MAP
                    keep = nms(fb, fs, iou_threshold=0.45)
                    preds_fmt.append({"boxes": fb[keep], "scores": fs[keep], "labels": fl[keep]})
                else:
                    preds_fmt.append({
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "scores": torch.empty((0,), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64)
                    })

                # Chuyển đổi nhãn thật cxcywh về xyxy pixel 224 để so khớp
                valid = t_labels_cpu[i] >= 0
                if valid.sum() > 0:
                    cx, cy, w, h = t_boxes_cpu[i][valid].T
                    xyxy = torch.stack([(cx - w/2)*IMG_SIZE, (cy - h/2)*IMG_SIZE,
                                         (cx + w/2)*IMG_SIZE, (cy + h/2)*IMG_SIZE], dim=1)
                    targets_fmt.append({"boxes": xyxy, "labels": t_labels_cpu[i][valid].long()})
                else:
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
    # Thay đổi đường dẫn đến file trọng số ResNet-50 mới của bạn
    DATASET_PATH = "D:\\football\\pascal-voc-2012\\val" 
    CHECKPOINT = "D:\\football\\voc_resnet50_anchor_best.pt"
    
    calculate_accuracy(DATASET_PATH, CHECKPOINT)