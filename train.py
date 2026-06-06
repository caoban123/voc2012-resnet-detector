import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def cxcywh_to_xyxy(boxes, img_size=224):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=1).to(boxes.device)
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "D:\\football\\pascal-voc-2012\\train\\images"
    label_dir = "D:\\football\\pascal-voc-2012\\train\\labels"

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = VOC2012Dataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = VOC2012Model(num_classes=20, max_objects=5).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion_box = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="runs/voc2012_experiment")
    metric = MeanAveragePrecision(iou_type="bbox")

    best_map = 0.0
    epochs = 100

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, colour="green")

        for images, targets in progress_bar:
            images = images.to(device)
            true_boxes = targets['boxes'].to(device)
            true_labels = targets['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            pred_boxes = outputs['pred_boxes']
            pred_classes = outputs['pred_classes']

            mask = true_labels >= 0

            if mask.sum() > 0:
                loss_box = criterion_box(pred_boxes[mask], true_boxes[mask])
                loss_class = criterion_class(pred_classes[mask], true_labels[mask])
            else:
                loss_box = torch.tensor(0.0, device=device, requires_grad=True)
                loss_class = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss = loss_box + (loss_class * 2.0)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            preds_fmt = []
            targets_fmt = []
            
            p_boxes_cpu = pred_boxes.detach().cpu()
            p_classes_cpu = pred_classes.detach().cpu()
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

            progress_bar.set_description(
                f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.4f}"
            )

        # Tổng kết epoch
        epoch_loss = running_loss / len(dataloader)
        result = metric.compute()
        epoch_map = result["map"].item()
        epoch_map50 = result["map_50"].item()
        metric.reset()

        print(f"=> Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | mAP: {epoch_map:.4f} | mAP@50: {epoch_map50:.4f}")

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("mAP/train", epoch_map, epoch)
        writer.add_scalar("mAP50/train", epoch_map50, epoch)

        is_best = False
        if epoch_map > best_map:
            best_map = epoch_map
            is_best = True

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map
        }
        
        torch.save(checkpoint, "voc_checkpoint_last.pt")
        if is_best:
            torch.save(checkpoint, "voc_checkpoint_best.pt")
            print(f"   ✓ Best model saved! mAP: {best_map:.4f}")

    writer.close()


if __name__ == "__main__":
    print("Starting VOC2012 Training")

    try:
        train_model()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting safely...")

    except Exception as e:
        print(f"\nTraining failed due to an error.")
        print(f"Error details: {e}")
        print("Please check your data paths or model configuration.")

    finally:
        print("\nProcess finished.")