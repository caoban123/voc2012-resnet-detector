import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

NUM_CLASSES = 20
NUM_ANCHORS = 3
GRID_SIZE = 7
IMG_SIZE = 224

ANCHORS = torch.tensor([
    [0.15, 0.22],
    [0.45, 0.50],
    [0.78, 0.82]
])

LOSS_BOX_W    = 5.0
LOSS_OBJ_W    = 1.0
LOSS_NOOBJ_W  = 0.5
LOSS_CLASS_W  = 1.0


def decode_boxes(pred_boxes, anchors, grid_size=GRID_SIZE):
    device = pred_boxes.device
    _, grid_h, grid_w, _, _ = pred_boxes.shape

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


def build_targets(true_boxes, true_labels, anchors, grid_size=GRID_SIZE):
    device = true_boxes.device
    batch_size, max_objects, _ = true_boxes.shape

    tgt_boxes  = torch.zeros(batch_size, grid_size, grid_size, NUM_ANCHORS, 4, device=device)
    tgt_conf   = torch.zeros(batch_size, grid_size, grid_size, NUM_ANCHORS, 1, device=device)
    tgt_classes = torch.full((batch_size, grid_size, grid_size, NUM_ANCHORS), -1, dtype=torch.long, device=device)

    anchors_np = anchors.tolist()

    for b in range(batch_size):
        for o in range(max_objects):
            if true_labels[b, o] < 0:
                continue

            gx, gy, gw, gh = true_boxes[b, o].tolist()
            bin_x = min(int(gx * grid_size), grid_size - 1)
            bin_y = min(int(gy * grid_size), grid_size - 1)

            best_anchor, min_diff = 0, float('inf')
            for a, (aw, ah) in enumerate(anchors_np):
                diff = abs(gw * gh - aw * ah)
                if diff < min_diff:
                    min_diff = diff
                    best_anchor = a

            aw, ah = anchors_np[best_anchor]
            tx = gx * grid_size - bin_x
            ty = gy * grid_size - bin_y
            tw = math.log(max(gw / aw, 1e-8))
            th = math.log(max(gh / ah, 1e-8))

            tgt_boxes[b, bin_y, bin_x, best_anchor]   = torch.tensor([tx, ty, tw, th], device=device)
            tgt_conf[b, bin_y, bin_x, best_anchor]     = 1.0
            tgt_classes[b, bin_y, bin_x, best_anchor]  = true_labels[b, o].long()

    return tgt_boxes, tgt_conf, tgt_classes


def format_predictions(decoded_boxes, pred_conf, pred_classes, true_boxes, true_labels):
    preds_fmt, targets_fmt = [], []
    p_conf  = torch.sigmoid(pred_conf).cpu()
    p_cls   = pred_classes.cpu()
    t_boxes = true_boxes.cpu()
    t_labels = true_labels.cpu()
    d_boxes = decoded_boxes.cpu()

    for i in range(len(d_boxes)):
        b_boxes  = d_boxes[i].view(-1, 4)
        b_confs  = p_conf[i].view(-1)
        b_cls    = p_cls[i].view(-1, NUM_CLASSES)

        box_scores, box_labels = torch.softmax(b_cls, dim=-1).max(dim=-1)
        final_scores = b_confs * box_scores

        mask = final_scores > 0.1
        if mask.sum() > 0:
            fb = b_boxes[mask]
            fs = final_scores[mask]
            fl = box_labels[mask]
            keep = nms(fb, fs, iou_threshold=0.45)
            preds_fmt.append({"boxes": fb[keep], "scores": fs[keep], "labels": fl[keep]})
        else:
            preds_fmt.append({
                "boxes":  torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,),   dtype=torch.float32),
                "labels": torch.empty((0,),   dtype=torch.int64)
            })

        valid = t_labels[i] >= 0
        if valid.sum() > 0:
            cx, cy, w, h = t_boxes[i][valid].T
            xyxy = torch.stack([(cx - w/2)*IMG_SIZE, (cy - h/2)*IMG_SIZE,
                                 (cx + w/2)*IMG_SIZE, (cy + h/2)*IMG_SIZE], dim=1)
            targets_fmt.append({"boxes": xyxy, "labels": t_labels[i][valid].long()})
        else:
            targets_fmt.append({
                "boxes":  torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,),   dtype=torch.int64)
            })

    return preds_fmt, targets_fmt


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir   = "D:\\football\\pascal-voc-2012\\train\\images"
    label_dir = "D:\\football\\pascal-voc-2012\\train\\labels"
    CHECKPOINT_PATH = "voc_checkpoint_best.pt"

    transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
    dataset   = VOC2012Dataset(img_dir, label_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model     = VOC2012Model(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion_box   = nn.MSELoss()
    criterion_conf  = nn.BCEWithLogitsLoss()
    criterion_noobj = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss(ignore_index=-1)

    writer = SummaryWriter(log_dir="runs/voc2012_anchor_experiment")
    metric = MeanAveragePrecision(iou_type="bbox")

    start_epoch = 0
    best_map    = 0.0
    epochs      = 100

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint  = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map    = checkpoint.get('best_map', 0.0)
        print(f"Tiếp tục từ epoch {start_epoch + 1}, best mAP: {best_map:.4f}")
    else:
        print("Bắt đầu huấn luyện từ đầu")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, colour="green")

        for images, targets in progress_bar:
            images      = images.to(device)
            true_boxes  = targets['boxes'].to(device)
            true_labels = targets['labels'].to(device)

            optimizer.zero_grad()

            outputs      = model(images)
            pred_boxes   = outputs['pred_boxes']
            pred_conf    = outputs['pred_conf']
            pred_classes = outputs['pred_classes']

            tgt_boxes, tgt_conf, tgt_classes = build_targets(true_boxes, true_labels, ANCHORS)

            obj_mask   = (tgt_conf == 1.0).squeeze(-1)
            noobj_mask = (tgt_conf == 0.0).squeeze(-1)

            if obj_mask.sum() > 0:
                loss_box   = criterion_box(pred_boxes[obj_mask], tgt_boxes[obj_mask])
                loss_class = criterion_class(pred_classes[obj_mask], tgt_classes[obj_mask])
            else:
                loss_box   = torch.tensor(0.0, device=device, requires_grad=True)
                loss_class = torch.tensor(0.0, device=device, requires_grad=True)

            loss_obj   = criterion_conf(pred_conf[obj_mask.unsqueeze(-1)],
                                        tgt_conf[obj_mask.unsqueeze(-1)])
            loss_noobj = criterion_noobj(pred_conf[noobj_mask.unsqueeze(-1)],
                                         tgt_conf[noobj_mask.unsqueeze(-1)])

            total_loss = (loss_box   * LOSS_BOX_W  +
                          loss_obj   * LOSS_OBJ_W   +
                          loss_noobj * LOSS_NOOBJ_W +
                          loss_class * LOSS_CLASS_W)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += total_loss.item()

            decoded = decode_boxes(pred_boxes.detach(), ANCHORS)
            preds_fmt, targets_fmt = format_predictions(decoded, pred_conf.detach(),
                                                        pred_classes.detach(),
                                                        true_boxes, true_labels)
            metric.update(preds_fmt, targets_fmt)

            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.4f}")

        scheduler.step()

        epoch_loss  = running_loss / len(dataloader)
        result      = metric.compute()
        epoch_map   = result["map"].item()
        epoch_map50 = result["map_50"].item()
        metric.reset()

        print(f"=> Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | mAP: {epoch_map:.4f} | mAP@50: {epoch_map50:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("Loss/train",  epoch_loss,  epoch)
        writer.add_scalar("mAP/train",   epoch_map,   epoch)
        writer.add_scalar("mAP50/train", epoch_map50, epoch)

        checkpoint = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map':             best_map
        }
        torch.save(checkpoint, "voc_resnet50_anchor_last.pt")

        if epoch_map > best_map:
            best_map = epoch_map
            torch.save(checkpoint, "voc_resnet50_anchor_best.pt")
            print(f"   ✓ Best mAP: {best_map:.4f}")

    writer.close()


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTiến trình bị hủy bởi người dùng.")
    except Exception as e:
        print(f"\nLỗi: {e}")
        raise