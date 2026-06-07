import os
import glob
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.ops import nms  # Khôi phục NMS để dọn dẹp hộp trùng khi test
from src.models import VOC2012Model

CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

NUM_CLASSES = 20
NUM_ANCHORS = 3
GRID_SIZE = 7
IMG_SIZE = 224

# Bộ 3 Anchor mẫu bắt buộc phải giống hệt file train
ANCHORS = torch.tensor([
    [0.15, 0.22],
    [0.45, 0.50],
    [0.78, 0.82]
])

def decode_boxes(pred_boxes, anchors, grid_size=GRID_SIZE):
    """Giải mã ma trận đặc trưng đầu ra thành tọa độ pixel thực trên màn hình 224x224"""
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

def predict_image(image_path, checkpoint_path, output_dir="results", conf_threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo model cấu hình chuẩn ResNet-50 + 3 Anchors
    model = VOC2012Model(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Lỗi: Không tìm thấy file trọng số tại {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Đọc và tiền xử lý ảnh thô
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    orig_h, orig_w, _ = original_img.shape

    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    transform = Compose([
            ToTensor(),                 # 🌟 ĐƯA TO_TENSOR LÊN ĐẦU để biến Numpy Array thành PyTorch Tensor trước
            Resize((IMG_SIZE, IMG_SIZE)) # Sau đó mới ép kích thước về 224x224
        ])
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 3. Đẩy ảnh qua mạng AI dự đoán (Inference)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_boxes = outputs['pred_boxes']
        pred_conf = outputs['pred_conf']
        pred_classes = outputs['pred_classes']

        # Giải mã tọa độ dựa trên Grid lưới và Anchor
        decoded_boxes = decode_boxes(pred_boxes, ANCHORS).squeeze(0) # Loại bỏ kích thước Batch_size -> [7, 7, 3, 4]
        p_conf = torch.sigmoid(pred_conf).squeeze(0)                 # [7, 7, 3, 1]
        p_cls = pred_classes.squeeze(0)                              # [7, 7, 3, 20]

    # 4. Làm phẳng cấu trúc lưới về dạng danh sách 147 hộp phẳng trên CPU
    b_boxes = decoded_boxes.view(-1, 4).cpu()
    b_confs = p_conf.view(-1).cpu()
    b_cls = p_cls.view(-1, NUM_CLASSES).cpu()

    # Tính điểm tự tin cuối cùng = Độ tự tin nền x Xác suất nhãn lớp cao nhất
    box_scores, box_labels = torch.softmax(b_cls, dim=-1).max(dim=-1)
    final_scores = b_confs * box_scores

    # Lọc thô theo ngưỡng cấu hình (Mạng Anchor nên đặt tầm 0.25 - 0.3 là vừa đẹp)
    mask = final_scores > conf_threshold
    
    drawn_objects = 0
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))

    if mask.sum() > 0:
        fb = b_boxes[mask]
        fs = final_scores[mask]
        fl = box_labels[mask]

        # 🌟 ÁP DỤNG BỘ LỌC CHÍ MẠNG NMS ĐỂ XÓA HỘP TRÙNG LÊN NHAU
        keep = nms(fb, fs, iou_threshold=0.45)

        print(f"\n--- Kết quả phát hiện cho {os.path.basename(image_path)} ---")
        
        for idx in keep:
            class_idx = fl[idx].item()
            class_name = CLASS_NAMES[class_idx]
            confidence = fs[idx].item()
            
            # Phóng đại tọa độ từ thang ảnh 224 về kích thước pixel gốc của ảnh ban đầu
            xmin = int(fb[idx][0].item() * (orig_w / IMG_SIZE))
            ymin = int(fb[idx][1].item() * (orig_h / IMG_SIZE))
            xmax = int(fb[idx][2].item() * (orig_w / IMG_SIZE))
            ymax = int(fb[idx][3].item() * (orig_h / IMG_SIZE))

            # Ép biên cố định nằm trong lòng bức ảnh
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)

            print(f"Vật thể {drawn_objects+1}: {class_name} | Độ tự tin: {confidence:.2f} | Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

            # Vẽ khung và viết chữ đè lên ảnh gốc bằng OpenCV
            color = (0, 255, 0) if class_name == "person" else (255, 0, 0) # Màu xanh lá cho người, xanh dương cho vật khác
            cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
            
            label_text = f"{class_name} {confidence:.2f}"
            cv2.putText(original_img, label_text, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            drawn_objects += 1

    cv2.imwrite(output_path, original_img)
    print(f"Đã lưu ảnh vẽ kết quả ({drawn_objects} vật thể) vào thư mục: {output_path}")

# --- KHỐI CHẠY DUYỆT THƯ MỤC VÀ DỪNG CHỜ LỆNH ---
if __name__ == "__main__":
    image_dir = "D:\\football\\pascal-voc-2012\\val\\images"
    weight_file = "D:\\football\\voc_resnet50_anchor_best.pt" # Trỏ thẳng vào file bộ não mới của bạn
    
    search_path = os.path.join(image_dir, "*.jpg")
    image_files = glob.glob(search_path)
    
    print(f"Tìm thấy {len(image_files)} hình ảnh trong thư mục Validation.")
    
    for img_path in image_files:
        print(f"\n=======================================================")
        print(f"Đang xử lý: {os.path.basename(img_path)}")
        
        # Gọi hàm dự đoán (Hạ conf_threshold xuống 0.3 vì điểm mAP tổng thể đạt 65.4% là rất mạnh rồi)
        predict_image(img_path, weight_file, output_dir="D:\\football\\predictions_result", conf_threshold=0.3)
        
        # Bắt Terminal dừng lại để bạn mở thư mục kết quả ra xem ảnh
        input("\nNhấn [ENTER] trên bàn phím để chuyển sang ảnh tiếp theo...")
        
    print("\nĐã duyệt và kiểm tra hết tất cả các ảnh thành công!")