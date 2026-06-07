import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from src.models import VOC2012Model

CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def cxcywh_to_xyxy_numpy(boxes, orig_w, orig_h):
    # Convert normalized cxcywh to absolute xyxy pixels based on original image size
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xmin = (cx - w / 2) * orig_w
    ymin = (cy - h / 2) * orig_h
    xmax = (cx + w / 2) * orig_w
    ymax = (cy + h / 2) * orig_h
    return np.stack([xmin, ymin, xmax, ymax], axis=1).astype(int)

def predict_image(image_path, checkpoint_path, output_path="result.jpg", conf_threshold=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = VOC2012Model(num_classes=20, max_objects=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")

    # 2. Read and Preprocess Image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    orig_h, orig_w, _ = original_img.shape

    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_boxes = outputs['pred_boxes'].squeeze(0).cpu().numpy()     # [max_objects, 4]
        pred_classes = outputs['pred_classes'].squeeze(0).cpu().numpy() # [max_objects, 20]

    # 4. Convert all predictions from cxcywh to xyxy based on original image dimensions
    pred_boxes_xyxy = cxcywh_to_xyxy_numpy(pred_boxes, orig_w, orig_h)

    print("\n--- Detection Results ---")
    drawn_objects = 0
    
    for i in range(pred_boxes.shape[0]):
        class_logits = pred_classes[i]
        class_idx = np.argmax(class_logits)
        
        # Calculate Softmax Confidence
        exp_logits = np.exp(class_logits - np.max(class_logits))
        scores = exp_logits / np.sum(exp_logits)
        confidence = scores[class_idx]

        # Filter out low confidence bounding boxes
        if confidence < conf_threshold:
            continue

        class_name = CLASS_NAMES[class_idx]
        box = pred_boxes_xyxy[i]
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        # Bound check constraints to stay within image boundaries
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(orig_w, xmax), min(orig_h, ymax)

        print(f"Object {drawn_objects+1}: {class_name} | Conf: {confidence:.2f} | Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
        cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        label_text = f"{class_name} {confidence:.2f}"
        cv2.putText(original_img, label_text, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        drawn_objects += 1

    cv2.imwrite(output_path, original_img)
    print(f"\nSaved visualization result ({drawn_objects} objects) to: {output_path}")

# if __name__ == "__main__":
#     test_img = "D:\\football\\pascal-voc-2012\\val\\images\\VOC2007_000008.jpg"       
#     weight_file = "D:\\football\\voc_checkpoint_best.pt" 
    
#     predict_image(test_img, weight_file, conf_threshold=0.99)
import glob
import os

if __name__ == "__main__":
    image_dir = "D:\\football\\pascal-voc-2012\\val\\images"
    weight_file = "D:\\football\\voc_checkpoint_best.pt"
    
    search_path = os.path.join(image_dir, "*.jpg")
    image_files = glob.glob(search_path)
    
    print(f"Tìm thấy {len(image_files)} hình ảnh.")
    
    for img_path in image_files:
        print(f"\nĐang xử lý: {os.path.basename(img_path)}")
        
        # Chạy dự đoán và hiển thị ảnh
        predict_image(img_path, weight_file, conf_threshold=0.99)
        
        # Dừng chương trình ở Terminal, chờ bạn nhấn Enter
        input("Nhấn [ENTER] trên bàn phím để chuyển sang ảnh tiếp theo...")
        
    print("Đã xem hết tất cả các ảnh!")