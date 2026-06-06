import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from src.models import VOC2012Model

# Class names mapping based on PASCAL VOC 2012 standards
# Focus tracking index 14 for 'person' (players)
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def predict_image(image_path, checkpoint_path, output_path="result.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model architecture and weights
    model = VOC2012Model(num_classes=20, max_objects=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")

    # 2. Read and preprocess the input image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Cannot read image from {image_path}")
        return

    orig_h, orig_w, _ = original_img.shape

    # Convert BGR to RGB for PyTorch transforms
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    transform = Compose([
        ToTensor(),         
        Resize((224, 224))  
    ])
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_boxes = outputs['pred_boxes'].squeeze(0).cpu().numpy()     # Shape: [max_objects, 4]
        pred_classes = outputs['pred_classes'].squeeze(0).cpu().numpy() # Shape: [max_objects, 20]

    print("\n--- Detection Results ---")
    # 4. Draw predicted bounding boxes onto the original image
    for i in range(pred_boxes.shape[0]):
        box = pred_boxes[i]
        class_logits = pred_classes[i]
        
        # Get the class with highest probability
        class_idx = np.argmax(class_logits)
        confidence = np.max(torch.softmax(torch.tensor(class_logits), dim=0).numpy())

        # Filter out low confidence detections (Threshold: 0.5)
        if confidence < 0.95:
            continue

        class_name = CLASS_NAMES[class_idx]
        
        # Denormalize coordinates back to original image size
        # Assuming model outputs normalized coordinates [xmin, ymin, xmax, ymax] between 0 and 1
        xmin = int(box[0] * orig_w)
        ymin = int(box[1] * orig_h)
        xmax = int(box[2] * orig_w)
        ymax = int(box[3] * orig_h)

        print(f"Object {i+1}: {class_name} | Conf: {confidence:.2f} | Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        # Select color: Green for players (person), Blue for others
        color = (0, 255, 0) if class_name == "person" else (255, 0, 0)

        # Draw Bounding Box
        cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw Text Label
        label_text = f"{class_name} {confidence:.2f}"
        cv2.putText(original_img, label_text, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Save the output visualized image
    cv2.imwrite(output_path, original_img)
    print(f"\nSaved visualization result to: {output_path}")

if __name__ == "__main__":
    # Test execution configuration
    test_img = "D:\\football\\pascal-voc-2012\\train\\images\\VOC2007_000005.jpg" 
    weight_file = "D:\\football\\voc_checkpoint_best.pt" # Put your downloaded weight file here
    
    predict_image(test_img, weight_file)