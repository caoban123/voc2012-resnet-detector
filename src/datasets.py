import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
import torch

class VOC2012Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.filenames = []
        for filename in os.listdir(img_dir):
            if filename.endswith('.jpg'):
                self.filenames.append(filename[:-4])
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.filenames[idx] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image_name = self.filenames[idx]
        label_path = os.path.join(self.label_dir, self.filenames[idx] + '.txt')

        boxes = []

        class_ids = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        class_ids.append(int(parts[0]))     # class_id
                        boxes.append(parts[1:]) # [x_center, y_center, width, height]
        max_objects = 5
        
        # Tạo mảng trống chứa dữ liệu padding
        padded_boxes = torch.zeros((max_objects, 4), dtype=torch.float32)
        padded_labels = torch.full((max_objects,), -1, dtype=torch.long)
        
        # Điền dữ liệu thực tế vào mảng trống
        num_objects = min(len(boxes), max_objects)
        if num_objects > 0:
            padded_boxes[:num_objects] = torch.tensor(boxes[:num_objects], dtype=torch.float32)
            padded_labels[:num_objects] = torch.tensor(class_ids[:num_objects], dtype=torch.long)

        target = {
            'boxes': padded_boxes,   # Kích thước cố định luôn là [5, 4]
            'labels': padded_labels  # Kích thước cố định luôn là [5]
        }

        if self.transform:
            image = self.transform(image)

        return image, target

if __name__ == "__main__":
    img_dir = "D:\\football\\pascal-voc-2012\\train\\images"
    label_dir = "D:\\football\\pascal-voc-2012\\train\\labels"

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = VOC2012Dataset(img_dir, label_dir, transform)
    print(f"Dataset size: {len(dataset)}")
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Target: {target}")
    print(image)
