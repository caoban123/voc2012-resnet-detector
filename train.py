import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

from src.datasets import VOC2012Dataset
from src.models import VOC2012Model

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    
    model.train() 
    epochs = 10

    best_acc = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        correct_labels = 0   
        total_labels = 0
        progress_bar = tqdm(dataloader, colour="green")
        
        for iter, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            true_boxes = targets['boxes'].to(device)  
            true_labels = targets['labels'].to(device) 

            optimizer.zero_grad()

            outputs = model(images)
            pred_boxes = outputs['pred_boxes']     
            pred_classes = outputs['pred_classes']

            loss_box = criterion_box(pred_boxes, true_boxes)
            mask = true_labels >= 0
            if mask.sum() > 0:
                loss_class = criterion_class(pred_classes[mask], true_labels[mask])
            else:
                loss_class = 0.0
            total_loss = loss_box + (loss_class * 2.0)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if mask.sum() > 0:
                preds = torch.argmax(pred_classes[mask], dim=1)
                correct_labels += (preds == true_labels[mask]).sum().item()
                total_labels += mask.sum().item()
            
            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Batch Loss: {total_loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = (correct_labels / total_labels) * 100 if total_labels > 0 else 0.0
        print(f" => Epoch [{epoch+1}/{epochs}] Hoàn thành, Loss trung bình: {epoch_loss:.4f}")
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        is_best = False
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            is_best = True
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc 
        }
        
        torch.save(checkpoint, "voc_checkpoint_last.pt")
        if is_best:
            torch.save(checkpoint, "voc_checkpoint_best.pt")

    writer.close()



# Đặt đoạn này ở cuối cùng file train.py của bạn

if __name__ == "__main__":
    print("Starting VOC2012")
    
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