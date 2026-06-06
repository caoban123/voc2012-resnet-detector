# VOC2012 ResNet18 Object Detector

A custom PyTorch implementation for object detection and localization on the PASCAL VOC 2012 dataset using a ResNet-18 backbone. This project is specifically configured to handle dynamic multi-object labels, utilize clean masking strategies to filter out padding/empty slots, track training metrics via TensorBoard, and leverage robust training mechanics (Full Checkpoints & Best/Last weight saving logic).

## 🚀 Project Overview

Unlike rigid, off-the-shelf detection frameworks, this repository demonstrates a custom-built object detection pipeline. It modifies the output layers of a ResNet-18 model to concurrently predict bounding box coordinates and class probabilities for multiple target objects per image.

Key features implemented in the pipeline:
- **Custom Loss Masking:** Dynamically masks out padded target slots (`-1`) to safely compute bounding box adjustments (`MSELoss`) and category scores (`CrossEntropyLoss`) without throwing runtime boundary exceptions.
- **Full Checkpoint Restoration:** Saves full dictionaries containing model state, optimizer states (SGD momentum context), current epoch, and record accuracy. This protects long training workloads from sudden failures or cloud platform session timeouts (e.g., Kaggle's 12-hour limit).
- **Dual Weight Tracking (`last` vs `best`):** Automatically saves continuous updates into `last` state checkpoints while safely archiving historical peaks into `best` state files based on training classification accuracy.

---

## 📂 Project Structure

Your project directory should follow this structural layout:

```text
voc2012-resnet18-detector/
├── src/
│   ├── datasets.py     # Custom Dataset loader for VOC2012 text/image pairs
│   └── models.py       # Modified ResNet-18 model architecture
├── runs/               # TensorBoard event logs (Auto-generated)
├── .gitignore          # File specifying untracked heavy/binary files
├── README.md           # Project documentation and guidelines
└── train.py            # Core execution script for the training cycle
