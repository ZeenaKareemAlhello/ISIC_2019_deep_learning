# ISIC Skin Lesion Classification with DenseNet121

This project implements a **multi-class skin lesion classification system** using the **ISIC 2019 dataset** and **transfer learning with DenseNet121** in PyTorch.  
The goal is to classify images into 7 skin lesion categories:

- Melanoma
- Nevus
- Basal Cell Carcinoma
- Actinic Keratosis
- Benign Keratosis
- Dermatofibroma
- Vascular Lesion

---

## 📁 Project Structure

project_root/
├── data/
│ └── slice/
│    └── ISIC_SUBSET/
│    ├── images/ # Images
│    ├── train.csv
│    └── val.csv
├── src/
│ ├── modules/
│ │ ├── init.py
│ │ ├── model.py # DenseNetClassifier
│ │ └── dataset.py # Custom Dataset class
│ │ └── trainer.py # 

│ └── scripts/
│ ├── init.py
│ ├── train.py 
│ └── predict.py 
├── notebooks/ # analysis / EDA
├── pyproject.txt
└── README.md
└── predictions_val.csv


---

## ⚡ Features

- **Transfer Learning**: DenseNet121 pretrained on ImageNet
- **Multi-class classification**: 7 skin lesion categories
- **Custom Dataset Loader**: Reads CSV + images
- **Training & Validation**: Modular OOP Trainer class
- **Prediction Script**: Reads validation CSV and outputs predictions + confidence
- **Professional save/load**: Full model state dict with metadata

---

## 🛠 Dependencies

```bash
Pillow
tqdm
black
ipython
jupyter
numpy
pandas
scikit-learn
tensorflow
torch
torchaudio
torchvision



