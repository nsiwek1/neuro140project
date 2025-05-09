# This code preprocesses CheXpert data, splits them into datasets and trains and validates a CheXpert model that is later evaluated.
# Later this model (that didn't see Padchest data) is tested on those data

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

# disease labels
disease_labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


pd.set_option('future.no_silent_downcasting', True)
base_dir = '/shared/home/nas6781/'

# chexpert class
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, transform=None, validate_files=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.ToTensor()
        self.disease_labels = disease_labels
        self.valid_indices = None

        if validate_files:
            self.valid_indices = self.validate_dataset()

    def validate_dataset(self):
        """Pre-filter dataset to only include valid images with labels"""
        print("Validating dataset files and labels...")
        valid_indices = []

        for idx, row in self.data.iterrows():
            if idx % 1000 == 0:
                print(f"Validated {idx}/{len(self.data)} entries...")

            image_path = row['Path']

            if image_path.startswith('._'):
                continue

            full_path = os.path.join(base_dir, image_path)

            if not os.path.isfile(full_path):
                continue

            labels = row[disease_labels].fillna(0).replace(-1, 0)
            if labels.isnull().all():
                continue

            valid_indices.append(idx)

        print(f"Found {len(valid_indices)} valid images out of {len(self.data)} entries")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices) if self.valid_indices is not None else len(self.data)

    def __getitem__(self, idx):
        if self.valid_indices is not None:
            data_idx = self.valid_indices[idx]
        else:
            data_idx = idx
    
        row = self.data.iloc[data_idx]
        image_path = row['Path']
        full_path = os.path.join(base_dir, image_path)
    
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a black placeholder
            return torch.zeros(3, 224, 224), torch.zeros(len(self.disease_labels))
    
        # Force resize BEFORE any other transform
        image = transforms.Resize((224, 224))(image)
        
        if self.transform:
            image = self.transform(image)
    
        labels = row[disease_labels].fillna(0).replace(-1, 0)
        labels = labels.infer_objects(copy=False).astype(float)
        return image, torch.FloatTensor(labels.values)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#50/50
def get_balanced_binary_datasets(dataset, train_ratio=0.7, val_ratio=0.15):
    
    print("\nCreating balanced binary datasets...")

    pos_indices = []
    neg_indices = []

    for i in range(len(dataset)):
        _, labels = dataset[i]

        if torch.sum(labels) > 0:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    print(f"Found {len(pos_indices)} positive samples and {len(neg_indices)} negative samples")

    min_count = min(len(pos_indices), len(neg_indices))
    balanced_indices = pos_indices[:min_count] + neg_indices[:min_count]
    print(f"Using {len(balanced_indices)} balanced samples ({min_count} per class)")

    np.random.shuffle(balanced_indices)

    total = len(balanced_indices)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size:train_size+val_size]
    test_indices = balanced_indices[train_size+val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Split into {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    return train_dataset, val_dataset, test_dataset

def binary_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    binary_labels = torch.stack([(label.sum() > 0).float().unsqueeze(0) for label in labels])
    return images, binary_labels


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json
from torchvision import transforms
from PIL import Image


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            if images.size(0) == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss, val_batches = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                if images.size(0) == 0:
                    continue

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_scores.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_scores = torch.cat(all_scores).numpy()

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auroc': roc_auc_score(all_labels, all_scores)
    }

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auroc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Finding', 'Finding'],
                yticklabels=['No Finding', 'Finding'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return metrics

def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = CheXpertDataset(csv_file='/shared/home/nas6781/CheXpert-v1.0-small/train.csv', transform=None, validate_files=True)
    train_dataset, val_dataset, test_dataset = get_balanced_binary_datasets(full_dataset)

    # transofmrations
    train_dataset.transform = train_transform
    val_dataset.transform = val_test_transform
    test_dataset.transform = val_test_transform

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=binary_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=binary_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=binary_collate_fn)
    # training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # No Sigmoid inside model!
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    plot_training_curves(train_losses, val_losses)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'chest_xray_binary_model.pth')

    metrics = evaluate_model(model, test_loader, device)
    with open('binary_evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == '__main__':
    main()



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json
from torchvision.models import resnet18
import pandas as pd
import ast
import os
import pandas as pd
import ast

#padchest data
df = pd.read_csv('/shared/home/nas6781/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
df = df.dropna(subset=['Labels', 'ImageID'])
df['ImageID'] = df['ImageID'].astype(str)
df['ImageID'] = df['ImageID'].str.replace('.dicom', '.png')
df['ImageID'] = df['ImageID'].str.replace('.dcm', '.png')
def safe_eval(val):
    try:
        return ast.literal_eval(str(val))
    except:
        return []

df['Labels'] = df['Labels'].apply(safe_eval)

df['no_findings'] = df['Labels'].apply(
    lambda x: 1 if len(x) == 1 and isinstance(x[0], str) and x[0].strip().lower() == 'normal' else 0
)
#dataset
class PadChestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['ImageID'])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['no_findings'], dtype=torch.float32)
        return image, label
from torchvision import transforms

from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = PadChestDataset(df, image_dir=image_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

images, labels = next(iter(loader))
print("Batch shape:", images.shape)
print("Batch labels:", labels[:5])


import os

image_dir = '/shared/home/nas6781/Padchest'
existing_images = set(os.listdir(image_dir))

df = df[df['ImageID'].isin(existing_images)]

def load_padchest_data():
    """Load and preprocess PadChest dataset"""
    print("Loading PadChest data...")
    df = pd.read_csv('/shared/home/nas6781/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    df = df.dropna(subset=['Labels', 'ImageID'])
    df['ImageID'] = df['ImageID'].astype(str)
    df['ImageID'] = df['ImageID'].str.replace('.dicom', '.png')
    df['ImageID'] = df['ImageID'].str.replace('.dcm', '.png')
    
    def safe_eval(val):
        try:
            return ast.literal_eval(str(val))
        except:
            return []
    
    df['Labels'] = df['Labels'].apply(safe_eval)
    
    # Create binary label for no findings
    df['no_findings'] = df['Labels'].apply(
        lambda x: 1 if len(x) == 1 and isinstance(x[0], str) and x[0].strip().lower() == 'normal' else 0
    )
    
    # Filter for existing images
    image_dir = '/shared/home/nas6781/Padchest'
    existing_images = set(os.listdir(image_dir))
    df = df[df['ImageID'].isin(existing_images)]
    
    print(f"Loaded {len(df)} PadChest images")
    return df
# pretrained model
def load_chexpert_model(model_path, device):
    """Load trained CheXpert model"""
    print("Loading CheXpert model...")
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    return model

def plot_roc_curve(labels, scores, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (CheXpert on PadChest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Finding', 'Finding'],
                yticklabels=['No Finding', 'Finding'])
    plt.title('Confusion Matrix (CheXpert on PadChest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()

def plot_metrics_bar(metrics, save_path):
    """Plot metrics as a bar chart"""
    plt.figure(figsize=(10, 6))
    metrics_names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = plt.bar(metrics_names, values)
    plt.title('Performance Metrics (CheXpert on PadChest)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_on_padchest(model, test_loader, device):
    """Evaluate model on PadChest test set"""
    print("Evaluating model on PadChest...")
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_scores.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_scores = torch.cat(all_scores).numpy()

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auroc': roc_auc_score(all_labels, all_scores)
    }

    # Create plots
    os.makedirs('evaluation_plots', exist_ok=True)
    plot_roc_curve(all_labels, all_scores, 'evaluation_plots/roc_curve.png')
    plot_confusion_matrix(all_labels, all_preds, 'evaluation_plots/confusion_matrix.png')
    plot_metrics_bar(metrics, 'evaluation_plots/metrics.png')

    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data and model
    df = load_padchest_data()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset
    padchest_dataset = PadChestDataset(
        dataframe=df,
        image_dir='/shared/home/nas6781/Padchest',
        transform=transform
    )

    # Get test dataset
    _, _, test_dataset = get_balanced_binary_datasets_fast(padchest_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=binary_collate_fn
    )

    # Load CheXpert model
    model = load_chexpert_model('chest_xray_binary_model.pth', device)

    # Evaluate
    metrics = evaluate_on_padchest(model, test_loader, device)

    # Save results
    with open('chexpert_on_padchest_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\nEvaluation Results (CheXpert on PadChest):")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    main() 
