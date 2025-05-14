import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, efficientnet_b0
import numpy as np
from PIL import UnidentifiedImageError
from tqdm import tqdm  
import pandas as pd
import ast
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json


# disease labels
disease_labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

pd.set_option('future.no_silent_downcasting', True)
base_dir = '/shared/home/nas6781/'

# custom dataset for chexpert
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, transform=None, validate_files=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.ToTensor()
        self.disease_labels = disease_labels

    def __len__(self):
        # return len(self.valid_indices) if self.valid_indices is not None else len(self.data)
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        image_path = row['Path']
        full_path = os.path.join(base_dir, image_path)
    
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a black placeholder
            return torch.zeros(3, 224, 224), torch.zeros(len(self.disease_labels))
    
        # resize
        image = transforms.Resize((224, 224))(image)
        
        if self.transform:
            image = self.transform(image)
    
        labels = row[disease_labels].fillna(0).replace(-1, 0)
        labels = labels.infer_objects(copy=False).astype(float)
        return image, torch.FloatTensor(labels.values)

#  Transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# specific diseases -> binary
def binary_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    binary_labels = torch.stack([(label.sum() > 0).float().unsqueeze(0) for label in labels])
    return images, binary_labels



# disease labels adjusting 
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

image_dir = '/shared/home/nas6781/Padchest'
existing_images = set(os.listdir(image_dir))

df = df[df['ImageID'].isin(existing_images)]



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


# caching for faster balancing
def cache_valid_indices_and_labels_combined(dataset):
    pos_indices = []
    neg_indices = []

    for i in tqdm(range(len(dataset)), desc="Caching labels"):
        try:
            _, labels = dataset[i]
            if torch.sum(labels) > 0:
                pos_indices.append(i)
            else:
                neg_indices.append(i)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    return pos_indices, neg_indices
# 50/50 splits
def get_balanced_binary_datasets_fast(dataset, train_ratio=0.7, val_ratio=0.15):
    print("\nCreating balanced binary datasets (fast)...")
    pos_indices, neg_indices = cache_valid_indices_and_labels_combined(dataset)

    print(f"Found {len(pos_indices)} positive and {len(neg_indices)} negative samples")
    min_count = min(len(pos_indices), len(neg_indices))
    balanced_indices = pos_indices[:min_count] + neg_indices[:min_count]
    np.random.shuffle(balanced_indices)

    total = len(balanced_indices)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size:train_size + val_size]
    test_indices = balanced_indices[train_size + val_size:]

    return (Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices))

# training
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.squeeze(1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels.squeeze(1))
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

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

# evaluation
def evaluate_model(model, test_loader, device, dataset_name):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            outputs = model(images).squeeze(1)
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

    #  ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auroc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve (Combined Model on {dataset_name})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'combined_model_{dataset_name.lower()}_roc.png')
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Finding', 'Finding'],
                yticklabels=['No Finding', 'Finding'])
    plt.title(f'Confusion Matrix (Combined Model on {dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'combined_model_{dataset_name.lower()}_cm.png')
    plt.close()

    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # chexpert dataset
    print("Loading CheXpert dataset...")
    chexpert_dataset = CheXpertDataset(
        csv_file='/shared/home/nas6781/CheXpert-v1.0-small/train.csv',
        transform=transform,
        validate_files=True
    )
    print(f"Loaded {len(chexpert_dataset)} CheXpert images")

    # padchest dataset
    print("Loading PadChest dataset...")
    padchest_dataset = PadChestDataset(
        dataframe=df,  
        image_dir='/shared/home/nas6781/Padchest',
        transform=transform
    )
    print(f"Loaded {len(padchest_dataset)} PadChest images")

    # combine balanced
    combined_dataset = ConcatDataset([chexpert_dataset, padchest_dataset])
    print(f"Combined dataset size: {len(combined_dataset)}")

    # splits
    print("Creating balanced splits...")
    train_dataset, val_dataset, test_dataset = get_balanced_binary_datasets_from_each(chexpert_dataset, padchest_dataset)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=binary_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=binary_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=binary_collate_fn
    )

    # Initialize model
    print("Initializing model...")
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Train model
    print("Training model...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'combined_model.pth')

    # Create separate test loaders for each dataset
    chexpert_test_loader = DataLoader(
        Subset(chexpert_dataset, range(len(chexpert_dataset))),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=binary_collate_fn
    )
    padchest_test_loader = DataLoader(
        Subset(padchest_dataset, range(len(padchest_dataset))),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=binary_collate_fn
    )

    # Evaluate on both datasets
    print("\nEvaluating on CheXpert...")
    chexpert_metrics = evaluate_model(model, chexpert_test_loader, device, "CheXpert")
    with open('combined_model_chexpert_results.json', 'w') as f:
        json.dump(chexpert_metrics, f, indent=2)

    print("\nEvaluating on PadChest...")
    padchest_metrics = evaluate_model(model, padchest_test_loader, device, "PadChest")
    with open('combined_model_padchest_results.json', 'w') as f:
        json.dump(padchest_metrics, f, indent=2)

    # Print results
    print("\nResults on CheXpert:")
    for metric, value in chexpert_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nResults on PadChest:")
    for metric, value in padchest_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    main() 
