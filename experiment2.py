# Training on padcchest data,testing on padchest and chexpert

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18, efficientnet_b0
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
from tqdm import tqdm 
from PIL import Image

# custom dataset class
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# data load 
dataset = PadChestDataset(df, image_dir=image_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

images, labels = next(iter(loader))
print("Batch shape:", images.shape)
print("Batch labels:", labels[:5])

def cache_valid_indices_and_labels(dataset):
    pos_indices = []
    neg_indices = []

    for i in tqdm(range(len(dataset)), desc="Caching labels"):
        row = dataset.df.iloc[i]
        img_path = os.path.join(dataset.image_dir, row['ImageID'])

        # Skip if image does not exist or cannot be opened
        if not os.path.exists(img_path):
            continue

        try:
            Image.open(img_path).verify()  # Very fast check without full load
        except (UnidentifiedImageError, OSError):
            continue

        label = torch.tensor(row['no_findings'], dtype=torch.float32)
        if label > 0:
            neg_indices.append(i)  # no findings = negative
        else:
            pos_indices.append(i)  # any finding = positive

    return pos_indices, neg_indices

# 50/50 split
def get_balanced_binary_datasets_fast(dataset, train_ratio=0.7, val_ratio=0.15):
    print("\nCreating balanced binary datasets (fast)...")
    pos_indices, neg_indices = cache_valid_indices_and_labels(dataset)

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

# specific disease -> findings 
def binary_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    binary_labels = torch.stack([(label.sum() > 0).float().unsqueeze(0) for label in labels])
    return images, binary_labels

# training
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if images.size(0) == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = (outputs > 0.5).float()
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

# testing
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_scores.append(outputs.cpu())

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

# Main function
def main():
    # Assuming the `df` dataframe and `image_dir` are defined correctly
    # For example, loading your dataframe might look like:
    # df = pd.read_csv('path_to_your_csv_file.csv') 
    # image_dir = '/path/to/images'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Initialize PadChestDataset with the correct arguments
    dataset = PadChestDataset(dataframe=df, image_dir=image_dir, transform=transform)

    # Split dataset into train, validation, and test sets
    # You can create a custom split or use sklearn.model_selection.train_test_split
    train_dataset, val_dataset, test_dataset = get_balanced_binary_datasets_fast(dataset)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ResNet18 model and modify the final layer for binary classification
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device)

    # Plot training curves
    plot_training_curves(train_losses, val_losses)

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'padchest_binary_model.pth')

    # Evaluate the model on the test set
    metrics = evaluate_model(model, test_loader, device)
    with open('binary_evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == '__main__':
    main()


# ---- testing on chexpert

# Define disease labels
disease_labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# Set pandas option to avoid warnings
pd.set_option('future.no_silent_downcasting', True)
base_dir = '/shared/home/nas6781/'

# custom dataset
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.ToTensor()
        self.disease_labels = disease_labels

    def __len__(self):
        # return len(self.valid_indices) if self.valid_indices is not None else len(self.data)
        return len(self.data)

    def __getitem__(self, idx):
        # if self.valid_indices is not None:
        #     data_idx = self.valid_indices[idx]
        # else:
        #     data_idx = idx
    
        # row = self.data.iloc[data_idx]
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

# transformation (preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def cache_valid_indices_and_labels_chexpert(dataset):
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

# with caching to speed up
def get_balanced_binary_datasets_fast(dataset, train_ratio=0.7, val_ratio=0.15):
    print("\nCreating balanced binary datasets (fast)...")
    pos_indices, neg_indices = cache_valid_indices_and_labels_chexpert(dataset)

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

# data from chexpert
def load_chexpert_data():
    # Load CheXpert dataset
    df = pd.read_csv('/shared/home/nas6781/CheXpert-v1.0-small/train.csv')
    
    # Define disease labels
    disease_labels = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    # Convert labels to binary (any finding vs no finding)
    df['no_findings'] = df[disease_labels].apply(
        lambda x: 1 if all(v == 0 for v in x) else 0, axis=1
    )
    
    return df
    
# padchest model
def load_padchest_model(model_path, device):
    # Initialize model architecture
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# testing 
def evaluate_on_chexpert(model, test_loader, device):
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

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auroc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (PadChest on CheXpert)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('padchest_on_chexpert_roc.png')
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Finding', 'Finding'],
                yticklabels=['No Finding', 'Finding'])
    plt.title('Confusion Matrix (PadChest on CheXpert)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('padchest_on_chexpert_cm.png')
    plt.close()

    return metrics

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
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading CheXpert dataset...")
    chexpert_dataset = CheXpertDataset(
        csv_file='/shared/home/nas6781/CheXpert-v1.0-small/train.csv',
        transform=transform,
        validate_files=True
    )
    print(f"Loaded {len(chexpert_dataset)} CheXpert images")


    print("Creating balanced test dataset...")
    _, _, test_dataset = get_balanced_binary_datasets_fast(chexpert_dataset)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2,
        collate_fn=binary_collate_fn
    )
    print(f"Created test dataset with {len(test_dataset)} images")
    print("Loading PadChest model...")
    model = load_padchest_model('efficientnet_binary_model.pth', device)

    print("Evaluating model...")
    metrics = evaluate_on_chexpert(model, test_loader, device)
    with open('padchest_on_chexpert_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation Results (PadChest on CheXpert):")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    main() 

