import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MalnutritionDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

class ViTMalnutritionModel:
    def __init__(self, target_variable='malnutrition', batch_size=32, learning_rate=1e-4, num_epochs=20):
        self.target_variable = target_variable
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ViT model and processor
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.threshold = 0.5  # Default threshold, will be tuned

    def compute_metrics(self, loader, phase='train'):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # Probability of positive class
                preds = (probs > self.threshold).long()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / len(loader)
        
        print(f'\n{phase.capitalize()} metrics:')
        print(f'Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC-ROC: {auc:.4f}')
        
        # Save confusion matrix for validation and test phases
        if phase in ['validation', 'test']:
            plot_confusion_matrix(
                all_labels, 
                all_preds,
                os.path.join('meron2', 'modeling', 'experiments', f'{phase}_confusion_matrix.png')
            )
        
        return avg_loss, accuracy, precision, recall, f1, auc, all_probs, all_labels

    def tune_threshold(self, probabilities, true_labels):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (probabilities > threshold).astype(int)
            f1 = f1_score(true_labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        return best_threshold

    def load_data(self):
        # Load metadata
        metadata_path = os.path.join('data', 'processed', 'malnutrition_flags.csv')
        df = pd.read_csv(metadata_path)
        print(f"Loaded {len(df)} records from metadata")
        
        # Get image paths
        image_dir = os.path.join('data', 'cropped_pictures')
        image_paths = []
        labels = []
        missing_images = []
        
        for photo_id in df['photo_id']:
            base_id = photo_id.replace('.jpg', '')
            image_path = os.path.join(image_dir, f"{base_id}.jpg")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(df[df['photo_id'] == photo_id][self.target_variable].values[0])
            else:
                missing_images.append(photo_id)
        
        print(f"Found {len(image_paths)} matching images")
        print(f"Missing {len(missing_images)} images")
        
        # Print class distribution
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} ({count/len(labels)*100:.2f}%)")
        
        if not image_paths:
            raise ValueError("No matching images found. Please check the paths and photo IDs.")
        
        # Compute class weights with stronger penalty for minority class
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        # Increase weight for minority class
        class_weights[1] *= 2.0  # Assuming 1 is the minority class
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            val_paths, val_labels, test_size=0.5, random_state=42, stratify=val_labels
        )
        
        print(f"\nSplit sizes:")
        print(f"Training set: {len(train_paths)}")
        print(f"Validation set: {len(val_paths)}")
        print(f"Test set: {len(test_paths)}")
        
        # Create datasets
        self.train_dataset = MalnutritionDataset(train_paths, train_labels, self.processor)
        self.val_dataset = MalnutritionDataset(val_paths, val_labels, self.processor)
        self.test_dataset = MalnutritionDataset(test_paths, test_labels, self.processor)
        
        # Create weighted sampler for training set
        train_labels = np.array(train_labels)
        class_sample_count = np.array([len(train_labels[train_labels == t]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.model.train()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=5)
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 20)
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Training')
            
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Compute metrics
            train_metrics = self.compute_metrics(self.train_loader, 'train')
            val_metrics = self.compute_metrics(self.val_loader, 'validation')
            
            train_loss = train_metrics[0]
            val_loss = val_metrics[0]
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Tune threshold using validation set
            self.threshold = self.tune_threshold(val_metrics[6], val_metrics[7])
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            early_stopping(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join('meron2', 'modeling', 'models', 'vit_malnutrition_model_best.pt'))
            
            if early_stopping.early_stop:
                print("\nEarly stopping triggered")
                break
        
        return train_losses, val_losses

    def evaluate(self):
        best_model_path = os.path.join('meron2', 'modeling', 'models', 'vit_malnutrition_model_best.pt')
        self.model.load_state_dict(torch.load(best_model_path))
        
        print("\nFinal Test Results:")
        print('-' * 20)
        metrics = self.compute_metrics(self.test_loader, 'test')
        
        return metrics[1:6]  # Return accuracy, precision, recall, f1, auc

    def plot_training_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.savefig(os.path.join('meron2', 'modeling', 'experiments', 'vit_training_curve.png'))
        plt.close()

def main():
    model = ViTMalnutritionModel(
        target_variable='malnutrition',
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=20
    )
    
    model.load_data()
    train_losses, val_losses = model.train()
    accuracy, precision, recall, f1, auc = model.evaluate()
    model.plot_training_curve(train_losses, val_losses)

if __name__ == '__main__':
    main() 