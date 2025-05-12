import argparse
import numpy as np

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from shallownet_pytorch.paths import list_images
from shallownet_pytorch.dataset import SimpleDataLoader, SimplePreprocessor, ImageToTensor
from shallownet_pytorch.models.shallownet import ShallowNet

def train():
    """
    Main function to train a model on image data.
    
    Handles:
    - Argument parsing
    - Data loading and preprocessing
    - Data splitting
    - Data conversion to PyTorch tensors
    - Dataset and DataLoader creation
    
    Returns:
        tuple: (train_loader, test_loader) PyTorch DataLoader objects
    """
    # Argument parsing with improved help messages
    ap = argparse.ArgumentParser(
        description="Train a model on image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "-d", "--dataset", 
        required=True, 
        help="Path to input dataset directory containing images"
    )
    
    ap.add_argument(
        "-b", "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    
    ap.add_argument(
        "--test-size", 
        type=float, 
        default=0.25,
        help="Fraction of data to use for testing"
    )
    
    ap.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )

    ap.add_argument(
        "--test-model", 
        type=bool, 
        default=False,
        help="Test model"
    )
    
    args = ap.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    try:
        # Load and preprocess images
        image_paths = list(list_images(dataset_path))
        if not image_paths:
            raise ValueError(f"No images found in dataset directory: {dataset_path}")
            
        print(f"Found {len(image_paths)} images in dataset")
        
        # initialize the image preprocessors
        sp = SimplePreprocessor(32, 32)
        iap = ImageToTensor()

        # load the dataset from disk then scale the raw pixel intensities
        # to the range [0, 1]
        sdl = SimpleDataLoader(preprocessors=[sp, iap])
        data, labels = sdl.load(image_paths, verbose=500)
        data = data.astype("float") / 255.0

        # Split data into training and testing sets
        trainX, testX, trainY, testY = train_test_split(
            data, 
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels  # Maintain class distribution
        )

        # Convert labels to one-hot vectors
        label_encoder = LabelEncoder()
        trainY = label_encoder.fit_transform(trainY)
        testY = label_encoder.transform(testY)  # Use same binarizer

        # Converter dados para tensores PyTorch
        trainX_tensor = torch.tensor(trainX, dtype=torch.float32)#.permute(0,3,1,2)  # de NHWC para NCHW
        trainY_tensor = torch.tensor(trainY, dtype=torch.long)  # labels como inteiros (não one-hot)

        testX_tensor = torch.tensor(testX, dtype=torch.float32)#.permute(0,3,1,2)
        testY_tensor = torch.tensor(testY, dtype=torch.long)

        # Criar datasets e dataloaders
        train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
        test_dataset = TensorDataset(testX_tensor, testY_tensor)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")

        # Model
        model = ShallowNet(width=32, height=32, depth=3, classes=3)

        if args.test_model:
            try:
                # Testar o modelo antes do treinamento
                model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout/batchnorm, se houver)

                # Pegamos um mini-batch do train_loader
                sample_batch = next(iter(train_loader))
                sample_images, _ = sample_batch

                # Passamos pelo modelo
                with torch.no_grad():  # Sem cálculo de gradientes
                    outputs = model(sample_images)

                print(f"Input shape: {sample_images.shape}")
                print(f"Output shape: {outputs.shape}")
            except Exception as e:
                print(f"Error ao testar modelo {e}")
                return
            
        # Escolher dispositivo (GPU se disponível)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Função de perda para classificação multi-classe (categorical cross-entropy)
        criterion = nn.CrossEntropyLoss()
        # Definir otimizador SGD com learning rate 0.005
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        NUM_EPOCHS = 1

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            # Validação
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f} - "
                f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        #Suponha que 'model' seja sua rede neural PyTorch
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    except Exception as e:
       print(f"[ERROR] {e}")

if __name__ == "__main__":
    train()