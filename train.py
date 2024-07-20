from transforms import transform_data
from model import build_model

import torch
import numpy as np
from tqdm import tqdm
import argparse

import os

def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, num_epochs, gpu=False):
    print("Using model architecture:", arch)
    print("Number of hidden units:", hidden_units)
    print("Learning rate:", learning_rate)
    print("Number of epochs:", num_epochs)
    if gpu:
        if torch.cuda.is_available():
            print("Using GPU for training.")
        else:
            print("GPU is not available. Using CPU for training.")
            gpu = False
    else:
        if torch.cuda.is_available():
            print("GPU is available but not selected for training. To use GPU, set the --gpu flag.")
        print("Using CPU for training.")

    train_dir = f'{data_dir}/train'
    valid_dir = f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'

    train_loader, valid_loader, test_loader, train_data = transform_data(train_dir, valid_dir, test_dir)

    model, criterion, optimizer = build_model(arch, hidden_units, learning_rate, gpu)

    train_losses, valid_losses = [], []
    patience = 5
    early_stop_counter = 0
    best_valid_loss = np.Inf
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in train_loader_tqdm:
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        valid_loader_tqdm = tqdm(valid_loader, total=len(valid_loader), desc="Validating")

        with torch.no_grad():
            for inputs, labels in valid_loader_tqdm:
                if gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                valid_loader_tqdm.set_postfix(loss=running_loss / len(valid_loader))

        valid_loss = running_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs.')
            break

    # On test

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc="Testing")

    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader_tqdm.set_postfix(loss=test_loss / len(test_loader), accuracy=100 * correct / total)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'architecture': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.fc if arch == 'resnet' else model.classifier,
        'class_to_idx': model.class_to_idx,
        'epochs': num_epochs,
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, f'{save_dir}/checkpoint-{arch}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the data.')
    parser.add_argument('--save_dir', type=str, default='checkpoints_folder', help='Path to the directory to save the model checkpoint.')
    parser.add_argument('--arch', type=str, default='efficientnet', help='Model architecture to use.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
