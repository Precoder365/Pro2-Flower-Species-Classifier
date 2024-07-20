from torchvision import models
from torch import nn, optim

def build_model(model_name, hidden_units, learning_rate, gpu=False):
    model = None

    if model_name == 'efficientnet':
        model = models.efficientnet_b2(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(1408, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
        )
    elif model_name == 'resnet':
        model = models.resnet50(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
    )
    elif model_name == 'vgg':
        model = models.vgg16(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
        )
    else:
        print("Model not recognized. Please choose from 'efficientnet', 'resnet', or 'vgg'.")
        exit()
    

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion, optimizer
