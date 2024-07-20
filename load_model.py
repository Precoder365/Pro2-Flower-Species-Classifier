import torch
from model import build_model

def load_checkpoint(filepath, gpu=False):
    checkpoint = torch.load(filepath, map_location='cuda' if gpu else 'cpu')

    arch = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']

    model, criterion, optimizer = build_model(arch, hidden_units, learning_rate, gpu)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model, optimizer, criterion


