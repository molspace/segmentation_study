import torch
from torch import nn

n_classes = 2
batch_size = 16
image_size = 256
data_dir = "./data/manual_segmentation"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'UNet'
n_epochs = 30
criterion = nn.CrossEntropyLoss()
lr = 0.0002
random_state = 42
visualize = False
augment = True