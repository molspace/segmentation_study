import torch
from torch import nn

batch_size = 32
image_size = 256
data_dir = "./data/manual_segmentation"
device = 'mps' if torch.cuda.is_available() else 'cpu'
model_name = 'UNet'
n_epochs = 10
criterion = nn.CrossEntropyLoss()
lr = 0.0002
random_state = 42