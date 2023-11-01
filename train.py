import os
import torch
import torch.nn as nn

import numpy as np
from model import UNet
from prepare_data import dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
from ultralytics import YOLO
from torchmetrics.functional.classification import dice

from configs import model_name, n_epochs, criterion, lr, device, batch_size, random_state



if model_name == 'UNet':
    model = UNet(n_base_channels=64).to(device)
elif model_name == 'DeepLabv3':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
elif model_name == 'YOLOv8':
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
elif model_name == 'sam':
    model = sam_model_registry["vit_h"](checkpoint="<path/to/checkpoint>")



# Setting up SummaryWriter for TensorBoard:
experiment_name = "{}@{}".format(model_name, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
print('experiment_name:', experiment_name)
writer = SummaryWriter(log_dir=os.path.join("./tb", experiment_name))


# Train and val splits
train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[0.8, 0.2], generator=torch.Generator().manual_seed(random_state))
print(f'Train data len: {len(train_dataset)}, validation data len: {len(valid_dataset)}')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

opt = torch.optim.Adam(model.parameters(), lr=lr)

# Train loop
for epoch in range(n_epochs):
    model.train()
    n_iters = 0
    batch_dices_train = []

    for batch in tqdm(train_dataloader):

        # unpack batch
        image_batch, mask_batch = batch
        image_batch, mask_batch = image_batch.to(device), mask_batch.to(device)
        
        # forward
        mask_pred_batch = model(image_batch)
        loss = criterion(mask_pred_batch, mask_batch)
        dice_score = dice(mask_pred_batch, mask_batch, average='micro') 
        batch_dices_train.append(dice_score.item())

        # optimize
        opt.zero_grad()
        # calculate gradients - backward pass
        loss.backward()
        # update weights
        opt.step()
        
        # dump statistics
        writer.add_scalar("train/loss", loss.item(), global_step=epoch * len(train_dataloader) + n_iters)
        writer.add_scalar("train/dice", dice_score.item(), global_step=epoch * len(train_dataloader) + n_iters)
        
        # if n_iters % 50 == 0:
        #     writer.add_image('train/image', make_grid(image_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        #     writer.add_image('train/mask_pred', make_grid(mask_pred_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        #     writer.add_image('train/mask_gt', make_grid(mask_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        n_iters += 1

    dice_averaged = np.mean(batch_dices_train)
    print(f"Epoch {epoch} train dice {dice_averaged}.")
    
    # validation
    with torch.no_grad():
        model.eval()
        n_iters = 0
        batch_losses = []
        batch_dices_val = []
        for batch in valid_dataloader:
            image_batch, mask_batch = batch
            image_batch, mask_batch = image_batch.to(device), mask_batch.to(device)

            mask_pred_batch = model(image_batch)

            loss = criterion(mask_pred_batch, mask_batch)
            batch_losses.append(loss.item())
            
            dice_score = dice(mask_pred_batch, mask_batch, average='micro')
            batch_dices_val.append(dice_score.item())
            
            # if n_iters < 5:
            #     writer.add_image(f'val_{n_iters}/image', make_grid(image_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
            #     writer.add_image(f'val_{n_iters}/mask_pred', make_grid(mask_pred_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
            #     writer.add_image(f'val_{n_iters}/mask_gt', make_grid(mask_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
            # n_iters += 1
    
    loss_averaged = np.mean(batch_losses)
    writer.add_scalar('val/loss_averaged', loss_averaged.item(), epoch)

    dice_averaged = np.mean(batch_dices_val)
    writer.add_scalar('val/dice_averaged', dice_averaged.item(), epoch)
    
    print(f"Epoch {epoch} validation dice {dice_averaged}.")