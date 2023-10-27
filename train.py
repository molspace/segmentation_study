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
train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[0.8, 0.2], generator=torch.Generator().manual_seed(random_state)
print(f'Train data len: {len(train_dataset)}, validation data len: {len(valid_dataset)}')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

opt = torch.optim.Adam(model.parameters(), lr=lr)

# Train loop
for epoch in range(n_epochs):
    model.train()
    n_iters = 0
    batch_losses = []

    for batch in tqdm(train_dataloader):

        # unpack batch
        image_batch, mask_batch = batch
        image_batch, mask_batch = image_batch.to(device), mask_batch.to(device)
        
        # forward
        mask_pred_batch = model(image_batch)
        loss = criterion(mask_pred_batch, mask_batch)
        
        # optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # dump statistics
        writer.add_scalar("train/loss", loss.item(), global_step=epoch * len(train_dataloader) + n_iters)
        
        # if n_iters % 50 == 0:
        #     writer.add_image('train/image', make_grid(image_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        #     writer.add_image('train/mask_pred', make_grid(mask_pred_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        #     writer.add_image('train/mask_gt', make_grid(mask_batch) * 0.5 + 0.5, epoch * len(train_dataloader) + n_iters)
        n_iters += 1
    
    
    # validation
    model.eval()
    n_iters = 0
    batch_losses = []
    for batch in valid_dataloader:
        image_batch, mask_batch = batch
        image_batch, mask_batch = image_batch.to(device), mask_batch.to(device)

        mask_pred_batch = model(image_batch)

        loss = criterion(mask_pred_batch, mask_batch)
        batch_losses.append(loss.item())
        
        # if n_iters < 5:
        #     writer.add_image(f'val_{n_iters}/image', make_grid(image_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
        #     writer.add_image(f'val_{n_iters}/mask_pred', make_grid(mask_pred_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
        #     writer.add_image(f'val_{n_iters}/mask_gt', make_grid(mask_batch) * 0.5 + 0.5, epoch * len(valid_dataloader) + n_iters)
        # n_iters += 1
    
    loss_averaged = np.mean(batch_losses)
    writer.add_scalar('val/loss_averaged', loss_averaged.item(), epoch)
    
    iou = dice(mask_pred_batch, mask_batch, average='micro')
    print(f"Epoch {epoch} validation iou {iou}.")






# def UnetLoss(preds, targets):
#     ce_loss = ce(preds, targets)
#     acc = (torch.max(preds, 1)[1] == targets).float().mean()
#     return ce_loss, acc


# class engine():
#   def train_batch(model, data, optimizer, criterion):
#       model.train()

#       ims, ce_masks = data
#       _masks = model(ims)
#       optimizer.zero_grad()

#       loss, acc = criterion(_masks, ce_masks)
#       loss.backward()
#       optimizer.step()

#       return loss.item(), acc.item()


#   @torch.no_grad()
#   def validate_batch(model, data, criterion):
#       model.eval()
#       ims, masks = data
#       _masks = model(ims)
#       loss, acc = criterion(_masks, masks)
#       return loss.item(), acc.item()


# def make_model():
#   model = UNet().to(config.DEVICE)
#   criterion = UnetLoss
#   optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
#   return model, criterion, optimizer


# model, criterion, optimizer = make_model()


# def run():
#   for epoch in range(config.N_EPOCHS):
#       print("####################")
#       print(f"       Epoch: {epoch}   ")
#       print("####################")

#       for bx, data in tqdm(enumerate(trn_dl), total = len(trn_dl)):
#           train_loss, train_acc = engine.train_batch(model, data, optimizer, criterion)

#       for bx, data in tqdm(enumerate(val_dl), total = len(val_dl)):
#           val_loss, val_acc = engine.validate_batch(model, data, criterion)

#       wandb.log(
#           {   
#               'epoch': epoch,
#               'train_loss': train_loss,
#               'train_acc': train_acc,
#               'val_loss': val_loss,
#               'val_acc': val_acc
#           }
#       )

#       print()
