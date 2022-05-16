import typing as t

from sklearn.utils import shuffle
from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils import DEVICE, create_fpn_label, synthesize_data, normalize, unnormalize, unnormalize_tensor, to_corners, calc_iou_array
from losses import modulated_loss, kfiou_loss
from models.StarModel import StarModel

# Dataloader
# Normalizes the label
class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000, has_star=True, mtype='localizer'):
        self.data_size = data_size
        self.has_star = has_star
        self.mtype=mtype

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=self.has_star)
        if self.mtype == 'fpn':
            label = create_fpn_label(label)
        else:
            label = normalize(label)
        return image[None], label

def train(model: StarModel, dl: StarDataset, num_epochs: int, optimizer, loss_fn, scheduler, regressor=True) -> StarModel:
    #Initiate model for training (Batchnorm and dropout)
    model.train()
    # Get model summary
    summary(model, (1,200,200))
    # Initialize total loss and iou

    for epoch in range(1,num_epochs+1):
        print(f"EPOCH: {epoch}")
        if epoch % 10 == 0:
            fn = 'model_' + str(epoch) + '.pickle'
            torch.save(model.state_dict(), fn)
        epoch_losses = []
        epoch_ious = []
        for image, labels in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(image)
            if regressor:
                loss = loss_fn(
                    preds,
                    labels,
                    unnormalize_tensor(preds).to(DEVICE),
                    unnormalize_tensor(labels).to(DEVICE)
                )
                loss = torch.mean(loss)
                preds8 = to_corners(torch.unsqueeze(unnormalize_tensor(preds), 1).to(DEVICE))
                labels8 = to_corners(torch.unsqueeze(unnormalize_tensor(labels), 1).to(DEVICE))
                iou = calc_iou_array(preds8, labels8).to(DEVICE)
                epoch_ious.append(iou.cpu().detach().numpy())
                epoch_losses.append(loss.cpu().detach().numpy())
            else:
                loss = loss_fn(preds, labels)
                epoch_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch)
        print(np.mean(epoch_losses))
        if regressor:
            print(np.mean(epoch_ious))
    return model

def main():
    # Initialize model and set to available device
    model = StarModel().to(DEVICE)
    # Initialize loss functions and optimizer for localization
    optimizer = torch.optim.Adam(model.parameters())
    loss_regressor = kfiou_loss
    epochs = 50
    batch_size = 128
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # Train the localizer
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(data_size=32000), batch_size=batch_size, shuffle=True),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_regressor, 
        scheduler=scheduler
    )

    # Freeze the backbone layers since they've been trained well enough on low level features during localizer training
    print("Classification Training")
    # Change head to classifier
    star_model.head = 'classifier'
    loss_classifier = torch.nn.BCELoss()

    for param in star_model.backbone.parameters():
        param.requires_grad = False

    for param in star_model.localizer.parameters():
        param.requires_grad = False

    epochs=5
    optimizer = torch.optim.Adam(star_model.parameters())
    star_model = train(
        star_model,
        torch.utils.data.DataLoader(StarDataset(data_size=16000, has_star=None), batch_size=batch_size, shuffle=True),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_classifier, 
        scheduler=scheduler, 
        regressor=False
    )
    torch.save(star_model.state_dict(), "model.pickle")

if __name__ == "__main__":
    main()
