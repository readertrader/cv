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
from losses import modulated_loss, kfiou_loss, kfiou_loss_fpn
from models.localizer import StarModel
from models.fpn import FPN

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

def train(model: StarModel, dl: StarDataset, num_epochs: int, optimizer, loss_fn, scheduler) -> StarModel:
    #Initiate model for training (Batchnorm and dropout)
    model.train()
    # Get model summary
    summary(model, (1,200,200))
    # Initialize total loss and iou

    for epoch in range(1,num_epochs+1):
        print(f"EPOCH: {epoch}")
        if epoch % 10 == 0:
            fn = 'model_fpn_' + str(epoch) + '.pickle'
            torch.save(model.state_dict(), fn)
        epoch_losses = []
        epoch_ious = []
        epoch_clf = []
        epoch_reg = []
        for image, labels in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            optimizer.zero_grad()
            preds = model(image)
            loss, lclf, lreg = loss_fn(
                preds, 
                labels, 
                unnormalize_tensor(preds[:,1:]).to(DEVICE),
                unnormalize_tensor(labels[:,1:]).to(DEVICE)
            )
            loss = torch.mean(loss)
            lclf = torch.mean(lclf)
            lreg = torch.mean(lreg)
            preds8 = to_corners(torch.unsqueeze(unnormalize_tensor(preds[:,1:]), 1).to(DEVICE))
            labels8 = to_corners(torch.unsqueeze(unnormalize_tensor(labels[:,1:]), 1).to(DEVICE))
            iou = calc_iou_array(preds8, labels8).to(DEVICE)
            epoch_ious.append(iou.cpu().detach().numpy())
            epoch_losses.append(loss.cpu().detach().numpy())
            epoch_clf.append(lclf.cpu().detach().numpy())
            epoch_reg.append(lreg.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("Epoch: ", epoch)
        print('Loss: ', np.mean(epoch_losses))
        print('Loss Classifier: ', np.mean(epoch_clf))
        print('Loss Regressor: ', np.mean(epoch_reg))
        print('IOU: ',np.mean(epoch_ious))

    return model

def main():
    # Initialize model and set to available device
    model = FPN().to(DEVICE)
    # Initialize loss functions and optimizer for localization
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = kfiou_loss_fpn
    epochs = 50
    batch_size = 128
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # Train the localizer
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(data_size=64000, has_star=None,mtype='fpn'), batch_size=batch_size),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn, 
        scheduler=scheduler
    )

    torch.save(star_model.state_dict(), "model_fpn.pickle")

if __name__ == "__main__":
    main()
