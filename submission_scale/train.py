import typing as t
from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils import DEVICE, synthesize_data, normalize, unnormalize, unnormalize_tensor, to_corners, calc_iou_array, norm_classifier
from losses import kfiou_loss
from models.StarModel import StarModel

# Dataloader
# Normalizes the label

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000, has_star=True, classifier=False):
        self.data_size = data_size
        self.has_star = has_star
        self.classifier=classifier

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=self.has_star)
        # If training the classifier hed then normalize the data accordingly
        if self.classifier:
            label = norm_classifier(label)
        else:
            label = normalize(label)
        return image[None], label

def train(model: StarModel, dl: StarDataset, num_epochs: int, optimizer, loss_fn, scheduler, regressor=True) -> StarModel:
    #Initiate model for training (Batchnorm, dropout)
    model.train()
    # Get model summary
    summary(model, (1,200,200))

    for epoch in range(1,num_epochs+1):
        print(f"EPOCH: {epoch}")
        # Save before the 10th epoch
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
            # If training the regressor use the appropriate loss function, kiou in this case and compute running iou
            if regressor:
                # Compute loss
                loss = loss_fn(
                    preds,
                    labels,
                    unnormalize_tensor(preds).to(DEVICE),
                    unnormalize_tensor(labels).to(DEVICE)
                )
                loss = torch.mean(loss)
                # Turn to corners to compute IOU
                preds8 = to_corners(torch.unsqueeze(unnormalize_tensor(preds), 1).to(DEVICE))
                labels8 = to_corners(torch.unsqueeze(unnormalize_tensor(labels), 1).to(DEVICE))
                iou = calc_iou_array(preds8, labels8).to(DEVICE)
                # Append running values to print end of the epoch
                epoch_ious.append(iou.cpu().detach().numpy())
                epoch_losses.append(loss.cpu().detach().numpy())
            else:
                # Training the classifier head
                loss = loss_fn(preds, labels)
                epoch_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Print stats end of epoch
        print("EPOCH: ", epoch)
        print("Loss: ", np.mean(epoch_losses))
        if regressor:
            # Print IOU only is training regressor
            print("IOU: ", np.mean(epoch_ious))
    return model

def main():
    # Initialize model and set to available device
    model = StarModel().to(DEVICE)
    # Initialize loss functions and optimizer for regressor
    optimizer = torch.optim.Adam(model.parameters())
    loss_regressor = kfiou_loss
    epochs = 50
    # ~ About 250 steps at this batch size and data size (time constraints for gpu) however I don't believe increasing would've helped by much
    batch_size = 128
    # Update learning rate after 20th and 40th epoch by factor of 10 
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # Train the regressor and store model under star model to then train classifier head 
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(data_size=32000), batch_size=batch_size, shuffle=True),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_regressor, 
        scheduler=scheduler
    )

    # Freeze the backbone layers since they've been trained well enough on low level features during localizer training
    # Change head to classifier
    star_model.head = 'classifier'
    # Use Binary Cross Entropy for classifier loss
    loss_classifier = torch.nn.BCELoss()
    # Iterate through the backbone and the regressor and freeze the params
    for param in star_model.backbone.parameters():
        param.requires_grad = False

    for param in star_model.localizer.parameters():
        param.requires_grad = False

    # Short training schedule for classifier since low level features which require the greater training are already trained
    epochs=5
    optimizer = torch.optim.Adam(star_model.parameters())
    star_model = train(
        star_model,
        torch.utils.data.DataLoader(StarDataset(data_size=16000, has_star=None, classifier=True), batch_size=batch_size),
        num_epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_classifier, 
        scheduler=scheduler, 
        regressor=False
    )
    torch.save(star_model.state_dict(), "model.pickle")

if __name__ == "__main__":
    main()
