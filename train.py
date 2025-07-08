import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset_util import platedataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from datetime import datetime

def get_model(num_classes):

    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model

def collate_fn(batch):

    return tuple(zip(*batch))

def get_dataloaders(dataset, batch_size):
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def valid_(model, val_loader, device):

    model.train()
    val_loss = 0.0

    loop = tqdm(val_loader, desc="🔍 Validating", ncols=100, leave=True)

    with torch.no_grad():

        for images, targets in loop:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            val_loss += total_loss.item()
            val_avg_so_far = val_loss / float(loop.n + 1)
            loop.set_postfix(val_loss = round(val_avg_so_far, 4))

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss


def train_(model, train_load, val_loader, num_epochs, save_loc, optimizer=torch.optim.AdamW, lr=1e-4, weight_decay=1e-4):

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optimizer(params=params, lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_load, desc=f"🚀 Epoch {epoch+1}/{num_epochs}", ncols=100, leave=True)

        for images, targets in loop:
            
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #forward pass
            loss_dict = model(images, targets) #loss dict
            loss = sum(loss for loss in loss_dict.values()) #total loss from all losses there are 4 losses
            epoch_loss += loss.item()

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_train_loss = epoch_loss / len(train_load)

            # Update tqdm display with individual losses
            loop.set_postfix({'batch_loss': round(loss.item(), 3),
                              "train_loss":round(avg_train_loss,3)})

        avg_val_loss = valid_(model, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        epoch_model_path = os.path.join(save_loc, f"train{epoch+1}.pth")
        torch.save(model, epoch_model_path)

        if avg_val_loss<best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_loc,"best.pth")
            torch.save(model, best_model_path)

        torch.cuda.empty_cache()

        lr_scheduler.step() #update learning rate after 5 epochs
            


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_run_loc = os.path.join("train", f"checkpoint_{timestamp}")
    os.makedirs(current_run_loc, exist_ok=True)

    model = get_model(num_classes=2).to(device=device)

    print("MODEL LOADED\n")

    dataset = platedataset(root="datasets", transforms=ToTensor())
    
    train, val = get_dataloaders(dataset=dataset, batch_size=8)

    print("DATASET LOADED\n")

    train_(model=model, train_load=train, val_loader=val, num_epochs=10, save_loc=current_run_loc)
