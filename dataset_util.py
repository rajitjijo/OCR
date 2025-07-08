import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as et
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

class platedataset(Dataset):

    def __init__(self, root, transforms = None):

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images")),key=self.extract_number))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations")),key=self.extract_number))
        self.class_to_idx = {"licence":1}

    def extract_number(self, filename):
        return int("".join(list(filter(lambda x:x.isdigit(), filename))))

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = Image.open(img_path).convert("RGB")

        tree = et.parse(annot_path)
        root = tree.getroot()

        boxes, labels = [], []

        for obj in root.findall("object"):

            name = obj.find("name").text.lower()

            if name not in self.class_to_idx:
                continue

            labels.append(self.class_to_idx[name])
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {

            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

        
def show_grid_samples(dataset, class_map={1:"license"}, indices=None):
    num_images = 16
    grid_size = 4

    if indices is None:
        indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()

        ax.imshow(image_np)
        ax.axis('off')

        for box, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box.tolist()
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            label_str = class_map.get(label.item(), f'class_{label.item()}')
            ax.text(xmin, ymin - 3, label_str, color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

        ax.set_title(f"#{idx}", fontsize=10)

    plt.suptitle(f"{grid_size} X {grid_size} images from the Plate Dataset")
    # plt.tight_layout()
    plt.show()

if __name__=="__main__":

    print("Running from Dataset_Util.py")
