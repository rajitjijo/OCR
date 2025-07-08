import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
import xml.etree.ElementTree as et
import torch
from torchvision.transforms import ToTensor
from PIL import Image

def visualize_pred(model, img_path, device="cuda", annot_path=None):

    model.roi_heads.detections_per_img = 1
    model.eval()
    transform = ToTensor()

    img = Image.open(img_path)
    
    if img.size[0] > 600 or img.size[1] > 600:

        img = img.resize((400,267), 1)
    
    img = img.convert("RGB")
    
    with torch.no_grad():
        img_tensor = [transform(img).to(device)]
        pred = model(img_tensor)[0]["boxes"].cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10,6))
    ax.imshow(img)

    for bbox in pred:
        
        x1, y1, x2, y2 = bbox.astype(int).round()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Pred", color='white',
                    fontsize=10, backgroundcolor="red")
    
    if annot_path:

        tree = et.parse(annot_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            
            bndbox = obj.find("bndbox")
            x1_ = int(bndbox.find("xmin").text)
            y1_ = int(bndbox.find("ymin").text)
            x2_ = int(bndbox.find("xmax").text)
            y2_ = int(bndbox.find("ymax").text)

            rect2 = patches.Rectangle((x1_, y1_), x2_ - x1_, y2_ - y1_, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect2)
            ax.text(x2_+3, y2_+5, f"gt", color='white', fontsize=10, backgroundcolor="green")
    
    plt.axis('off')
    plt.title("Prediction")
    plt.show()

if __name__ == "__main__":

    trained_model = torch.load("train/checkpoint_2025-07-07_19-56-37/best.pth", weights_only=False)

    # visualize_pred(model=trained_model, img_path=r"C:\Users\rajme\Documents\liscense_plate_detector\datasets\images\Cars100.png",
    #                                     annot_path=r"C:\Users\rajme\Documents\liscense_plate_detector\datasets\annotations\Cars100.xml")

    visualize_pred(model=trained_model, img_path="test_images/two.png")