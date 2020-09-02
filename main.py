import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

#データセットの定義
class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images")))) #images
        self.txt_files = list(sorted(os.listdir(os.path.join(root, "annotations")))) #bounding boxes and labels

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx]) #self.imgs, self.masks are lists defined in init
        txt_path = os.path.join(self.root, "annotations", self.txt_files[idx])
        img = Image.open(img_path).convert("RGB")

        obj_ids=np.arange(12)
        """
        ignored regions(0), pedestrian(1), 
        people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
        others(11)
        """
        
              
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        with open(txt_path) as f:
            for line in f:
              x, y, w, h, s,c,l3,l4 = line.split(",")
              xmin = int(x)
              ymin = int(y)
              xmax = int(x+w)
              ymax = int(y+h)
              boxes.append([xmin, ymin, xmax, ymax])
        
        #convert everything into a torch.tensor type
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # ? change
        labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #no keypoints for this excercise
        """
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        """
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#データセットの作成
dataset = DroneDataset('VisDrone2019-DET-train/')
print("dataset created")
print(dataset[0])
"""
# FastRCNN
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
    """