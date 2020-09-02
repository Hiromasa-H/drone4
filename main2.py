import os
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
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
        img_path = os.path.join(self.root, "images", self.imgs[idx]) #get path of image with idx
        txt_path = os.path.join(self.root, "annotations", self.txt_files[idx]) #get path of txt with idx
        img = Image.open(img_path).convert("RGB") #open the image and convert it to RGB

        obj_ids=np.arange(12) #ids are 0 ~ 11, as listed below.
        """
        ignored regions(0), pedestrian(1),people(2), bicycle(3), car(4), van(5),
        truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
        others(11)
        """
        
              
        # get every bounding box coordinate from the txt file.
        num_objs = len(obj_ids)
        boxes = []
        labels = []
        with open(txt_path) as f:
            for line in f:
                x, y, w, h, score,category,l3,l4 = line.split(",")
                xmin = int(x)
                ymin = int(y)
                xmax = int(x) + int(w) #int(x+w)
                ymax = int(y) + int(h)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(category))
                
                
        # create mask image
        width,height = img.size
        bbmasks = []
        for box in boxes:
            bbmask = np.zeros((height, width))
            bbmask = bbmask.astype(int)
            #bb = box#.astype(np.int)
            bbmask[box[1]:box[3], box[0]:box[2]] = int(1.) #set the area of the bounding box to 1
            bbmasks.append(bbmask)
            
        #convert everything into a torch.tensor type
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        bbmasks = torch.as_tensor(bbmasks, dtype=torch.uint8)
        # ? change later
        #labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["masks"] = bbmasks
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

#maskとbounding boxを相互に変換するための関数
def bb_to_rect(box): #take in 'boxes' item from dataset
    box = np.array(box, dtype=np.float32) #convert from tensor to numpy array
    #box: array([6.84000e+02, 8.00000e+00, 6.84273e+05, 8.11600e+03], dtype=float32)
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], color='red',fill=False, lw=1)

#def mask_to_bb():

def show_one_bb(image,box):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = bb_to_rect(box = box)
    ax.add_patch(rect)
    return plt.show()

def show_all_bb(image,boxes):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        ax.add_patch(bb_to_rect(box = box))
    return plt.show()

"""
#データセットの作成
dataset = DroneDataset('VisDrone2019-DET-train/')
print("dataset created")
"""

#pretrain 済みのmodelを引っ張り出す
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
print("got the pretrained model")

#データのaugmentation, transformation.
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

#構築
# use our dataset and defined transformations
dataset = DroneDataset('VisDrone2019-DET-train/', get_transform(train=True))
dataset_test = DroneDataset('VisDrone2019-DET-train/', get_transform(train=False))
print("split the dataset into training and testing")
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist() #returns a list of indexes.
dataset = torch.utils.data.Subset(dataset, indices[:-50]) #exclude last 50
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
print("excluded last 50")

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
print("set data loaders")

#GPU環境の整備
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 1

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)# the one we defined
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
print("setup GPU environment")

#10 epoch 分の訓練
num_epochs = 10
print("starting training")
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
print("finished training")

#結果を見てみる。
# pick one image from the test set
img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# prediction
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(fp="output/image.jpg")
Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()).save(fp="output/mask.jpg")
"""
image  = dataset[0][0]
boxes  = dataset[0][1]["boxes"]
show_all_bb(image,boxes)
"""


"""
in the pytorch tutorial the masks are images with different colors for each mask.
in the 前処理段階 it splits each color into different binary masks

in the pytorch tutorial the masks rarely overlap, and every instance in a certain category is put into one mask. For example, if there are 3 people in an image, only one mask is generated for the category "people" with all 3 people on that mask. (i.e. only 1 mask is generated, not 3.)
Here I've made an individual mask for each instance, and have not been able to link the category labels to each instance. 


"""