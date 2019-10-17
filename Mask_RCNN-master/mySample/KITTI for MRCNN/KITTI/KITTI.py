"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
    
"""
"""
    本python文件有bollon.py更改得到，为了训练标准的KITTI数据集，由于KITTI数据集并未给出mask，
    因此根据需要，读取kitti label 文件的 box坐标，并且将box作为mask     by 杨彬
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class KITTIConfig(Config):
    """Configuration for training on the KITTI dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "KITTI"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background +  1:'Pedestrian', 2:'Car', 3:'Cyclist'

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    

############################################################
#  Dataset
############################################################

class KITTIDataset(utils.Dataset):
    #数据集属性
    trainDataNum = 170   #训啦集数目
    trainBeginID = 0     #训练集开始的id num  eg : 000000.png  -> 0 
    valDataNum =30
    valBeginID = 170    # 000170.png -> 170
    
    classStrList = ['BG','Pedestrian','Car','Cyclist']
    classDict = {'Pedestrian':1,'Car':2,'Cyclist':3}
    def load_KITTI(self, dataset_dir, subset):
        """Load a subset of the KITTI dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # 添加class ,  1:'Pedestrian', 2:'Car', 3:'Cyclist'
        self.add_class("KITTI", 1, "Pedestrian")
        self.add_class("KITTI",2,'Car')
        self.add_class("KITTI",3,'Cyclist')
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        '''
        加载 KITTI的 label
               KITTI label:
               1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                                 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                 'Misc' or 'DontCare'
               1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                                 truncated refers to the object leaving image boundaries
               1    occluded     Integer (0,1,2,3) indicating occlusion state:
                                 0 = fully visible, 1 = partly occluded
                                 2 = largely occluded, 3 = unknown
               1    alpha        Observation angle of object, ranging [-pi..pi]
               4    bbox         2D bounding box of object in the image (0-based index):
                                 contains left, top, right, bottom pixel coordinates
               3    dimensions   3D object dimensions: height, width, length (in meters)
               3    location     3D object location x,y,z in camera coordinates (in meters)
               1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
               1    score        Only for results: Float, indicating confidence in
                                 detection, needed for p/r curves, higher is better.        
           
        '''
        
        #按照数据集参数加载 label 和 image
        label_dir = os.path.join(dataset_dir, "labels")

        if subset == 'train':
            fileIDList = os.listdir(label_dir);
            #fileIDList = [str(id).zfill(6) for id in range(self.trainBeginID,self.trainBeginID+self.trainDataNum)]
            #按照fileIDList 加载 一张image的label 以及添加照片
            for fileName in fileIDList:
                fileName = os.path.splitext(fileName)[0]
                labelPath = os.path.join(label_dir,fileName+".txt")
                imagePath = os.path.join(dataset_dir,fileName+".png")
                if not os.path.exists(imagePath):
                    continue
                #box  x0,y0 左上角坐标（x是横轴 y是纵轴） x1,y1右下角坐标
                rectBoxs=[]  
                classIDs=[]  #类别 默认是 0：背景
                classNames=[]
                #读取label 
                with open(labelPath) as labelFile:
                    #逐行读取,读取单个实例
                    for labelLine in labelFile.readlines():
                        attributes = labelLine.split(' ')
                        #读取类别
                        className = attributes[0]
                        if className in self.classDict.keys():
                            classIDs.append( self.classDict[className])
                            classNames.append(className)
                            rectBox={}
                            rectBox['x0'] = round(float(attributes[4]))
                            rectBox['y0'] = round(float(attributes[5]))
                            rectBox['x1'] = round(float(attributes[6]))
                            rectBox['y1'] = round(float(attributes[7]))
                            #添加box
                            rectBoxs.append(rectBox)
                        else:
                            continue

                #添加单张image
                image = skimage.io.imread(imagePath) 
                height, width = image.shape[:2]
                self.add_image(
                    "KITTI",
                    image_id=fileName+'.png',  # use file name as a unique image id
                    path=imagePath,
                    width=width, height=height,
                    classIDs = classIDs,
                    classNames= classNames,
                    rectBoxs=rectBoxs)

        elif subset == 'val':
            fileIDList = os.listdir(label_dir);
            #fileIDList = [str(id).zfill(6) for id in range(self.valBeginID,self.valBeginID+self.valDataNum)]
            #按照fileIDList 加载 一张image的label 以及添加照片
            for fileName in fileIDList:
                fileName = os.path.splitext(fileName)[0]
                labelPath = os.path.join(label_dir,fileName+".txt")
                imagePath = os.path.join(dataset_dir,fileName+".png")
                if not os.path.exists(imagePath):
                    continue
                #box  x0,y0 左上角坐标（x是横轴 y是纵轴） x1,y1右下角坐标
                rectBoxs=[]  
                classIDs=[]  #类别 默认是 0：背景
                classNames=[]
                #读取label 
                with open(labelPath) as labelFile:
                    #逐行读取,读取单个实例
                    for labelLine in labelFile.readlines():
                        attributes = labelLine.split(' ')
                        #读取类别
                        className = attributes[0]
                        if className in self.classDict.keys():
                            classIDs.append( self.classDict[className])
                            classNames.append(className)
                            rectBox={}
                            rectBox['x0'] = round(float(attributes[4]))
                            rectBox['y0'] = round(float(attributes[5]))
                            rectBox['x1'] = round(float(attributes[6]))
                            rectBox['y1'] = round(float(attributes[7]))
                            #添加box
                            rectBoxs.append(rectBox)
                        else:
                            continue
                #添加单张image
                image = skimage.io.imread(imagePath) 
                height, width = image.shape[:2]
                self.add_image(
                    "KITTI",
                    image_id=fileName+'.png',  # use file name as a unique image id
                    path=imagePath,
                    width=width, height=height,
                    classIDs = classIDs,
                    classNames= classNames,
                    rectBoxs=rectBoxs)
      
                  
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a KITTI dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "KITTI":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["rectBoxs"])],
                        dtype=np.bool)
        #绘制矩形框框
        for i, box in enumerate(info["rectBoxs"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.rectangle((box['y0'],box['x0']),end=(box['y1'],box['x1']))
            mask[rr, cc, i] = True
        #因为根据之前的实验结果在KITTI里遮挡问题比较严重，因此需要处理一下遮挡
        #当实例不止一个的时候，遍历所有的实例
        if len(info["rectBoxs"]) >1:
            for i in range(0,len(info["rectBoxs"])):
                #计算遮挡
                occusionList = list(range(0,len(info["rectBoxs"])))  #去掉 i 因为 i是我们需要的
                occusionList.remove(i)
                #去除了 下标为 i 的实例 外 所有实例的公共部分
                occusion = mask[:,:,occusionList[-1]]
                occusionList.pop();   #去掉最后一个元素
                for count in occusionList:
                    occusion = occusion | mask[:,:,count]  #求并集
                occusion = np.logical_not(occusion )  #求 非 ，和 1异或相当于求非
                #去除公共部分，即去除遮挡,实现方法是做与操作
                mask[:,:,i] = mask[:,:,i] & occusion
        
    
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool),  np.array(info['classIDs'],dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "KITTI":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = KITTIDataset()
    dataset_train.load_KITTI(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = KITTIDataset()
    dataset_val.load_KITTI(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect KITTI.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/KITTI/dataset/",
                        help='Directory of the KITTI dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = KITTIConfig()
    else:
        class InferenceConfig(KITTIConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
