import torch
import torchvision.transforms as transforms
import numpy as np
import glob
import matplotlib.pyplot as plt

randCrop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomCrop((64, 64)),
    transforms.Resize((128,128))
])

hFlip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((128,128))
])

vFlip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((128,128))
])

rotate = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.Resize((128,128))
])



def augment_data(img_dir,dest_img_dir):
    image_list = glob.glob(img_dir+"*.jpg")
    for i in image_list:
        image = plt.imread(i)
        image = Image.fromarray(image).convert('RGB')
        image.save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_0.jpg"))
        image = np.asarray(image).astype(np.uint8)
        randCrop(image).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_1.jpg"))
        vFlip(image).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_2.jpg"))
        hFlip(image).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_3.jpg"))
        rotate(image).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_4.jpg"))
        randCrop(np.asarray(vFlip(image)).astype(np.uint8)).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_5.jpg"))
        randCrop(np.asarray(hFlip(image)).astype(np.uint8)).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_6.jpg"))
        rotate(np.asarray(randCrop(image)).astype(np.uint8)).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_7.jpg"))
        rotate(np.asarray(vFlip(image)).astype(np.uint8)).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_8.jpg"))
        rotate(np.asarray(hFlip(image)).astype(np.uint8)).save((i.replace(img_dir,dest_img_dir)).replace(".jpg","_9.jpg"))

        
