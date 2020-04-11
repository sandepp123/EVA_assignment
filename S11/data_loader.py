
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations.pytorch as A2

class Cifar10Augmentation:

  def __init__(self, transforms_list=[]):
    transforms_list.append(A2.ToTensor())
    
    self.transforms = A.Compose(transforms_list)


  def __call__(self, img):
    img = np.array(img)
    return self.transforms(image=img)['image']


def data_loader_cifar(k=32):
   
  mean = (0.4914, 0.4822, 0.4465)
  std_dev = (0.2023, 0.1994, 0.2010)
# Train Phase transformations
  train_transforms = Cifar10Augmentation([
                                      #  A.PadIfNeeded(min_height=36, min_width=36, border_mode=4, always_apply=True, p=1.0),
                                       A.RandomCrop(height=32,width=32,always_apply=False,p=0.7),
                                       A.HorizontalFlip(p=0.7),
                                      #  A.RGBShift(r_shift_limit=45, g_shift_limit=45, b_shift_limit=45, p=0.3),
                                       
                                       A.Normalize(mean=mean, std=std_dev),
                                       A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=0.7)
                                      #  A.RandomCrop(8,8,p=0.3)

                                       ])

  



# Test Phase transformations
  test_transforms = Cifar10Augmentation([A.Normalize(mean=mean, std=std_dev)])

 
  transform1 = transforms.Compose([transforms.ToTensor()])
  
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=train_transforms)

  
  
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=4)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True,transform=test_transforms)
                                    
  testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=True, num_workers=4)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader,testloader,classes