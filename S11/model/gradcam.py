import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
import random
class gradcam_resnet18(nn.Module):
    def __init__(self,model):

        super(gradcam_resnet18, self).__init__()
        
        # get the pretrained VGG19 network
        self.model2 = model
        # print(self.model2.conv1,self.model2.bn1,self.model2.layer1,self.model2.layer2,self.model2.layer3,self.model2.layer4)
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(self.model2.conv1,
                                           self.model2.bn1,
                                           self.model2.layer1,
                                           self.model2.layer2,
                                           self.model2.layer3,
                                           self.model2.layer4
                                           )
        # print(self.features)
        self.linear = self.model2.linear
        self.gradients = None
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=4, padding=0, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.model2.linear
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self,grad):
      # print(grad)
      self.gradients = grad
        
    def forward(self, x):
      x = self.features_conv(x)
        
        # register the hook
      h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
      x = x = self.max_pool(x)
      x = x.view((x.size(0), -1))
      x = self.classifier(x)
      return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
      return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
      return self.features_conv(x)

def GradCam_(model,classes,img,device,target):
  target = target.to(device)
  network2 = gradcam_resnet18(model)
  network2.eval()
  # network.eval()
  pred = network2(img.cuda())
  output = model(img.to(device))
  pred2 = output.argmax(dim=1, keepdim=True)
  
  # print(output)

  is_correct = pred2.eq(target.view_as(pred2))
  # print(is_correct)
  layer_needed = int(np.array(pred.cpu().argmax(dim=1)))#pred.argmax(dim=1).tolist()[0]
  # print(pred[:,layer_needed])
  pred[:, layer_needed].backward()
  
  # print(is_correct)
  gradients = network2.get_activations_gradient()
  # print(gradients)
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = network2.get_activations(img.cuda()).detach()
  for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = np.maximum(heatmap.cpu(), 0)
  heatmap /= torch.max(heatmap)
  return heatmap,is_correct,pred2


import  matplotlib.pyplot as plt
import numpy as np
def reconstruct(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))
def plot1(img,heat_map,classes,_,pred2):
# axes[0].matshow(heat_map.squeeze())
  for i in img:
    img2 = reconstruct(i)
    # axes[1].imshow(img2, cmap='gray', interpolation='bicubic')
  # print(img2.shape)
  import cv2
  # img = cv2.imread('./data/Elephant/data/05fig34.jpg')
  heat_map = np.array(heat_map)
  heat_map = cv2.resize(heat_map, (32,32))
  heat_map = np.uint8(255 * heat_map)
  heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
  superimposed_img = heat_map*0.9 + img2
  with warnings.catch_warnings():

    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("target : "+classes[_]+" prediction : "+classes[pred2])
    axes[0].matshow(heat_map)
    # axes[0].matshow(img[0]) out.astype('uint8')
    axes[1].imshow(img2, cmap='gray', interpolation='bicubic')
    axes[2].imshow(superimposed_img)
    metric = "fig_"+ str(random.randint(0,1000))
    fig.savefig(f'%s_change.png' % (metric.lower()))
