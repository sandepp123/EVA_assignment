
import torch
import torchvision
import torchvision.transforms as transforms

def data_loader_cifar():
  
  #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  transform1 = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform1)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=4)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform1)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=False, num_workers=4)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader,testloader,classes