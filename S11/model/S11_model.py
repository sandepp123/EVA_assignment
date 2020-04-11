import torch
import torch.nn as nn
import torch.nn.functional as F


class S11Model(nn.Module):

    def __init__(self):     

        super(S11Model, self).__init__()
        self.fc = nn.Linear(512, 10)

        self.prep = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            
        )

        self.layer1_p1 = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            
        )


        self.layer1_p2 = nn.Sequential( 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential( 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        self.layer3_p1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )
        self.layer3_p2 = nn.Sequential( 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.cov = nn.Sequential(
          nn.Conv2d(in_channels=512,out_channels = 10, kernel_size = 3,padding=0)
        )

    
    def forward(self, x):
        x1 = self.prep(x)
        
        x2 = self.layer1_p1(x1)
        x3 = self.layer1_p2(x2)
        

        xt = x2+x3
        
        x4 = self.layer_2(xt)
        x5 = self.layer3_p1(x4)
        x6 = self.layer3_p2(x5)

        x7 = x5+x6
        # print(x7.shape)
        x8 = F.max_pool2d(x7,4)
        # print(x8.shape)
        x9 = x8.view(-1, 512)
        # print(x9.shape)
        x9 = self.fc(x9)
        x9 = F.softmax(x9,dim=-1)
        return x9

        
        
        
