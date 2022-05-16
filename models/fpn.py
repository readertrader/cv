import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # Base
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 8, 3, stride=1,padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2,2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=1,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv2_1 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2,padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2,padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.conv4_1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2,padding=1), nn.BatchNorm2d(128), nn.ReLU())

        self.conv5_1 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2,padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv5_2 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2,padding=1), nn.BatchNorm2d(256), nn.ReLU())

        self.top_layer = nn.Conv2d(256, 256, 1)

        self.latlayer2 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # nn.Sigmoid() ## using BCE with logits
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv1_1(x)
        u1 = self.conv1_2(x)
        u11 = self.conv2_1(u1)
        u2 = self.conv2_2(u11)
        u22 = self.conv3_1(u2)
        u3 = self.conv3_2(u22)
        u33 = self.conv4_1(u3)
        u4 = self.conv4_2(u33)
        u44 = self.conv5_1(u4)
        u5 = self.conv5_2(u44)

        t5 = self.top_layer(u5)
    
        t4 = self._upsample_add(t5, self.latlayer4(u4))
        t3 = self._upsample_add(t4, self.latlayer3(u3))
        t2 = self._upsample_add(t3, self.latlayer2(u2))

        t2 = self.smooth2(t2)

        classification = self.classifier(self.avg_pooling(t5))
        regression = self.regressor(self.avg_pooling(t2))

        prob = classification.view(x.shape[0], 1)
        preds = regression.view(x.shape[0], 5)
        prediction = torch.cat([prob,preds], dim=1)
        return prediction

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y