import os,torch
from torch import nn
import torch.nn.functional as F

#from pytorch_lib import*

input_size = (1,28,28)

class FNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

        self.fc1 = nn.Linear(784, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x#,x

class FNN_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

        self.fc1 = fnn_cell(784,32,10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x,x

class FNN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x,x

class CNN_1(nn.Module):
    class Cell(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn1 = cnn_cell(1,10,3,1,1)
            self.cnn2 = cnn_cell(10,10,3,1,1)
            self.cnn3 = cnn_cell(10,5,3,1,1)
            self.cnn4 = cnn_cell(5,5,3,1,1)
            self.cnn5 = cnn_cell(5,5,3,1,1)

            self.fc1 = nn.Linear(1620,1)

        def forward(self,x):
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x)
            x = self.cnn4(x)
            x = self.cnn5(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

        self.layers = nn.ModuleList([self.Cell() for i in range(10)])

    def forward(self,x):
        result = []
        for layer in self.layers: result.append(layer(x))
        x = torch.cat(result, dim=1)
        return x,x

class ResNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

        self.conv1 = cnn_cell(input_size[0],8,7,2,3,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = residual_cell(8, 16, 2, 1)
        self.layer2 = residual_cell(16, 32, 2, 2)
        self.layer3 = residual_cell(32, 32, 2, 2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # because MNIST is already 1x1 here: disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

class Vit_1(nn.Module):
    def __init__(self,*,image_size=28,patch_size=28,input_channel=1,att_dim=64,depth=8,heads=1,mlp_dim=128,num_cls=10):
        super().__init__()
        self.name = type(self).__name__+'.'+f'{patch_size},{att_dim},{mlp_dim},{depth},{heads}'
        self.input_size = (1,28,28)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patche = (image_size // patch_size) ** 2
        self.patch_dim = input_channel * patch_size ** 2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim,att_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patche + 1, att_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, att_dim))
        #self.extra_token = nn.Parameter(torch.randn(1, 15, att_dim))
        self.transformer = Transformer(att_dim, depth, heads, mlp_dim)
        self.mlp_head = fnn_cell(att_dim,mlp_dim,num_cls)
        #self.mlp_head = fnn_cell(1024,mlp_dim,num_cls)
        self.to_cls_token = nn.Identity()
    
    def forward(self, img, mask=None):
        x = self.patch_to_embedding(img)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        #extra_tokens = self.extra_token.expand(img.shape[0], -1, -1)
        #x = torch.cat((cls_tokens, x,extra_tokens), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        #print(x.shape)
        x = self.transformer(x, mask)
        #print(x.shape)
        #input()

        x = self.to_cls_token(x[:, 0])
        #print(x.shape)
        #input()
        return self.mlp_head(x),x