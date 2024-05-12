import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class CNN(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride,pedding,activation_function=True,bias=True):
        super().__init__()
        self.cnn = nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias)
        self.bn = nn.BatchNorm2d(output_channel,track_running_stats=True)
        self.activation_function = activation_function
        if activation_function: self.af = nn.GELU()
        #if activation_function: self.af = nn.ReLU()
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        if self.activation_function: x = self.af(x)
        return x

class Residual(nn.Module):
    def __init__(self, layers,transform=None):
        super().__init__()
        self.layers = layers
        self.transform = transform

    def forward(self, x):
        residual = x
        if self.transform is not None: residual = self.transform(residual)
        return self.layers(x) + residual

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        self.output = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)
        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        # Apply mask (if provided)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        # Apply softmax
        attention = F.softmax(scores, dim=-1)
        # Multiply weights with values
        output = torch.matmul(attention, values)
        output = self.output(output)
        return output, attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attention = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class VIT(nn.Module):
    def __init__(self,image_size,patch_size,input_channel,att_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patche = (image_size // patch_size) ** 2
        self.patch_dim = input_channel * patch_size ** 2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
            nn.Linear(self.patch_dim,att_dim),)
        self.cls_token = nn.Parameter(torch.randn(1, 1, att_dim))
        
        pass

def residual_cell(input_channel,output_channel,number,stride):
        def create_cell(input_channel,output_channel,stride):
            layers = [cnn_cell(input_channel,output_channel,3,stride,1),
                      cnn_cell(output_channel,output_channel,3,1,1,relu=False)]
            return nn.Sequential(*layers)
        
        downsample = None
        if stride != 1 or input_channel != output_channel:
            downsample = cnn_cell(input_channel,output_channel,1,stride,0,relu=False)
        layers = []
        layers.append(Residual(create_cell(input_channel,output_channel,stride),downsample))
        for i in range(1, number):
            layers.append(Residual(create_cell(output_channel,output_channel,1)))
        return nn.Sequential(*layers)

def cnn_cell(input_channel,output_channel,kernel_size,stride,pedding,relu=True,bias=True):
    layers = [nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),
              nn.BatchNorm2d(output_channel,track_running_stats=True)]
    #if relu: layers.append(nn.ReLU())
    if relu: layers.append(nn.GELU())
    return nn.Sequential(*layers)

def initialize_cnn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**.5)

if __name__ == "__main__":
    input = torch.randn(3,3,784)
    #input = torch.flatten(input, 1)
    model = MultiHeadSelfAttention(784)
    output = model(input)
    print(output.shape)
    