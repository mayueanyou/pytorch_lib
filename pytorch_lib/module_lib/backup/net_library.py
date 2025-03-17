import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def fnn_cell(dim_in,dim_hidden,dim_out):
    layers = [nn.Linear(dim_in,dim_hidden),nn.GELU(),nn.Linear(dim_hidden,dim_out)]
    return nn.Sequential(*layers)


def cnn_cell(input_channel,output_channel,kernel_size,stride,pedding,activation_function=True,bias=True):
    layers = [nn.Conv2d(input_channel,output_channel,kernel_size,stride,pedding,bias=bias),
              nn.BatchNorm2d(output_channel,track_running_stats=True)]
    #if activation_function: layers.append(nn.ReLU())
    if activation_function: layers.append(nn.GELU())
    return nn.Sequential(*layers)

class Residual(nn.Module):
    def __init__(self, layers,transform=None):
        super().__init__()
        self.layers = layers
        #self.gamma = nn.Parameter(torch.zeros(1))
        self.transform = transform

    def forward(self, x):
        residual = x
        if self.transform is not None: residual = self.transform(residual)
        return self.layers(x) + residual

def residual_cell(input_channel,output_channel,number,stride):
    def create_cell(input_channel,output_channel,stride):
        layers = [cnn_cell(input_channel,output_channel,3,stride,1),
                    cnn_cell(output_channel,output_channel,3,1,1,activation_function=False)]
        return nn.Sequential(*layers)
    
    downsample = None
    if stride != 1 or input_channel != output_channel:
        downsample = cnn_cell(input_channel,output_channel,1,stride,0,activation_function=False)
    layers = []
    layers.append(Residual(create_cell(input_channel,output_channel,stride),downsample))
    for i in range(1, number):
        layers.append(Residual(create_cell(output_channel,output_channel,1)))
    return nn.Sequential(*layers)

class ResNet_18(nn.Module):
    def __init__(self,num_cls):
        super().__init__()
        self.name = type(self).__name__
        self.input_size = (1,28,28)

        self.conv1 = cnn_cell(self.input_size[0],64,7,2,3,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = residual_cell(64, 64, 2, 1)
        self.layer2 = residual_cell(64, 128, 2, 2)
        self.layer3 = residual_cell(128, 256, 2, 2)
        self.layer4 = residual_cell(256, 512, 2, 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128, num_cls)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here: disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
    def __init__(self, dim, heads=1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        #self.to_qkv = nn.Linear(dim, dim * 3 * heads, bias=False)
        #self.to_out = nn.Linear(dim * heads, dim)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        #print('x',x.shape)
        qkv = self.to_qkv(x)
        #print('akv',qkv.shape)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        #print('q',q.shape)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #print('dots',dots.shape)

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attention = dots.softmax(dim=-1)
        print('att',attention.shape)
        print('v',v.shape)
        input()
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                Residual(PreNorm(dim, MultiHeadSelfAttention(dim, heads = heads))),
                Residual(PreNorm(dim, fnn_cell(dim, mlp_dim, dim)))
            ))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, mask=None):
        x = self.layers(x)
        return x

class ImageToPatches(nn.Module):
    def __init__(self,image_size,patch_size,dim_out,input_channel=3,position=True,cls_token=True):
        super().__init__()
        num_patche = (image_size // patch_size) ** 2
        patch_dim = input_channel * patch_size ** 2
        self.position = position
        self.cls_token = cls_token
        extra_pos = 0
        if cls_token: 
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim_out))
            extra_pos = 1
        
        if position: self.pos_embedding = nn.Parameter(torch.randn(1, num_patche + extra_pos, dim_out))
        
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim,dim_out))
        
    def forward(self,x):
        x = self.patch_to_embedding(x)
        if self.cls_token: x = torch.cat((self.cls_embedding.expand(x.shape[0], -1, -1), x), dim=1)
        if self.position: x  += self.pos_embedding
        return x
        

class VIT(nn.Module):
    def __init__(self,*,image_size,patch_size,input_channel,att_dim,depth,heads,mlp_dim,num_cls):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patche = (image_size // patch_size) ** 2
        self.patch_dim = input_channel * patch_size ** 2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim,att_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patche + 1, att_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, att_dim))
        self.transformer = Transformer(att_dim, depth, heads, mlp_dim)
        self.mlp_head = fnn_cell(att_dim,mlp_dim,num_cls)
        self.to_cls_token = nn.Identity()
    
    def forward(self, img, mask=None):
        #p = self.patch_size
        #x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(img)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def initialize_cnn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**.5)

if __name__ == "__main__":
    input = torch.randn(1,1,28,28)
    #input = torch.flatten(input, 1)
    model = VIT(image_size=28, patch_size=7, num_cls=10, input_channel=1,
            att_dim=64, depth=6, heads=8, mlp_dim=128)
    output = model(input)
    print(output.shape)
    