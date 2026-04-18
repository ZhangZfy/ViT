import torch
from torch import nn

class ViT(nn.Module):
    def __init__(self, embedding_size = 16):
        super().__init__()
        self.patch_size = 4
        self.patch_count = 28 // self.patch_size # 7

        # Convolution层实际上就是一个线性层，做了linear projection，卷积核里的参数实际上就是线性层的权重和偏置
        # [b,1,28,28] -> [b,16,7,7]
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size**2, 
                              kernel_size=self.patch_size, stride = self.patch_size, padding=0)
        # CLS token是一个可学习的参数，初始化为随机值，维度为(1, 1, embedding_size)，在训练过程中会被优化
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        # 位置编码也是一个可学习的参数，初始化为随机值，维度为(1, patch_count**2 + 1, embedding_size)，其中+1是为了CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_count**2+1, embedding_size))
        # Transformer Encoder层，输入维度为embedding_size，输出维度也为embedding_size
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=2, batch_first=True), num_layers=3)
        # MLP head, 把CLS token的输出映射到10类的分类任务上
        self.mlp_head = nn.Linear(in_features=embedding_size, out_features=10)
    
    def forward(self, x):
        x = self.conv(x)
        
        x = x.view(x.size(0), x.size(1), self.patch_count**2)
        x = x.permute(0, 2, 1)

        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_embedding + x

        x = self.transformer_encoder(x)
        logits = self.mlp_head(x[:,0,:])    # 只对cls token做映射
        return logits
    
if __name__ == '__main__':
    vit = ViT()
    x = torch.randn(5,1,28,28)
    y = vit(x)
    print(y.shape)