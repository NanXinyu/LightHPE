import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    def __init__(
        self,
        in_dim,
        embedding_dim,
        input_resolution,
        stride):
        super(PatchMerging, self).__init__()
        self.H, self.W = input_resolution
        self.stride = stride
        self.squeeze = nn.Conv2d(in_dim * stride * stride, 
                                embedding_dim, 
                                kernel_size = 1, 
                                bias = False)
    def forward(self, x):
        _,_,h,w = x.shape
        assert h == self.H and w == self.W, "INPUT SIZE WRONG"
        x_split = []
        for i in range(self.stride):
            for j in range(self.stride):
                x_split.append(
                    x[:, :, j::self.stride, i::self.stride]
                )
        
        print(len(x_split))
        x = torch.cat(x_split, dim = 1)
        
        x = self.squeeze(x)
    
        return x

class Conv2dBNActivation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        groups = 1,
        dilation = 1,      
        norm_layer = nn.BatchNorm2d,
        activation = 'RE',
    ):
        super(Conv2dBNActivation, self).__init__()
        self.activation = activation
        padding = (kernel_size - 1)//2 * dilation
    
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation, 
            groups = groups, bias = False)

        self.norm_layer = norm_layer(out_channels, eps=0.01, momentum=0.01)
        if activation == 'PRE':
            self.acti_layer = nn.PReLU()
        elif activation == 'HS':
            self.acti_layer = nn.Hardswish(inplace = True)  
        else:
            self.acti_layer = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        if self.activation is not None:
            x = self.acti_layer(x)
        return x

class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        activation) -> None:
        super(Bottleneck, self).__init__()
        self.res_connect = stride == 1 and in_channels == out_channels
        layers = []
        if mid_channels != in_channels:
            layers.append(
                Conv2dBNActivation(
                in_channels,
                mid_channels,
                1,
                1,
                activation=activation
                )
            )
        layers.append(
            Conv2dBNActivation(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            activation=activation
            )
        )
        layers.append(
            Conv2dBNActivation(
            mid_channels,
            out_channels,
            1,
            1,
            activation=None
            )
        )
        self.block = nn.Sequential(*layers)
        self.act = nn.Hardswish(inplace = True)
    def forward(self, x):
        residual = x
        x = self.block(x)
        if self.res_connect:
            x = self.act(x + residual)
        else:
            x = self.act(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    pm = PatchMerging(3, 1024, [256, 256], 16)
    x = pm(x)
    print(x.shape)

    