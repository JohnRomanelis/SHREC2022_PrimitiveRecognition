import MinkowskiEngine as ME
import torch.nn as nn

def get_mlp_block(in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

def get_conv_block(in_channel, out_channel, kernel_size, stride, D=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dimension=D,
        ),
        ME.MinkowskiBatchNorm(out_channel),
        ME.MinkowskiLeakyReLU(),
    )

class Regressor3(nn.Module):
    
    def __init__(self, embedding_channel, k, intermediate_channel=512):
        
        super(Regressor3, self).__init__()
        
        self.head = nn.Sequential(
            get_mlp_block(embedding_channel * 2, intermediate_channel),
            ME.MinkowskiDropout(),
            get_mlp_block(intermediate_channel, intermediate_channel),
            ME.MinkowskiLinear(intermediate_channel, k, bias=True),
        )
    
    def forward(self, x):
        
        return self.head(x)

class Regressor2(nn.Module):
    
    def __init__(self, embedding_channel, k, intermediate_channel=512):
        
        super(Regressor2, self).__init__()
        
        self.head = nn.Sequential(
            get_mlp_block(embedding_channel*2, intermediate_channel),
            ME.MinkowskiDropout(),
            ME.MinkowskiLinear(intermediate_channel, k, bias=True)
        )
    
    def forward(self, x):
        
        return self.head(x)
    
class Regressor1(nn.Module):
    
    def __init__(self, embedding_channel, k):
        
        super(Regressor1, self).__init__()
        
        self.head = ME.MinkowskiLinear(embedding_channel*2, k, bias=False)
    
    def forward(self, x):
        
        return self.head(x)


class MinkowskiMR(ME.MinkowskiNetwork):
    
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()


    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = get_mlp_block(in_channel, channels[0])
        self.conv1 = get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
            D=D
        )
        self.conv2 = get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
            D=D
        )

        self.conv3 = get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
            D=D
        )

        self.conv4 = get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
            D=D
        )
        self.conv5 = nn.Sequential(
            get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
                D=D
            ),
            get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
                D=D
            ),
            get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
                D=D
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.cls_head = Regressor3(embedding_channel, 5, 512)
        self.plane_reg = Regressor2(embedding_channel, 6, 512)
        self.cylinder_reg = Regressor2(embedding_channel, 7, 512)
        self.cone_reg = Regressor2(embedding_channel, 7, 512)
        self.sphere_reg = Regressor2(embedding_channel, 4, 512)
        self.torus_reg = Regressor2(embedding_channel, 8, 512)

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        feat_vec = ME.cat(x1, x2)
        
        
        return self.cls_head(feat_vec).F, self.plane_reg(feat_vec).F, self.cylinder_reg(feat_vec).F,\
                self.cone_reg(feat_vec).F, self.sphere_reg(feat_vec).F, self.torus_reg(feat_vec).F