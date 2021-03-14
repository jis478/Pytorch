# From MUNIT

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


    
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

    
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
    

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

        self.adain_param_num = 2 * style_dim   # (gamma,beta;2) * (IL+LN;2) * (dims)  2 x 32 = 64
#         self.adain_param_num = 2 * style_dim * 2  # (gamma,beta;2) * (IL+LN;2) * (dims)  2 x 2 x 8 = 32
        self.mlp = MLP(style_dim, self.adain_param_num, mlp_dim, 3, norm='none', activ=activ)

    def forward(self, x):
        style_code = self.model(x)
        print('style_code', style_code.shape)
        adain_params = self.mlp(style_code)                                                                             # 각 이미지 마다 feature maps x 2
        return adain_params


class PoLIN(nn.Module):
    def __init__(self, dim):
        super(PoLIN, self).__init__()
        self.conv1x1 = nn.Conv2d(dim*2, dim, 1, 1, 0)

    def forward(self, input):
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)
        LIN = torch.cat((IN,LN),dim=1)
        result = self.conv1x1(LIN)
        return result

class Ada_PoLIN(nn.Module):
    def __init__(self, dim):
        super(Ada_PoLIN, self).__init__()
        self.Conv1x1 = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=False)

    def forward(self, input, params):
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)                                               # 16 x 512 x 8 x 8
        LIN = torch.cat((IN,LN),dim=1)
        b,c,w,h = LIN.size()
        print('LIN size', LIN.size())
        print('params_zie', params.size())
        mid = params.size()[1] // 2
        gamma = params[:, :mid]                                                                                               # b x 2048
        beta = params[:, mid:]                                                                                                # b x 2048
        c = self.Conv1x1(LIN)
        print(c.size())
        print(gamma.size())
        print(beta.size())
#         result = gamma[0] * c + beta    
        gamma= gamma.repeat(1,w*w).reshape((1,mid,w,w))
        beta= beta.repeat(1,w*w).reshape((1,mid,w,w))
        result = gamma * c + beta
#         result = gamma.expand(-1,-1,w,h) + c + beta.expand(-1,-1,w,h)                                                                           # (16 x 1024) * (16 x 1024 x 8 x 8) : bcast 테스트필요
        return result


class ASC_block(nn.Module):
    def __init__(self, input_dim, dim, num_ASC_layers, activation, pad_type):
        super(ASC_block, self).__init__()
        self.input_dim = input_dim
        self.num_ASC_layers = num_ASC_layers
        self.activation = activation
        self.pad_type = pad_type
        self.ConvLayer = []
        self.NormLayer = []
        self.Ada_PoLINLayer = []
        for _ in range(self.num_ASC_layers):
            self.ConvLayer += [Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm='none', activation=self.activation, pad_type=self.pad_type)]                       # activation 먼저 하는게 맞는지?
            self.NormLayer += [Ada_PoLIN(dim)]
            self.input_dim = dim
        self.ConvLayer = nn.ModuleList(self.ConvLayer)
        self.NormLayer = nn.ModuleList(self.NormLayer)

    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
        for ConvLayer, NormLayer in zip(self.ConvLayer, self.NormLayer):
            print('x_size', x.size())
            x = ConvLayer(x)
            print('x_size', x.size())
            print('ada',Ada_PoLIN_params.size() )
            x = NormLayer(x,Ada_PoLIN_params)                                                                               # Ada_PoLIN_params: feature maps x 2
        return x


class FST_block(nn.Module):
    def __init__(self, input_dim, dim, activation, pad_type):
        super(FST_block, self).__init__()
        self.input_dim = input_dim
        self.activation = activation
        self.pad_type = pad_type
        self.block = []
        self.block += [nn.Upsample(scale_factor=2)]
        self.block += [Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm='none', activation=self.activation, pad_type=self.pad_type)]
        self.block += [PoLIN(dim)]
        self.block += [Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm='none', activation=self.activation, pad_type=self.pad_type)]
        self.block = nn.Sequential(*self.block)
        self.Ada_PoLIN = Ada_PoLIN(dim)                                                                                     # buffer register 해야하는지?


    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
        x = self.block(x)
        x = self.Ada_PoLIN(x, Ada_PoLIN_params)
        return x



# From MUNIT
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, input_dim=256, num_ASC_layers=4, num_FST_blocks=2, activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        self.model = []
        
        # ASC Blocks
        self.model += [ASC_block(input_dim, dim, num_ASC_layers, activ, pad_type)]

        # FST Blocks
        for i in range(num_FST_blocks):
            self.model += [FST_block(dim, dim, activ, pad_type)]

        self.model = nn.ModuleList(self.model)

        # Last Convlayer
        self.conv = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x, Ada_PoLIN_params):
        for block in self.model:
#             print('block', block)
            x = block(x, Ada_PoLIN_params)
#             print('block_output', x.size())
        x = self.conv(x)
        return x

