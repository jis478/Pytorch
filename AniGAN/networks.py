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

        self.adain_param_num = 2 * self.output_dim
        self.mlp = MLP(style_dim, self.adain_param_num, mlp_dim, 3, norm='none', activ=activ)

    def forward(self, x):
        style_code = self.model(x)
        adain_params = self.mlp(style_code)                                                                             # 각 이미지 마다 feature maps x 2
        return adain_params


class PoLIN(nn.Module):
    def __init__(self, dim):
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, input):
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)
        LIN = torch.cat((IN,LN),dim=1)
        result = self.conv1x1(LIN)
        return result

class Ada_PoLIN(nn.Module):
    def __init__(self, dim):
        self.Conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, input, params):
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)                                               # 16 x 512 x 8 x 8
        LIN = torch.cat((IN,LN),dim=1)                                                                                     # 16 x 1024 x 8 x 8      
        gamma = params[:, :feature_maps]                                                                                    # 16 x 512 (one for each fmap)
        beta = params[:, feature_maps:]                                                                                     # 16 x 512
        result = gamma * self.Conv1x1(LIN) + beta                                                                           # (16 x 512) * (16 x 1024 x 8 x 8) : bcast 테스트필요
        return result

        super(StyleEncoder, self).__init__()
        self.model = []

self.gamma.expand(input.shape[0], -1, -1, -1)

class ASC_block(nn.Module):
    def __init__(self, input_dim, dim, num_blocks, norm, activation, pad_type):
        super(ASC_block, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.norm = norm
        self.activation = activation
        self.pad_type = pad_type
        self.ConvLayer = []
        self.Ada_PoLINLayer = []
        for _ in range(self.num_blocks):
            self.ConvLayer += Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm=self.norm, activation=self.activation, pad_type=self.pad_type)
            self.NormLayer += Ada_PoLIN(dim)
            self.input_dim = dim
        self.ConvLayer = nn.ModuleList(self.ConvLayer)
        self.NormLayer = nn.ModuleList(self.NormLayer)

    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
        idx = 0
        for ConvLayer, NormLayer in zip(self.ConvLayer, self.NormLayer):
            x = ConvLayer(x)
            x = NormLayer(x,Ada_PoLIN_params)  #????????????????????????????????????????????????????????????????????????????? # Ada_PoLIN_params: feature maps x 2
            # x = NormLayer(x,Ada_PoLIN_params[idx],Ada_PoLIN_params[idx+1])
            idx += 2
        return x


# Gamma, Beta가 Feature maps 만큼 있어야 한다.
class FST_block(nn.Module):
    def __init__(self, input_dim, dim, num_blocks, norm, activation, pad_type):
        super(FST_block, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.norm = norm
        self.activation = activation
        self.pad_type = pad_type
        self.block = []
        self.block += Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm=self.norm, activation=self.activation, pad_type=self.pad_type)
        self.block += PoLIN(dim)
        self.block += Conv2dBlock(self.input_dim, dim, 3, 1, 1, norm=self.norm, activation=self.activation, pad_type=self.pad_type)
        self.block += Ada_PoLIN(dim)
        self.NormLayer = nn.ModuleList(self.NormLayer)

    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
        idx = 0
        for ConvLayer, Ada_PoLINLayer in zip(self.ConvLayer, self.Ada_PoLINLayer):
            x = ConvLayer(x)
            x = Ada_PoLINLayer(x,Ada_PoLIN_params[idx],Ada_PoLIN_params[idx+1])
            idx += 2
        return x






class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"



class AdaPoLIN(nn.Module):





class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


# From MUNIT
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ASC_block(nn.Module):
    return None



# class Content_encoder(nn.Module):
#     def __init__(self, in_channels):
#         super(Encoder, self).__init__()
#
#         self.in_channels = in_channels
#         self.conv1 = ConvLayer(3, self.in_channels, 1, stride=1)                                                                      # 512x512x32
#
#         self.blocks = []
#         for i in range(4):
#             self.blocks.append(ResBlock(self.in_channels, self.in_channels*2, padding="reflect"))
#             self.in_channels = self.in_channels * 2
#         self.blocks = nn.Sequential(*self.blocks)
#
#         self.structure = nn.Sequential(ConvLayer(self.in_channels, self.in_channels, 1, stride=1, padding="valid"),
#                                        ConvLayer(512, 8, 1, stride=1, padding="valid")
#                                        )
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.blocks(out)
#         c_code = self.structure(out)
#         return c_code
#

# class Style_encoder(nn.Module):
#     def __init__(self, in_channels):
#         super(Encoder, self).__init__()
#
#         self.in_channels = in_channels
#         self.conv1 = ConvLayer(3, self.in_channels, 1, stride=1)                                                                      # 512x512x32
#
#         self.blocks = []
#         for i in range(4):
#             self.blocks.append(ResBlock(self.in_channels, self.in_channels*2, padding="reflect"))
#             self.in_channels = self.in_channels * 2
#         self.blocks = nn.Sequential(*self.blocks)
#
#         self.texture = nn.Sequential(ConvLayer(self.in_channels, self.in_channels * 2, 3, stride=2, padding="valid"),
#                                      ConvLayer(self.in_channels * 2, self.in_channels * 4, 3, stride=2, padding="valid"),    # 7x7x2048
#                                      nn.AdaptiveAvgPool2d(1),                                                              # 1x1x2048
#                                      nn.Flatten(),                                                                         # 2048
#                                      # EqualLinear(2048, 2048)                                                               # 2048
#                                      ConvLayer(2048, 2048, 1, stride=1, padding="valid")
#                                      )
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.blocks(out)
#         t_code = self.texture(out)
#         return t_code
#

class ASC_block(nn.Module):
    return None

class FST_block(nn.Module):
    return None

class Decoder(nn.Module):
    return None


