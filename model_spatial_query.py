import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from utils.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


store = []


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        state_dict = torch.load('./vgg19-dcbb9e9d.pth', map_location="cpu")
        vgg19.load_state_dict(state_dict)
        vgg19 = vgg19.cuda().eval()
        vgg_pretrained_features = vgg19

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        self.eps = 1e-5

    def norm(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean) / torch.sqrt(self.std + self.eps)
        return x

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(self.norm(x * 0.5 + 0.5)), self.vgg(self.norm(y * 0.5 + 0.5))
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class PixelNorm(nn.Module): 
    def __init__(self,pixel_norm_op_dim):
        super().__init__()
        self.pixel_norm_op_dim = pixel_norm_op_dim

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=self.pixel_norm_op_dim, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) 
        weight = self.scale * self.weight * style 

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8) 
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1) 

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        ) 

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.reshape(1, batch * in_channel, height, width) 
            out = F.conv2d(input, weight, padding=self.padding, groups=batch) 
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_() 

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            layer_noise_injection=True
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.layer_noise_injection = layer_noise_injection
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        if self.layer_noise_injection:
            out = self.noise(out, noise=noise)
            out = self.activate(out)
        else:
            out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            param_dim,
            token_dim,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            layer_noise_injection = False,
            use_spatial_mapping=True,
            num_region=1,
            n_trans=4,
            pixel_norm_op_dim=2,
            no_trans=False
    ):
        super().__init__()

        self.size = size
        self.lr_mlp = lr_mlp
        self.n_trans = n_trans
        self.no_trans = no_trans

        self.style_dim = style_dim 
        self.param_dim = param_dim 
        self.token_dim = token_dim 
        self.layer_noise_injection = layer_noise_injection

        self.style = None

        self.use_spatial_mapping = use_spatial_mapping
        self.num_region = num_region
        
        self.num_spatial_mapping = int(16/num_region) 
        self.num_style_mapping = self.num_spatial_mapping 
        self.pixel_norm_op_dim = pixel_norm_op_dim

        # initialize mapping network for spacial code, 
        if self.use_spatial_mapping:
            self.spatial_mapping_network=self.spatial_mapping()

        self.style_mapping_network = self.style_mapping()
        

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.adjust_style = EqualLinear(in_dim = 16, out_dim = self.token_dim)

        # self.input = ConstantInput(self.channels[4]) # 512*4*4
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel,
            layer_noise_injection=self.layer_noise_injection
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    layer_noise_injection=self.layer_noise_injection
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    layer_noise_injection=self.layer_noise_injection
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.register_buffer('token', torch.eye(self.token_dim))
    
        self.register_buffer('token_spatial', torch.eye(16))

        self.trans_interact = not self.no_trans
        if self.trans_interact:
            self.interact=self.interaction_network()


    # mapp network for Z code 
    def style_mapping(self):
        layers=[PixelNorm(self.pixel_norm_op_dim)]
        for i in range(self.num_style_mapping):
            layers.append(
                EqualLinear(
                    in_dim=self.style_dim, out_dim=self.style_dim, lr_mul=self.lr_mlp, activation='fused_lrelu'
                )
            )
        return nn.Sequential(*layers)

    # mapping function for the spatial (P) code  (as in SNI)
    def spatial_mapping(self):
        layers=[PixelNorm(self.pixel_norm_op_dim)]
        for i in range(self.num_spatial_mapping):
            layers.append(
                EqualLinear(
                    in_dim=self.style_dim, out_dim=self.style_dim, lr_mul=self.lr_mlp, activation='fused_lrelu'
                )
            )
        return nn.Sequential(*layers)

    # the interaction module
    def interaction_network(self):
        layers = [
            AttentionBlock(in_dim = self.style_dim + 16, param_dim=self.param_dim + 16, out_dim=self.style_dim, lr_mul=self.lr_mlp,) 
        ]
        for i in range(1, self.n_trans):
            layers.append(
                AttentionBlock(in_dim = self.style_dim, param_dim=self.param_dim, out_dim=self.style_dim, lr_mul=self.lr_mlp, )
            )
        return nn.Sequential(*layers)

    def make_noise(self):
        device = 'cuda'

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises
   
   
    def forward(
            self,
            style,
            op_param,
            return_latents=False,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            return_style=False,
            return_p_latent=False,
            return_only_style=False,
            return_only_style_latent=False,
            return_only_mapped_p=False,
            return_only_mapped_z=False,
            use_spatial_mapping=True,
            use_style_mapping=True,
            trans_interact=True,
            return_mapped_codes=False,
    ):
        global store
        store = []
       

        trans_interact = trans_interact
        if self.no_trans:
            trans_interact = False
        
        if input_is_latent:
            use_spatial_mapping = True
            use_style_mapping = False
            trans_interact = False
        # op_param: [8,512,16], style: [8,512,16]  

        # the pre-mapping process for op_param(using mlp)
        # input_device = style.device
        if use_spatial_mapping:
            N,D,C = op_param.shape # op_param: [8,512,16] 
            # pixel norm
            op_param = self.spatial_mapping_network[0](op_param)
            spatialcode = torch.zeros(N, D, C).cuda()
            for i in range(self.num_spatial_mapping):
                spatialcode[:,:,i]=self.spatial_mapping_network[i+1](op_param[:,:,i])
       
        else:
            spatialcode = op_param
        
        # the pre-mapping process for style(using mlp)
        if use_style_mapping:
            N,D,C = style.shape # style: [8,512,16] 
            # pixel norm
            style = self.style_mapping_network[0](style)
            stylecode = torch.zeros(N, D, C).cuda()
            for i in range(self.num_style_mapping):
                stylecode[:,:,i]=self.style_mapping_network[i+1](style[:,:,i])
        else:
            stylecode = style

        if return_mapped_codes:
            return stylecode, spatialcode

        if return_only_mapped_p:
            return spatialcode

        if return_only_mapped_z:
            return stylecode
        

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]
        # style : [bs, 512, 14]
        
        stylecode = stylecode.permute(0,2,1) # bs,16,512
        spatialcode = spatialcode.permute(0,2,1) # bs, 16,512
        # the interaction process
        if trans_interact:
            similarity_list = []
            input_stylecode = torch.cat([stylecode, self.token_spatial.repeat(stylecode.size()[0], 1, 1)], 2)
            input_spatialcode = torch.cat([spatialcode, self.token_spatial.repeat(spatialcode.size()[0], 1, 1)], 2)
        
            x, similarity = self.interact[0](input_stylecode, input_spatialcode,return_similarity=True) # x is now normalized

            for i in range(1, self.n_trans):
                x, similarity = self.interact[i](x, spatialcode, return_similarity=True)
                similarity_list.append(similarity)
                
        # x: bs, 16*512
        if self.no_trans:
            latent = self.adjust_style(stylecode.permute(0,2,1)).permute(0,2,1) # latent: [bs, 14,512]
        else:
            if not input_is_latent:
                latent = self.adjust_style(x.permute(0,2,1)).permute(0,2,1) # latent: [bs, 14,512]
            else:
                latent = style

        if return_only_style_latent:
            return latent
        
        if return_only_style:
            return latent
       
        batch = spatialcode.shape[0]

        # spatialcode bs, 16, 512
        out = spatialcode.permute(0,2,1).reshape((batch,512,4,4))
        

        out = self.conv1(out, latent[:, 0], noise=noise[0]) # torch.Size([8, 512, 4, 4])

        skip = self.to_rgb1(out, latent[:, 1]) # torch.Size([8, 3, 4, 4])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_style:
            return image, latent
        
        if return_p_latent:
            return image, spatialcode

        if return_latents:
            return image, latent, None

        else:
            return image, None, None


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, blur_kernel=blur_kernel)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, blur_kernel=blur_kernel, bias=False, activate=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3) # EqualConv2d(513, 512, 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), # EqualLinear(8192, 512)
            EqualLinear(channels[4], 1), #  EqualLinear(512, 1)
        )

    def forward(self, input):
        out = self.convs(input) # torch.Size([8, 512, 4, 4])

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width) # torch.Size([8, 1, 4, 4])
        out = torch.cat([out, stddev], 1) # torch.Size([8, 513, 4, 4])

        out = self.final_conv(out) # torch.Size([8, 512, 4, 4])

        out = out.view(batch, -1)
        out = self.final_linear(out) # torch.Size([8, 1])

        return out


class Attention(nn.Module):
    def __init__(self, in_dim, param_dim, out_dim, lr_mul=1.0, groups=4, compress=4):
        assert out_dim % (groups * compress) == 0
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.param_dim = param_dim
        self.out_dim = out_dim
        self.compress = compress
        self.groups = groups
        self.planes = out_dim // compress
        self.group_planes = self.planes // groups
        self.scale = self.planes ** -0.5
        

        
        self.q_transform = EqualLinear(param_dim, self.planes, lr_mul=lr_mul)
        self.k_transform = EqualLinear(in_dim, self.planes, lr_mul=lr_mul)
        self.v_transform = EqualLinear(in_dim, self.planes, lr_mul=lr_mul)
       
        self.proj = EqualLinear(self.planes, out_dim, lr_mul=lr_mul)

    def forward(self, attention, op_param, return_similarity=False): # attention: 14*512 op_para:16*512

        N, L, C = attention.shape  # N, L, C
        _, M, _ = op_param.shape
        
        q = self.q_transform(op_param).reshape(N, M, self.groups, self.group_planes).permute(0, 2, 3, 1)  # N, g, gp, M
        k = self.k_transform(attention).reshape(N, L, self.groups, self.group_planes).permute(0, 2, 3, 1)  # N, g, gp, L
        v = self.v_transform(attention).reshape(N, L, self.groups, self.group_planes).permute(0, 2, 3, 1)  # N, g, gp, L
        qk = torch.einsum('abcd,abce->abde', q, k) * self.scale  # N, g, M, L
        similarity = F.softmax(qk, dim=3)  # N, g, M, L
        sv = torch.einsum('abcd,abed->abec', similarity, v)  # N, g, gp, M
        stacked_output = sv.reshape(N, self.planes, L).permute(0, 2, 1)  # N, M, g*gp
        
        output = self.proj(stacked_output)

        if return_similarity:
            return output, similarity
        else:
            return output


class AttentionBlock(nn.Module):
    def __init__(self, in_dim, param_dim, out_dim, lr_mul=1.0, groups=4):
        super(AttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.param_dim = param_dim

        self.atten = Attention(in_dim, param_dim, out_dim, lr_mul=lr_mul, groups=groups)
        self.mlp = nn.Sequential(
            EqualLinear(out_dim, out_dim, lr_mul=lr_mul),
            nn.GELU(),
            EqualLinear(out_dim, out_dim, lr_mul=lr_mul)
        )
        if out_dim != in_dim:
            self.proj = EqualLinear(in_dim, out_dim, lr_mul=lr_mul)

    def forward(self, x, op_param, return_similarity=False):
    
        similarity = None
        if return_similarity:
            attention, similarity = self.atten(F.layer_norm(x, x.size()[1:]), op_param, return_similarity=True)
        else:
            attention = self.atten(F.layer_norm(x, x.size()[1:]),op_param)
        if self.out_dim != self.in_dim:
            x = self.proj(x) + attention
        else:
            x = x + attention
        x = x + self.mlp(F.layer_norm(x, x.size()[1:]))

        if return_similarity:
            return x, similarity
        else:
            return x
