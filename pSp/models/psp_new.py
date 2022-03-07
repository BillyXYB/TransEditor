"""
This file defines the core research contribution
"""
import math

import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
import sys

sys.path.append('.')
sys.path.append('../../')

import torch
from torch import nn
from pSp.models.encoders import psp_encoders_new as psp_encoders
from model_spatial_query import Generator
from pSp.configs.paths_config import model_paths
from utils.sample import prepare_noise_new, prepare_param


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.device = self.opts.device
        self.encoder = self.set_encoder()
        self.token = 2 * (int(math.log(self.opts.output_size, 2)) - 1)

        # the decoder is TransEditor
        self.decoder = Generator(
            opts.size, opts.latent, opts.latent, opts.token, 
            channel_multiplier=opts.channel_multiplier,layer_noise_injection = opts.inject_noise,
            use_spatial_mapping=opts.use_spatial_mapping, num_region=opts.num_region, n_trans=opts.num_trans,
            pixel_norm_op_dim=opts.pixel_norm_op_dim, no_trans=opts.no_trans
        ).to(self.device)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):

        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)

            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.ckpt)
            # self.decoder.load_state_dict(torch.load(self.opts.ckpt)['g_ema'])
            self.decoder.load_state_dict(ckpt['g_ema'])
            self.decoder.eval()
            self.decoder = self.decoder.to(self.device)

            self.__load_latent_avg(ckpt)

    
    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, only_encode=False,
                
                ):
        
        if input_code:
            codes = x
        else:
            z_code, p_code = self.encoder(x) # z_code, p_code : bs, 512, 16
            # normalize with respect to the center of an average face
            # the z_latent_avg should be shape of 1，512， 16
            if self.opts.start_from_latent_avg:
                if self.opts.from_plus_space:
                    z_code = z_code + self.z_plus_latent_avg.repeat(z_code.shape[0], 1, 1)
                    p_code = p_code + self.p_plus_latent_avg.repeat(p_code.shape[0], 1, 1)
                else:
                    z_code = z_code + self.z_latent_avg.repeat(z_code.shape[0], 1, 1)
                    p_code = p_code + self.p_latent_avg.repeat(p_code.shape[0], 1, 1)
            if only_encode:
                return z_code, p_code
        # input_is_latent = not input_code
        if not self.opts.from_plus_space: # p,z are not plus, use spatial and style mappings
            images, _, _ = self.decoder(z_code, p_code, return_latents=False)
        else:
            images, _, _ = self.decoder(z_code, p_code, 
                                use_spatial_mapping=False,use_style_mapping=False, 
                                return_latents=False)

        if resize:
            images = self.face_pool(images)

        return images, z_code, p_code

    def only_decode(self, z_code, p_code):
            if self.opts.from_plus_space:
                images, _, _ = self.decoder(z_code, p_code, 
                                    use_spatial_mapping=False,use_style_mapping=False, 
                                    return_latents=False)
            return images, z_code, p_code


    def only_map(self,z_code, p_code):
        return self.decoder(z_code, p_code, return_mapped_codes=True)
        
    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if self.opts.from_plus_space and ('z_plus_latent_avg' in ckpt) and ('p_plus_latent_avg' in ckpt):
            self.p_plus_latent_avg = ckpt['p_plus_latent_avg'].to(self.opts.device)
            self.z_plus_latent_avg = ckpt['z_plus_latent_avg'].to(self.opts.device)
            print("load average p_plus and z_plus latent succesully")

        elif ('z_latent_avg' in ckpt) and ('p_latent_avg' in ckpt):
            self.p_latent_avg = ckpt['p_latent_avg'].to(self.opts.device)
            self.z_latent_avg = ckpt['z_latent_avg'].to(self.opts.device)
            print("load average p and z latent succesully")
        else:
            # generate 10000 random latent codes and calculate latent_avg
            print("Generating 10000 random samples to calculate latent_avg...")

            p_latents = []
            z_latents = []

            p_plus_latents = []
            z_plus_latents = []

            self.decoder.eval()
            with torch.no_grad():
                for i in tqdm(range(10000)):
                    
                    sample_param = prepare_param(10, self.opts, self.device, "spatial") # bs, 512, 16
                    sample_z = prepare_noise_new(10, self.opts, self.device, "query")

                    if self.opts.from_plus_space:
                        stylecode, spatialcode = self.decoder(sample_z, sample_param, return_mapped_codes = True)
                        p_plus_latents.append(spatialcode.cpu())
                        z_plus_latents.append(stylecode.cpu())
                    else:
                        p_latents.append(sample_param.cpu())
                        z_latents.append(sample_z.cpu())

            if self.opts.from_plus_space:
                print("using the plus space")
                self.p_plus_latent_avg = torch.mean(torch.cat(p_plus_latents, dim=0), dim=0, keepdim=True).to(self.opts.device)
                self.z_plus_latent_avg = torch.mean(torch.cat(z_plus_latents, dim=0), dim=0, keepdim=True).to(self.opts.device)
            else:
                self.p_latent_avg = torch.mean(torch.cat(p_latents, dim=0), dim=0, keepdim=True).to(self.opts.device)
                self.z_latent_avg = torch.mean(torch.cat(z_latents, dim=0), dim=0, keepdim=True).to(self.opts.device)
                   
