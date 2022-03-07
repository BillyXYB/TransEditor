from abc import ABC, abstractmethod
import numpy as np
import torch
import os


def define_generator(nettype, domain, ckpt_path=None,
                     load_encoder=False, device='cuda'):
    if 'celebahq' in domain:
        domain = 'ffhq'
        print("Mapping celebahq dataset (with labels) to FFHQ generators (unlabeled)")
    if nettype == 'stylegan2':
        # stylegan2 generators from
        # https://github.com/rosinality/stylegan2-pytorch
        return Stylegan2Net(domain, ckpt_path, load_encoder, device)
    elif nettype == 'stylegan2-cc':
        # class conditional cifar10 stylegan2 generator from
        # https://github.com/NVlabs/stylegan2-ada-pytorch
        return Stylegan2CCNet(domain)
    elif nettype == 'stylegan-idinvert':
        # 256x256 ffhq stylegan generator from
        # https://github.com/genforce/idinvert_pytorch
        return StyleganIDInvertNet(domain)
    else:
        raise NotImplementedError


class BaseNet(ABC):

    @abstractmethod
    def sample_zs(self, n, seed):
        pass

    @abstractmethod
    def seed2w(self, n, seed):
        pass

    @abstractmethod
    def zs2image(self, zs):
        pass

    @abstractmethod
    def seed2image(self, n, seed):
        pass

    @abstractmethod
    def encode(self, image, mask):
        pass

    @abstractmethod
    def decode(self, latent):
        pass

    @abstractmethod
    def optimize(self, image, mask):
        pass

    @abstractmethod
    def perturb_isotropic(self, latent, layer, eps, n, is_eval):
        pass

    @abstractmethod
    def perturb_pca(self, latent, layer, eps, n, is_eval):
        pass

    @abstractmethod
    def perturb_stylemix(self, latent, layer, mix_latent, n, is_eval):
        pass


class Stylegan2Net(BaseNet):
    def __init__(self, domain, ckpt_path=None,
                 load_encoder=False, device='cuda'):
        from .stylegan2 import stylegan2_networks
        from . import perturb_settings
        setting = stylegan2_networks.stylegan_setting(domain)
        self.generator = stylegan2_networks.load_stylegan(
            domain, size=setting['outdim']).eval().to(device)
        if load_encoder:
            self.encoder = stylegan2_networks.load_stylegan_encoder(
                domain, nz=setting['nlatent'], outdim=setting['outdim'],
                use_RGBM=True, use_VAE=False,
                resnet_depth=setting['resnet_depth'],
                ckpt_path=ckpt_path).eval().to(device)
        self.setting = setting
        self.device = device
        self.perturb_settings = perturb_settings.stylegan2_settings[domain]
        self.pca_stats = None

    def sample_zs(self, n=100, seed=1, device=None):
        depth = self.setting['nz']
        rng = np.random.RandomState(seed)
        result = torch.from_numpy(
            rng.standard_normal(n * depth)
            .reshape(n, depth)).float()
        if device is None:
            result = result.to(self.device)
        else:
            result = result.to(device)
        return result

    def seed2w(self, n, seed):
        zs = self.sample_zs(n, seed)
        ws = self.generator.gen.style(zs)
        return ws

    def zs2image(self, zs):
        ws = self.generator.gen.style(zs)
        return self.generator(ws)

    def seed2image(self, n, seed):
        zs = self.sample_zs(n, seed)
        return self.zs2image(zs)

    def encode(self, image, mask=None):
        if mask is None:
            mask = torch.ones_like(image)[:, :1, :, :]
        # stylegan mask is [0, 1]
        if torch.min(mask) == -0.5:
            mask += 0.5
        assert(torch.min(mask >= 0))

        net_input = torch.cat([image, mask], dim=1)

        encoded = self.encoder(net_input)

        return encoded

    def decode(self, latent):
        return self.generator(latent)

    def optimize(self, image, mask):
        assert(self.encoder)  # check encoder is loaded
        from utils import inversions
        checkpoint_dict, losses = inversions.invert_lbfgs(
            self, image, mask, num_steps=500)
        return checkpoint_dict, losses

    def perturb_isotropic(self, latent, layer, eps, n=8, is_eval=True):
        # is_eval applies multiple perturbations to the same latent
        w_inv_batch = latent.repeat(n, 1, 1) if is_eval else latent
        noisy = w_inv_batch + torch.randn_like(w_inv_batch) * eps
        if layer == 'fine':
            fine_layer = self.perturb_settings['fine_layer']
            w_mix = torch.cat([w_inv_batch[:, :fine_layer, :],
                               noisy[:, fine_layer:, :]], dim=1)
        elif layer == 'coarse':
            coarse_layer = self.perturb_settings['coarse_layer']
            w_mix = torch.cat([noisy[:, :coarse_layer, :],
                               w_inv_batch[:, coarse_layer:, :]], dim=1)
        assert(w_mix.shape[1] == latent.shape[1])
        jittered_im = self.generator(w_mix)
        return jittered_im

    def perturb_pca(self, latent, layer, eps, n=8, is_eval=True):
        if self.pca_stats is None:
            # load pca stats if have not already
            pca_stats = np.load(self.perturb_settings['pca_stats'])
            components = torch.from_numpy(pca_stats['components']).float().to(latent.device)
            stddev = torch.from_numpy(pca_stats['stddev']).float().to(latent.device)
            valid_components = 20
            self.pca_stats = {'components': components,
                              'stddev': stddev,
                              'valid_components': valid_components}

        w_inv_batch = latent.repeat(n, 1, 1) if is_eval else latent
        # pick some random components [N]
        comps = np.random.choice(self.pca_stats['valid_components'], n)
        # pick random direction and magnitude [Nx1]
        sigma = torch.from_numpy(np.random.uniform(
            -eps, eps, n)).float().to(latent.device).view(-1, 1)
        # scale for each component, reshape to [Cx1]
        act_stdev = self.pca_stats['stddev'].view(-1, 1)
        # compute scaled displacement, multiplied by random magnitude
        # size: [Nx512]
        delta = self.pca_stats['components'][comps] * act_stdev[comps] * sigma
        # repeat it across the style layers dimension
        delta = delta[:, None, :].repeat(1, latent.shape[1], 1)
        noisy = w_inv_batch + delta
        if layer == 'fine':
            fine_layer = self.perturb_settings['fine_layer']
            w_mix = torch.cat([w_inv_batch[:, :fine_layer, :],
                               noisy[:, fine_layer:, :]], dim=1)
        elif layer == 'coarse':
            coarse_layer = self.perturb_settings['coarse_layer']
            w_mix = torch.cat([noisy[:, :coarse_layer, :],
                               w_inv_batch[:, coarse_layer:, :]], dim=1)
        assert(w_mix.shape[1] == latent.shape[1])
        jittered_im = self.generator(w_mix)
        return jittered_im

    def perturb_stylemix(self, latent, layer, mix_latent, n=8, is_eval=True):
        # replicate  batch dimension if necessary
        w_inv_batch = latent.repeat(n, 1, 1) if is_eval else latent
        assert(mix_latent.shape[0] == n)  # sanity check batch dimension
        mix_latent = mix_latent[:, None, :].repeat(
            1, latent.shape[1], 1)  # replicate style dimension
        if layer == 'fine':
            fine_layer = self.perturb_settings['fine_layer']
            w_mix = torch.cat([w_inv_batch[:, :fine_layer, :],
                               mix_latent[:, fine_layer:, :]], dim=1)
        elif layer == 'coarse':
            coarse_layer = self.perturb_settings['coarse_layer']
            w_mix = torch.cat([mix_latent[:, :coarse_layer, :],
                               w_inv_batch[:, coarse_layer:, :]], dim=1)
        assert(w_mix.shape[1] == latent.shape[1])
        jittered_im = self.generator(w_mix)
        return jittered_im


class StyleganIDInvertNet(BaseNet):
    def __init__(self, domain, ckpt_path=None,
                 load_encoder=False, device='cuda'):
        from . import perturb_settings
        from resources.idinvert_pytorch.utils.inverter import StyleGANInverter
        # note: need to symlink resources/idinvert_pytorch/models to
        # base directory and download the pretrained model
        assert(domain == 'ffhq')
        if (not os.path.isfile('models/pretrain/styleganinv_ffhq256_encoder.pth')
                or not os.path.isfile('models/pretrain/styleganinv_ffhq256_generator.pth')):
            print("Missing pretrained models in directory models/pretrain")
            raise FileNotFoundError
        inverter = StyleGANInverter('styleganinv_ffhq256')
        generator = inverter.G.net.synthesis
        self.inverter = inverter
        self.encoder = inverter.E
        self.generator = generator
        self.device = device
        self.perturb_settings = perturb_settings.stylegan_idinvert_settings[domain]
        self.pca_stats = None

    def sample_zs(self, n=100, seed=1, device=None):
        # note: this is not seeded
        zs = torch.from_numpy(self.inverter.G.easy_sample(
            n, latent_space_type='z'))
        return zs

    def seed2w(self, seed, n):
        # note: this samples to wp space and is not seeded
        wp = torch.from_numpy(self.inverter.G.easy_sample(
            n, latent_space_type='wp')).cuda()
        return wp

    def zs2image(self, zs):
        raise NotImplementedError

    def seed2image(self, n, seed):
        raise NotImplementedError

    def encode(self, image, mask=None):
        raise NotImplementedError

    def decode(self, latent):
        return self.generator(latent)

    def optimize(self, image, mask):
        # optimized separately using settings from idinvert implementation
        raise NotImplementedError

    def perturb_isotropic(self, latent, layer, eps, n=8, is_eval=True):
        raise NotImplementedError

    def perturb_pca(self, latent, layer, eps, n=8, is_eval=True):
        raise NotImplementedError

    def perturb_stylemix(self, latent, layer, mix_latent, n=8, is_eval=True):
        # replicate  batch dimension if necessary
        w_inv_batch = latent.repeat(n, 1, 1) if is_eval else latent
        assert(mix_latent.shape[0] == n)  # sanity check batch dimension
        assert(mix_latent.shape[1] == latent.shape[1])  # sanity check style dimension
        if layer == 'fine':
            fine_layer = self.perturb_settings['fine_layer']
            w_mix = torch.cat([w_inv_batch[:, :fine_layer, :],
                               mix_latent[:, fine_layer:, :]], dim=1)
        elif layer == 'coarse':
            coarse_layer = self.perturb_settings['coarse_layer']
            w_mix = torch.cat([mix_latent[:, :coarse_layer, :],
                               w_inv_batch[:, coarse_layer:, :]], dim=1)
        assert(w_mix.shape[1] == latent.shape[1])
        jittered_im = self.generator(w_mix)
        return jittered_im


class Stylegan2CCNet(BaseNet):
    def __init__(self, domain, ckpt_path=None,
                 load_encoder=False, device='cuda'):
        from . import perturb_settings
        import sys
        sys.path.append('resources/stylegan2-ada-pytorch')
        import legacy
        import dnnlib
        import click

        assert(domain == 'cifar10')
        network_pkl = {
            'cifar10': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl',
        }[domain]
        device = torch.device(device)
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        G = G.eval()
        self.generator = G
        self.encoder = None  # no pretrained encoder for cifar10 model
        self.device = device
        self.perturb_settings = perturb_settings.stylegan2_cc_settings[domain]
        self.pca_stats = None
        self.cc_mean_w = None  # mean w latent per class

    def sample_zs(self, n=100, seed=1, device=None):
        if device is None:
            device = self.device
        z_rand = (torch.from_numpy(np.random.RandomState(seed).randn(
            n, self.generator.z_dim)).to(device))
        return z_rand

    def seed2w(self, n, seed, labels):
        # ws has dim [batch, nlayers, ndim]
        zs = self.sample_zs(n, seed)
        assert(zs.shape[0] == labels.shape[0])
        ws = self.generator.mapping(zs, labels, truncation_psi=1.0)
        return ws

    def zs2image(self, zs, labels):
        ws = self.generator.mapping(zs, labels, truncation_psi=1.0)
        return self.generator.synthesis(ws, noise_mode='const')

    def seed2image(self, n, seed, labels):
        ws = self.seed2w(n, seed, labels)
        return self.generator.synthesis(ws, noise_mode='const')

    def encode(self, image, mask=None):
        raise NotImplementedError

    def decode(self, latent):
        return self.generator.synthesis(latent, noise_mode='const')

    def optimize(self, image, label, mask=None):
        from utils import inversions
        if self.cc_mean_w is None:
            self.cc_mean_w = torch.from_numpy(np.load(
                self.perturb_settings['cc_mean_w'])['wmeans']).float().to(self.device)
        mean_latent = self.cc_mean_w[label]
        checkpoint_dict, losses = inversions.invert_lbfgs(
            self, image, mask=None,
            num_steps=200, initial_latent=mean_latent)
        return checkpoint_dict, losses

    def perturb_isotropic(self, latent, layer, eps, n=8, is_eval=True):
        raise NotImplementedError

    def perturb_pca(self, latent, layer, eps, n=8, is_eval=True):
        raise NotImplementedError

    def perturb_stylemix(self, latent, layer, mix_latent, n=8, is_eval=True):
        # replicate  batch dimension if necessary
        w_inv_batch = latent.repeat(n, 1, 1) if is_eval else latent
        assert(mix_latent.shape[0] == n)  # sanity check batch dimension
        assert(mix_latent.shape[1] == latent.shape[1])  # sanity check style dimension
        assert(layer == 'fine')
        if layer == 'fine':
            fine_layer = self.perturb_settings['fine_layer']
            w_mix = torch.cat([w_inv_batch[:, :fine_layer, :],
                               mix_latent[:, fine_layer:, :]], dim=1)
        assert(w_mix.shape[1] == latent.shape[1])
        return self.generator.synthesis(w_mix, noise_mode='const')
