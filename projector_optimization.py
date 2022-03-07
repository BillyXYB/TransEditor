import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model_spatial_query import Generator
from train_spatial_query import data_sampler, sample_data
from utils.sample import prepare_param, prepare_noise_new
from utils import lpips
from utils.dataset_projector import MultiResolutionDataset


def noise_regularize_(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * torch.unsqueeze(strength, -1)

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to('cpu')
            .numpy()
    )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--para_num', type=int, default=16)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./projection/optimization')
    parser.add_argument('--pixel_norm_op_dim', type=int, default=1)
    parser.add_argument('--num_trans', type=int, default=8)
    parser.add_argument('--old_version', action='store_true', default=False)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--use_noise', action='store_true', default=False)
    parser.add_argument('--no_trans', action='store_true', default=False)
    parser.add_argument('--no_spatial_map', action='store_true', default=False)
    parser.add_argument('--num_region', type=int, default=1)
    parser.add_argument('--inject_noise', action='store_true', default=False)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()


    n_mean_latent = 10000
    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    
    args.use_spatial_mapping = not args.no_spatial_map
    

    g_ema = Generator(
        args.size, args.latent, args.latent, args.token,
        channel_multiplier=args.channel_multiplier,layer_noise_injection = args.inject_noise, 
        use_spatial_mapping=args.use_spatial_mapping, num_region=args.num_region, n_trans=args.num_trans,
        pixel_norm_op_dim=args.pixel_norm_op_dim, no_trans=args.no_trans
    ).to(device)

    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)

    ckpt_name = os.path.basename(args.ckpt)
    iter_ckpt = int(os.path.splitext(ckpt_name)[0])
    exp_name = str(args.ckpt).split('/')[-3]
    args.output_dir = os.path.join(args.output_dir, exp_name, f'{iter_ckpt}')
    sample_path = args.output_dir
    os.makedirs(sample_path, exist_ok=True)

    dataset = MultiResolutionDataset(args.dataset_dir ,resolution=args.size)

    loader = DataLoader(dataset, shuffle=False, batch_size= 1 , num_workers=4, drop_last = False)

    percept = lpips.PerceptualLoss(
                model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
            )
    
    res_latent = []
    res_param = []
    res_perceptual_values = []
    res_noise_values = []
    res_mse_values = []

    for it, imgs in enumerate(iter(loader)):
        imgs = imgs.to(device)
        noise_sample = prepare_noise_new(n_mean_latent, args, device,"query",truncation=args.truncation)
        para_base = prepare_param(n_mean_latent, args, device, method='spatial',truncation = args.truncation)

        z_plus = g_ema(noise_sample, para_base,return_only_mapped_z=True)
        p_plus = g_ema(noise_sample, para_base,return_only_mapped_p=True)

        latent_mean = z_plus.mean(0)

        latent_std = ((z_plus - latent_mean).pow(2).sum([0, 2]) / n_mean_latent) ** 0.5

        param_mean = p_plus.mean(0)
        param_std = ((p_plus - param_mean).pow(1).sum([0, 1]) / n_mean_latent) ** 0.5

        noise_single = g_ema.make_noise()
        noises = []
        for noise in noise_single:
            noises.append(noise.repeat(args.batch, 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch, 1, 1) # torch.Size([8, 512, 16])

        latent_in.requires_grad = True

        param_in = param_mean.detach().clone().unsqueeze(0).repeat(args.batch, 1, 1) # torch.Size([8, 512, 16])

        param_in.requires_grad = True

        if args.use_noise:
            for noise in noises:
                noise.requires_grad = True
            optimizer = optim.Adam([latent_in] + [param_in] + noises, lr=args.lr)
        else:
            for noise in noises:
                noise.requires_grad = False
            optimizer = optim.Adam([latent_in] + [param_in], lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []
        param_path = []
        perceptual_values = []
        noise_values = []
        mse_values = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr, rampdown=args.lr_rampdown, rampup=args.lr_rampup)
            optimizer.param_groups[0]['lr'] = lr
            if args.use_noise:
                noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                latent_n = latent_noise(latent_in, noise_strength)
                img_gen, _ ,_ = g_ema(latent_n, param_in, use_spatial_mapping=False, use_style_mapping=False, noise=noises)
            else:
                img_gen, _ ,_ = g_ema(latent_in, param_in, use_spatial_mapping=False, use_style_mapping=False)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize_(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            if args.use_noise:
                loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss
            else:
                loss = p_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())
                param_path.append(param_in.detach().clone())

            if (i + 1) % 10 == 0:
                perceptual_values.append(p_loss.item())
                noise_values.append(n_loss.item())
                mse_values.append(mse_loss.item())

            pbar.set_description(
                (
                    f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
                    f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'
                )
            )

        if args.use_noise:
            img_gen, _ , _ = g_ema(latent_path[-1], param_path[-1],use_spatial_mapping=False, use_style_mapping=False, noise=noises)
        else:
            img_gen, _ , _ = g_ema(latent_path[-1], param_path[-1],use_spatial_mapping=False, use_style_mapping=False)

        img_or = make_image(imgs)
        img_ar = make_image(img_gen)

        res_latent.append(latent_path[-1])
        res_param.append(param_path[-1])
        res_perceptual_values.append(perceptual_values[-1])
        res_noise_values.append(noise_values[-1])
        res_mse_values.append(mse_values[-1])

        img1 = Image.fromarray(img_or[0])
        img1.save(os.path.join(sample_path, f'origin_{it}.png'))
        img2 = Image.fromarray(img_ar[0])
        img2.save(os.path.join(sample_path, f'project_{it}.png'))

    res_latent = torch.cat(res_latent)
    res_param = torch.cat(res_param)
    print('res_latent.shape',res_latent.shape)
    print('res_param.shape',res_param.shape)

    np.save(os.path.join(sample_path, f'latents.npy'), res_latent.cpu().numpy())
    np.save(os.path.join(sample_path, f'param.npy'), res_param.cpu().numpy())
    np.save(os.path.join(sample_path, f'perceptual.npy'), res_perceptual_values)
    np.save(os.path.join(sample_path, f'noise.npy'), res_noise_values)
    np.save(os.path.join(sample_path, f'mse.npy'), res_mse_values)


# python projector_optimization.py --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --dataset_dir ffhq/test/images
# python projector_optimization.py --ckpt ./out/trans_spatial_squery_fixed_celeb/checkpoint/370000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --dataset_dir celeba_hq/test/images
