import argparse
import os
import pickle
import math

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append("./")
sys.path.append("../")

from metrics.calc_inception import load_patched_inception_v3


from model_spatial_query import Generator
from utils.sample import prepare_noise_new, prepare_param


@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_spatial_latent, truncation_style_latent, batch_size, n_sample,args, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        sample_param = prepare_param(batch, args, device, method="spatial") * truncation
        latent = prepare_noise_new(batch, args, device, method="query") * truncation

        img, _, _ = generator(latent, sample_param)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None, required=True)
    parser.add_argument('--ckpt', default='./checkpoint')
    
    parser.add_argument('--para_num', type=int, default=16)
   

    parser.add_argument('--output_dir', type=str, default='./new_generation')

    parser.add_argument('--channel_multiplier', type=int, default=2)

    parser.add_argument('--inject_noise', action='store_true', default=False)

    parser.add_argument('--num_region', type=int, default=1)
    parser.add_argument('--no_spatial_map', action='store_true', default=False)


    parser.add_argument('--num_trans', type=int, default=8)
    parser.add_argument('--no_trans', action='store_true', default=False)

    parser.add_argument('--pixel_norm_op_dim', type=int, default=1)

    args = parser.parse_args()
    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
  
    args.use_spatial_mapping = True

    _tensorboard_path = "tensorboard_log"
    tensorboard_writer = SummaryWriter(_tensorboard_path)
    if not os.path.exists(_tensorboard_path):
        os.makedirs(_tensorboard_path)

    if os.path.isdir(args.ckpt):
        files = os.listdir(args.ckpt)
        ckpt = sorted([os.path.join(args.ckpt, x) for x in files])
        ckpt = list(filter(lambda x: int(x.split('/')[-1].split('.')[0])>=args.start_num, ckpt))
        print(args.ckpt)
    else:
        ckpt = [args.ckpt]

    print(ckpt)

    for model_path in ckpt:
        iteration = int(os.path.splitext(os.path.basename(model_path))[0])
        print(f'Iteration = {iteration}')
        g = Generator(
            args.size, args.latent, args.latent, args.token, 
            channel_multiplier=args.channel_multiplier, layer_noise_injection=args.inject_noise,
            use_spatial_mapping=args.use_spatial_mapping, num_region=args.num_region, n_trans=args.num_trans,
            pixel_norm_op_dim=args.pixel_norm_op_dim, no_trans=args.no_trans
        ).to(device)

        model = torch.load(model_path, map_location='cpu')
        g.load_state_dict(model['g_ema'])
        g = nn.DataParallel(g)
        g.eval()

        """
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g.mean_latent(args.truncation_mean)

        else:
            mean_latent = None
        
        if args.truncation < 1:
            spatial_mean = prepare_param(args.truncation_mean, args, device, method="spatial").mean(0, keepdim=True)
            style_mean = prepare_noise_new(args.truncation_mean, args, device, method="query").mean(0, keepdim=True)
        """
        spatial_mean = None
        style_mean = None

        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()

        features = extract_feature_from_samples(
            g, inception, args.truncation, spatial_mean, style_mean , args.batch, args.n_sample, args, device
        ).numpy() # features.shape: (50000, 2048)
        print(f'extracted {features.shape[0]} features')

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)


        with open(args.inception, 'rb') as f:
            embeds = pickle.load(f)
            real_mean = embeds['mean'] # (2048,)
            real_cov = embeds['cov'] # (2048, 2048)

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

        print('fid:', fid)
        tensorboard_writer.add_scalar("FID", fid, iteration)
    tensorboard_writer.close()

# srun --partition=pat_pluto --gres=gpu:1 --ntasks-per-node=1 -n1 python metrics/fid_query.py --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --batch 64 --inception metrics/inception_ffhq.pkl --truncation 0.5
# spring.submit run --gpu -n1 "python metrics/fid.py --ckpt /mnt/lustre/share_data/xuyanbo/to_yueqin/out/trans_spatial_nonoise_singlemap/checkpoint/580000.pt --n_mlp 8 --truncation 1 --batch 64 --para_num 16 --inception inception_ffhq.pkl --mode old"
# srun --partition=pat_pluto --gres=gpu:1 --ntasks-per-node=1 -n1 python metrics/fid.py --ckpt /mnt/lustre/share_data/xuyanbo/to_yueqin/out/trans_spatial_nonoise_singlemap/checkpoint/580000.pt --n_mlp 8 --truncation 1 --batch 64 --para_num 16 --inception inception_ffhq.pkl --mode old