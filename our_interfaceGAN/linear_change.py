import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys

from interfaceGAN.linear_interpolation import linear_interpolate
from interfaceGAN.train_boundary import train_boundary
from model_allattention import Generator as Allatt_G
from model_noresidual import Generator as NoResidual_G
from model_paramchange import Generator as Param_G
from model_newdia import Generator as Dia_G
from model_onlyinput import Generator as Only_G
from model_changepz import Generator as Change_G
from model_noposition import Generator as Position_G
from model_allpos import Generator as Allpos_G
from model import Generator as G
from projector_z import make_image
from utils import dex
from train import prepare_param

if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--z_latent', type=str, required=True)
    parser.add_argument('--p_latent', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./random_interpolate')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--num_sample', type=int, default=100000)
    parser.add_argument('--para_num', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--start_distance', type=int, default=-30)
    parser.add_argument('--end_distance', type=int, default=30)
    parser.add_argument('--steps', type=int, default=61)
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('--old_version', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='allatt',choices=['allatt','noresidual','paramchange','newdia','onlyinput','changepz','noposition', 'allpos', 'ori'])
    parser.add_argument('--attribute_name', type=str, default='age')

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.mode)

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    args.n_mlp = 8
    args.w_space = False
    new_version = not args.old_version

    ckpt_name = os.path.basename(args.ckpt)
    iter = int(os.path.splitext(ckpt_name)[0])
    exp_name = str(args.ckpt).split('/')[-3]
    sample_path = os.path.join(args.output_dir, exp_name, f'{iter}',args.attribute_name)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    if args.mode == "noresidual":
        g_ema = NoResidual_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "paramchange":
        g_ema = Param_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "allatt":
        g_ema = Allatt_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "newdia":
        g_ema = Dia_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "onlyinput":
        g_ema = Only_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "changepz":
        g_ema = Change_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "noposition":
        g_ema = Position_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "allpos":
        g_ema = Allpos_G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)
    elif args.mode == "ori":
        g_ema = G(args.size, args.latent, args.latent, args.token, args.n_mlp, w_space=args.w_space, new_version=new_version)

    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)

    batch_size = args.batch_size
    num_batch = args.num_sample // batch_size
    last_batch = args.num_sample - (batch_size * num_batch)

    z_latents = []
    p_latents = []
    ages = []
    genders = []
    dex.eval(args.attribute_name)


    z_latent_projected = np.load(args.z_latent)
    p_latent_projected = np.load(args.p_latent)
    count = z_latent_projected.shape[0]
    z_latent_projected = np.reshape(z_latent_projected, (count, -1))
    p_latent_projected = np.reshape(p_latent_projected, (count, -1))

    for num in range(15):
        z_boundary_age = torch.randn(1, z_latent_projected.shape[1], device=device).cpu().numpy()
        p_boundary_age = torch.randn(1, p_latent_projected.shape[1], device=device).cpu().numpy()
        start_distance = args.start_distance
        end_distance = args.end_distance
        steps = args.steps
        with torch.no_grad():
            z_store_path = os.path.join(sample_path,f'{num}','z')
            p_store_path = os.path.join(sample_path,f'{num}','p')
            if not os.path.exists(z_store_path):
                os.makedirs(z_store_path)
            if not os.path.exists(p_store_path):
                os.makedirs(p_store_path)
            for i in tqdm(range(count)):
                # edit age
                z_latent_interpolated = linear_interpolate(z_latent_projected[i:i + 1],
                                                        z_boundary_age,
                                                        start_distance=start_distance,
                                                        end_distance=end_distance,
                                                        steps=steps)
                for j in range(steps):
                    z_latent = torch.from_numpy(z_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    p_input = torch.from_numpy(p_latent_projected[i:i+1].reshape(1,-1,args.latent)).to(device)
                    img_gen, _, _ = g_ema(z_latent, p_input)
                    image = img_gen[:, [2, 1, 0], :, :]
                    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                    age = dex.estimate_age(image)
                    img_ar = make_image(img_gen)
                    img = Image.fromarray(img_ar[0])
                    img.save(os.path.join(z_store_path, f'origin_{i}_edit_{j}_age_{round(age.cpu().numpy()[0])}.png'))
            
            for i in tqdm(range(count)):
                # edit age
                p_latent_interpolated = linear_interpolate(p_latent_projected[i:i + 1],
                                                        p_boundary_age,
                                                        start_distance=start_distance,
                                                        end_distance=end_distance,
                                                        steps=steps)
                for j in range(steps):
                    p_latent = torch.from_numpy(p_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    z_input = torch.from_numpy(z_latent_projected[i:i+1].reshape(1,-1,args.latent)).to(device)
                    img_gen, _, _ = g_ema(z_input, p_latent)
                    image = img_gen[:, [2, 1, 0], :, :]
                    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                    age = dex.estimate_age(image)
                    img_ar = make_image(img_gen)
                    img = Image.fromarray(img_ar[0])
                    img.save(os.path.join(p_store_path, f'origin_{i}_edit_{j}_age_{round(age.cpu().numpy()[0])}.png'))
    
    print('Done!')

# srun --partition=ha_vug --mpi=pmi2 --gres=gpu:1 --job-name=only --ntasks-per-node=1 -n1 python interfaceGAN/linear_change.py --ckpt ./new_out/onlyinput/onlyinput_para_space/checkpoint/570000.pt --z_latent ./projection_w/onlyinput/onlyinput_para_space/570000/latents.npy --p_latent ./projection_w/onlyinput/onlyinput_para_space/570000/param.npy --num_sample 100000 --batch_size 25 --start_distance -30 --end_distance 30 --steps 61 --ratio 0.02 --mode onlyinput --old_version
# srun --partition=ha_vug --mpi=pmi2 --gres=gpu:1 --job-name=only --ntasks-per-node=1 -n1 python interfaceGAN/linear_change.py --ckpt ./new_out/allpos/para_space/checkpoint/170000.pt --z_latent ./projection_w/allpos/para_space/170000/latents.npy --p_latent ./projection_w/allpos/para_space/170000/param.npy --num_sample 100000 --batch_size 25 --start_distance -30 --end_distance 30 --steps 61 --ratio 0.02 --mode allpos
# srun --partition=ha_vug --mpi=pmi2 --gres=gpu:1 --job-name=only --ntasks-per-node=1 -n1 python interfaceGAN/linear_change.py --ckpt ./new_out/changepz/para_space/checkpoint/500000.pt --z_latent ./projection_w/changepz/para_space/500000/latents.npy --p_latent ./projection_w/changepz/para_space/500000/param.npy --num_sample 100000 --batch_size 25 --start_distance -30 --end_distance 30 --steps 61 --ratio 0.02 --mode changepz