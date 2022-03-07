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

from model_spatial_query import Generator
from utils.sample import prepare_noise_new, prepare_param
from torchvision import transforms, utils
from our_interfaceGAN.linear_interpolation import linear_interpolate


def sample_generation(args, sample_path, g_ema):
    sample_param = prepare_param(args.n_sample, args, device,method="spatial", truncation=0.7)
    for i in range(args.loop_num):
        sample_z = prepare_noise_new(args.n_sample, args, device, method="query", truncation=0.7)
        sample, _, _ = g_ema(sample_z, sample_param)
        utils.save_image(
            sample,
            sample_path + f'/{str(i)}.png',
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1, 1),
        )

def swap_z(args, sample_path, g_ema):
    para_base = prepare_param(args.n_sample, args, device, method="spatial",truncation=args.truncation)
    results = []
    store = []

    for i in range(args.loop_num):
        tmp_z = prepare_noise_new(args.n_sample, args, device,"query",truncation=args.truncation)

        sample, _, _ = g_ema(tmp_z, para_base)
        results.append(sample)

    results_tensor = torch.cat(results)
    utils.save_image(
        results_tensor,
        sample_path + '/swap_z.png',
        nrow=int(args.n_sample),
        normalize=True,
        range=(-1, 1),
        padding = 0
    )

def swap_p(args, sample_path, g_ema):
    sample_z = prepare_noise_new(args.n_sample, args, device,"query",truncation=args.truncation)
    results = []
    store = []
    for i in range(args.loop_num):
        para_tmp = prepare_param(args.n_sample, args, device, method="spatial",truncation=args.truncation)

        sample, _, _ = g_ema(sample_z, para_tmp)
        results.append(sample)
       
    results_tensor = torch.cat(results)
    utils.save_image(
        results_tensor,
        sample_path + '/swap_p.png',
        nrow=int(args.n_sample),
        normalize=True,
        range=(-1, 1),
        padding = 0
    )


def interpolate_style_many(args, sample_path, g_ema, num_tests=10):
    sample_path = os.path.join(sample_path, "interp_many", args.interp_space)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    # import pdb; pdb.set_trace()
    for j in range(num_tests):
        para_base = prepare_param(10, args, device,method='spatial_same',truncation=args.truncation) # [bs, 512, 16]
        z_base = prepare_noise_new(8, args, device,"query",truncation=args.truncation) # [bs, 512, 16]

        boundary_z = torch.randn(1, args.latent)# .repeat(args.n_sample, 1,1 )
        results_z = []
        if args.interp_space == "z":
            z_base = z_base.transpose(1,2) # bs, 16, 512 
        # the mapped z space, before trans
        if args.interp_space == "z+": 
            z_plus = g_ema(z_base, para_base[0].repeat(8,1,1),return_only_mapped_z=True).transpose(1,2) # bs, 16, 512 
            z_base = z_plus.clone()
        elif args.interp_space == "w":
            z_w = g_ema(z_base, para_base[0].repeat(8,1,1),return_only_style_latent=True) # [bs, 14, 512]
            z_base = z_w.clone()
            
        for i in range(8):
            interpolated_latent = linear_interpolate(z_base[i:i+1].cpu().numpy(), boundary_z.cpu().numpy(), start_distance = -1, end_distance=1)
            if args.interp_space == "z":
                sample, _, _ = g_ema(torch.from_numpy(interpolated_latent).transpose(1,2).cuda(), para_base)
            elif args.interp_space == "z+":
                sample, _, _ = g_ema(torch.from_numpy(interpolated_latent).transpose(1,2).cuda(), para_base, use_style_mapping=False)
            elif args.interp_space == "w":
                sample, _, _ = g_ema(torch.from_numpy(interpolated_latent).cuda(), para_base, input_is_latent=True)

            results_z.append(sample)

        results_tensor = torch.cat(results_z)
        utils.save_image(
            results_tensor,
            sample_path + f"/interp_{args.interp_space}_{j}.png",
            nrow=int(10),
            normalize=True,
            range=(-1, 1),
        )

def interpolate_style_dat(args, sample_path, g_ema, num_tests=10):
    sample_path = os.path.join(sample_path, "interp_many_dat", args.interp_space)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    for j in range(num_tests):
        para_base = prepare_param(6, args, device,method='spatial',truncation=args.truncation) # [10, 512, 16]
        z_base1 = prepare_noise_new(6, args, device,"query_same",truncation=args.truncation) # [8, 512, 16]
        z_base2 = prepare_noise_new(6, args, device,"query_same",truncation=args.truncation) # [8, 512, 16]

        results_z = []

        if args.interp_space == "z+": 
            z_plus1 = g_ema(z_base1, para_base,return_only_mapped_z=True)  # bs, 512, 16
            z_plus2 = g_ema(z_base2, para_base,return_only_mapped_z=True)  # bs, 512, 16
        
        for i in range(4):
            if args.interp_space == "z":
                interpolated_latent = torch.lerp(z_base1, z_base2, 0.25*(i+1))
                sample, _, _ = g_ema(interpolated_latent, para_base)
            elif args.interp_space == "z+":
                interpolated_latent = torch.lerp(z_plus1, z_plus2, 0.25*(i+1))
                sample, _, _ = g_ema(interpolated_latent, para_base, use_style_mapping=False)
         
            results_z.append(sample)

        results_tensor = torch.cat(results_z)
        utils.save_image(
            results_tensor,
            sample_path + f"/interp_{args.interp_space}_{j}.png",
            nrow=int(6),
            normalize=True,
            range=(-1, 1),
        )


def interpolate_spatial_many(args, sample_path, g_ema, num_tests=10):
    sample_path = os.path.join(sample_path, "interp_many", args.interp_space)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    # import pdb; pdb.set_trace()
    for j in range(num_tests):
        para_base = prepare_param(8, args, device,method='spatial',truncation=args.truncation) # [bs, 512, 16]
        z_base = prepare_noise_new(10, args, device,"query_same",truncation=args.truncation) # [bs, 512, 16]

        
        boundary_para = torch.randn(1, args.latent)# .repeat(args.n_sample, 1,1 )

        results_para = []
        if args.interp_space == "p":
            para_base = para_base.transpose(1,2) # bs, 16, 512 
        # the mapped z space, before trans
        if args.interp_space == "p+": 
            p_plus = g_ema(z_base[0].repeat(8,1,1), para_base,return_only_mapped_p=True).transpose(1,2) # bs, 16, 512 
            para_base = p_plus.clone()
        
            
        for i in range(8):
            interpolated_latent = linear_interpolate(para_base[i:i+1].cpu().numpy(), boundary_para.cpu().numpy(), start_distance = -1, end_distance=1)
            if args.interp_space == "p":
                sample, _, _ = g_ema(z_base,torch.from_numpy(interpolated_latent).transpose(1,2).cuda())
            elif args.interp_space == "p+":
                sample, _, _ = g_ema(z_base, torch.from_numpy(interpolated_latent).transpose(1,2).cuda(), use_spatial_mapping=False)
            

            results_para.append(sample)

        results_tensor = torch.cat(results_para)
        utils.save_image(
            results_tensor,
            sample_path + f"/interp_{args.interp_space}_{j}.png",
            nrow=int(10),
            normalize=True,
            range=(-1, 1),
        )


def interpolate_spatial_dat(args, sample_path, g_ema, num_tests=10):
    sample_path = os.path.join(sample_path, "interp_many_dat", args.interp_space)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    for j in range(num_tests):
        z_base = prepare_noise_new(6, args, device,"query",truncation=args.truncation) # [bs, 512, 16]
        para_base1 = prepare_param(6, args, device,method='spatial_same',truncation=args.truncation) # [10, 512, 16]
        para_base2 = prepare_param(6, args, device,method='spatial_same',truncation=args.truncation) # [10, 512, 16]
      
        results_para = []

        if args.interp_space == "p+": 
            p_plus1 = g_ema(z_base, para_base1,return_only_mapped_p=True)
            p_plus2 = g_ema(z_base, para_base2,return_only_mapped_p=True)
          
        for i in range(4):
            if args.interp_space == "p":
                interpolated_latent = torch.lerp(para_base1, para_base2, 0.25*(i+1))
                sample, _, _ = g_ema(z_base,interpolated_latent)
            elif args.interp_space == "p+":
                interpolated_latent = torch.lerp(p_plus1, p_plus2, 0.25*(i+1))
                sample, _, _ = g_ema(z_base, interpolated_latent, use_spatial_mapping=False)

            results_para.append(sample)

        results_tensor = torch.cat(results_para)
        utils.save_image(
            results_tensor,
            sample_path + f"/interp_{args.interp_space}_{j}.png",
            nrow=int(6),
            normalize=True,
            range=(-1, 1),
        )





if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--loop_num', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./generation')
    parser.add_argument('--para_num', type=int, default=16)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--num_trans', type=int, default=8)
    parser.add_argument('--no_trans', action='store_true', default=False)

    parser.add_argument('--pixel_norm_op_dim', type=int, default=1)
    parser.add_argument('--inject_noise', action='store_true', default=False)

    parser.add_argument('--num_region', type=int, default=1)
    parser.add_argument('--no_spatial_map', action='store_true', default=False)

    # test options
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--swap_z', action='store_true', default=False)
    parser.add_argument('--swap_p', action='store_true', default=False)

    
    parser.add_argument('--interp', action='store_true', default=False)
    parser.add_argument('--interp_space', type=str, default="p")

    parser.add_argument('--dat_interp', action='store_true', default=False)
     # output number for z, z+, w, p, p+ spaces
    parser.add_argument('--interp_num', type=int, default=30)

    parser.add_argument('--truncation', type=float, default=1.0)
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, "visual")
    
    args.check_noise = args.inject_noise

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
    iter = int(os.path.splitext(ckpt_name)[0])
    exp_name = str(args.ckpt).split('/')[-3]
    sample_path = os.path.join(args.output_dir, exp_name, f'{iter}')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    with torch.no_grad():
        if args.sample:
            sample_generation(args, sample_path, g_ema)
        if args.swap_z:
            swap_z(args, sample_path, g_ema)
        if args.swap_p:
            res4 = swap_p(args, sample_path, g_ema)

        if args.interp:
            args.interp_space = 'z'
            interpolate_style_many(args, sample_path, g_ema, num_tests=args.interp_num)
            args.interp_space = 'z+'
            interpolate_style_many(args, sample_path, g_ema,num_tests=args.interp_num//2)
            args.interp_space = 'w'
            interpolate_style_many(args, sample_path, g_ema,num_tests=args.interp_num//3)
            args.interp_space = 'p'
            interpolate_spatial_many(args, sample_path, g_ema,num_tests=args.interp_num)
            args.interp_space = 'p+'
            interpolate_spatial_many(args, sample_path, g_ema,num_tests=args.interp_num//2)

       
        
        if args.dat_interp:
            args.interp_space = 'z'
            interpolate_style_dat(args, sample_path, g_ema, num_tests=args.interp_num//2)
            args.interp_space = 'z+'
            interpolate_style_dat(args, sample_path, g_ema,num_tests=args.interp_num//2)
            args.interp_space = 'p'
            interpolate_spatial_dat(args, sample_path, g_ema,num_tests=args.interp_num//2)
            args.interp_space = 'p+'
            interpolate_spatial_dat(args, sample_path, g_ema,num_tests=args.interp_num//2)
       

        print('Test done!')

# python test_spatial_query.py --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --sample