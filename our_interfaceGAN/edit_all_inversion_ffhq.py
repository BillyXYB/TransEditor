import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
import json

sys.path.append("./")
sys.path.append("../")
from our_interfaceGAN.linear_interpolation import linear_interpolate
from our_interfaceGAN.train_boundary import train_boundary

from model_spatial_query import Generator
from utils.editing_utils import make_image, visualize
from ffhq_utils import dex
from torchvision import transforms, utils
from utils.sample import prepare_param, prepare_noise_new
from glob import glob



if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--z_latent', type=str, default='./projection/encoder_inversion/ffhq_encode/encoded_z.npy')
    parser.add_argument('--p_latent', type=str, default='./projection/encoder_inversion/ffhq_encode/encoded_p.npy')
    parser.add_argument('--output_dir', type=str, default='./editing_inversion')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--num_sample', type=int, default=150000)
    parser.add_argument('--para_num', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--start_distance', type=int, default=-30)
    parser.add_argument('--end_distance', type=int, default=30)
    parser.add_argument('--steps', type=int, default=61)
    parser.add_argument('--ratio', type=float, default=0.02)
    parser.add_argument('--old_version', action='store_true', default=False)
    parser.add_argument('--attribute_name', type=str, default='age',choices=['gender','pose','age'])
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--retrain_boundrary', action='store_true', default=False)
    parser.add_argument('--pixel_norm_op_dim', type=int, default=1)
    parser.add_argument('--num_trans', type=int, default=8)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--inject_noise', action='store_true', default=False)
    parser.add_argument('--num_region', type=int, default=1)
    parser.add_argument('--no_spatial_map', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0) # 0:encoder 1:optimize
    parser.add_argument('--no_trans', action='store_true', default=False)


    args = parser.parse_args()

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    # args.n_mlp = 8
    args.w_space = False

    new_version = not args.old_version 

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
    args.output_dir = os.path.join(args.output_dir, exp_name, f'{iter}')


    batch_size = args.batch_size
    num_batch = args.num_sample // batch_size
    last_batch = args.num_sample - (batch_size * num_batch)

    w_latents = []
    p_latents = []
    ages = []
    genders = []
    dex.eval(args.attribute_name)
    boundrary_path = os.path.join(args.output_dir,f'boundrary/{args.attribute_name}')

    os.makedirs(boundrary_path, exist_ok=True)

    z_boundrary_file = os.path.join(boundrary_path, "z_boundrary.npy")
    p_boundrary_file = os.path.join(boundrary_path, "p_boundrary.npy")

    if os.path.exists(z_boundrary_file):
        z_boundary_age = np.load(z_boundrary_file)
    if os.path.exists(p_boundrary_file):
        p_boundary_age = np.load(p_boundrary_file)

    if not os.path.exists(z_boundrary_file) or args.retrain_boundrary:
        print(f"Starting to generate {args.num_sample} random samples...")
        with torch.no_grad():
            for b in tqdm(range(num_batch)):
                noise = prepare_noise_new(batch_size, args, device,"query",truncation=args.truncation)
                para_base = prepare_param(batch_size, args, device, method='spatial',truncation = args.truncation)

                z_plus = g_ema(noise, para_base,return_only_mapped_z=True)
                p_plus = g_ema(noise, para_base,return_only_mapped_p=True)
                img, _ ,_ = g_ema(z_plus, p_plus, use_spatial_mapping=False, use_style_mapping=False)
                w_latents.append(z_plus.transpose(1,2).cpu())
                p_latents.append(p_plus.transpose(1,2).cpu())
            
                # change from RGB to GBR
                image = img[:, [2, 1, 0], :, :]
                # normalize to [0, 255]
                image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                # estimate age
                if args.attribute_name == "age":
                    age = dex.estimate_age(image)
                else:
                    age = dex.estimate_gender(image)
                ages.append(age.cpu())

            if last_batch != 0:
                noise = prepare_noise_new(last_batch, args, device,"query",truncation=args.truncation)
                para_base = prepare_param(last_batch, args, device, method='spatial',truncation = args.truncation)

                z_plus = g_ema(noise, para_base,return_only_mapped_z=True)
                p_plus = g_ema(noise, para_base,return_only_mapped_p=True)
                img, _ ,_ = g_ema(z_plus, p_plus, use_spatial_mapping=False, use_style_mapping=False)
                w_latents.append(z_plus.transpose(1,2).cpu())
                p_latents.append(p_plus.transpose(1,2).cpu())

                # change from RGB to GBR
                image = img[:, [2, 1, 0], :, :]
                # normalize to [0, 255]
                image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                # estimate age
                if args.attribute_name == "age":
                    age = dex.estimate_age(image)
                else:
                    age = dex.estimate_gender(image)
                ages.append(age.cpu())

        print(f"{args.num_sample} random samples generated...")
        w_latent_codes = torch.cat(w_latents, dim=0).reshape(args.num_sample, -1).numpy()
        p_latent_codes = torch.cat(p_latents, dim=0).reshape(args.num_sample, -1).numpy() 
        scores_age = torch.cat(ages, dim=0).reshape(args.num_sample, -1).numpy() # (1000, 1)

        chosen_num_or_ratio = args.ratio # 0.02
        split_ratio = 0.7
        invalid_value = None
        z_boundary_age = train_boundary(latent_codes=w_latent_codes,
                                    scores=scores_age,
                                    chosen_num_or_ratio=chosen_num_or_ratio,
                                    split_ratio=split_ratio,
                                    invalid_value=invalid_value)
        print(f"{args.attribute_name} Boundary trained for style space...")
        np.save(z_boundrary_file, z_boundary_age)

        p_boundary_age = train_boundary(latent_codes=p_latent_codes,
                                    scores=scores_age,
                                    chosen_num_or_ratio=chosen_num_or_ratio,
                                    split_ratio=split_ratio,
                                    invalid_value=invalid_value)
        print(f"{args.attribute_name} Boundary trained for param space...")

        np.save(p_boundrary_file, p_boundary_age)
    else:
        print('boundary exists!')


    z_latent_projected = torch.from_numpy(np.load(args.z_latent))
    p_latent_projected = torch.from_numpy(np.load(args.p_latent))

    count = z_latent_projected.shape[0]


    z_latent_projected = np.reshape(z_latent_projected.transpose(1,2).cpu().detach().numpy(), (count, -1))
    p_latent_projected = np.reshape(p_latent_projected.transpose(1,2).cpu().detach().numpy(), (count, -1))


    config_path = './our_interfaceGAN/config_inversion'
    config_path = os.path.join(config_path, f"{args.attribute_name}.json")
    with open(config_path, 'r') as j:
        step_dict = json.loads(j.read())
    
    steps = args.steps
    
    for e_s in step_dict['style_end_distance']:
        for e_c in step_dict['content_end_distance']:

            w_store_path = os.path.join(args.output_dir, f"{args.attribute_name}/{e_s}_{e_c}")

            for i in tqdm(range(count)):
                tmp_p_plus_path = os.path.join(w_store_path, f"image_{i}", "p_plus" )
                tmp_z_plus_path = os.path.join(w_store_path, f"image_{i}", "z_plus")
                tmp_pz_plus_path = os.path.join(w_store_path, f"image_{i}", "pz_plus")
                os.makedirs(tmp_p_plus_path, exist_ok=True)
                os.makedirs(tmp_z_plus_path, exist_ok=True)
                os.makedirs(tmp_pz_plus_path, exist_ok=True)

                z_latent_interpolated = linear_interpolate(z_latent_projected[i:i + 1],
                                                        z_boundary_age,
                                                        start_distance=-int(e_s),
                                                        end_distance=int(e_s),
                                                        steps=steps)
                p_latent_interpolated = linear_interpolate(p_latent_projected[i:i + 1],
                                                        p_boundary_age,
                                                        start_distance=-int(e_c),
                                                        end_distance=int(e_c),
                                                        steps=steps)

                # edit in two space
                for j in range(steps):
                    z_latent = torch.from_numpy(z_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    p_latent = torch.from_numpy(p_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    z_latent = z_latent.transpose(1,2)
                    p_latent = p_latent.transpose(1,2)
                    img_gen, _, _ = g_ema(z_latent, p_latent,use_style_mapping=False, use_spatial_mapping=False)
                    image = img_gen[:, [2, 1, 0], :, :]
                    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                    if args.attribute_name == "age":
                        age = dex.estimate_age(image)
                    else:
                        age = dex.estimate_gender(image)
                    img_ar = make_image(img_gen)
                    img = Image.fromarray(img_ar[0])
                    img.save(os.path.join(tmp_pz_plus_path, f'origin_{i}_edit_{j}_{args.attribute_name}_{round(age.cpu().numpy()[0])}.png'))
                
                for j in range(steps):
                    p_latent = torch.from_numpy(p_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    z_input = torch.from_numpy(z_latent_projected[i:i+1]).reshape(1, -1, args.latent).to(device)
                    img_gen, _, _ = g_ema(z_input.transpose(1,2), p_latent.transpose(1,2),use_style_mapping=False, use_spatial_mapping=False)
                    image = img_gen[:, [2, 1, 0], :, :]
                    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                    if args.attribute_name == "age":
                        age = dex.estimate_age(image)
                    else:
                        age = dex.estimate_gender(image)
                    img_ar = make_image(img_gen)
                    img = Image.fromarray(img_ar[0])
                    img.save(os.path.join(tmp_p_plus_path, f'origin_{i}_edit_{j}_{args.attribute_name}_{round(age.cpu().numpy()[0])}.png'))
                
                for j in range(steps):
                    z_latent = torch.from_numpy(z_latent_interpolated[j:j + 1]).reshape(1, -1, args.latent).to(device)
                    p_input = torch.from_numpy(p_latent_projected[i:i+1]).reshape(1, -1, args.latent).to(device)
                    z_latent = z_latent.transpose(1,2)
                    p_input = p_input.transpose(1,2)
                    img_gen, _, _ = g_ema(z_latent, p_input,use_style_mapping=False, use_spatial_mapping=False)

                    image = img_gen[:, [2, 1, 0], :, :]
                    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
                    if args.attribute_name == "age":
                        age = dex.estimate_age(image)
                    else:
                        age = dex.estimate_gender(image)
                    img_ar = make_image(img_gen)
                    img = Image.fromarray(img_ar[0])
                    img.save(os.path.join(tmp_z_plus_path, f'origin_{i}_edit_{j}_{args.attribute_name}_{round(age.cpu().numpy()[0])}.png'))

                visualize(tmp_p_plus_path)
                visualize(tmp_z_plus_path)
                visualize(tmp_pz_plus_path)

            print(f"{steps} interpolation generated for {count} samples...")

# python our_interfaceGAN/edit_all_inversion_ffhq.py --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --attribute_name pose --z_latent ./projection/encoder_inversion/ffhq_encode/encoded_z.npy --p_latent ./projection/encoder_inversion/ffhq_encode/encoded_p.npy
