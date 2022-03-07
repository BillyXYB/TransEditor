from torch.utils.tensorboard import SummaryWriter
import os
import sys
import argparse
import pickle
import math

from torch.nn import functional as F
import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append("./")
sys.path.append("../")

from model_spatial_query import Generator
from utils.sample import prepare_noise_new, prepare_param
from fid_query import extract_feature_from_samples, calc_fid
from calc_inception import load_patched_inception_v3
from lpips import LPIPS
import utils.lpips as lpips


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


# evaluate the fid
@torch.no_grad()
def evaluate_fid(args, g, dataset = 'ffhq'):
    spatial_mean = None
    style_mean = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    if dataset == 'ffhq':
        args.n_sample = 69000
        fid_file = args.inception
    elif dataset == 'celebahq':
        args.n_sample = 29000
        fid_file = "./celeba_hq_stats_256_29000.pkl"

    features = extract_feature_from_samples(
            g, inception, args.truncation, spatial_mean, style_mean , args.batch, args.n_sample, args, device
        ).numpy() # features.shape: (50000, 2048)
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)


    with open(fid_file, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean'] # (2048,)
        real_cov = embeds['cov'] # (2048, 2048)

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid:', fid)
    return fid


@torch.no_grad()
def calculate_lpips_given_images(group_of_images,lpips):
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value

# evaluate the lpips
@torch.no_grad()
def evaluate_lpips(args, generator):
    device = 'cuda'
    lpips = LPIPS().eval().to(device)
    all_lpips_values = []
    fix_p_lpips_values = []
    fix_z_lpips_values = []

    num_paris = 40 
    total_batch = 1000 # 40,000 = 40*1000 as in the DAT
    truncation = 1.0

    for i in tqdm(range(total_batch)):
        # calculate the random pairwise lpips
        sample_param = prepare_param(num_paris, args, device, method="spatial") * truncation
        latent = prepare_noise_new(num_paris, args, device, method="query") * truncation

        img, _, _ = generator(latent, sample_param)
        lpips_value = calculate_lpips_given_images(img, lpips)
        all_lpips_values.append(lpips_value)

        # calculate the pairwise when para fixed, or the lpips of style space
        sample_param = prepare_param(num_paris, args, device, method="spatial_same") * truncation
        latent = prepare_noise_new(num_paris, args, device, method="query") * truncation

        img, _, _ = generator(latent, sample_param)
        lpips_value = calculate_lpips_given_images(img, lpips)
        fix_z_lpips_values.append(lpips_value)
        # calculate the pairwise when z fixed, or the lpips of spatial space
        sample_param = prepare_param(num_paris, args, device, method="spatial") * truncation
        latent = prepare_noise_new(num_paris, args, device, method="query_same") * truncation

        img, _, _ = generator(latent, sample_param)
        lpips_value = calculate_lpips_given_images(img, lpips)
        fix_p_lpips_values.append(lpips_value)

    all_lpips_value = torch.mean(torch.stack(all_lpips_values, dim=0))
    fix_p_lpips_value = torch.mean(torch.stack(fix_p_lpips_values, dim=0))
    fix_z_lpips_value = torch.mean(torch.stack(fix_z_lpips_values, dim=0))
    return all_lpips_value, fix_z_lpips_value, fix_p_lpips_value
    
@torch.no_grad()
def evaluate_ppl(args, generator, space="all", eval_plus=False, crop=False, use_slerp=False):
    # import pdb; pdb.set_trace()
    device = "cuda"
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    args_n_sample = 10000
    args_batch = 64
    args_sampling = "end"
    args_eps = 1e-4

    n_batch = args_n_sample // args_batch
    resid = args_n_sample - (n_batch * args_batch)
    batch_sizes = [args_batch] * n_batch + [resid]
    distances = []
    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            # the total PPL i.e. need to change both spaces
            if space == 'z': # fix p 
                inputs_z = prepare_noise_new(batch * 2, args, device, method="query")
                inputs_p = prepare_param(batch * 2, args, device, method="spatial_same")
            elif space == 'p': # fix z
                inputs_z = prepare_noise_new(batch * 2, args, device, method="query_same")
                inputs_p = prepare_param(batch * 2, args, device, method="spatial")
            else: # all
                inputs_z = prepare_noise_new(batch * 2, args, device, method="query")
                inputs_p = prepare_param(batch * 2, args, device, method="spatial")
            
            if eval_plus:
                inputs_z, inputs_p = generator(inputs_z, inputs_p, return_mapped_codes=True)

            if args_sampling == "full":
                lerp_t = torch.rand(1, device=device)
            else:
                lerp_t = torch.zeros(1, device=device)
                
            # lerp the z code
            if space == 'all':
                z_t0, z_t1 = inputs_z[::2], inputs_z[1::2] # z_t0: bs,512,16 
                p_t0, p_t1 = inputs_p[::2], inputs_p[1::2]
                if not use_slerp:
                    z_e0 = lerp(z_t0, z_t1, lerp_t)
                    z_e1 = lerp(z_t0, z_t1, lerp_t + args_eps)

                    p_e0 = lerp(p_t0, p_t1, lerp_t)
                    p_e1 = lerp(p_t0, p_t1, lerp_t + args_eps)

                else:
                    z_e0 = slerp(z_t0, z_t1, lerp_t)
                    z_e1 = slerp(z_t0, z_t1, lerp_t + args_eps)

                    p_e0 = slerp(p_t0, p_t1, lerp_t)
                    p_e1 = slerp(p_t0, p_t1, lerp_t + args_eps)

                lerped_z = torch.stack([z_e0, z_e1], 1).view(*inputs_z.shape)
                lerped_p = torch.stack([p_e0, p_e1], 1).view(*inputs_p.shape)
            elif space == 'z':
                z_t0, z_t1 = inputs_z[::2], inputs_z[1::2] # z_t0: bs,512,16 
                if not use_slerp:
                    z_e0 = lerp(z_t0, z_t1, lerp_t)
                    z_e1 = lerp(z_t0, z_t1, lerp_t + args_eps)
                else:
                    z_e0 = slerp(z_t0, z_t1, lerp_t)
                    z_e1 = slerp(z_t0, z_t1, lerp_t + args_eps)
                lerped_z = torch.stack([z_e0, z_e1], 1).view(*inputs_z.shape)

                lerped_p = inputs_p
            else: # 'p'
                p_t0, p_t1 = inputs_p[::2], inputs_p[1::2]
                if not use_slerp:
                    p_e0 = lerp(p_t0, p_t1, lerp_t)
                    p_e1 = lerp(p_t0, p_t1, lerp_t + args_eps)
                else:
                    p_e0 = slerp(p_t0, p_t1, lerp_t)
                    p_e1 = slerp(p_t0, p_t1, lerp_t + args_eps)

                lerped_p = torch.stack([p_e0, p_e1], 1).view(*inputs_p.shape)

                lerped_z = inputs_z
            

            if not eval_plus:
                image, _, _ = generator(lerped_z, lerped_p)
            else:
                image, _, _ = generator(lerped_z, lerped_p, use_style_mapping=False, use_spatial_mapping=False)

            if crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]
            
            factor = image.shape[2] // 256
            

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )
            
            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args_eps ** 2
            )
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    # print("eval_plus:", eval_plus, "crop:", crop, "use_slerp:",use_slerp)
    print("ppl_{}:".format(space), filtered_dist.mean())

    return filtered_dist.mean()



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
    parser.add_argument("--dataset", type=str, default='ffhq')

    parser.add_argument('--para_num', type=int, default=16)

    parser.add_argument('--output_dir', type=str, default='./new_generation')

    parser.add_argument('--channel_multiplier', type=int, default=2)

    parser.add_argument('--inject_noise', action='store_true', default=False)

    parser.add_argument('--num_region', type=int, default=1)
    parser.add_argument('--no_spatial_map', action='store_true', default=False)


    parser.add_argument('--num_trans', type=int, default=8)
    parser.add_argument('--no_trans', action='store_true', default=False)

    parser.add_argument('--pixel_norm_op_dim', type=int, default=1)
    # test_options
    parser.add_argument('--fid', action='store_true', default=False)
    parser.add_argument('--lpips', action='store_true', default=False)
    parser.add_argument('--ppl_all', action='store_true', default=False)
    parser.add_argument('--ppl', action='store_true', default=False)


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

    fid_list = []
    i = 0
    print(ckpt)
    
    # import pdb; pdb.set_trace()

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
        # fid
        if args.fid:
            fid = evaluate_fid(args, g, args.dataset)
            fid_list.append(fid)

        # lpips
        if args.lpips:
            all_lpips_value, fix_p_lpips_value, fix_z_lpips_value = evaluate_lpips(args, g)
            print("lpips:", all_lpips_value, fix_p_lpips_value, fix_z_lpips_value)
        
        if args.ppl:
            space_list = ["all", "p", "z"]
            """
            print("p and z space, using slerp, crop")
            for space in space_list:
                evaluate_ppl(args, g, space=space, eval_plus=False, 
                                    use_slerp=True, crop=True)
            """
            print("p+ and z+ space, using lerp, crop")
            for space in space_list:
                evaluate_ppl(args, g, space=space, eval_plus=True, 
                                    use_slerp=False, crop=True)

        # ppl_all
        if args.ppl_all:
            space_list = ["all", "p", "z"]
            plus_list = [True, False]
            slerp_list = [True, False]
            crop_list = [True, False]
            for use_crop in crop_list:
                for use_slerp in slerp_list:
                    for eval_plus in plus_list:
                        for space in space_list:
                            print("space:", space, "eval_plus:", eval_plus, 
                                "crop:", use_crop, "use_slerp:",use_slerp)
                            evaluate_ppl(args, g, space=space, eval_plus=eval_plus, 
                                use_slerp=use_slerp, crop=use_crop)
        


    if args.fid:
        best_fid = fid_list[0]
        best_model_itr = 0
        for model_path, fid in zip(ckpt,fid_list):
            iteration = int(os.path.splitext(os.path.basename(model_path))[0])
            if fid< best_fid:
                best_fid = fid
                best_model_itr = iteration
        print("best fid is:", best_fid," on the model: ", best_model_itr)
    
# spring.submit run --gpu -n1 "python metrics/evaluate_query.py --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 --batch 64 --inception metrics/inception_ffhq.pkl --truncation 1"