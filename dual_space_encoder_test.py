from dual_space_encoder import DualSpaceEncoder
from psp_testing_options import TestOptions
import os
import math
from utils import lpips
from torch.utils.data import DataLoader
from utils.dataset_projector import MultiResolutionDataset
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

  
def make_noise(log_size = 8):
    device = 'cuda'

    noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
    return noises

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


def editing(z_code, p_code):
    return z_code, p_code


if __name__ == '__main__':
    device = 'cuda'

    psp_config = TestOptions()

    # for Optimization configuration
    psp_config.parser.add_argument('--lr_rampup', type=float, default=0.05)
    psp_config.parser.add_argument('--lr_rampdown', type=float, default=0.25)
    psp_config.parser.add_argument('--lr', type=float, default=0.1)
    psp_config.parser.add_argument('--noise', type=float, default=0.05)
    psp_config.parser.add_argument('--noise_ramp', type=float, default=0.75)
    psp_config.parser.add_argument('--noise_regularize', type=float, default=1e5)
    psp_config.parser.add_argument('--mse', type=float, default=0)
    psp_config.parser.add_argument('--encode_batch', type=int, default=8) # batch encoder
    psp_config.parser.add_argument('--optimization_batch', type=int, default=1) # optimize one by one
    psp_config.parser.add_argument('--seed', type=int, default=1) # different seed for each experiments
    psp_config.parser.add_argument('--dataset_dir', type=str, required=True) # the path stores the ffhq test 500/celeba test 500 images
    psp_config.parser.add_argument('--loop', type=int, default=2000) # 


    args = psp_config.parse()

    args.output_dir = os.path.join(args.output_dir, "encoder_inversion", f"{args.dataset_type}")

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    
    args.use_spatial_mapping = not args.no_spatial_map  

    os.makedirs(args.output_dir, exist_ok=True)

    args.encoded_z_npy = os.path.join(args.output_dir, "encoded_z.npy")
    args.encoded_p_npy = os.path.join(args.output_dir, "encoded_p.npy")

    network = DualSpaceEncoder(args)

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    res_latent = []
    res_param = []
    res_perceptual_values = []
    res_mse_values = []

    dataset = MultiResolutionDataset(args.dataset_dir ,resolution=args.size)

    if os.path.isfile(args.encoded_z_npy):
        print("encoded_npy exists!")
        z_latent_codes = np.load(args.encoded_z_npy)
        p_latent_codes = np.load(args.encoded_p_npy)
    else:
        loader = DataLoader(dataset, shuffle=False, batch_size=args.encode_batch, num_workers=4, drop_last = False)

        z_latent_codes_enc = []
        p_latent_codes_enc = []

        with tqdm(desc='Generation', unit='it', total=len(loader)) as pbar_1:
            for it, images in enumerate(iter(loader)):
                images = images.to(device)
                z_code, p_code = network.encode(images)
                z_latent_codes_enc.append(z_code.cpu().detach().numpy())
                p_latent_codes_enc.append(p_code.cpu().detach().numpy())

                pbar_1.update()
            
            z_latent_codes = np.concatenate(z_latent_codes_enc, axis=0)
            p_latent_codes = np.concatenate(p_latent_codes_enc, axis=0)
            # save the encodeded latent codes
            np.save(f"{args.encoded_z_npy}", z_latent_codes)
            np.save(f"{args.encoded_p_npy}", p_latent_codes)
            
    print('z_latent_codes.shape',z_latent_codes.shape)
    print('p_latent_codes.shape',p_latent_codes.shape)
    
    # below is the optimization inversion based on the perivious inverted codes
    # z_latent_codes = torch.from_numpy(z_latent_codes) # [500, 512, 16]
    # p_latent_codes = torch.from_numpy(p_latent_codes) # [500, 512, 16]
    # latent_mean = z_latent_codes.mean(0).to(device)
    # param_mean = p_latent_codes.mean(0).to(device)
    # n_mean_latent = z_latent_codes.shape[0]

    # loader = DataLoader(dataset, shuffle=False, batch_size=args.optimization_batch, num_workers=4, drop_last = False)

    # res_latent = []
    # res_param = []
    # res_perceptual_values = []
    # res_noise_values = []
    # res_mse_values = []

    # for it, imgs in enumerate(iter(loader)):
    #     imgs = imgs.to(device)
    #     latent_in = z_latent_codes[it: it+1].to(device)
    #     param_in = p_latent_codes[it: it+1].to(device)
    #     latent_std = ((latent_in - latent_mean).pow(1).sum([0, 1]) / n_mean_latent) ** 0.5
    #     param_std = ((param_in - param_mean).pow(1).sum([0, 1]) / n_mean_latent) ** 0.5
    #     noise_single = make_noise()
    #     noises = []
    #     for noise in noise_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())

    #     latent_in.requires_grad = True
    #     param_in.requires_grad = True
    #     if args.inject_noise:
    #         for noise in noises:
    #             noise.requires_grad = True
    #         optimizer = optim.Adam([latent_in] + [param_in] + noises, lr=args.lr)
    #     else:
    #         for noise in noises:
    #             noise.requires_grad = False
    #         optimizer = optim.Adam([latent_in] + [param_in], lr=args.lr)
    
    #     pbar = tqdm(range(args.loop))
    #     latent_path = []
    #     param_path = []
    #     perceptual_values = []
    #     noise_values = []
    #     mse_values = []

    #     for i in pbar:
    #         t = i / args.loop
    #         lr = get_lr(t, args.lr, rampdown=args.lr_rampdown, rampup=args.lr_rampup)
    #         optimizer.param_groups[0]['lr'] = lr
    #         if args.inject_noise:
    #             noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
    #             latent_n = latent_noise(latent_in, noise_strength)
    #             img_gen = network.decode(latent_n, param_in)
    #         else:
    #             img_gen = network.decode(latent_in, param_in) # noise ???

    #         batch, channel, height, width = img_gen.shape

    #         if height > 256:
    #             factor = height // 256

    #             img_gen = img_gen.reshape(
    #                 batch, channel, height // factor, factor, width // factor, factor
    #             )
    #             img_gen = img_gen.mean([3, 5])

    #         p_loss = percept(img_gen, imgs).sum()
    #         n_loss = noise_regularize_(noises)
    #         mse_loss = F.mse_loss(img_gen, imgs)

    #         if args.inject_noise:
    #             loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss
    #         else:
    #             loss = p_loss + args.mse * mse_loss

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         noise_normalize_(noises)

    #         if (i + 1) % 100 == 0:
    #             latent_path.append(latent_in.detach().clone())
    #             param_path.append(param_in.detach().clone())

    #         if (i + 1) % 10 == 0:
    #             perceptual_values.append(p_loss.item())
    #             noise_values.append(n_loss.item())
    #             mse_values.append(mse_loss.item())

    #         pbar.set_description(
    #             (
    #                 f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
    #                 f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'
    #             )
    #         )

    #     if args.inject_noise:
    #         img_gen = network.decode(latent_path[-1], param_path[-1]) # noise ??? 
    #     else:
    #         img_gen = network.decode(latent_path[-1], param_path[-1])

    #     img_or = make_image(imgs)
    #     img_ar = make_image(img_gen)

    #     res_latent.append(latent_path[-1])
    #     res_param.append(param_path[-1])
    #     res_perceptual_values.append(perceptual_values[-1])
    #     res_noise_values.append(noise_values[-1])
    #     res_mse_values.append(mse_values[-1])

    #     batch_size = 1
    #     for i in range(batch):
    #         index = i + it * batch_size
    #         img1 = Image.fromarray(img_or[i])
    #         img1.save(os.path.join(sample_path, f'origin_{index}.png'))
    #         img2 = Image.fromarray(img_ar[i])
    #         img2.save(os.path.join(sample_path, f'project_{index}.png'))

    # res_latent = torch.cat(res_latent)
    # res_param = torch.cat(res_param)
    # print('res_latent.shape',res_latent.shape)
    # print('res_param.shape',res_param.shape)

    # np.save(os.path.join(sample_path, f'latents.npy'), res_latent.cpu().numpy())
    # np.save(os.path.join(sample_path, f'param.npy'), res_param.cpu().numpy())
    # np.save(os.path.join(sample_path, f'perceptual.npy'), res_perceptual_values)
    # np.save(os.path.join(sample_path, f'noise.npy'), res_noise_values)
    # np.save(os.path.join(sample_path, f'mse.npy'), res_mse_values)
