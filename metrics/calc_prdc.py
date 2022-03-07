import argparse
import os

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, models
from tqdm import tqdm

from metrics.prdc import compute_prdc
from model import Generator
from train import data_sampler, sample_data
from utils.dataset import MultiResolutionDataset


@torch.no_grad()
def extract_feature_from_samples(generator, vgg, batch_size, n_sample, device):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid == 0:
        batch_sizes = [batch_size] * n_batch
    else:
        batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 14, 512, device=device)
        img, _ = generator(latent)
        feat = vgg(img)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


@torch.no_grad()
def extract_feature_from_data(dataset, vgg, batch_size, n_sample, device):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid == 0:
        batch_sizes = [batch_size] * n_batch
    else:
        batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        loader = data.DataLoader(
            dataset,
            batch_size=batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
        )
        loader = sample_data(loader)
        img = next(loader).to(device)
        feat = vgg(img)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--ckpt', default='./checkpoint')
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    nearest_k = 3
    resize = min(args.size, 256)

    if os.path.isdir(args.ckpt):
        files = os.listdir(args.ckpt)
        ckpt = sorted([os.path.join(args.ckpt, x) for x in files])
        ckpt = list(filter(lambda x: int(x.split('/')[-1].split('.')[0]) >= args.start_num, ckpt))
        print(args.ckpt)
    else:
        ckpt = [args.ckpt]

    print(ckpt)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = MultiResolutionDataset(args.dataset, transform, args.size)

    model_vgg16 = models.vgg16(pretrained=True)
    model_vgg16.classifier = model_vgg16.classifier[:-1]
    model_vgg16 = nn.DataParallel(model_vgg16).to(device)
    model_vgg16.eval()

    for model_path in ckpt:
        iteration = int(os.path.splitext(os.path.basename(model_path))[0])
        print(f'Iteration = {iteration}')

        g = Generator(args.size, 512, 8).to(device)
        model = torch.load(model_path, map_location='cpu')
        g.load_state_dict(model['g_ema'])
        g = nn.DataParallel(g)
        g.eval()

        fake_features = extract_feature_from_samples(g, model_vgg16, args.batch, args.n_sample, device).numpy()
        print(f'extracted {fake_features.shape[0]} fake features')
        real_features = extract_feature_from_data(dataset, model_vgg16, args.batch, args.n_sample, device).numpy()
        print(f'extracted {real_features.shape[0]} real features')

        metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)
        print(metrics)
