import torch

def prepare_param(n_sample, args, device, method="batch_same", truncation=1.0):
    if method == "batch_same":
        return torch.randn(args.para_num, args.latent, device=device).repeat(n_sample, 1, 1) * truncation
    elif method == "batch_diff":
        return torch.randn(n_sample, args.para_num, args.latent, device=device) * truncation
    elif method == "spatial":
        # batch, 512，16
        return torch.randn(n_sample, args.latent, args.para_num, device=device) * truncation
    elif method == "spatial_same":
        return torch.randn(args.latent, args.para_num, device=device).repeat(n_sample,1,1) * truncation
    


def prepare_noise_new(n_sample, args, device, method="multi", truncation=1.0, mode = 'train'):
    # used for train_spatial_query, returns (bs, 512, 16)
    if method == 'query':
        return torch.randn(n_sample, args.latent, args.para_num, device=device)  * truncation
    elif method == 'query_same':
        return torch.randn(args.latent, args.para_num, device=device).repeat(n_sample,1,1) * truncation