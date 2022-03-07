import os
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pSp.utils import common, train_utils
from pSp.criteria import id_loss, w_norm
from pSp.configs import data_configs
from pSp.datasets.images_dataset import ImagesDataset
from pSp.criteria.lpips.lpips import LPIPS
from pSp.models.psp_new import pSp
from pSp.training.ranger import Ranger
from tqdm import tqdm

from torchvision import transforms
from torch.utils import data


sys.path.append("../../")
from utils.dataset import MultiResolutionDataset
from utils.sample import prepare_noise_new, prepare_param

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device

        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.p_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
            self.z_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        # if self.opts.use_fake_lambda > 0:
            
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        """
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        """
        self.train_dataset, self.test_dataset = self.configure_datasets_lmdb()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            sampler=data_sampler(self.train_dataset, shuffle=True, distributed=self.opts.distributed),
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.opts.test_batch_size,
            sampler=data_sampler(self.test_dataset, shuffle=True, distributed=self.opts.distributed),
            drop_last=True,
        )

        # Initialize logger
        log_dir = os.path.join(self.opts.output_dir, self.opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(self.opts.output_dir, self.opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        
        self.net.train()
        loader = sample_data(self.train_dataloader)
        for global_step in tqdm(range(self.opts.max_steps)):
        # while self.global_step < self.opts.max_steps:
            # with tqdm(desc='psp train', unit='it', total=len(self.train_dataloader)) as pbar:
                # for batch_idx, batch in enumerate(self.train_dataloader):
            # x: from_domain img, y: target_domain image, in single domain case, x and y are same
            # x, y = batch
            # x, y = x.to(self.device).float(), y.to(self.device).float()
            

            self.optimizer.zero_grad()
            batch = next(loader)
            real_img = batch 
            real_img = real_img.to(self.device)
            inversed_img, z_code,p_code = self.net.forward(real_img, return_latents=False) # this will inverse x, then generate
            loss, loss_dict, id_logs = self.calc_loss(real_img, inversed_img, z_code, p_code)
            loss.backward()
            self.optimizer.step()

            loss_fake = 0
            if (self.global_step % self.opts.fake_every == 0) and self.opts.use_fake_lambda>0:
                self.optimizer.zero_grad()

                sample_z = prepare_noise_new(self.opts.batch_size, self.opts, self.device, method="query")  # torch.Size([64, 512,14])
                sample_param = prepare_param(self.opts.batch_size ,self.opts, self.device, method="spatial")

                if self.opts.from_plus_space:
                    sample_z_plus, sample_param_plus = self.net.only_map(sample_z,sample_param)
                    fake_real_img, _, _ = self.net.only_decode(sample_z_plus, sample_param_plus)

                    inversed_fake_img, inversed_z_code,inversed_p_code = self.net.forward(fake_real_img, return_latents=False)
                    loss_fake =  self.opts.use_fake_lambda * self.calc_fake_loss(sample_z_plus, sample_param_plus, inversed_z_code,inversed_p_code)
                    loss_fake.backward()
                    self.optimizer.step()
            loss_dict["fake_guide"] = float(loss_fake)
            

            # Logging related
            if self.global_step % self.opts.image_interval == 0 or (
                self.global_step < 1000 and self.global_step % 25 == 0):
                self.parse_and_log_images(id_logs, real_img, real_img, inversed_img, title='images/train/faces')
            if self.global_step % self.opts.board_interval == 0:
                self.print_metrics(loss_dict, prefix='train')
                self.log_metrics(loss_dict, prefix='train')

            # Validation related
            val_loss_dict = None
            if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                val_loss_dict = self.validate()
                if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

            if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                if val_loss_dict is not None:
                    self.checkpoint_me(val_loss_dict, is_best=False)
                else:
                    self.checkpoint_me(loss_dict, is_best=False)

            if self.global_step == self.opts.max_steps:
                print('OMG, finished training!')
                break

            self.global_step = global_step

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            real_img = batch 

            with torch.no_grad():
                real_img = real_img.to(self.device)
                inversed_img, z_code,p_code = self.net.forward(real_img, return_latents=False) # this will inverse x, then generate
                loss, cur_loss_dict, id_logs = self.calc_loss(real_img, inversed_img, z_code, p_code)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx % 1000 == 0:
                self.parse_and_log_images(id_logs, real_img, real_img, inversed_img,
                                          title='images/test/faces',
                                          subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset_celeba = ImagesDataset(source_root=dataset_args['train_source_root'],
                                             target_root=dataset_args['train_target_root'],
                                             source_transform=transforms_dict['transform_source'],
                                             target_transform=transforms_dict['transform_gt_train'],
                                             opts=self.opts)
        test_dataset_celeba = ImagesDataset(source_root=dataset_args['test_source_root'],
                                            target_root=dataset_args['test_target_root'],
                                            source_transform=transforms_dict['transform_source'],
                                            target_transform=transforms_dict['transform_test'],
                                            opts=self.opts)
        train_dataset = train_dataset_celeba
        test_dataset = test_dataset_celeba
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset
    
    def configure_datasets_lmdb(self):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        train_dataset = MultiResolutionDataset(self.opts.path, transform, self.opts.size)
        test_dataset = MultiResolutionDataset(self.opts.test_path, transform_test, self.opts.size)

        return train_dataset, test_dataset
        
        
    def calc_fake_loss(self, sample_z_plus, sample_param_plus, inversed_z_code,inversed_p_code):
        fake_loss  = self.mse_loss(sample_z_plus,inversed_z_code) + self.mse_loss(sample_param_plus,inversed_p_code)

        return fake_loss  

        
    def calc_loss(self, real_img, inversed_img, z_code,p_code):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(inversed_img, real_img, real_img)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(inversed_img, real_img)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(inversed_img, real_img)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(inversed_img[:, :, 35:223, 32:220], real_img[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(inversed_img[:, :, 35:223, 32:220], real_img[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        # the regu loss 
        # todo: the p+ and z+ case, not added yet
        if self.opts.w_norm_lambda > 0:
            loss_p_norm = self.p_norm_loss(p_code, self.net.p_latent_avg)
            loss_z_norm = self.z_norm_loss(z_code, self.net.z_latent_avg)
            norm = loss_z_norm + loss_p_norm
            loss_dict['loss_w_norm'] = float(loss_p_norm) + float(loss_z_norm)
            loss += norm * self.opts.w_norm_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            if self.opts.from_plus_space:
                save_dict['z_plus_latent_avg'] = self.net.z_plus_latent_avg
                save_dict['p_plus_latent_avg'] = self.net.p_plus_latent_avg
            else:  
                save_dict['z_latent_avg'] = self.net.z_latent_avg 
                save_dict['p_latent_avg'] = self.net.p_latent_avg
        return save_dict
