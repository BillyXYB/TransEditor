from argparse import ArgumentParser

from pSp.configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # self.parser.add_argument('--ckpt', type=str, required=True)
        self.parser.add_argument('path', type=str)
        self.parser.add_argument('--test_path', type=str, required=True)
        self.parser.add_argument('--ckpt', type=str, required=False)
        self.parser.add_argument('--size', type=int, default=256)
        self.parser.add_argument('--n_sample', type=int, default=8)
        self.parser.add_argument('--loop_num', type=int, default=10)
        self.parser.add_argument('--output_dir', type=str, default='./psp_out')
        self.parser.add_argument('--para_num', type=int, default=16)
        """
        self.parser.add_argument('--sample_z', action='store_true', default=False)
        self.parser.add_argument('--change_para', action='store_true', default=False)
        self.parser.add_argument('--change_z', action='store_true', default=False)
        self.parser.add_argument('--swap_z', action='store_true', default=False)
        self.parser.add_argument('--swap_para', action='store_true', default=False)
        self.parser.add_argument('--yyq_param', action='store_true', default=False)
        self.parser.add_argument('--interp', action='store_true', default=False)
        self.parser.add_argument('--change_region_code', action='store_true', default=False)
        self.parser.add_argument('--mix_region_code', action='store_true', default=False)

        self.parser.add_argument('--old_version', action='store_true', default=False)
        self.parser.add_argument('--store_transout', action='store_true', default=False)
        self.parser.add_argument('--mode', type=str, default='allatt',
                                 choices=['allatt', 'noresidual', 'paramchange', 'newdia', 'onlyinput', 'changepz',
                                          'noposition',
                                          'allpos'])
        """

        self.parser.add_argument('--channel_multiplier', type=int, default=2)

        self.parser.add_argument('--inject_noise', action='store_true', default=False)

        self.parser.add_argument('--num_region', type=int, default=1)
        self.parser.add_argument('--no_spatial_map', action='store_true', default=False)

        self.parser.add_argument('--num_trans', type=int, default=8)
        self.parser.add_argument('--no_trans', action='store_true', default=False)

        self.parser.add_argument('--pixel_norm_op_dim', type=int, default=1)

        # for psp
        self.parser.add_argument('--exp_dir', type=str, default= "psp_training_dir",help='Path to experiment output directory')

        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=3, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int,
                                 help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=256, type=int, help='Output size of generator')

        self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=8, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')
        self.parser.add_argument('--fake_every', default = 10, type=int)
        self.parser.add_argument("--use_fake_lambda", default = 0.0, type=float)
            
        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=1000, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=2500, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=5000, type=int, help='Model checkpoint interval')

        # arguments for super-resolution
        self.parser.add_argument('--resize_factors', type=str, default=None,
                                 help='For super-res, comma-separated resize factors to use for inference.')

        self.parser.add_argument('--from_plus_space', action='store_true') # invert to p+ and z+
        

    def parse(self):
        opts = self.parser.parse_args()
        return opts
