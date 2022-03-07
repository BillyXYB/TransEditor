import json
import os
import pprint
import sys
import argparse
import math


sys.path.append("./pSp")

from psp_training_options import TrainOptions
from pSp.training.coach_new import Coach


if __name__ == '__main__':
    device = 'cuda'

    args = TrainOptions().parse()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)
    
    args.use_spatial_mapping = not args.no_spatial_map

    coach = Coach(args)
    coach.train()


# python psp_spatial_train.py ffhq/LMDB_train/ --test_path ffhq/LMDB_test/ --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1"
# python psp_spatial_train.py ffhq/LMDB_train/ --test_path ffhq/LMDB_test/ --ckpt ./out/trans_spatial_squery_multimap_fixed/checkpoint/790000.pt --num_region 1 --num_trans 8 --pixel_norm_op_dim 1 