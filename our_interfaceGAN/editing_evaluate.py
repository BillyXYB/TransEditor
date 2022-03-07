import argparse
import math
import os
from re import I

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
import json

from our_interfaceGAN.linear_interpolation import linear_interpolate
from our_interfaceGAN.train_boundary import train_boundary

from model_spatial_query import Generator
from celebahq_utils import dex
from torchvision import transforms, utils
from utils.sample import prepare_param, prepare_noise_new
from glob import glob

from PIL import Image
from ffhq_utils.dex.models import Age, Gender, ClassifyModel18


attribute_list = [
                "Male",
                "Smiling",
                "Attractive",
                "Wavy_Hair",
                "Young",
                "5_o_Clock_Shadow",
                "Arched_Eyebrows",
                "Bags_Under_Eyes",
                "Bald",
                "Bangs",
                "Big_Lips",
                "Big_Nose",
                "Black_Hair",
                "Blond_Hair",
                "Blurry",
                "Brown_Hair",
                "Bushy_Eyebrows",
                "Chubby",
                "Double_Chin",
                "Eyeglasses",
                "Goatee",
                "Gray_Hair",
                "Heavy_Makeup",
                "High_Cheekbones",
                "Mouth_Slightly_Open",
                "Mustache",
                "Narrow_Eyes",
                "No_Beard",
                "Oval_Face",
                "Pale_Skin",
                "Pointy_Nose",
                "Receding_Hairline",
                "Rosy_Cheeks",
                "Sideburns",
                "Straight_Hair",
                "Wearing_Earrings",
                "Wearing_Hat",
                "Wearing_Lipstick",
                "Wearing_Necklace",
                "Wearing_Necktie",
                "Pose"
            ]

base_dir = "/mnt/lustre/xuyanbo/Controllable-Face-Generation/our_interfaceGAN/utils/dex"
pose_model_path = os.path.join(base_dir, 'pth/classifier/pose/weight.pkl')
age_model_path =  os.path.join(base_dir,'pth/age_sd.pth')
gender_model_path = os.path.join(base_dir, 'pth/gender_sd.pth')


test_attribute_list = ["Male", "Smiling", "Wavy_Hair", "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Young", "pose", "age", "gender"]


def calculate_score(attr_classifier, img, no_soft=True):
    # change from RGB to GBR
    image = img[:, [2, 1, 0], :, :]
    # normalize to [0, 255]
    image = image.clamp(min=-1, max=1).add(1).div_(2).mul(255).round()
    # get the score of the imaeg
    score = dex.estimate_score(attr_classifier, image, no_soft=no_soft)
    
    return score


def read_image_to_tensor(img_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )

    image = Image.open(img_dir).convert('RGB')
    tensor = transform(image)
    return tensor

def estimate_pose(pose_model, img):
    tensor = transforms.CenterCrop(224)(img)
    with torch.no_grad():
        output = pose_model(tensor)[:,0] 
    return output



if __name__ == '__main__':

    device = 'cuda'
    parser = argparse.ArgumentParser()

    parser.add_argument("--attribute", type=str, required=True)
    parser.add_argument("--test_num", type=int, default=500)
    parser.add_argument("--method_dir", type=str, default="Controllable-Face-Generation")
    parser.add_argument("--softmax", action='store_true', default=False)
    args = parser.parse_args()

    total_num = args.test_num
    eval_base_dir = os.path.join("/mnt/lustre/xuyanbo",args.method_dir, "editing_evaluation")
    eval_dir = os.path.join(eval_base_dir, args.attribute)

    single_space = False
    no_soft = not args.softmax

    if args.method_dir in ["StyleMapGAN", "stylegan2-pytorch"]:
        single_space = True

    
    origin_image_dir = os.path.join(eval_dir, "origin_image")

    result_list = []

    p_plus_dir = os.path.join(eval_dir, "p_plus") 
    pz_plus_dir = os.path.join(eval_dir, "pz_plus")  
    z_plus_dir = os.path.join(eval_dir, "z_plus")

    img_list = list(range(total_num))
    # read the images
    # ori_imgs # 8, 3, 256,256
    # p_imgs # 8, 6, 3, 256, 256
    # pz_imgs # 8, 6, 3, 256, 256
    # z_imgs # 8, 6, 3, 256, 256

    # define the attribute classifier
    for img_idx in range(8*len(img_list)):
        tmp_dict = {}
        result_list.append(tmp_dict)

    for attribute_name in test_attribute_list:
        if attribute_name in ["pose", "age", "gender"]:
            continue

        attr_classifier = dex.eval(attribute_name)
        
        for img_num in tqdm(img_list):
            ori_imgs = read_image_to_tensor(os.path.join(origin_image_dir,"{}.png".format(img_num))).reshape(3,8,256,256).permute(1,0,2,3).to(device=device)*255
            z_imgs = read_image_to_tensor(os.path.join(z_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255
            if not single_space:
                p_imgs = read_image_to_tensor(os.path.join(p_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255
                pz_imgs = read_image_to_tensor(os.path.join(pz_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255

            for img_idx in range(8):
                p_tmp_list = []
                pz_tmp_list = []
                z_tmp_list = []
                tmp_dict = {}
                ori_score = calculate_score(attr_classifier, ori_imgs[img_idx].unsqueeze(0), no_soft).detach().cpu().numpy()
                
                if not single_space:
                    p_score = calculate_score(attr_classifier, p_imgs[img_idx], no_soft).detach().cpu().numpy()
                    pz_score = calculate_score(attr_classifier, pz_imgs[img_idx], no_soft).detach().cpu().numpy()
                z_score = calculate_score(attr_classifier, z_imgs[img_idx], no_soft).detach().cpu().numpy()

                for i in range(0,3):
                    if not single_space:
                        p_tmp_list.append(p_score[i])
                        pz_tmp_list.append(pz_score[i])
                    z_tmp_list.append(z_score[i])
                if not single_space:
                    p_tmp_list.append(ori_score[0])
                    pz_tmp_list.append(ori_score[0])
                z_tmp_list.append(ori_score[0])

                for j in range(3,6):
                    if not single_space:
                        p_tmp_list.append(p_score[j])
                        pz_tmp_list.append(pz_score[j])
                    z_tmp_list.append(z_score[j])

                tmp_dict["p"] =  p_tmp_list
                tmp_dict["pz"] =  pz_tmp_list
                tmp_dict["z"] =  z_tmp_list

                result_list[img_idx + img_num*8][attribute_name] = tmp_dict

    for attribute_name in test_attribute_list:
        if attribute_name == "pose":
            binarymodel_path = pose_model_path
            pose_model = ClassifyModel18()
            pose_model.load_state_dict(torch.load(binarymodel_path))
            model = pose_model.eval().to(device)

        elif attribute_name == "gender":
            gender_model = Gender()
            gender_model.load_state_dict(torch.load(gender_model_path))
            model = gender_model.eval().to(device)
        
        elif attribute_name == "age":
            age_model = Age()
            age_model.load_state_dict(torch.load(age_model_path))
            model = age_model.eval().to(device)
        else:
            continue

        for img_num in tqdm(img_list):
            ori_imgs = read_image_to_tensor(os.path.join(origin_image_dir,"{}.png".format(img_num))).reshape(3,8,256,256).permute(1,0,2,3).to(device=device)*255
            if not single_space:
                p_imgs = read_image_to_tensor(os.path.join(p_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255
                pz_imgs = read_image_to_tensor(os.path.join(pz_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255
            z_imgs = read_image_to_tensor(os.path.join(z_plus_dir,"{}.png".format(img_num))).reshape(3,8,256,6,256).permute(1,3,0,2,4).to(device=device)*255

            for img_idx in range(8):
                p_tmp_list = []
                pz_tmp_list = []
                z_tmp_list = []
                tmp_dict = {}
                ori_score = estimate_pose(model, ori_imgs[img_idx].unsqueeze(0)).detach().cpu().numpy()
                if not single_space:
                    p_score = estimate_pose(model, p_imgs[img_idx]).detach().cpu().numpy()
                    pz_score = estimate_pose(model, pz_imgs[img_idx]).detach().cpu().numpy()
                z_score = estimate_pose(model, z_imgs[img_idx]).detach().cpu().numpy()
                for i in range(0,3):
                    if not single_space:
                        p_tmp_list.append(p_score[i])
                        pz_tmp_list.append(pz_score[i])
                    z_tmp_list.append(z_score[i])

                if not single_space:
                    p_tmp_list.append(ori_score[0])
                    pz_tmp_list.append(ori_score[0])
                z_tmp_list.append(ori_score[0])

                for j in range(3,6):
                    if not single_space:
                        p_tmp_list.append(p_score[j])
                        pz_tmp_list.append(pz_score[j])
                    z_tmp_list.append(z_score[j])

                tmp_dict["p"] =  p_tmp_list
                tmp_dict["pz"] =  pz_tmp_list
                tmp_dict["z"] =  z_tmp_list

                result_list[img_idx + img_num*8][attribute_name] = tmp_dict


    if args.softmax:
        np.save(os.path.join(eval_dir, "test_dict_softmax.npy"), result_list)
    else:
        np.save(os.path.join(eval_dir, "test_dict.npy"), result_list)
 
