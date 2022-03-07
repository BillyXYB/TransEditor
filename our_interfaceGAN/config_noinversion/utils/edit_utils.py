import torch
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np


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

def visualize(img_path):
    img_list=os.listdir(img_path)
    img_list.sort()
    img_list.sort(key = lambda x: (x[:-4])) ##文件名按数字排序
    b = ['0', '12', '24', '36', '48', '60']
    dir = []
    img_nums=len(img_list)
    res = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )

    for i in range(img_nums):
        img_name=os.path.join(img_path, img_list[i])
        ll = img_name.split('/')[-1].split('_')[3]
        if ll in b:
            dir.append(img_name)
            img = Image.open(img_name).convert('RGB')
            img2 = transform(img)
            array = np.asarray(img2)
            data = torch.from_numpy(array).unsqueeze(0)
            res.append(data)
    sample = torch.cat(res, dim=0)
    utils.save_image(
                            sample,
                            os.path.join(img_path,'edit.png'),
                            nrow=int(6),
                            normalize=True,
                            range=(-1, 1),
                        )
