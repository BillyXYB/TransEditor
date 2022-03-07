import torch
import os
from . import attribute_classifier
import glob

softmax = torch.nn.Softmax(dim=1)

def downsample(images, size=256):
    # Downsample to 256x256. The attribute classifiers were built for 256x256.
    # follows https://github.com/NVlabs/stylegan/blob/master/metrics/linear_separability.py#L127
    if images.shape[2] > size:
        factor = images.shape[2] // size
        assert(factor * size == images.shape[2])
        images = images.view([-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
        images = images.mean(dim=[3, 5])
        return images
    else:
        assert(images.shape[-1] == 256)
        return images


def get_logit(net, im):
    im_256 = downsample(im)
    logit = net(im_256)
    return logit


def get_softmaxed(net, im):
    logit = get_logit(net, im)
    softmaxed = softmax(torch.cat([logit, -logit], dim=1))[:, 1]
    # logit is (N,) softmaxed is (N,)
    return logit[:, 0], softmaxed


def load_attribute_classifier(attribute, ckpt_path=None):
    if ckpt_path is None:        
        base_path = os.path.abspath(__file__+"/../../../pth_celeba")
        attribute_pkl = os.path.join(base_path, attribute, 'net_best.pth')
        ckpt = torch.load(attribute_pkl)
    else:
        ckpt = torch.load(ckpt_path)
    # print("Using classifier at epoch: %d" % ckpt['epoch'])
    if 'valacc' in ckpt.keys():
        print("Validation acc on raw images: %0.5f" % ckpt['valacc'])
    detector = attribute_classifier.from_state_dict(
        ckpt['state_dict'], fixed_size=True, use_mbstd=False).cuda().eval()
    return detector


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier_name, ckpt_path=None, device='cuda'):
        super(ClassifierWrapper, self).__init__()
        self.net = load_attribute_classifier(classifier_name, ckpt_path).eval().to(device)

    def forward(self, ims, no_soft=False):
        # returns (N,) softmax values for binary classification
        if not no_soft:
            return get_softmaxed(self.net, ims)[1]
        else:
            return get_softmaxed(self.net, ims)[0]
