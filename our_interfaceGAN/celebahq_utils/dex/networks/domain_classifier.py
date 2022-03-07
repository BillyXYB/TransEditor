import torch


def define_classifier(domain, classifier_name=None,
                      ckpt_path=None, device='cuda'):
    # check that name of pretrained model, or direct ckpt path is provided
    assert(classifier_name or ckpt_path)
    # load the trained classifiers
    if 'celebahq' in domain:
        from .classifiers import attribute_utils
        return attribute_utils.ClassifierWrapper(classifier_name,
                                                 ckpt_path=ckpt_path,
                                                 device=device)
    elif domain == 'cat' or domain == 'car':
        import torchvision.models
        if ckpt_path is None:
            ckpt = torch.load('results/pretrained_classifiers/%s/%s/net_best.pth' %
                              (domain, classifier_name))
        else:
            ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        # determine num_classes from the checkpoint
        num_classes = state_dict['fc.bias'].shape[0]
        net = torchvision.models.resnet18(num_classes=num_classes)
        net.load_state_dict(state_dict)
        return net.eval().to(device)
    elif domain == 'cifar10':
        from .classifiers import cifar10_resnet
        net = cifar10_resnet.ResNet18()
        if ckpt_path is None:
            ckpt = torch.load('results/pretrained_classifiers/cifar10/%s/ckpt.pth' %
                              classifier_name)['net']
        else:
            ckpt = torch.load(ckpt_path)['net']
        net.load_state_dict(ckpt)
        return net.eval().to(device)


softmax = torch.nn.Softmax(dim=-1)


def postprocess(classifier_output):
    # multiclass classification N x labels
    if len(classifier_output.shape) == 2:
        postprocessed_outputs = softmax(classifier_output)
    # binary classification output, (N,)
    # the softmax should already be applied in ClassifierWrapper
    elif len(classifier_output.shape) == 1:
        postprocessed_outputs = classifier_output
    # sanity check
    assert(torch.min(postprocessed_outputs) >= 0.)
    assert(torch.max(postprocessed_outputs) <= 1.)
    return postprocessed_outputs
