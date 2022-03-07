
from pSp.models.psp_new import pSp


def set_test_options(opts):
    opts.start_from_latent_avg = True


    return opts


class DualSpaceEncoder():
    def __init__(self, opts):
        self.device = 'cuda'
        self.opts = opts
        self.opts.device = self.device
        self.net = pSp(self.opts).to(self.device)
        self.net.eval()

    def encode(self, real_img):
        z_code, p_code = self.net(real_img, only_encode=True)
        return z_code, p_code
        
    def decode(self,z_code,p_code, plus_sapce=True):
        if plus_sapce:
            images, _, _ = self.net.decoder(z_code, p_code, 
                                use_spatial_mapping=False,use_style_mapping=False, 
                                return_latents=False)
        else:
            images = self.net.decoder(z_code, p_code, return_latents=False)
        
        return images
