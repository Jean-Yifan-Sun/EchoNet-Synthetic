import pytest
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from einops import rearrange
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
import lpips
# from vae_rec_ import DeNormalize

# Import the function to test
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# Define the test function
def test_vae_rec():
    # Setup
    mean = [0.2641497552394867, 0.2641497552394867, 0.2641497552394867]
    std = [0.42522257566452026, 0.42522257566452026, 0.42522257566452026]
    denormalize = DeNormalize(mean, std)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained('sd-legacy/stable-diffusion-v1-5', subfolder='vae')
    vae = vae.to(device, weight_dtype)

    img = Image.open("/rds/projects/c/chenhp-dpmodel/FinetuneVAE-SD/Data/ACDC/preprocessed/png/training/Unlabeled/patient-1-frame1_slice_2.png")
    img = np.array(img)
    img = torch.from_numpy(img)

    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.to(device, weight_dtype) / 256 - 1
    imgt = img.clone()

    with torch.no_grad():
        lat2d = vae.encode(img).latent_dist.sample()
        rec2d = vae.decode(lat2d).sample

    rect = rec2d.clone()
    rec = denormalize(rec2d).clamp(0, 1)
    rec = torch.mean(rec, dim=1)
    rec = torch.stack([rec, rec, rec], dim=1)
    rec = (rec * 255).to(torch.uint8).squeeze(0)
    rec = rearrange(rec, "c h w ->c h w")
    rec = transforms.functional.to_pil_image(rec)

    # Compute SSIM
    img_np = imgt.squeeze().permute(1,2,0).cpu().numpy()
    if img_np.max() <= 1:
        img_np = img_np * 255
        img_np = img_np.astype(np.uint8)
    rec_np = np.array(rec)
    print(img_np.shape,rec_np.shape) 
    ssim_value = ssim(img_np, rec_np, multichannel=True,channel_axis=2,data_range=1)
    print(f"SSIM: {ssim_value}")

    # Compute FID
    # fid_value = fid_score.calculate_fid_given_paths(['/path/to/original/images', '/path/to/reconstructed/images'], batch_size=50, device=device, dims=2048)
    # print(f"FID: {fid_value}")

    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    lpips_value = loss_fn(imgt, rect)
    print(f"LPIPS: {lpips_value.item()}")

    assert ssim_value > 0.5, "SSIM is too low"
    # assert fid_value < 50, "FID is too high"
    assert lpips_value < 0.5, "LPIPS is too high"

if __name__ == "__main__":
    test_vae_rec()