import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# globals
categories = {
    'Airplane',
    'Animal',
    'Automobile',
}
categories_scenes = {
    'Airplane': {'Cloud', 'Runway'},
    'Animal': {'Beach', 'Desert', 'Forest'},
    'Automobile': {'City', 'Highway'},
} 


def save_imgs(imgs, basedir, img_names, original):
    for category in categories:
        if original:
            for i, img in enumerate(imgs[category]):
                img_name = img_names[category][i]
                save_path = f"{basedir}/{category}/{img_name}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, img)
        else:
            for i, scene in enumerate(categories_scenes[category]):
                for i, img in enumerate(imgs[category][scene]):
                    img_name = img_names[category][i]
                    save_path = f"{basedir}/{category}/{scene}/{img_name}"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, img)
        
def load_original_images(orig_imgs_dir):
    orig_imgs = {}
    img_names = {}
    for category in categories:
        orig_imgs[category] = []
        img_names[category] = []

        dir_path = os.path.join(orig_imgs_dir, category)
        # loop over all images in the directory
        img_names_list = sorted([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
        img_names[category] = img_names_list
        for img_name in img_names_list:
            img_path = os.path.join(dir_path, img_name)  
            img = cv2.imread(img_path)
            orig_imgs[category].append(img)      
    return orig_imgs, img_names

def load_experiment_images(experiment_dir, to_list=False):
    experiment_imgs = {}
    for category in categories:
        experiment_imgs[category] = {}
        for scene in categories_scenes[category]:
            experiment_imgs[category][scene] = []
            dir_path = os.path.join(experiment_dir, category, scene)
            # make sure only images 
            imgs_list = sorted([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
            for img_name in imgs_list:
                img = cv2.imread(os.path.join(experiment_dir, category, scene, img_name))
                experiment_imgs[category][scene].append(img)
            if not to_list:
                experiment_imgs[category][scene] = np.stack(experiment_imgs[category][scene], axis=0)
    return experiment_imgs

def show_img(img, save_path=None, show = True):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # save with cv2 to avoid changes in the image
        cv2.imwrite(save_path, img)

    if show:
        # Get image dimensions
        height, width, depth = img.shape

        # Set DPI and calculate figure size in inches
        dpi = 100  
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure with the specified size and DPI
        plt.figure(figsize=figsize, dpi=dpi)

        # Display image in notebook
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        plt.axis('off') 

        
        plt.show()
        plt.close()

def standardize_sizes(imgs, dimensions=(512, 512, 3), img_names=None, save_dir=None, original=False):
    new_imgs = {}
    for category in imgs:
        # Original images do not have scenes
        if original:
            new_imgs[category] = []
            for idx, img in enumerate(imgs[category]):
                new_img = cv2.resize(img, dimensions[:2], interpolation=cv2.INTER_AREA)
                new_imgs[category].append(new_img)
                if save_dir:
                    save_dir_category = os.path.join(save_dir, category)
                    os.makedirs(save_dir_category, exist_ok=True)
                    cv2.imwrite(f"{save_dir_category}/{img_names[category][idx]}", new_img)
        else:
            new_imgs[category] = {}
            for scene in categories_scenes[category]:
                new_imgs[category][scene] = []
                for i, img in enumerate(imgs[category][scene]):
                    new_imgs[category][scene].append(cv2.resize(img, dimensions[:2], interpolation=cv2.INTER_AREA))
    return new_imgs


def imgs_compare(orig_imgs: list, experiment_imgs, index: int, category, scene, show=True, save_dir=None,
                 experiment_names = ['Stable Diffusion','Aggregated Attention', 'Scene-Based'],
                 img_name = None):

    original_img = orig_imgs[category][index]
    experiment_imgs = [expt_imgs[category][scene][index] for expt_imgs in experiment_imgs]
    all_imgs = [original_img] + experiment_imgs

    # Target sizes for resizing
    target_height = min([img.shape[0] for img in all_imgs])
    target_width = min(img.shape[1] for img in [original_img] + experiment_imgs)
    target_size = (target_width, target_height)

    # Resize images to common size
    original_img_resized = cv2.resize(original_img, target_size, interpolation=cv2.INTER_AREA)
    experiment_imgs_resized = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in experiment_imgs]

    # Create a figure to display the images
    fig, axs = plt.subplots(1, len(experiment_imgs_resized) + 1, figsize=(20, 20)) 

    # Display original image
    axs[0].imshow(cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[0].axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

    # Display experimental images
    for i, img in enumerate(experiment_imgs_resized):
        axs[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(experiment_names[i])
        axs[i + 1].axis('off') 

    # Save subplot to file
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_name = img_name if img_name else index
        plt.savefig(os.path.join(save_dir, f'{img_name}.png'), bbox_inches='tight', pad_inches=0.1)

    if show:
        plt.show() 
    plt.close(fig)  



def compute_is_score(images):
    images = np.transpose(images, (0, 3, 1, 2))  # Channels-last to channels-first
    images = torch.tensor(images, dtype=torch.uint8)  # Convert to torch tensor
    # Initialize InceptionScore metric
    is_metric = InceptionScore(feature=2048)  # Using default feature extraction
    is_metric.update(images)  # Update metric with generated images
    is_score = is_metric.compute()  # Compute Inception Score

    return is_score[0].item(), is_score[1].item()  # Return mean and standard deviation

def compute_fid_score(real_imgs, gen_imgs):
    real_imgs = torch.tensor(real_imgs, dtype=torch.uint8)
    gen_imgs = torch.tensor(gen_imgs, dtype=torch.uint8)

    # Batch_size * C * H * W -> Batch_size * H * W * C
    real_imgs = real_imgs.permute(0, 3, 1, 2)
    gen_imgs = gen_imgs.permute(0, 3, 1, 2)

    # Repeat images for FID calculation  --- why?
    # real_imgs = real_imgs.repeat(2, 1, 1, 1)
    # gen_imgs = gen_imgs.repeat(2, 1, 1, 1)    

    # ratio of batch size
    ratio = real_imgs.shape[0] / gen_imgs.shape[0]
    if ratio > 1:
        gen_imgs = gen_imgs.repeat(int(ratio), 1, 1, 1)
    elif ratio < 1:
        real_imgs = real_imgs.repeat(int(1/ratio), 1, 1, 1)

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_imgs, real=True)
    fid.update(gen_imgs, real=False)
    fid_score = fid.compute()
    return fid_score.item()


def compute_psnr(pred_imgs, orig_imgs):
    if np.max(pred_imgs) > 1:        
        p_imgs = np.array(pred_imgs)/255.0
    if np.max(orig_imgs) > 1:
        o_imgs = np.array(orig_imgs)/255.0

    p_imgs = torch.tensor(p_imgs, dtype=torch.float64)
    o_imgs = torch.tensor(o_imgs, dtype=torch.float64)

    # Batch_size * C * H * W -> Batch_size * H * W * C
    p_imgs = p_imgs.permute(0, 3, 1, 2)

    o_imgs = o_imgs.permute(0, 3, 1, 2)
    psnr_metric = PeakSignalNoiseRatio()
    psnr_metric.update(p_imgs, o_imgs)
    psnr_score = psnr_metric.compute()
    return psnr_score.item()

    
def compute_ssim(pred_imgs, orig_imgs):
    # normalize to [0,1]
    if np.max(pred_imgs) > 1:        
        p_imgs = np.array(pred_imgs)/255.0
    if np.max(orig_imgs) > 1:
        o_imgs = np.array(orig_imgs)/255.0

    p_imgs = torch.tensor(p_imgs, dtype=torch.float64)
    o_imgs = torch.tensor(o_imgs, dtype=torch.float64)

    # Batch_size * C * H * W -> Batch_size * H * W * C
    p_imgs = p_imgs.permute(0, 3, 1, 2)
    o_imgs = o_imgs.permute(0, 3, 1, 2)
    
    ssim_metric = StructuralSimilarityIndexMeasure()
    ssim_metric.update(p_imgs, o_imgs)
    ssim_score = ssim_metric.compute()
    return ssim_score.item()