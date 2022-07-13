from matplotlib import pyplot as plt
from astropy.visualization import make_lupton_rgb
import numpy as np

from io import BytesIO
from PIL import Image

import seaborn as sns

def combine_images(im1, im2):
    
    im1 = Image.open(BytesIO(im1.data))
    im2 = Image.open(BytesIO(im2.data))
    
    images = [im1, im2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
        
    return new_im


def format_axes(a, log=False):
    """
    MNRAS style asks for axis ticks on every side of the plot, not just
    bottom+left. Run this on an axis object to set the right format, e.g.
    fig, ax = plt.subplots()
    ax.plot(...)
    format_axes(ax)
    fig.savefig(...)

    """

    if log:
        formatter = LogFormatter(10, labelOnlyBase=True)
        a.set_yscale('log')
    
    a.tick_params(axis='y', which='major', direction='in', color='k',
                  length=5.0, width=1.0, right=True)
    a.tick_params(axis='y', which='minor', direction='in', color='k',
                  length=3.0, width=1.0, right=True)
    a.tick_params(axis='x', which='major', direction='in', color='k',
                  length=5.0, width=1.0, top=True)
    a.tick_params(axis='x', which='minor', direction='in', color='k',
                  length=3.0, width=1.0, top=True)

def show_morph_properties(data_morph_full):
    
    width = 20 
    height = 11

    stat = 'probability'
    
    with sns.plotting_context("paper", font_scale=2.0):

        # Dark theme
        # sns.set(rc={'axes.facecolor':'#02010D', 'figure.facecolor':'#02010D', 'axes.labelcolor' : "white", 'text.color' : "white"}, font_scale=1.8)

        # White theme
        # sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.labelcolor' : "black", 'text.color' : "black"}, font_scale=1.6)

        hue_order = ['StyleGAN', 'SKIRT']

        fig = plt.figure(constrained_layout=False, figsize=(width, height))
        gs = fig.add_gridspec(nrows=2, ncols=4, left=0.05, right=0.95,
                              hspace=0.25, wspace=0.25)

        axes = {0 :{}, 1:{}}
        # fig, axes = plt.subplots(2,4, figsize=(15,10))
        ax_ = fig.add_subplot(gs[0, 0])
        axes[0][0] = ax_
        ax = sns.histplot(x="asymmetry", hue='source', stat = stat, data=data_morph_full, 
                          common_norm=False, ax=axes[0][0], kde=True, bins=50, element='step', hue_order = hue_order)
        axes[0][0].set(xlabel='Asymmetry')
        axes[0][0].legend([],[], frameon=False)
        axes[0][0].set_ylim([0.0, 0.4])
        axes[0][0].set_xlim([-0.5, 1.0])
        # axes[0][0].set(adjustable='box', aspect='equal')
        axes[0][0].set_xticks(np.linspace(-0.5, 1.0, 6))
        axes[0][0].set_yticks(np.linspace(0.0, 0.4, 6))
        format_axes(axes[0][0])

        ax_ = fig.add_subplot(gs[0, 1])
        axes[0][1] = ax_
        sns.histplot(x="sersic_n", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[0][1], legend=False, kde=True, bins=50, element='step', hue_order = hue_order)
        axes[0][1].set(xlabel="Sérsic index n")
        axes[0][1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.20), ncol=4, labels=hue_order[::-1])
        axes[0][1].set_ylabel(None)
        # axes[0][1].set(adjustable='box', aspect='equal')
        axes[0][1].set_xlim([0.0, 10.0])
        axes[0][1].set_yticks(np.linspace(0.0, 0.20, 6))
        axes[0][1].set_xticks(np.linspace(0.0, 10.0, 6))
        format_axes(axes[0][1])

        ax_ = fig.add_subplot(gs[0, 2])
        axes[0][2] = ax_
        sns.histplot(x="orientation_centroid", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[0][2], kde=True, bins=50, element='step', hue_order = hue_order)

        axes[0][2].set(xlabel='Orientation centroid')
        axes[0][2].legend([],[], frameon=False)
        axes[0][2].set_xlim([-1.6, 1.6])
        # axes[0][2].set(adjustable='box', aspect='equal')
        axes[0][2].set_ylabel(None)
        axes[0][2].set_yticks(np.linspace(0.0, 0.040, 6))
        axes[0][2].set_xticks(np.linspace(-1.6, 1.6, 6))
        axes[0][2].set_xticklabels([r'$0^{\circ}$', r'$72^{\circ}$', r'$144^{\circ}$', r'$216^{\circ}$', r'$288^{\circ}$', r'$360^{\circ}$'])
        format_axes(axes[0][2])

        ax_ = fig.add_subplot(gs[0, 3])
        axes[0][3] = ax_
        sns.histplot(x="ellipticity_centroid", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[0][3], kde=True, bins=50, element='step', hue_order = hue_order)
        axes[0][3].set(xlabel='Ellipticity centroid')
        axes[0][3].legend([],[], frameon=False)
        axes[0][3].set_xlim([0.0, 1.0])
        axes[0][3].set_ylim([0.0, 0.080])
        axes[0][3].set_ylabel(None)
        # axes[0][3].set(adjustable='box', aspect='equal')
        # axes[0][3].set_yticks(np.linspace(0.0, 0.040, 6))
        axes[0][3].set_yticks(np.linspace(0.0, 0.080, 6))
        axes[0][3].set_xticks(np.linspace(0.0, 1.0, 6))
        # axes[0][3].set_xticklabels([r'$0^{\circ}$', r'$72^{\circ}$', r'$144^{\circ}$', r'$216^{\circ}$', r'$288^{\circ}$', r'$360^{\circ}$'])
        format_axes(axes[0][3])

        ax_ = fig.add_subplot(gs[1, 0])
        axes[1][0] = ax_
        sns.histplot(x="gini", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[1][0], kde=True, bins=50, element='step', hue_order = hue_order)
        axes[1][0].set(xlabel='Gini')
        axes[1][0].legend([],[], frameon=False)
        axes[1][0].set_xlim([0.3, 0.7])
        # axes[1][0].set(adjustable='box', aspect='equal')
        axes[1][0].set_yticks(np.linspace(0.0, 0.1, 6))
        axes[1][0].set_xticks(np.linspace(0.3, 0.7, 6))
        format_axes(axes[1][0])

        ax_ = fig.add_subplot(gs[1, 1])
        axes[1][1] = ax_
        sns.histplot(x="smoothness", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[1][1], kde=True, bins=50, element='step', hue_order = hue_order)
        # axes[1][1].set_xlim([-0.050,0.025])
        axes[1][1].set_ylabel(None)
        axes[1][1].set(xlabel='Smoothness')
        axes[1][1].legend([],[], frameon=False)
        # axes[1][1].set(adjustable='box', aspect='equal')
        axes[1][1].set_xlim([-0.05, 0.2])
        axes[1][1].set_yticks(np.linspace(0.0, 0.3, 6))
        axes[1][1].set_xticks(np.linspace(-0.05, 0.2, 6))
        format_axes(axes[1][1])

        ax_ = fig.add_subplot(gs[1, 2])
        axes[1][2] = ax_
        sns.histplot(x="concentration", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[1][2], kde=True, bins=50, element='step', hue_order = hue_order)
        axes[1][2].set_ylabel(None)
        axes[1][2].set(xlabel='Concentration')
        axes[1][2].legend([],[], frameon=False)
        axes[1][2].set_xlim([1, 6])
        # axes[1][2].set(adjustable='box', aspect='equal')
        axes[1][2].set_yticks(np.linspace(0.0, 0.15, 6))
        axes[1][2].set_xticks(np.linspace(1, 6, 6))
        format_axes(axes[1][2])

        ax_ = fig.add_subplot(gs[1, 3])
        axes[1][3] = ax_
        sns.histplot(x="rhalf_circ", hue='source', stat = stat, data=data_morph_full, 
                     common_norm=False, ax=axes[1][3], kde=True, bins=50, element='step', hue_order = hue_order)
        axes[1][3].set(xlabel=r'$R_\mathrm{half}$')
        axes[1][3].legend([],[], frameon=False)
        axes[1][3].set_xlim([0, 36])
        axes[1][3].set_ylim([0.0, 0.25])
        # axes[1][3].set(adjustable='box', aspect='equal')
        axes[1][3].set_ylabel(None)
        axes[1][3].set_yticks(np.linspace(0.0, 0.25, 6))
        axes[1][3].set_xticks(np.linspace(0, 36, 6))
        # axes[0][3].set_xticklabels([r'$0^{\circ}$', r'$72^{\circ}$', r'$144^{\circ}$', r'$216^{\circ}$', r'$288^{\circ}$', r'$360^{\circ}$'])
        format_axes(axes[1][3])

        for i in range(2):
            for j in range(4):
                axes[i][j].tick_params(axis='x', colors='black')
                axes[i][j].tick_params(axis='y', colors='black')

        plt.savefig('pictures/morph_stylegan.svg', transparent=True)

        plt.show()
        
        
def predict_denoising_images(keys, gen, loader, augmentation=None, name=None):
    if name is None:
        name = gen.name
    
    NUM_SAMPLES = 200
    video_log = {}
    video_log[f'{name}_plots'] = []
    video_log[f'{name}_r_channel_orig'] = []
    video_log[f'{name}_r_channel_noised'] = []
    video_log[f'{name}_r_channel_denoised'] = []
    video_log[f'{name}_orig'] = []
    video_log[f'{name}_noised'] = []
    video_log[f'{name}_denoised'] = []

    
    for id in range(min(NUM_SAMPLES,len(keys))):
        key = keys[id]
        noised, orig = gen.load(key, transform=False, augmentation=augmentation)

        
        channel_g_orig = orig[...,0]
        channel_r_orig = orig[...,1]
        channel_i_orig = orig[...,2]
        channel_z_orig = orig[...,3]

        channel_g_noised = noised[...,0]
        channel_r_noised = noised[...,1]
        channel_i_noised = noised[...,2]
        channel_z_noised = noised[...,3]

        denoised = loader.model.predict(noised[None,...])[0]

        channel_g_denoised = denoised[...,0]
        channel_r_denoised = denoised[...,1]
        channel_i_denoised = denoised[...,2]
        channel_z_denoised = denoised[...,3]

        orig_stacked = np.concatenate([channel_g_orig, channel_r_orig, channel_i_orig, channel_z_orig], axis=1)
        denoised_stacked = np.concatenate([channel_g_denoised, channel_r_denoised, channel_i_denoised, channel_z_denoised], axis=1)
        noised_stacked = np.concatenate([channel_g_noised, channel_r_noised, channel_i_noised, channel_z_noised], axis=1)

        cm = plt.cm.ScalarMappable(None, cmap='bwr')
        cmap = cm.get_cmap()
        residual_stacked = (orig_stacked - denoised_stacked) / np.std(noised) # ** 2
        residual_stacked = cmap(residual_stacked)


        cm = plt.cm.ScalarMappable(None, cmap='bone')
        # cm = plt.cm.ScalarMappable(None, cmap='magma')
        cmap = cm.get_cmap()
        pic1 = np.concatenate([orig_stacked, denoised_stacked, noised_stacked], axis=0)
        pic1 = (pic1 - np.min(pic1)) / (np.max(pic1) - np.min(pic1))
        pic1 = cmap(pic1)

        pic2 = np.concatenate([pic1, residual_stacked], axis=0)

        video_log[f'{name}_plots'].append(pic2)
        video_log[f'{name}_r_channel_orig'].append(channel_r_orig)
        video_log[f'{name}_r_channel_noised'].append(channel_r_noised)
        video_log[f'{name}_r_channel_denoised'].append(channel_r_denoised)
        
        video_log[f'{name}_orig'].append(orig)
        video_log[f'{name}_noised'].append(noised)
        video_log[f'{name}_denoised'].append(denoised)

    return video_log



def image_to_rgb(im, r_norm=None, g_norm=None, i_norm=None):
    r_channel = im[:,:,1]
    g_channel = im[:,:,0]
    i_channel = im[:,:,2]

    epsilon = 1e-6
    
    if not r_norm is None:
        r_max, r_min = r_norm
        r_channel = np.minimum(r_channel, r_max)
        r_channel = np.maximum(r_channel, r_min)
    else:
        r_max = np.max(r_channel)
        r_min = np.min(r_channel)
    if not g_norm is None:
        g_max, g_min = g_norm
        g_channel = np.minimum(g_channel, g_max)
        g_channel = np.maximum(g_channel, g_min)
    else:
        g_max = np.max(g_channel)
        g_min = np.min(g_channel)
    if not i_norm is None:
        i_max, i_min = i_norm
        i_channel = np.minimum(i_channel, i_max)
        i_channel = np.maximum(i_channel, i_min)
    else:
        i_max = np.max(i_channel)
        i_min = np.min(i_channel)
    
    r_channel = (r_channel - r_min) / (r_max - r_min) # np.maximum(r_channel,0) + epsilon
    g_channel = (g_channel - g_min) / (g_max - g_min) # np.maximum(g_channel,0) + epsilon
    i_channel = (i_channel - i_min) / (i_max - i_min) # np.maximum(i_channel,0) + epsilon

    rgb_default = np.zeros(shape=(r_channel.shape[0], r_channel.shape[1], 3))
    rgb_default[...,0] = 0.5 * r_channel + 0.5 * i_channel
    rgb_default[...,1] = 0.5 * g_channel + 0.5 * r_channel
    rgb_default[...,2] = g_channel
    
    rgb_default[...,0] = np.sqrt(rgb_default[...,0])
    rgb_default[...,1] = np.sqrt(rgb_default[...,1])
    rgb_default[...,2] = np.sqrt(rgb_default[...,2])

    rgb_default[...,0] = rgb_default[...,0] / np.max(rgb_default[...,0])
    rgb_default[...,1] = rgb_default[...,1] / np.max(rgb_default[...,1])
    rgb_default[...,2] = rgb_default[...,2] / np.max(rgb_default[...,2])
    
    return rgb_default # make_lupton_rgb(i_channel, r_channel, g_channel)
    

def plot_denoising_example(onlySourceData, onlyGeneratedData, index = 0):

    width = 12
    
    noAugmentation_source, Augmentation1_source, Augmentation2_source, Augmentation3_source = onlySourceData
    noAugmentation_generated, Augmentation1_generated, Augmentation2_generated, Augmentation3_generated = onlyGeneratedData
    
    fig, ax = plt.subplots(6,4, figsize = (width, width * 1.5))
    
    max_value = max(np.max(noAugmentation_source['test_r_channel_orig'][index]), np.max(Augmentation1_source['test_r_channel_orig'][index]), 
                    np.max(Augmentation2_source['test_r_channel_orig'][index]), np.max(Augmentation3_source['test_r_channel_orig'][index]), 
                    np.max(noAugmentation_source['test_r_channel_noised'][index]), np.max(Augmentation1_source['test_r_channel_noised'][index]), 
                    np.max(Augmentation2_source['test_r_channel_noised'][index]), np.max(Augmentation3_source['test_r_channel_noised'][index]),
                    np.max(noAugmentation_source['test_r_channel_denoised'][index]), np.max(Augmentation1_source['test_r_channel_denoised'][index]), 
                    np.max(Augmentation2_source['test_r_channel_denoised'][index]), np.max(Augmentation3_source['test_r_channel_denoised'][index]),
                    np.max(noAugmentation_generated['test_r_channel_denoised'][index]), np.max(Augmentation1_generated['test_r_channel_denoised'][index]), 
                    np.max(Augmentation2_generated['test_r_channel_denoised'][index]), np.max(Augmentation3_generated['test_r_channel_denoised'][index]))
                    
    r_norm = (50, 0)
    g_norm = (50, 0)
    i_norm = (50, 0)
    
    ax[0][0].imshow(image_to_rgb(noAugmentation_source['test_orig'][index]))
    ax[0][1].imshow(image_to_rgb(Augmentation1_source['test_orig'][index]))
    ax[0][2].imshow(image_to_rgb(Augmentation2_source['test_orig'][index]))
    ax[0][3].imshow(image_to_rgb(Augmentation3_source['test_orig'][index]))
     
    ax[1][0].imshow(image_to_rgb(noAugmentation_source['test_noised'][index], r_norm = r_norm, i_norm = i_norm, g_norm = g_norm))
    ax[1][1].imshow(image_to_rgb(Augmentation1_source['test_noised'][index], r_norm = r_norm, i_norm = i_norm, g_norm = g_norm))
    ax[1][2].imshow(image_to_rgb(Augmentation2_source['test_noised'][index], r_norm = r_norm, i_norm = i_norm, g_norm = g_norm))
    ax[1][3].imshow(image_to_rgb(Augmentation3_source['test_noised'][index], r_norm = r_norm, i_norm = i_norm, g_norm = g_norm))
    
    ax[2][0].imshow(image_to_rgb(noAugmentation_source['test_denoised'][index]))
    ax[2][1].imshow(image_to_rgb(Augmentation1_source['test_denoised'][index]))
    ax[2][2].imshow(image_to_rgb(Augmentation2_source['test_denoised'][index]))
    ax[2][3].imshow(image_to_rgb(Augmentation3_source['test_denoised'][index]))
    
    ax[3][0].imshow(image_to_rgb(noAugmentation_generated['test_denoised'][index]))
    ax[3][1].imshow(image_to_rgb(Augmentation1_generated['test_denoised'][index]))
    ax[3][2].imshow(image_to_rgb(Augmentation2_generated['test_denoised'][index]))
    ax[3][3].imshow(image_to_rgb(Augmentation3_generated['test_denoised'][index]))

    vmax = 25
    vmin = -25
    
    ax[4][0].imshow(noAugmentation_source['test_r_channel_denoised'][index] - noAugmentation_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    ax[4][1].imshow(Augmentation1_source['test_r_channel_denoised'][index] - Augmentation1_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    ax[4][2].imshow(Augmentation2_source['test_r_channel_denoised'][index] - Augmentation2_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    ax[4][3].imshow(Augmentation3_source['test_r_channel_denoised'][index] - Augmentation3_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    
    ax[5][0].imshow(noAugmentation_generated['test_r_channel_denoised'][index] - noAugmentation_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    ax[5][1].imshow(Augmentation1_generated['test_r_channel_denoised'][index] - Augmentation1_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    ax[5][2].imshow(Augmentation2_generated['test_r_channel_denoised'][index] - Augmentation2_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    im = ax[5][3].imshow(Augmentation3_generated['test_r_channel_denoised'][index] - Augmentation3_source['test_r_channel_orig'][index], cmap='bwr', vmin = vmin, vmax = vmax)
    
    for i in range(6):
        for j in range(4):
            format_axes(ax[i][j])
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
    
    ax[5][0].set_xlabel('(i) No augmentations')
    ax[5][1].set_xlabel('(ii) Increased physical pixel size')
    ax[5][2].set_xlabel('(iii) Increased background noise')
    ax[5][3].set_xlabel('(iv) Augmentations (ii) + (iii)')
    
    ax[0][0].set_ylabel('Ground truth')
    ax[1][0].set_ylabel('Network input: noise + PSF')
    ax[2][0].set_ylabel(r'Only source data $(\alpha = 0.0)$')
    ax[3][0].set_ylabel(r'Only generated data $(\alpha = 1.0)$')
    
    ax[4][0].set_ylabel(r'Prediction error $(\alpha = 0.0)$')
    ax[5][0].set_ylabel(r'Prediction error $(\alpha = 1.0)$')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([1.00, 0.027, 0.03, 0.31])
    
    labels_cb = np.arange(vmin,vmax+5,5)
    loc_cb    = labels_cb 
    
    format_axes(cbar_ax)
    
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks(loc_cb)
    cb.set_ticklabels(labels_cb)
    
    plt.tight_layout()
    
    plt.savefig(f'pictures/denoising_comparison_{index}.pdf', bbox_inches='tight')
    
    # plt.clf()