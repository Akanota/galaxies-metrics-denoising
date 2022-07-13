# Copyright (c) 2022, Benjamin Holzschuh
#

import h5py
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch

import data_iterator

from IPython.display import Image
import scipy.stats as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from radially_averaged_power_spectrum import *



def show_gallery(file, start=0, transform=True):
    iterator = data_iterator.H5PyIterator(file, transform=transform)
    data_list = []
    rows = 5
    for i in range(start-1):
        iterator.__next__()
        
    for i in range(rows * 5):
        data_list.append(iterator.__next__())
    
    
    fig, ax = plt.subplots(rows,5, figsize=(40, 40))
    for i in range(rows):
        for j in range(5):
          
            ax[i][j].imshow(data_list[0][:,:,1], cmap='magma')
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            del data_list[0]
    
    # fig.patch.set_facecolor('#02010D')
    
    plt.show()

def get_mode_dist(powerspectra, mode):
    return [x[mode] for x in powerspectra if mode < x.shape[0]]

# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def density_estimates(powerspectra, mode_1, mode_2):
    
    if mode_1 == mode_2:
        x = get_mode_dist(powerspectra, mode_1)
        return None, x, x
    
    x = get_mode_dist(powerspectra, mode_1)
    y = get_mode_dist(powerspectra, mode_2)
    xmean, xvar = np.mean(x), np.sqrt(np.var(x))
    ymean, yvar = np.mean(y), np.sqrt(np.var(y))
    xmin, xmax = xmean - 3 * xvar, xmean + 3 * xvar
    ymin, ymax = ymean - 3 * yvar, ymean + 3 * yvar
    
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    min_dim = min(len(x), len(y))
    values = np.vstack([x[:min_dim], y[:min_dim]])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    return f, x, y
    

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    
def plot_avg_powerspectra_Sersic(file, transform=False, max_size=2516, resize_transform=256, downscale=4, ticks_n=128, 
                          statsfile=None, savename=None):   
    pow_avg = np.zeros(shape=(max_size, max_size))
    
    iterator = data_iterator.H5PyIterator(file, transform=transform, resize_transform=resize_transform)
    
    if statsfile:
        
        with open(statsfile, 'rb') as handle:
            mean, std = pickle.load(handle)
        
        normalization=True
        
    else:
        
        normalization=False
    
    c = 0
    for image in tqdm(iterator):
         
         if normalization:
            image = (image-mean) / std
            
         if image.shape[2] == 1:
            image = np.tile(image, (1,1,2))
         r_channel = image[:,:,1]
    
         dim = r_channel.shape[0]
    
         galaxy_fft = fftpack.fft2(r_channel)
         galaxy_shiftfft = np.fft.fftshift(galaxy_fft)
    
         galaxy_shiftfft_normalize = galaxy_shiftfft / (dim * dim)
    
         img = np.pad(abs(galaxy_shiftfft_normalize), pad_width=(int((max_size-dim) / 2), int(np.ceil((max_size-dim) / 2))))
    
         pow_avg += img
    
         c += 1
        
         if c >= 10000:
                break
    
    pow_avg = pow_avg / c #len_dataset(file)
    pow_avg_small = rebin(pow_avg, (max_size//downscale, max_size//downscale)) + 1e-90
    
    colorbar = True
    if colorbar:
        colorbar_args = dict(
                tick0=0,
                dtick=10,
                tickvals=[-10 * i for i in range(1,9)],
                ticktext=[f"{-10 * i}dB" for i in range(1,9)]
             )

    
    fig = go.Figure(data =
        go.Contour(
            z=10 * np.log10(pow_avg_small),
            colorscale='Blues',
            line_smoothing=0.85, 
            contours=dict(
                start=0,
                end=40,
                size=5,
            ),
            colorbar=dict(
                tick0=0,
                dtick=5,
                tickvals=[5 * (i - 2) for i in range(0,12)],
                ticktext=[f"{5 * (i - 2)}dB" for i in range(0,12)]
             ),
        ))
   
    xtickval = [pow_avg_small.shape[0] // 2]
    for i in range(10):
        xtickval.append(pow_avg_small.shape[0] // 2 + (i+1) * ticks_n - 1) # get 128 xtick at the right end
        xtickval.append(pow_avg_small.shape[0] // 2 - (i+1) * ticks_n)
    
    xtext = [0]
    for i in range(4):
        xtext.append((i+1) * ticks_n * downscale)
        xtext.append(-1 * (i+1) * ticks_n * downscale)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * 1 * mnras_col_wid) 
    
    fig.update_layout(
        template="simple_white",
        autosize=False,
        font_family='Serif', 
        width=fig_width,
        height=fig_width,
        xaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color='black'),
                tickvals = xtickval,
                mirror='allticks', ticks='inside', showline=True,
                ticktext = xtext
        ),
        yaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color='black'),
                mirror='allticks', ticks='inside', showline=True,
                tickvals = xtickval,
                ticktext = xtext
        )
    )
    
    if savename:
        fig.write_image(savename)
    fig.show()
    
    
def plot_avg_powerspectra_COSMOS(file, transform=False, max_size=2516, resize_transform=256, downscale=4, ticks_n=128, 
                          statsfile=None, savename=None):   
    pow_avg = np.zeros(shape=(max_size, max_size))
    
    iterator = data_iterator.H5PyIterator(file, transform=transform, resize_transform=resize_transform)
    
    if statsfile:
        
        with open(statsfile, 'rb') as handle:
            mean, std = pickle.load(handle)
        
        normalization=True
        
    else:
        
        normalization=False
    
    c = 0
    for image in tqdm(iterator):
         
         if normalization:
            image = (image-mean) / std
            
         if image.shape[2] == 1:
            image = np.tile(image, (1,1,2))
         r_channel = image[:,:,1]
    
         dim = r_channel.shape[0]
    
         galaxy_fft = fftpack.fft2(r_channel)
         galaxy_shiftfft = np.fft.fftshift(galaxy_fft)
    
         galaxy_shiftfft_normalize = galaxy_shiftfft / (dim * dim)
    
         img = np.pad(abs(galaxy_shiftfft_normalize), pad_width=(int((max_size-dim) / 2), int(np.ceil((max_size-dim) / 2))))
    
         pow_avg += img
    
         c += 1
        
         if c >= 10000:
                break
    
    pow_avg = pow_avg / c #len_dataset(file)
    pow_avg_small = rebin(pow_avg, (max_size//downscale, max_size//downscale)) + 1e-90
    
    colorbar = True
    if colorbar:
        colorbar_args = dict(
                tick0=0,
                dtick=10,
                tickvals=[-10 * i for i in range(1,9)],
                ticktext=[f"{-10 * i}dB" for i in range(1,9)]
             )

    
    fig = go.Figure(data =
        go.Contour(
            z=10 * np.log10(pow_avg_small),
            colorscale='Blues',
            line_smoothing=0.85, 
            contours=dict(
                start=-50,
                end=-30,
                size=5,
            ),
            colorbar=dict(
                tick0=0,
                dtick=5,
                tickvals=[-5 * i for i in range(1,18)],
                ticktext=[f"{-5 * i}dB" for i in range(1,18)]
            ),
        ))
   
    xtickval = [pow_avg_small.shape[0] // 2]
    for i in range(10):
        xtickval.append(pow_avg_small.shape[0] // 2 + (i+1) * ticks_n - 1) # get 128 xtick at the right end
        xtickval.append(pow_avg_small.shape[0] // 2 - (i+1) * ticks_n)
    
    xtext = [0]
    for i in range(4):
        xtext.append((i+1) * ticks_n * downscale)
        xtext.append(-1 * (i+1) * ticks_n * downscale)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * 1 * mnras_col_wid) 
    
    fig.update_layout(
        template="simple_white",
        autosize=False,
        font_family='Serif', 
        width=fig_width,
        height=fig_width,
        xaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color='black'),
                tickvals = xtickval,
                mirror='allticks', ticks='inside', showline=True,
                ticktext = xtext
        ),
        yaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color='black'),
                mirror='allticks', ticks='inside', showline=True,
                tickvals = xtickval,
                ticktext = xtext
        )
    )
    
    if savename:
        fig.write_image(savename)
    fig.show()
    
def plot_avg_powerspectra(file, transform=False, max_size=2516, resize_transform=256, downscale=4, ticks_n=128, 
                          statsfile=None, savename=None, colorbar=False, style='black'):
    
    if style=='white':
        color_ = 'white'
        template = 'plotly_dark'
    else:
        color_ = 'black'
        template = 'simple_white'
    
    pow_avg = np.zeros(shape=(max_size, max_size))
    
    iterator = data_iterator.H5PyIterator(file, transform=transform, resize_transform=resize_transform)
    
    if statsfile:
        
        with open(statsfile, 'rb') as handle:
            mean, std = pickle.load(handle)
        
        normalization=True
        
    else:
        
        normalization=False
    
    c = 0
    for image in tqdm(iterator):
         
        if normalization:
            image = (image-mean) / std
            
        if image.shape[2] == 1:
            image = np.tile(image, (1,1,2))
        r_channel = image[:,:,1]
    
        dim = r_channel.shape[0]
    
        galaxy_fft = fftpack.fft2(r_channel)
        galaxy_shiftfft = np.fft.fftshift(galaxy_fft)
    
        galaxy_shiftfft_normalize = galaxy_shiftfft / (dim * dim)
    
        img = np.pad(abs(galaxy_shiftfft_normalize), pad_width=(int((max_size-dim) / 2), int(np.ceil((max_size-dim) / 2))))
    
        pow_avg += img
    
        c += 1
        
        if c >= 10000:
            break
    
    pow_avg = pow_avg / c #len_dataset(file)
    pow_avg_small = rebin(pow_avg, (max_size//downscale, max_size//downscale)) + 1e-90
    
    if colorbar:
        colorbar_args = dict(
                tick0=0,
                dtick=10,
                tickvals=[-10 * i for i in range(1,9)],
                ticktext=[f"{-10 * i}dB" for i in range(1,9)]
             )

    
    fig = go.Figure(data =
        go.Contour(
            z=10 * np.log10(pow_avg_small),
            colorscale='Blues',
            line_smoothing=0.85, 
            contours=dict(
                start=-40,
                end=-5,
                size=5,
            ),
            colorbar=dict(
                tick0=0,
                dtick=10,
                tickvals=[-10 * i for i in range(1,9)],
                ticktext=[f"{-10 * i}dB" for i in range(1,9)]
             ),
        ))
   
    xtickval = [pow_avg_small.shape[0] // 2]
    for i in range(10):
        xtickval.append(pow_avg_small.shape[0] // 2 + (i+1) * ticks_n - 1) # get 128 xtick at the right end
        xtickval.append(pow_avg_small.shape[0] // 2 - (i+1) * ticks_n)
    
    xtext = [0]
    for i in range(4):
        xtext.append((i+1) * ticks_n * downscale)
        xtext.append(-1 * (i+1) * ticks_n * downscale)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * 1 * mnras_col_wid) 
    
    fig.update_layout(
        template=template,
        autosize=False,
        font_family='Serif', 
        width=fig_width,
        height=fig_width,
        xaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color=color_),
                tickvals = xtickval,
                mirror='allticks', ticks='inside', showline=True,
                ticktext = xtext
        ),
        yaxis = dict(
                tickmode = 'array',
                tickfont=dict(family='Serif', size=16, color=color_),
                mirror='allticks', ticks='inside', showline=True,
                tickvals = xtickval,
                ticktext = xtext
        )
    )
    
    if savename:
        fig.write_image(savename)
        
    img_bytes = fig.to_image(format="png")
      
    return Image(img_bytes)
        
    # fig.show()

# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def density_estimates(powerspectra, mode_1, mode_2, num_partitions=15):
    
    power_partition = powerspectra_to_partition(powerspectra, partition=get_partition_list(num_partitions=num_partitions))
    
    if mode_1 == mode_2:
        x = power_partition[mode_1] # get_mode_dist(powerspectra, mode_1)
        return None, x, x
    
    x = power_partition[mode_1] # get_mode_dist(powerspectra, mode_1)
    y = power_partition[mode_2] # get_mode_dist(powerspectra, mode_2)
    xmean, xvar = np.mean(x), np.sqrt(np.var(x))
    ymean, yvar = np.mean(y), np.sqrt(np.var(y))
    xmin, xmax = xmean - 3 * xvar, xmean + 3 * xvar
    ymin, ymax = ymean - 3 * yvar, ymean + 3 * yvar
    
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    min_dim = min(len(x), len(y))
    values = np.vstack([x[:min_dim], y[:min_dim]])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    return f, x, y

def cor_powerspectra_stylegan(powerspectra_SKIRT, powerspectra_StyleGAN, modes, savefig=None, dataset_name='SKIRT', label_function=None):
    
    pixel_size=0.276
    
    font_size = 12
    title_font_size = 12
    
    fig = make_subplots(rows=len(modes), cols=len(modes), start_cell="bottom-left",
                   horizontal_spacing = 0.025, vertical_spacing = 0.025) # horizontal_spacing = 0.075, vertical_spacing = 0.075) #, shared_yaxes=True, shared_xaxes=True)
   
    # fig, axes = plt.subplots(nrows=len(modes_target), ncols=len(modes_source), squeeze=False)
    template = 'plotly_dark'
    
    def format_mode_label(mode):
        if not label_function is None:
            return label_function(mode)
            
        if int(mode) == 0:
            str_ = u"\u221E" # r'\infty' 
        else:
            rad = (1 / int(mode)) * pixel_size 
            str_ = u"{:.2f}".format(rad)
        # return r'log-cycles at ' + str_ + r'$ckpch^{-1}$' + f' [{mode}]'
        # return r'{}\, {} {}'.format(r'${} '.format(str_), r'\text{ckpch}^{-1}', r' [{}]$'.format(mode))
        # return u'{} [{} ckpch<sup>-1</sup>]'.format(mode, str_)
        return u'Mode {}'.format(mode)
    
    color_alae = '#ff7f0e' #'rgb(127, 60, 141)'
    color_skirt = 'coral'# '#d62728' #'rgb(17, 165, 121)'
    color_stylegan = 'darkcyan'# '#2ca02c' # 'rgb(57, 105, 172)'
    color_vae = '#1f77b4' # 'rgb(242, 183, 1)'
    # min_v = -.5
    # max_v = 8
    showlegend = True
    for mode_target, i in zip(modes, range(len(modes))):
        for mode_source, j in zip(modes, range(len(modes))):
            
            if mode_target < mode_source:
                continue
            
            f_SKIRT, x_skirt, y_skirt = density_estimates(powerspectra_SKIRT, mode_target, mode_source, num_partitions=15)
            f_StyleGAN, x_stylegan, y_stylegan = density_estimates(powerspectra_StyleGAN, mode_target, mode_source, num_partitions=15)
            
                
          
            x_skirt = np.log10(x_skirt)
            y_skirt = np.log10(y_skirt)
            x_stylegan = np.log10(x_stylegan)
            y_stylegan = np.log10(y_stylegan)
          
            
           
            weights_skirt, edges_skirt = np.histogram(x_skirt, bins=20)
            weights_stylegan, edges_stylegan = np.histogram(x_stylegan, bins=20) 
            

      
            weights_y_skirt, edges_y_skirt = np.histogram(y_skirt, bins=20)
            weights_y_stylegan, edges_y_stylegan = np.histogram(y_stylegan, bins=20) 
        


   
            min_v_skirt = edges_skirt[0]
            min_v_stylegan = edges_stylegan[0]
        

          
            max_v_skirt = edges_skirt[-1]
            max_v_stylegan = edges_stylegan[-1]
           
            min_v = min(min_v_skirt, min_v_stylegan)
            max_v = max(max_v_skirt, max_v_stylegan)
                
            bins_barplot = np.linspace(min_v, max_v, 15)
            
            if mode_target == mode_source:
                # bins = np.linspace(min_v, max_v, 30)
                # bins_barplot = np.linspace(min_v, max_v, 30-1)
                # weights_t, edges_t = np.histogram(x_t, bins=bins) 
                # weights_s, edges_s = np.histogram(x_s, bins=bins) 
                
             
                weights_skirt, edges_skirt = np.histogram(x_skirt, bins=bins_barplot)
                weights_stylegan, edges_stylegan = np.histogram(x_stylegan, bins=bins_barplot) 
        
                
              
                weights_y_skirt, edges_y_skirt = np.histogram(y_skirt, bins=bins_barplot)
                weights_y_stylegan, edges_y_stylegan = np.histogram(y_stylegan, bins=bins_barplot) 
            
                
                
              
                weights_skirt = weights_skirt / np.sum(weights_skirt)
                weights_stylegan = weights_stylegan / np.sum(weights_stylegan)
           
                
         
                weights_y_skirt = weights_y_skirt / np.sum(weights_y_skirt)
                weights_y_stylegan = weights_y_stylegan / np.sum(weights_y_stylegan)
                
                max_x = max(max(x_skirt), max(x_stylegan))
                max_y = max(max(y_skirt), max(y_stylegan))
                min_x = min(min(x_skirt), min(x_stylegan))
                min_y = min(min(y_skirt), min(y_stylegan))
                
               
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_skirt,
                    legendgroup='group2',
                    marker_color = color_skirt,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_stylegan,
                    legendgroup='group3',
                    marker_color = color_stylegan,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                
                fig.update_xaxes(row=len(modes)-i, range=[min_y, max_y], col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, showticklabels=False)
                fig.update_yaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, side = 'right')
                if j==0:
                    fig.update_yaxes(title_text=format_mode_label(mode_target), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : title_font_size}, side= 'left')
                    fig.update_xaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5)   
                if i == len(modes) - 1 and j == len(modes) - 1:
                    fig.update_yaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, side='right')
                    fig.update_xaxes(title_text=format_mode_label(mode_source), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : title_font_size}, showticklabels=True)
                    
            
            else: 
            
                
             

                fig.add_trace(go.Histogram2dContour(
                        x = y_skirt,
                        y = x_skirt,
                        opacity=1,
                        name=dataset_name,
                        ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : color_skirt}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group2',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)
                
                fig.add_trace(go.Histogram2dContour(
                        x = y_stylegan,
                        y = x_stylegan,
                        opacity=1,
                        name='StyleGAN',
                        ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : color_stylegan}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group3',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)
                
                showlegend = False
                
                max_ = max(max(x_skirt), max(y_skirt), max(x_stylegan), max(y_stylegan))
                min_ = min(min(x_skirt), min(y_skirt), min(x_stylegan), min(y_stylegan))
                
                max_x = max(max(x_skirt), max(x_stylegan))
                max_y = max(max(y_skirt), max(y_stylegan))
                min_x = min(min(x_skirt), min(x_stylegan))
                min_y = min(min(y_skirt), min(y_stylegan))
                
                fig.update_traces(showscale=False, row=len(modes)-i, col=j+1)
          
                fig.update_xaxes(range=[min_y, max_y], row=len(modes)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 4, showticklabels=False)
                fig.update_yaxes(range=[min_x, max_x], row=len(modes)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 4, showticklabels=False)
                if i == len(modes) - 1:
                    fig.update_xaxes(range=[min_y, max_y], title_text=format_mode_label(mode_source), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 4, titlefont = {'size' : title_font_size}, showticklabels=True)
                if j == 0:
                    fig.update_yaxes(range=[min_x, max_x], title_text=format_mode_label(mode_target), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 4, titlefont = {'size' : title_font_size}, showticklabels=True)
                
    fig.update(
        layout_showlegend=True,
        layout_coloraxis_showscale=False)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * (2*mnras_col_wid + mnras_col_sep) * 0.65) # smaller than the all plot 
    fig_height = fig_width
    
    fig.update_layout(
        autosize=True,
        font_family="Serif",
        font_size=10, 
        width=fig_width,
        height=fig_height,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        title_font_size=10,
        legend=dict(
            yanchor="top",
            y=1.008,
            xanchor="left",
            x=0.275
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_layout(
        template=template) 
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='white', gridcolor='white')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='white', gridcolor='white')
    
    if savefig:
        fig.write_image(savefig)
    
    return fig

# https://stackoverflow.com/questions/68163066/plotly-latex-fonts-different-in-jupyter-and-exported-pdf-or-png
# At the moment plotly shouldn't ne ised together with latex, which is very unfortunate, but latex string are not exported with good quality

def cor_powerspectra_all(powerspectra_SKIRT, powerspectra_ALAE, powerspectra_StyleGAN, powerspectra_VAE, modes, savefig=None, dataset_name='SKIRT', label_function=None):
    
    pixel_size=0.276
    
    font_size = 13
    title_font_size = 13
    
    fig = make_subplots(rows=len(modes), cols=len(modes), start_cell="bottom-left",
                   horizontal_spacing = 0.025, vertical_spacing = 0.025) # horizontal_spacing = 0.075, vertical_spacing = 0.075) #, shared_yaxes=True, shared_xaxes=True)
   
    # fig, axes = plt.subplots(nrows=len(modes_target), ncols=len(modes_source), squeeze=False)
    template = 'simple_white'
    
    def format_mode_label(mode):
        if not label_function is None:
            return label_function(mode)
            
        if int(mode) == 0:
            str_ = u"\u221E" # r'\infty' 
        else:
            rad = (1 / int(mode)) * pixel_size 
            str_ = u"{:.2f}".format(rad)
        # return r'log-cycles at ' + str_ + r'$ckpch^{-1}$' + f' [{mode}]'
        # return r'{}\, {} {}'.format(r'${} '.format(str_), r'\text{ckpch}^{-1}', r' [{}]$'.format(mode))
        # return u'{} [{} ckpch<sup>-1</sup>]'.format(mode, str_)
        return u'Mode {}'.format(mode)
    
    color_alae = '#ff7f0e' #'rgb(127, 60, 141)'
    color_skirt = '#d62728' #'rgb(17, 165, 121)'
    color_stylegan = '#2ca02c' # 'rgb(57, 105, 172)'
    color_vae = '#1f77b4' # 'rgb(242, 183, 1)'
    # min_v = -.5
    # max_v = 8
    showlegend = True
    for mode_target, i in zip(modes, range(len(modes))):
        for mode_source, j in zip(modes, range(len(modes))):
            
            if mode_target < mode_source:
                continue
            
            f_ALAE, x_alae, y_alae = density_estimates(powerspectra_ALAE, mode_target, mode_source)
            f_SKIRT, x_skirt, y_skirt = density_estimates(powerspectra_SKIRT, mode_target, mode_source)
            f_StyleGAN, x_stylegan, y_stylegan = density_estimates(powerspectra_StyleGAN, mode_target, mode_source)
            f_VAE, x_vae, y_vae = density_estimates(powerspectra_VAE, mode_target, mode_source)
                
            x_alae = np.log10(x_alae)
            y_alae = np.log10(y_alae)
            x_skirt = np.log10(x_skirt)
            y_skirt = np.log10(y_skirt)
            x_stylegan = np.log10(x_stylegan)
            y_stylegan = np.log10(y_stylegan)
            x_vae = np.log10(x_vae)
            y_vae = np.log10(y_vae)
            
            weights_alae, edges_alae = np.histogram(x_alae, bins=20) 
            weights_skirt, edges_skirt = np.histogram(x_skirt, bins=20)
            weights_stylegan, edges_stylegan = np.histogram(x_stylegan, bins=20) 
            weights_vae, edges_vae = np.histogram(x_vae, bins=20)

            weights_y_alae, edges_y_alae = np.histogram(y_alae, bins=20) 
            weights_y_skirt, edges_y_skirt = np.histogram(y_skirt, bins=20)
            weights_y_stylegan, edges_y_stylegan = np.histogram(y_stylegan, bins=20) 
            weights_y_vae, edges_y_vae = np.histogram(y_vae, bins=20)


            min_v_alae = edges_alae[0]
            min_v_skirt = edges_skirt[0]
            min_v_stylegan = edges_stylegan[0]
            min_v_vae = edges_vae[0]

            max_v_alae = edges_alae[-1]
            max_v_skirt = edges_skirt[-1]
            max_v_stylegan = edges_stylegan[-1]
            max_v_vae = edges_vae[-1]

            min_v = min(min_v_alae, min_v_skirt, min_v_stylegan, min_v_vae)
            max_v = max(max_v_alae, max_v_skirt, max_v_stylegan, max_v_vae)
                
            bins_barplot = np.linspace(min_v, max_v, 15)
            
            if mode_target == mode_source:
                # bins = np.linspace(min_v, max_v, 30)
                # bins_barplot = np.linspace(min_v, max_v, 30-1)
                # weights_t, edges_t = np.histogram(x_t, bins=bins) 
                # weights_s, edges_s = np.histogram(x_s, bins=bins) 
                
                weights_alae, edges_alae = np.histogram(x_alae, bins=bins_barplot) 
                weights_skirt, edges_skirt = np.histogram(x_skirt, bins=bins_barplot)
                weights_stylegan, edges_stylegan = np.histogram(x_stylegan, bins=bins_barplot) 
                weights_vae, edges_vae = np.histogram(x_vae, bins=bins_barplot)
                
                weights_y_alae, edges_y_alae = np.histogram(y_alae, bins=bins_barplot) 
                weights_y_skirt, edges_y_skirt = np.histogram(y_skirt, bins=bins_barplot)
                weights_y_stylegan, edges_y_stylegan = np.histogram(y_stylegan, bins=bins_barplot) 
                weights_y_vae, edges_y_vae = np.histogram(y_vae, bins=bins_barplot)
                
                
                weights_alae = weights_alae / np.sum(weights_alae)
                weights_skirt = weights_skirt / np.sum(weights_skirt)
                weights_stylegan = weights_stylegan / np.sum(weights_stylegan)
                weights_vae = weights_vae / np.sum(weights_vae)
                
                weights_y_alae = weights_y_alae / np.sum(weights_y_alae)
                weights_y_skirt = weights_y_skirt / np.sum(weights_y_skirt)
                weights_y_stylegan = weights_y_stylegan / np.sum(weights_y_stylegan)
                weights_y_vae = weights_y_vae / np.sum(weights_y_vae)
                
            
                
                max_x = max(max(x_alae), max(x_skirt), max(x_stylegan), max(x_vae))
                max_y = max(max(y_alae), max(y_skirt), max(y_stylegan), max(y_vae))
                min_x = min(min(x_alae), min(x_skirt), min(x_stylegan), min(x_vae))
                min_y = min(min(y_alae), min(y_skirt), min(y_stylegan), min(y_vae))
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_alae,
                    legendgroup='group1',
                    marker_color = color_alae,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_skirt,
                    legendgroup='group2',
                    marker_color = color_skirt,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_stylegan,
                    legendgroup='group3',
                    marker_color = color_stylegan,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_y_vae,
                    legendgroup='group4',
                    marker_color = color_vae,
                    showlegend = False
                ), row = len(modes)-i, col=j+1)
                
                fig.update_xaxes(row=len(modes)-i, range=[min_y, max_y], col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, showticklabels=False)
                fig.update_yaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, side = 'right')
                if j==0:
                    fig.update_yaxes(title_text=format_mode_label(mode_target), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : title_font_size}, side= 'left')
                    fig.update_xaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5)   
                if i == len(modes) - 1 and j == len(modes) - 1:
                    fig.update_yaxes(row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 5, side='right')
                    fig.update_xaxes(title_text=format_mode_label(mode_source), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : title_font_size}, showticklabels=True)
                    
            
            else: 
            
                
                fig.add_trace(go.Histogram2dContour(
                        x = y_alae,
                        y = x_alae,
                        opacity=1,
                        name='ALAE',
                        ncontours = 5,
                    
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : color_alae}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group1',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)

                fig.add_trace(go.Histogram2dContour(
                        x = y_skirt,
                        y = x_skirt,
                        opacity=1,
                        name=dataset_name,
                        ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : color_skirt}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group2',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)
                
                fig.add_trace(go.Histogram2dContour(
                        x = y_stylegan,
                        y = x_stylegan,
                        opacity=1,
                        name='StyleGAN',
                        ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : color_stylegan}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group3',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)
                
                fig.add_trace(go.Histogram2dContour(
                        x = y_vae,
                        y = x_vae,
                        name='VAE',
                        opacity=1,
                        ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        # colorscale = 'Oranges',
                        line = {'color' : color_vae}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group4',
                        showlegend = showlegend
                ), row=len(modes)-i, col=j+1)
                
                showlegend = False
                
                max_ = max(max(x_alae), max(y_alae), max(x_skirt), max(y_skirt), max(x_stylegan), max(y_stylegan), max(x_vae), max(y_vae))
                min_ = min(min(x_alae), min(y_alae), min(x_skirt), min(y_skirt), min(x_stylegan), min(y_stylegan), min(x_vae), min(y_vae))
                
                max_x = max(max(x_alae), max(x_skirt), max(x_stylegan), max(x_vae))
                max_y = max(max(y_alae), max(y_skirt), max(y_stylegan), max(y_vae))
                min_x = min(min(x_alae), min(x_skirt), min(x_stylegan), min(x_vae))
                min_y = min(min(y_alae), min(y_skirt), min(y_stylegan), min(y_vae))
                
                fig.update_traces(showscale=False, row=len(modes)-i, col=j+1)
          
                fig.update_xaxes(range=[min_y, max_y], row=len(modes)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 4, showticklabels=False)
                fig.update_yaxes(range=[min_x, max_x], row=len(modes)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 4, showticklabels=False)
                if i == len(modes) - 1:
                    fig.update_xaxes(range=[min_y, max_y], title_text=format_mode_label(mode_source), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 4, titlefont = {'size' : title_font_size}, showticklabels=True)
                if j == 0:
                    fig.update_yaxes(range=[min_x, max_x], title_text=format_mode_label(mode_target), row=len(modes)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 4, titlefont = {'size' : title_font_size}, showticklabels=True)
                
                
                #data1 = pd.DataFrame()
                #data1['x'] = x_t
                #data1['y'] = y_t
                #data1['source'] = 'generated'
                #data2 = pd.DataFrame()
                #data2['x'] = x_s
                #data2['y'] = y_s
                #data2['source'] = 'truth'

                #data = pd.concat([data1, data2])
                #data = data.reset_index()

                #sns.kdeplot( 
                #    x=x_t, 
                #    y=y_t, color='orange', ax=axes[i,j])

                #sns.kdeplot( 
                #    x=x_s, 
                #    y=y_s, color='blue', ax=axes[i,j])

                #axes[i,j].set(ylim=(-.5, 3))
                #axes[i,j].set(xlim=(-.5, 3))

                #fig.add_trace(go.Contour(
                #        z=f,
                #        colorbar=dict(
                #            title='Color bar title', # title here
                #            titleside='right',
                #            titlefont=dict(
                #                size=14,
                #                family='Arial, sans-serif')
                #        )))
            
    
    fig.update(
        layout_showlegend=True,
        layout_coloraxis_showscale=False)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * (2*mnras_col_wid + mnras_col_sep)) 
    fig_height = fig_width
    
    fig.update_layout(
        template=template, 
        autosize=True,
        font_family="Serif",
        font_color="black",
        font_size=10, 
        width=fig_width,
        height=fig_height,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        title_font_size=10,
        legend=dict(
            yanchor="top",
            y=1.008,
            xanchor="left",
            x=0.175
        )
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    
    if savefig:
        fig.write_image(savefig)
    
    return fig
    
def cor_powerspectra(powerspectra_target, powerspectra_source, modes_target, modes_source, pixel_size=0.276, source_label='Source', generated_label='Generated', savefig=None):
    
    fig = make_subplots(rows=len(modes_source), cols=len(modes_target), start_cell="bottom-left",
                   horizontal_spacing = 0.075, vertical_spacing = 0.075) #, shared_yaxes=True, shared_xaxes=True)
   
    # fig, axes = plt.subplots(nrows=len(modes_target), ncols=len(modes_source), squeeze=False)
    template = 'plotly_white'
    
    def format_mode_label(mode):
        if int(mode) == 0:
            str_ = r'\infty'
        else:
            rad = (1 / int(mode)) * pixel_size 
            str_ = "{:.2f}".format(rad)
        # return r'log-cycles at ' + str_ + r'$ckpch^{-1}$' + f' [{mode}]'
        return r'{}\, {} {}'.format(r'${} '.format(str_), r'\text{ckpch}^{-1}', r' [{}]$'.format(mode))
    
    # min_v = -.5
    # max_v = 8
    showlegend = True
    for mode_target, i in zip(modes_target, range(len(modes_target))):
        for mode_source, j in zip(modes_source, range(len(modes_source))):
            
            if mode_target < mode_source:
                continue
            
            f_target, x_t, y_t = density_estimates(powerspectra_target, mode_target, mode_source)
            f_source, x_s, y_s = density_estimates(powerspectra_source, mode_target, mode_source)
                
            x_t = np.log10(x_t)
            y_t = np.log10(y_t)
            x_s = np.log10(x_s)
            y_s = np.log10(y_s)
            
            if mode_target == mode_source:
                # bins = np.linspace(min_v, max_v, 30)
                # bins_barplot = np.linspace(min_v, max_v, 30-1)
                # weights_t, edges_t = np.histogram(x_t, bins=bins) 
                # weights_s, edges_s = np.histogram(x_s, bins=bins) 
                
                weights_t, edges_t = np.histogram(x_t, bins=30) 
                weights_s, edges_s = np.histogram(x_s, bins=30)
                
                weights_y_t, edges_y_t = np.histogram(y_t, bins=30) 
                weights_y_s, edges_y_s = np.histogram(y_s, bins=30)
                
                
                min_v_t = edges_t[0]
                min_v_s = edges_s[0]
                max_v_t = edges_t[-1]
                max_v_s = edges_s[-1]
                min_v = min(min_v_t, min_v_s)
                max_v = max(max_v_t, max_v_s)
                
                bins_barplot = np.linspace(min_v, max_v, 30-1)
                
                weights_t = weights_t / np.sum(weights_t)
                weights_s = weights_s / np.sum(weights_s)
                
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_t,
                    legendgroup='group2',
                    marker_color = 'orange',
                    showlegend = False
                ), row = len(modes_target)-i, col=j+1)
              
                fig.add_trace(go.Bar(
                    x = bins_barplot,
                    y = weights_s,
                    legendgroup='group1',
                    marker_color = 'blue',
                    showlegend = False
                ), row = len(modes_target)-i, col=j+1)
                
                fig.update_xaxes(row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7)
                fig.update_yaxes(row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7)
                if j==0:
                    fig.update_yaxes(title_text=format_mode_label(mode_target), row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : 14})
                    fig.update_xaxes(row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7)   
                if i == len(modes_target) - 1 and j == len(modes_source) - 1:
                    fig.update_yaxes(row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7)
                    fig.update_xaxes(title_text=format_mode_label(mode_source), row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : 14})
                    
            
            else: 
            
                
                fig.add_trace(go.Histogram2dContour(
                        x = x_s,
                        y = y_s,
                        opacity=1,
                        name=source_label,
                        # ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        line = {'color' : 'blue'}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group1',
                        showlegend = showlegend
                ), row=len(modes_target)-i, col=j+1)

                fig.add_trace(go.Histogram2dContour(
                        x = x_t,
                        y = y_t,
                        name=generated_label,
                        opacity=1,
                        # ncontours = 5,
                        # nbinsx = 100,
                        # nbinsy = 100, 
                        # colorscale = 'Oranges',
                        line = {'color' : 'orange'}, # 'smoothing' : 1.3},
                        contours={'coloring' : 'none'},
                        legendgroup='group2',
                        showlegend = showlegend
                ), row=len(modes_target)-i, col=j+1)
                
                showlegend = False
                
                max_ = max(max(x_s), max(y_s), max(x_t), max(y_t))
                min_ = min(min(x_s), min(y_s), min(x_t), min(y_t))
                
                fig.update_traces(showscale=False, row=len(modes_target)-i, col=j+1)
                print([min_v, max_v])
                fig.update_xaxes(range=[min_, max_], row=len(modes_target)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 7)
                fig.update_yaxes(range=[min_, max_], row=len(modes_target)-i, col=j+1, gridcolor='rgba(211,211,211,1)', gridwidth=.2, mirror='allticks', ticks='inside', showline=True, nticks = 7)
                if i == len(modes_target) - 1:
                    fig.update_xaxes(title_text=format_mode_label(mode_source), row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : 14})
                if j == 0:
                    fig.update_yaxes(title_text=format_mode_label(mode_target), row=len(modes_target)-i, col=j+1, mirror='allticks', ticks='inside', showline=True, nticks = 7, titlefont = {'size' : 14})
                
                
                #data1 = pd.DataFrame()
                #data1['x'] = x_t
                #data1['y'] = y_t
                #data1['source'] = 'generated'
                #data2 = pd.DataFrame()
                #data2['x'] = x_s
                #data2['y'] = y_s
                #data2['source'] = 'truth'

                #data = pd.concat([data1, data2])
                #data = data.reset_index()

                #sns.kdeplot( 
                #    x=x_t, 
                #    y=y_t, color='orange', ax=axes[i,j])

                #sns.kdeplot( 
                #    x=x_s, 
                #    y=y_s, color='blue', ax=axes[i,j])

                #axes[i,j].set(ylim=(-.5, 3))
                #axes[i,j].set(xlim=(-.5, 3))

                #fig.add_trace(go.Contour(
                #        z=f,
                #        colorbar=dict(
                #            title='Color bar title', # title here
                #            titleside='right',
                #            titlefont=dict(
                #                size=14,
                #                family='Arial, sans-serif')
                #        )))
            
    
    fig.update(
        layout_showlegend=True,
        layout_coloraxis_showscale=False)
    
    mnras_col_wid = 3.3258 * 1.3
    mnras_col_sep = 0.3486 * 1.3
    fig_width = int(96 * (2*mnras_col_wid + mnras_col_sep)) 
    fig_height = fig_width
    
    fig.update_layout(
        template=template, 
        autosize=True,
        font_family="Serif",
        font_color="black",
        font_size=14, 
        width=fig_width,
        height=fig_height,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        title_font_size=14,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.2
        )
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', nticks = 10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', nticks = 10)
    
    if savefig:
        fig.write_image(savefig)
    
    return fig