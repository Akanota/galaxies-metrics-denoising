import pandas as pd
import ot

   
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../metrics')

from metrics import data_iterator
from metrics.radially_averaged_power_spectrum import *

from astropy.visualization import make_lupton_rgb
    
def show_gallery(file, start=0, rows=5, columns=5, transform=True, style='lupton', data='skirt'):
    iterator = data_iterator.H5PyIterator(file, transform=transform)
    data_list = []

    for i in range(start-1):
        iterator.__next__()
        
    for i in range(rows * columns):
        data_list.append(iterator.__next__())
    
    
    fig, ax = plt.subplots(rows, columns, squeeze=False, figsize=(columns*40/5, rows*40/5))
    for i in range(rows):
        for j in range(columns):
            
            if len(data_list[0].shape) == 2:
                copy_elem = np.array(data_list[0])
                data_list[0] = np.stack([copy_elem] * 3, axis=2)
            
            epsilon = 1e-6
            
            if data == 'skirt':
            
                r_channel = np.maximum(data_list[0][:,:,1],0) + epsilon
                g_channel = np.maximum(data_list[0][:,:,0],0) + epsilon
                i_channel = np.maximum(data_list[0][:,:,2],0) + epsilon

                rgb_default = make_lupton_rgb(i_channel, r_channel, g_channel)

                if style != 'lupton':
                    
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
                    
                ax[i][j].imshow(rgb_default)
            
            else:
                
                image_channel = data_list[0] 
                
                ax[i][j].imshow(image_channel[:,:,0], cmap='bone')
        
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            del data_list[0]
    
    
    plt.tight_layout()
    plt.show()
    
def get_powerspectrum_dataframe(results_dict):

    dataframe = pd.DataFrame()
    
    modes = list(range(15))

    for k in modes:
        dataframe[k] = None
    dataframe['name'] = None
    dataframe['avg'] = None
    
    w_dist = {'name' : 'w_dist'}
    mean_generated = {'name' : 'mean_generated'}
    std_generated = {'name' : 'std_generated'}

    mean_training = {'name' : 'mean_training'}
    std_training = {'name' : 'std_training'}

    power_partition_source = powerspectra_to_partition(results_dict['raps'][1]['SOURCE'][2])
    power_partition_target = powerspectra_to_partition(results_dict['raps'][1]['TARGET'][2])

    for mode in modes:

        bins = np.linspace(-5, 4, 1000)
        bins_midpoints = bins[:999] + bins[1] - bins[0]

        log_modes_source = np.log10(np.array(power_partition_source[mode]))
        log_modes_target = np.log10(np.array(power_partition_target[mode]))

        mean_training[mode] = log_modes_source.mean()
        std_training[mode] = np.sqrt(log_modes_source.var())

        mean_generated[mode] = log_modes_target.mean()
        std_generated[mode] = np.sqrt(log_modes_target.var())

        hist_source, edges = np.histogram(log_modes_source, bins=bins)
        hist_target, edges = np.histogram(log_modes_target, bins=bins)

        a1 = hist_source / np.sum(hist_source)
        a2 = hist_target / np.sum(hist_target)

        real_line = bins_midpoints
        _, r = ot.emd_1d(real_line, real_line, a1, a2, metric='euclidean', log=True)

        w_dist[mode] = r['cost']

    dataframe = dataframe.append(w_dist, ignore_index=True)
    dataframe = dataframe.append(mean_training, ignore_index=True)
    dataframe = dataframe.append(std_training, ignore_index=True)
    dataframe = dataframe.append(mean_generated, ignore_index=True)
    dataframe = dataframe.append(std_generated, ignore_index=True)

    dataframe = dataframe.set_index('name')
    
    avg_values = dataframe[modes].values.mean(axis=1)
    dataframe['avg'] = avg_values
    
    return dataframe

