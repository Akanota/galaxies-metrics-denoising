## Realistic galaxy images and improved robustness in machine learning tasks from generative modelling

Code repository for our paper [Realistic galaxy images and improved robustness in machine learning tasks from generative modelling](https://arxiv.org/abs/2203.11956) by Benjamin Holzschuh, [Conor O'Riordan](https://www.mpa-garching.mpg.de/person/109805), [Simona Vegetti](https://www.mpa-garching.mpg.de/person/44138), [Vicente Rodriguez-Gomez](https://vrodgom.github.io/) and [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/) published in [MNRAS](https://doi.org/10.1093/mnras/stac1188). This repository contains all scripts for downloading and preprocessing our training data sets, an adaption of the code by [Karras et al. (2020)](https://github.com/NVlabs/stylegan) for calculating the metrics introduced in the paper, and an implementation of the denoising models.

![StyleGAN_512_TNG_preview_colour_2](https://user-images.githubusercontent.com/16702943/169419393-9682ffc0-b5df-4539-8100-ce1b9c424388.png)
![TNG_preview_colour_2](https://user-images.githubusercontent.com/16702943/169419651-9316b1e6-0892-4ede-a31f-cd4b03432f3e.png)

------

### Main results
#### Data sets
We compare different approaches for generating galaxy images and demonstrate how generated images can be used to supplement existing data sets in downstream machine learning tasks. A particular focus lies on approaches based on generative adversarial networks. 
To evaluate the data quality of the trained models, we consider a variety of metrics based on feature-extraction networks (FID, KID), the power spectrum and morphological measurements. In this GitHub repository, we focus on generated data from an architecture based on StyleGAN, however our paper contains additional results for adversarial latent auto encoders and variational autoencoders. 

 <sup>SKIRT</sup> | <sup>COSMOS</sup> | <sup>Sérsic profiles</sup> |
:---------------------------------:|:---------------------------------:|:---------------------------------:
<img src="https://user-images.githubusercontent.com/16702943/177981070-f950a790-2ccf-4ab9-afe9-c038b9e29462.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177981151-73800a97-e36c-4003-91b6-ce3097f908d7.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177981203-ac0e3e4a-b029-4e2a-bbb1-754ed3c61a35.png" width="100%" />
<img src="https://user-images.githubusercontent.com/16702943/177983004-5f8c06da-ccc0-48de-89cc-7f55ac576360.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177983072-95d85fa7-aeb1-49b4-b0de-12e27ea7b367.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177983123-b62415c9-9be0-4d8a-ae59-6e8f4d39129c.png" width="100%" />

The images above show a sample of galaxies from the training data set (top row) and generated galaxies (bottom row). SKIRT comprises synthetic galaxy images obtained by running the [SKIRT code](https://skirt.ugent.be/root/_landing.html) on data from the [IllustrisTNG simulation](https://www.tng-project.org/). COSMOS refers to a data set of galaxy observations from the Hubble telescope. Finally, Sérsic profiles are a collection of galaxy images based on an analytic description of how the light intensity of the galaxy varies depending on the distance from its center. All images have a size of 256 x 256 with either 1 channel (COSMOS, Sérsic profiles) or 5 channels (SKIRT, 4 bands g, r, i, z + 1 channel for encoding additional size information).

#### Evaluation

We evaluate the quality of the generated data using a number of different metrics and provide results in Table 2 in our paper. 
The training data set size is *9 564* for SKIRT, *20 114* for COSMOS and *50 000* for Sérsic profiles, therefore SKIRT has by far the smallest number of data samples. 
In addition to that, SKIRT is comprised of galaxies with the most detailed morpgological structures, while Sérsic profiles are the most simple. 
Both those circumstances are reflected in the FID and KID. For example with our StyleGAN-based model, we obtain a FID of *7.76* for SKIRT, *1.45* for COSMOS and *0.55* for Sérsic profiles (lower is better). A similar results is obtained from our analysis of the 2D power spectrum. 
Overall, this shows that data sets comparable to COSMOS can already be learned extremely well using modern architectures for generative modelling. 
Nonetheless, more difficult data sets with a smaller number of samples, like SKIRT, already show very convincing results that are visually very hard or impossible to distinguish from the ground truth data. We expect that this will improve even more with larger data set sizes and further progress in methods for generative modelling in the near future.

Below, we show a comparison of the contour plots of the 2D power spectrum between generated data and the training data set for the SKIRT and COSMOS data set. 

 <sup>SKIRT</sup> | <sup>COSMOS</sup> | 
:---------------------------------:|:---------------------------------:
<img src="https://user-images.githubusercontent.com/16702943/177984328-be75b271-3b1b-447f-b7d9-0f7c7ade52a4.png" width="500px" /> | <img src="https://user-images.githubusercontent.com/16702943/177985611-3389ab34-6f75-4906-9326-0776457a1fd6.png" width="500px" />


Another cornerstone of our evaluation is a comparison based on morphological properties, i.e. properties such as the smoothness or half-light radius.
We compute the 1D Wasserstein distance between the histograms of the generated and ground truth data for each considered property as part of our evaluation metrics in Table 2 in our paper.
A subset of properties and their corresponding histograms for the SKIRT data is shown below. Overall, the generated data reproduces the distribution of morphological properties very well. 

![Download](https://user-images.githubusercontent.com/16702943/177980557-53feca8d-c3b7-472b-8924-897f2a61d4f2.png)

#### Denoising

A further method to evaluate the quality of the generated data is to test if the ground truth data can be replaced by generated data in downstream tasks. If the distribution of the generated data matches the ground truth data very well, we expect that the performance on the trained downstream tasks will be the same or very similar. Below we show the data set splits and methodology we adopt. 

<p align='center'>
   <img src="https://user-images.githubusercontent.com/16702943/178718932-c01532ac-85da-4ef9-b4cb-90546f034362.png" width="80%" />
</p>

As downstream task, we consider image denoising. A convolutional neural network is trained to remove the noise and PSF (point spread function, effectively a blurring of the image) from the images, which is artifically added to them. The parameter $\alpha$ controls the probability with which the next sample is drawn from the generated data set or the ground truth data set. If $\alpha = 0$, then the network is trained on the ground truth training data only (only SKIRT data) and if $\alpha = 1$, then the network is trained on generated data only. For this task, we consider the SKIRT data set and a data set of generated SKIRT images with 50 000 samples. Below, we show an example inputs and predictions of the networks. 

  &nbsp;&nbsp;&nbsp;&nbsp;<sup>Ground truth</sup>&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>noise + PSF</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | <sup>Prediction (only SKIRT data)</sup> | <sup>Prediction (only generated data)</sup>
:---------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:
<img src="https://user-images.githubusercontent.com/16702943/178508554-e019aab5-cc8e-4fa9-97bb-b689cc0cf976.png" /> | <img src="https://user-images.githubusercontent.com/16702943/178508580-04469b6d-cc1b-48b1-9bfd-d6f17980c6da.png" /> | <img src="https://user-images.githubusercontent.com/16702943/178508594-99a8edcb-13dd-4cce-9100-990923e3f782.png" /> | <img src="https://user-images.githubusercontent.com/16702943/178508606-a63ad13c-1c09-4d41-b79b-f48fa11288c9.png"  />

We evaluate the trained networks using the MSE (mean squared error) on a test data set. We are interested in the robustness of the networks with respect to data augmentations. For this, we consider four cases: 

- (i) no augmentation
- (ii) downsample the images to half the resolution (increased physical pixel size)
- (iii) increase the noise level (quadruple the standard deviation)
- (iv) combine augmentations (ii) and (iii)

For each combination of $\alpha$ and augmentation, we trained and evaluated 5 networks. We show the results of our evaluation below. 

 <sup>(i)</sup> | <sup>(ii)</sup> | <sup>(iii)</sup> | <sup>(iv)</sup>
:---------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:
<img src="https://user-images.githubusercontent.com/16702943/177776235-df252fe3-75e5-4746-933e-69ab749a17ac.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177776307-5127b2d3-c5e2-4bab-9940-2e598eab8cee.png" width="100%" />  | <img src="https://user-images.githubusercontent.com/16702943/177776338-3e791bcf-e1f2-4643-a2fe-a7d54ad2e3c1.png" width="100%" /> | <img src="https://user-images.githubusercontent.com/16702943/177776365-49c87c92-c78b-4121-9881-2e4d4c384a71.png" width="100%" />

If there are no augmentations in the test data set, the networks attain the same performance ($\alpha \leq 0.8$) or the performance is only marginally worse ($\alpha = 1.0$, only generated data). For (ii), (iii), (iv), we see that the addition of generated data improves the MSE considerably. In fact, training only on generated data gives the best performance in those case.   

### Requirements

The code in this repository was tested using Python 3.7, cuda 10.0 and cudnn 7.6.5 on a GeForce RTX 2080 Ti. More recent versions of cuda and cudnn might also work. 
To install required packages, run 
```bash
> pip install -r requirements.txt
```

### Create training data sets

This section describes how each of our training data sets are created. `notebooks/open_datasets.ipynb` shows how the .h5 files can be accessed and visualized.

#### SKIRT

- Create an account on the official [IllustrisTNG website](https://www.tng-project.org/) 

- Once your account has been approved, you will receive an API key, which can be used to download the SKIRT data set

- Download and extract the .tar archive

  ```bash
  > cd data
  > export TNG_API=<YOUR API KEY>
  > wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header="API-Key: $TNG_API" "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar" > skirt_images_sdss.99.tar
  > tar -xvf skirt_images_sdss.99.tar
  ```

- The extracted files will be located at `data/sdss`

- Run the pre-processing scripts to create .h5, .tfrecords and stats files

  ```bash
  > cd preprocessing 
  > python preprocessing_h5py_tfrecrods.py
  ```
  
  The generated files will be located at `data/SKIRT`

#### COSMOS

Download and unpack the raw COSMOS files provided by [Mandelbaum et al. (2011)](https://irsa.ipac.caltech.edu/data/COSMOS/images/shera_galaxy_postage_stamps/index.html)

```bash
> cd data/COSMOS
> wget "https://irsa.ipac.caltech.edu/data/COSMOS/images/shera_galaxy_postage_stamps/cosmos_galaxy_postage_stamps.tar"
> tar -xvf "cosmos_galaxy_postage_stamps.tar"
> cd ../..
```

Run the preprocessing script to generate .h5, .tfrecords and stats files 

```bash
> cd preprocessing/COSMOS
> python preprocessing.py
```

The generated files will be located at `data/COSMOS`

#### Sérsic profiles

To generate the .h5, .tfrecords and stats files, run

```bash
> cd preprocessing/Sersic
> python preprocessing.py
```

The generated files will be located at `data/Sersic`

### Download generated data sets

We provide the generated images from our StyleGAN-based model. This encompasses ca. 26.4GB for the SKIRT data, 11.3GB for the COSMOS data and 12.3GB for the generated Sérsic profiles. Instructions for downloading can be found [here](https://dataserv.ub.tum.de/index.php/s/m1661654). 
For the notebooks, place the downloaded files in the directory `data/generated`.

### Metrics

In this section, we describe how to calculate different metrics for comparing the training data sets with the generated data using our provided code. We show how to generate several plots in `notebooks/metrics.ipynb`. All examples are assuming that the training data is located at `data/SKIRT/SKIRT_training.h5` and the generated data at `data/SKIRT/SKIRT_generated.h5`.

The examples can be modified easily for the COSMOS and Sérsic profiles data sets. 

- Run the evaluation script 

  ```bash
  > cd metrics
  > python run_metrics.py -h
  
  usage: run_metrics.py [-h] --generated-file GENERATED_FILE --training-file
                      TRAINING_FILE --config CONFIG

  Parameter Parser

  optional arguments:
    -h, --help            show this help message and exit
    --generated-file GENERATED_FILE
                          Generated data file (default: None)
    --training-file TRAINING_FILE
                          Training data file (default: None)
    --config CONFIG       Either "SKIRT", "COSMOS" or "Sersic" (default: None)
  ```
  For example, for the SKIRT data
  ```bash
  > python run_metrics.py --generated-file ../data/SKIRT/SKIRT_generated.h5 --training-file ../data/SKIRT/SKIRT_training.h5 --config SKIRT
  ```
  This will calculate the 2D power spectra, the FID and KID as well as the morphological measurements of the galaxies. 

- The results will be cached at `data/cache`. 

- In `notebooks/metrics.ipynb` we evaluate the computed metrics and show some plots. 

### Galaxy Denoising

Before the denoising networks can be trained, we need an additional preprocessing step to apply the PSF (point spread function, blurring) to the images. We found that doing this once before the training saves time significantly. 

```bash
> cd preprocessing/SKIRT
> python preprocessing_channel_h5py_full_convolved.py
> python preprocessing_channel_h5py_full_convolved_gen.py
> cd ../..
```

This will create two new files `data/SKIRT/SKIRT_training_psf.h5` and `data/SKIRT/SKIRT_training_gen_psf.h5`.

The code for galaxy denoising is stored at `denoising/`. To train a model, run the following

```bash
> cd denoising
> python train.py -h

usage: train.py [-h] [--architecture ARCHITECTURE] [--name NAME] [--file FILE]
                [--gen-file GEN_FILE] [--continue-id CONTINUE_ID] [--gpu GPU]
                [--alpha-mixing ALPHA_MIXING] [--weight-decay WEIGHT_DECAY]
                [--api-key API_KEY]

Parameter Parser

optional arguments:
  -h, --help            show this help message and exit
  --architecture ARCHITECTURE
                        Architecture of transfer network (default: ResNetV1)
  --name NAME           Name of experiment (default: None)
  --continue-id CONTINUE_ID
                        ID of run to continue (default: None)
  --gpu GPU             Visible GPUs (default: None)
  --alpha-mixing ALPHA_MIXING
                        Dataset mixing coefficient (default: 0.0)
  --api-key API_KEY     Wandb API key (default: None)
  --file FILE           Data file (training data) (default: None)
  --gen-file GEN_FILE   Data file (generated data) (default: None)

```

- currently `ResNetV1` is the only available architecture. See our paper for details.
- we employ wandb for logging our results. If no API key (`api-key`) is specified, everything will be logged offline, which requires no wandb account. Optionally you can change the wandb entity when syncing your runs to the cloud in the train.py file by modifying the variable `ENTITY`.
- every run gets a unique ID assigned. Runs that are aborted before they are finished can be continued by specifying the `continue-id`
- `alpha-mixing` is the variable $\alpha$ in our paper

For example, we can run 

```bash
> python train.py --alpha-mixing 0.5 --file ../data/SKIRT/SKIRT_training_denoising.h5 --gen-file ../data/SKIRT/SKIRT_generated_denoising.h5
```

Additionally to wandb, training metrics and final evaluation are stored in `store/<ID>.py`  where `<ID>` is the ID assigned to the run by wandb

We show some visualizations and how to load trained models in `notebooks/denoising.ipynb`

### Citation

If find our work helpful to your research and/or using some of the code in this repository, please cite our paper
```tex
@article{10.1093/mnras/stac1188,
    author = {Holzschuh, Benjamin J and O’Riordan, Conor M and Vegetti, Simona and Rodriguez-Gomez, Vicente and Thuerey, Nils},
    title = {Realistic galaxy images and improved robustness in machine learning tasks from generative modelling},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {515},
    number = {1},
    pages = {652-677},
    year = {2022},
    month = {05},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac1188},
    url = {https://doi.org/10.1093/mnras/stac1188},
    eprint = {https://academic.oup.com/mnras/article-pdf/515/1/652/45026955/stac1188.pdf},
}
```
