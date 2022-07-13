# Copyright (c) 2022, Conor O'Riordan

import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import brenth
from astropy.convolution import Gaussian2DKernel, convolve


def pixels(pix_width, fov_width):
    """
    Creates a grid of pixels each of width pix_width
    (arcsec) spanning fov_width (edge to edge)
    """

    num_pix = int(fov_width / pix_width)
    limit = (fov_width / 2.0) - (pix_width / 2.0)

    # Define 1D grid (force float32 for gpu usage)
    x = np.linspace(- limit, limit, num_pix, dtype='float32')

    return np.meshgrid(x, x)


def snr_set(img0, target_snr=100, sig=2.0, stddev=2.0):
    """
    Finds the level of noise which sets the integrated SNR within
    the 2-sigma contours to the specified value, via an interpolation.
    """
    # Integrate signal for signal
    total_sig = img0.sum()

    # Set possible noise levels according to total signal
    levels = np.logspace(-10.0, 6.0, 101) * total_sig

    # Calculate the snr at all noise levels
    snrs = np.array([snr_find(img0 + np.random.normal(0.0, n, img0.shape), n, sig=sig, stddev=stddev)[0]
                     for n in levels])

    # Remove NaN values
    levels = levels[np.isfinite(snrs)]
    snrs = snrs[np.isfinite(snrs)]

    # Interpolate a function f(noise) = SNR(noise) - SNR_Target
    f = interp1d(levels, snrs - target_snr, kind='linear')

    # Find the root
    r = brenth(f, levels[0], levels[-1])

    # Return both the noise levels and the mask from the convolved image
    return r, snr_find(img0, r, sig)[1]


def snr_find(image, nlevel, sig=2.0, stddev=2.0):
    """
    Calculates the integrated snr in within the 2-sigma contours.
    """
    # Initialise kernal and convolve
    g = Gaussian2DKernel(x_stddev=stddev, y_stddev=stddev)
    img1 = convolve(image, g, boundary='extend')

    # Take the 2-sigma contour of the convolved image
    mask = (img1 > sig * nlevel).astype('float')

    # Calculate snr of original image within the contours bof the convolved image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = (mask * image).sum() / ((mask * nlevel ** 2.0).sum() ** 0.5)

    return snr, mask


def counts_AB(magnitude, instrument):
    """
    Converts AB magnitude to counts using the instrument
    exposure time and zero point
    """
    return instrument['exposure_time'] * 10 ** (
        0.4 * (instrument['zero_point'] - magnitude)
    )
