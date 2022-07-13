# Copyright (c) 2022, Conor O'Riordan

from lensmcmc.models.sources.SourceModel import *
from scipy.special import gamma


class Sersic(SourceModel):

    def __init__(self, params):

        # Units arcsec
        self.x = params['x_position']
        self.y = params['y_position']
        self.r = params['radius']

        # Dimensionless
        self.n = params['sersic_index']

        # Set by peak brightness
        if 'peak_brightness' in params.keys():
            # Units W m^-2 arcsec^-2
            self.I0 = params['peak_brightness']
            self.fT = self.I0 * self.brightness_constant()

        # Set by total flux
        elif 'total_flux' in params.keys():
            # Units W m^-2
            self.fT = params['total_flux']
            self.I0 = self.fT / self.brightness_constant()

        # Set by total counts
        else:
            self.fT = params['total_counts']
            self.I0 = self.fT / self.brightness_constant()

        # Account for ellipticity/position angle transformation
        if 'ellipticity_x' in params.keys():
            self.q = 1.0 - np.hypot(params['ellipticity_x'],
                                    params['ellipticity_y'])
            self.a = np.arctan2(params['ellipticity_y'],
                                params['ellipticity_x'])

        else:
            self.q = params['axis_ratio']
            self.a = params['position_angle']

    def brightness_constant(self):
        # Units of arcsec^2
        return (self.r ** 2) * (2 * np.pi * self.n / (sersic_b(self.n) ** (2 * self.n))) * gamma(2 * self.n)

    def brightness_profile(self, beta):
        """
        Creates a Sersic source centered on (srcx, srcy)
        with a radius srcr, peak brightness srcb and Sersuc index m.
        """

        # Shift to source centre
        z = beta - self.x - 1j * self.y

        # Rotate
        z *= np.exp(1j * self.a)

        # Transform to elliptical coords.
        rootq = np.sqrt(self.q)
        r = np.hypot(rootq * z.real, z.imag / rootq)

        # Call the sersic profile
        return self.I0 * np.exp(- sersic_b(self.n) * (r / self.r) ** (1.0 / self.n))


def sersic_b(n):
    """
    Half light radius constant
    """
    return (2 * n) - (1.0 / 3.0) + (4.0 / (405.0 * n)) + \
           (46.0 / (25515.0 * n ** 2)) + (131.0 / (148174.0 * n ** 3))
