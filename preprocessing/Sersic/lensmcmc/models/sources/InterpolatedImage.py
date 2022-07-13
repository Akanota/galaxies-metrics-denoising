# Copyright (c) 2022, Conor O'Riordan

from lensmcmc.models.sources.SourceModel import *
from scipy.interpolate import RectBivariateSpline


class InterpolatedImage(SourceModel):

    def __init__(self, params, x, y, img):
        """
        Source model with brightness function interpolated
        from img, with pixel coordinates x and y.
        
        Source params gives centre of source (x and y position)
        and a scale factor r which scales the image in the source
        plane.
        """

        # Parameters
        self.x = params['x_position']
        self.y = params['y_position']
        self.r = params['radius']  # Arbitrary scale factor for source

        # Initialise interpolation function
        self.source_function = RectBivariateSpline(x, y, img)

        # Normalise profile if total flux given
        if 'total_flux' in params.keys():
            self.total_flux = params['total_flux']

            # Get bounds for integral
            x0, x1 = x.min(), x.max()
            y0, y1 = y.min(), y.max()

            # Integrate to set total flux
            self.norm = self.total_flux / self.source_function.integral(x0, x1, y0, y1)

        else:
            self.norm = 1.0

    def brightness_profile(self, beta):

        # Shift to source centre
        z = beta - self.x - 1j * self.y

        # Evaluate brightness function
        surface_brightness = self.source_function(z.real.flatten() / self.r, z.imag.flatten() / self.r, grid=False)

        # Normalise and remove negative values
        return np.clip(surface_brightness.reshape(z.shape) * self.norm, 0.0, None)
