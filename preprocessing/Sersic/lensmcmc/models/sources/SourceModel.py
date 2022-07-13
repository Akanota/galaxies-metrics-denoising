# Copyright (c) 2022, Conor O'Riordan

import numpy as np


class SourceModel:
    """
    Generic class for handling source models.
    Specific classes must return brightness_profile
    """

    def ray_trace(self, grid, sub=1, N = None):
        """
        Finds the source plane surface brightness with
        sub-pixelisation. Use sub=1 for no sub-pixelisation
        """

        if sub == 1:
            beta = grid[0] + 1j * grid[1]
            return self.brightness_profile(beta)

        # Storage for grids and constants
        x1, x2 = [], []
        pwd = np.abs(grid[0][0, 0] - grid[0][0, 1])
        fov = grid[0].max() - grid[0].min() + pwd
        x_min, x_max = grid[0].min(), grid[0].max()
        y_min, y_max = grid[1].min(), grid[1].max()
        
        if N is None:
            N = int(np.ceil(fov/pwd)) # int(fov / pwd)

        # Loop over sub-pixelisation size and create
        # up to p.sub sub-pixelised grids
        for s in range(1, sub + 1):
            # Define 1D grid
            x = np.linspace(x_min, x_max, N * s)
            y = np.linspace(y_min, y_max, N * s)

            # Define 2D grid and reshape
            xx, yy = map(lambda x: x.reshape(N, s, N, s), np.meshgrid(x, y))

            # Add to list
            x1.append(xx)
            x2.append(yy)

        # Initial calculation with no sub-pixelisation
        img_ = np.sqrt(self.ray_trace((x1[0], x2[0]), sub=1))
        img = img_.mean(axis=3).mean(axis=1)

        # Find the level of detail (0-sub) necessary for that pixel
        msk = np.clip(np.array(np.ceil(sub * img / np.max(img)), dtype='int') - 1, 0, sub - 1)

        # Loop over the pixels and extract the necessary sub-grid for that pixel
        x1_sub, x2_sub, p_ix, t = [], [], [], 0
        for i in range(N):
            for j in range(N):
                # Pick out sub pixel coordinates for this pixel
                u = x1[msk[i, j]][i, :, j, :].flatten()
                v = x2[msk[i, j]][i, :, j, :].flatten()

                # Add coordinates and parent pixel indices to list
                x1_sub.append(u)
                x2_sub.append(v)

                p_ix.append([x for x in range(t, t + len(u))])
                t += len(u)

        # Collapse above into flat array
        x1_sub = np.array([x for s in x1_sub for x in s])
        x2_sub = np.array([x for s in x2_sub for x in s])

        # Calculate brightness in sub-pixels
        img_sub = self.ray_trace((x1_sub, x2_sub), sub=1)

        # Take the mean of the sub pixels for each pixel
        img = np.array([img_sub[p].mean() for p in p_ix])

        # Transform 1D -> 2D
        return img.reshape(N, N)
