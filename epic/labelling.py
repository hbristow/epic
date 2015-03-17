import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import imread

from epic import io


# ----------------------------------------------------------------------------
# Propagate Flow
# ----------------------------------------------------------------------------
class Tracer(object):
    def __init__(self, base, image=None, flow=None, occlusion=None):
        self.base_dir = base
        self.image_dir = image if image else os.path.join(base, 'images')
        self.flow_dir = flow if flow else os.path.join(base, 'flow')
        self.occ_dir  = occlusion if occlusion else os.path.join(base, 'occlusions')

    def trace(self, scene, start, end, flow=False):
        # get the component directories
        scene_images = os.path.join(self.image_dir, scene)
        scene_flow = os.path.join(self.flow_dir, scene)
        scene_occ = os.path.join(self.occ_dir, scene)

        # load the flow fields and occlusion mask
        frame = 'frame_{:04d}.{ext}'
        first = imread(os.path.join(scene_images, frame.format(start, ext='png')))
        last  = imread(os.path.join(scene_images, frame.format(end, ext='png')))
        flows, occlusions = [], []
        for n in range(start, end):
            with io.openflow(os.path.join(scene_flow, frame.format(n, ext='flo'))) as f:
                flows.append(f.copy())
            with open(os.path.join(scene_occ, frame.format(n, ext='png'))) as f:
                occlusions.append(imread(f).astype(bool))

        # initialize the displacement as spanning the image
        (M,N) = occlusions[0].shape
        x0,y0 = np.meshgrid(np.arange(N), np.arange(M))
        dx,dy = x0.copy(), y0.copy()

        # precompute the reference points
        x0y0 = np.dstack((x0,y0)).reshape(M*N,2)

        for (fx,fy), occlusion in zip(flows, occlusions):
            # invalidate occluded pixels
            fx[occlusion] = np.nan
            fy[occlusion] = np.nan

            # interpolate the flow at the displacement locations
            ix = map_coordinates(fx+occlusion, (dy,dx), order=1, cval=np.nan)
            iy = map_coordinates(fy+occlusion, (dy,dx), order=1, cval=np.nan)

            # update the displacement
            dx,dy = dx+ix,dy+iy

        # compute the invalid regions
        invalid = np.isnan(dx) & np.isnan(dy)
        dx,dy = (dx-x0,dy-y0) if flow else (dx,dy)

        # return the flow, mask and endpoint images
        return dx,dy,invalid,first,last
