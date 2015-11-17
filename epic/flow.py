from __future__ import division
import os
import numpy as np
from scipy.ndimage import imread, zoom
from scipy.ndimage.interpolation import map_coordinates


# ----------------------------------------------------------------------------
# Flow <--> Correspondence
# ----------------------------------------------------------------------------
def stretchgrid(I1, I2, integer_coordinates=False):
    """Create a mesh the size of the first image that spans the second image

    Args:
        I1, I2: The two images (ndarray) or image shapes (tuple). The output
            meshes x,y are the same shape as I1, and numerically span the
            shape of I2.

    Keyword Args:
        integer_coordinates: Return the mesh rounded to integer
            pixel coordinates (default: False)
    """
    M1,N1 = I1[:2] if isinstance(I1, tuple) else I1.shape[:2]
    M2,N2 = I2[:2] if isinstance(I2, tuple) else I2.shape[:2]
    x,y   = np.meshgrid(np.linspace(0,N2-1,N1), np.linspace(0,M2-1,M1))
    if integer_coordinates:
        return np.round(x).astype(int), np.round(y).astype(int)
    else:
        return x,y

def resize_flow(I1, I2, fx, fy):
    """Resize flow and scale the resultant vectors"""
    old_shape = fx.shape
    new_shape = I1.shape[:2]
    new_scale = I2.shape[:2]
    dy,dx = np.array(new_shape) / np.array(old_shape)
    sy,sx = np.array(new_scale) / np.array(old_shape)
    return zoom(dx*(fx), (dy,dx)), zoom(dy*(fy), (dy,dx))

def flow_to_correspondence(I1,I2,fx,fy):
    """Convert flow fields to correspondence fields"""
    x0,y0 = stretchgrid(I1,I2,True)
    dx,dy = x0+fx,y0+fy
    return dx,dy

def correspondence_to_flow(I1,I2,dx,dy):
    """Convert flow fields to dense correspondence fields"""
    x0,y0 = stretchgrid(I1,I2,True)
    fx,fy = dx-x0,dy-y0
    return fx,fy

def center_unary(unary):
    """Center the unary volume so the center pixel in each slice corresponds to zero flow

    Args:
        unary: The MxNxPxQ unary field
    """
    M,N,P,Q = unary.shape
    Qs,Ps = stretchgrid((M,N), (P,Q), True)
    pc,qc = np.floor(P/2), np.floor(Q/2)
    flow  = np.zeros_like(unary)

    # iterate over each slice (each pixel in the reference/source image)
    for m in range(M):
        for n in range(N):
            # get the reference pixel
            p = Ps[m,n]
            q = Qs[m,n]
            # compute the valid region around the pixel
            p0,p1 = min(p,pc), min(P-p,pc)
            q0,q1 = min(q,qc), min(Q-q,qc)

            # shift the region
            flow[m,n,pc-p0:pc+p1,qc-q0:qc+q1] = unary[m,n,p-p0:p+p1,q-q0:q+q1]

    return flow


# ----------------------------------------------------------------------------
# Propagate Flow
# ----------------------------------------------------------------------------
def propagate_flow(flows, occlusions=None, correspondence=False):
    """Propagate the flow through a series of images

    Given a sequence of flow fields and (optionally) occlusion masks,
    propagate the pixels from the first image to the last.

    Args:
        flows: The sequence of flow fields [(fx,fy), (fx,fy), ...]

    Keyword Args:
        occlusions: The sequence of occlusion masks (default: None)
        correspondence: Return the propagated field as correspondence
            values rather than flow fields (default: False)
    """
    dx,dy = None,None
    occlusions = [[] for n in range(len(flows))] if occlusions is None else occlusions
    for (fx,fy), occlusion in zip(flows,occlusions):

        # define the initial displacement
        if dx is None or dy is None:
            dx,dy = x0,y0 = np.meshgrid(np.arange(fx.shape[1]), np.arange(fx.shape[0]))

        # invalidate the occluded pixels
        fx[occlusion] = np.nan
        fy[occlusion] = np.nan

        # interpolate the flow at the displacment locations
        ix = map_coordinates(fx, (dy,dx), order=1, cval=np.nan)
        iy = map_coordinates(fy, (dy,dx), order=1, cval=np.nan)

        # update the displacement
        dx,dy = dx+ix,dy+iy

    # compute the invalid regions
    invalid = np.isnan(dx) & np.isnan(dy)
    dx,dy = (dx,dy) if correspondence else (dx-x0,dy-y0)

    # return the flow and mask
    return dx,dy,invalid


class SintelPropagator(object):
    def __init__(self, base, images=None, flow=None, occlusions=None):
        """Propagate flow fields across the Sintel dataset

        Args:
            base: The path to the base Sintel directory

        Keyword Args:
            image,flow,occlusion: Custom paths to the image, flow and occlusion
                directories if they are not under the root as 'images/',
                'flow/' and 'occlusions/'
        """
        self.base_dir = base
        self.image_dir = images if images else os.path.join(base, 'images')
        self.flow_dir = flow if flow else os.path.join(base, 'flow')
        self.occ_dir  = occlusions if occlusions else os.path.join(base, 'occlusions')

    def propagate_flow(scene, start, end, correspondence=False):
        """Propagate flow fields between arbitrary images in a scene

        Args:
            scene: The name of the scene (e.g. 'alley_1')
            start: The starting frame number as an integer
            end:   The ending frame number as an integer

        Returns:
            first,last: The last and last images
            fx,fy: The computed flow field from the start to end frame
            invalid: The invalid pixels
        """

        from epic import io
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

        # call the global function
        fx,fy,invalid = propagate_flow(flows, occlusions, correspondence=correspondence)
        return first,last, fx,fy, invalid
