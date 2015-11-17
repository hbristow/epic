import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import imread
from matplotlib import lines, pyplot as plt

from epic import io, vis


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


# ----------------------------------------------------------------------------
# Mechanical Turk Labelling
# ----------------------------------------------------------------------------
class Turker(vis.CorresponderBase):

    class Event(object):
        def __init__(self, xdata, ydata, inaxes):
            self.xdata = xdata
            self.ydata = ydata
            self.inaxes = inaxes

    def __init__(self, I1, I2, fig=None):
        super(Turker, self).__init__(I1, I2, fig=fig)
        self.fig.canvas.mpl_connect('close_event', self.onclose)
        self.active = True
        self.clear()

    def clear(self):
        super(Turker, self).clear()
        self.n = 0
        self.left = None
        self.correspondences = []

    def clearlast(self):
        lines = (self.fig.lines, self.ax1.lines, self.ax2.lines)
        self.n = n = max(len(self.fig.lines)-1, 0)
        for line in lines:
            line[n:] = []
        self.correspondences[n:] = []
        plt.draw()

    def onclose(self, event):
        self.active = False

    def onclick(self, event, auto=False):
        left  = self.left = (event.xdata, event.ydata) if auto else self.left
        right = (event.xdata, event.ydata) if not auto else None
        if left and right:
            self.correspondences.append((left,right))
        self.plot(left, right)

    def plot(self, xy_left=None, xy_right=None):
        """Plot the correspondence given a point in the left and right frame"""

        # compute the line in the global reference frame
        if xy_left and xy_right:
            inv = self.fig.transFigure.inverted()
            gx1,gy1 = inv.transform(self.ax1.transData.transform(xy_left))
            gx2,gy2 = inv.transform(self.ax2.transData.transform(xy_right))
            line = lines.Line2D((gx1,gx2), (gy1,gy2),
                transform=self.fig.transFigure,
                color=(0.85,0.85,0.85,0.3))
            self.fig.lines.append(line)

        # plot the input and output points
        if xy_left and not xy_right:
            self.ax1.plot(xy_left[0],xy_left[1],'ro', markersize=6)
        elif xy_left and xy_right:
            self.ax1.lines.pop()
            self.ax1.plot(xy_left[0],xy_left[1],'o',color=(0.3,0,0), markersize=5)
            self.ax2.plot(xy_right[0],xy_right[1],'o',color=(0.3,0,0), markersize=5)
        plt.draw()

    def random_access(self, iterable):
        while self.n < len(iterable) and self.active:
            n, self.n = self.n, self.n+1
            yield iterable[n]

    @classmethod
    def label(cls, name, sources, targets, correspondences):
        N = len(sources)
        for n, source, target, corr in zip(range(N), sources, targets, correspondences):
            print('  Pair {n} of {N}'.format(n=n+1,N=N))

            # load the images and ground-truth
            I1 = imread(source)
            I2 = imread(target)
            base,ext = os.path.splitext(corr)
            corr = np.load(corr)

            # reset/initialize the figure
            try:
                labeller = cls(I1,I2,fig=labeller.fig)
            except UnboundLocalError:
                labeller = cls(I1,I2)

            # draw a point and ask the user to complete the correspondence
            for (x1,y1),(x2g,y2g) in labeller.random_access(corr):
                # auto
                labeller.onclick(Turker.Event(x1,y1,labeller.ax1), auto=True)
                # manual
                plt.waitforbuttonpress()

            # save the correspondences
            if len(labeller.correspondences) == len(corr):
                np.save('{base}_{name}{ext}'.format(base=base,name=name,ext=ext),
                    labeller.correspondences)

            # check for early exit condition
            if not labeller.active:
                return


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import glob
    import argparse
    # load the resources
    sources = sorted(glob.glob('groundtruth/*_source.jpg'))
    targets = sorted(glob.glob('groundtruth/*_target.jpg'))
    correspondences = sorted(glob.glob('groundtruth/*_correspondences.npy'))
    print """
---------------------------------------------------------------------
            Semantic Correspondence Labeller
---------------------------------------------------------------------

Task:       The computer will display a series of image pairs.
            For each image pair it will highlight (in red) a semantic
            keypoint in the first image. Your task is to click on
            the same semantic point in the second image.

            There are no right or wrong answers. Some correspondences
            may be ambiguous. In such a case, label what seems most
            reasonble to you.

Controls:   [mouse]     To click on the corresponding point
            [backspace] To undo the last click that you performed
            [escape]    To reset an entire image pair
            [close fig] If you're sick of being a guinea pig ;)

Usage:      python -m epic.labelling
            python -m epic.labelling --resume 5  # resume at pair 5

---------------------------------------------------------------------
"""
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Semantic Correspondence Labeller')
    parser.add_argument('-r', '--resume', type=int, default=0, help='Resume at a particular pair')
    args = parser.parse_args()

    # truncate the sources on resume
    resume  = max(args.resume-1,0)
    sources = sources[resume:]
    targets = targets[resume:]
    correspondences = correspondences[resume:]

    # begin the interaction
    name = raw_input('Please enter your name: ').lower().replace(' ','_')
    print 'Starting session...'
    Turker.label(name, sources, targets, correspondences)
    print 'Session complete'
    print 'Thanks for your help :)'
