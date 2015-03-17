from __future__ import division
import time
import collections
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import maximum_filter
from scipy.misc import imresize
from scipy.signal import sawtooth, gaussian
from matplotlib import gridspec, lines, cm, pyplot as plt


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def cycle(iterable):
    N,n = len(iterable)-1, 0
    while True:
        idx,n = (N/2)*sawtooth(n/N*np.pi, width=0.5)+(N/2), n+1
        yield iterable[int(round(idx))]

def savefig(figure, filename):
    figure.savefig(filename, bbox_inches='tight', transparent=True)


# ----------------------------------------------------------------------------
# Keypoint Estimation
# ----------------------------------------------------------------------------
def edges(I):
    I = grayscale(I)
    M,N = I.shape
    w = gaussian(M, M/np.pi).reshape(M,1) * gaussian(N, N/np.pi)
    gy, gx = np.gradient(I)
    return w*(gx**2 + gy**2)


def keypoints(I, number=50):
    I = grayscale(I)
    M,N = I.shape
    w = gaussian(M, M/np.pi).reshape(M,1) * gaussian(N, N/np.pi)
    gy, gx = np.gradient(I)
    edges  = w*(gx**2 + gy**2)
    maxima = maximum_filter(edges, min(M,N)/20)
    m, n  = (edges == maxima).nonzero()
    edges = edges[m,n]
    edges,m,n = zip(*sorted(zip(edges,m,n), reverse=True, key=lambda x:x[0]))
    m,n = m[:number], n[:number]
    n,m = zip(*sorted(zip(n,m)))
    return m,n


# ----------------------------------------------------------------------------
# Geometric Changes
# ----------------------------------------------------------------------------
def stretchgrid(I1, I2):
    """Create a mesh the size of the first image that spans the second image

    Args:
        I1, I2: The two images (ndarray) or image shapes (tuple). The output
            meshes x,y are the same shape as I1, and numerically span the
            shape of I2.
    """
    M1,N1 = I1[:2] if isinstance(I1, tuple) else I1.shape[:2]
    M2,N2 = I2[:2] if isinstance(I2, tuple) else I2.shape[:2]
    x,y   = np.meshgrid(np.linspace(0,N2-1,N1), np.linspace(0,M2-1,M1))
    return x,y


def rescale(x, min=0.0, max=1.0):
    """Recale an array into the range [min, max]

    Args:
        min,max: The range of the output array. The output is always returned
            in floating point format.
    """
    range = max - min
    xnorm = (x-x.min()) / (x.max()-x.min())
    return range*xnorm + min


def resizeflow(I1, I2, fx, fy):
    """Resize flow and scale the resultant vectors"""
    old_shape = fx.shape
    new_shape = I1.shape[:2]
    new_scale = I2.shape[:2]
    dy,dx = np.array(new_shape) / np.array(old_shape)
    sy,sx = np.array(new_scale) / np.array(old_shape)
    return zoom(dx*(fx), (dy,dx)), zoom(dy*(fy), (dy,dx))


# ----------------------------------------------------------------------------
# Image and Flow Manipulation
# ----------------------------------------------------------------------------
def grayscale(rgb):
    """Convert an image to RGB"""
    try:
        return np.dot(rgb, [0.299, 0.587, 0.144])
    except (IndexError, ValueError):
        return rgb


def colorflow(fx, fy, rgba=False, zero_mean=True):
    """Create a colored flow field from (fx,fy) flow

    Optical flow fields are represented with color as angle and saturation
    as magnitude. This implementation optionally represents magntiude in
    the alpha channel for better blending with the original image.

    Args:
        fx,fy: The flow fields

    Keyword Args:
        rgba: If True, represent magnitude with an alpha channel rather than
            saturation (default: False)
        zero_mean: Remove the mean of the flow field before computing the
            visualization. This remove the global interpretability of the flow
            field, but makes better utilization of the color space
    """

    if zero_mean:
        fx, fy = fx-fx.mean(), fy-fy.mean()

    # angle is color, magnitude is saturation
    angle, magnitude  = np.arctan2(fy,fx), np.sqrt(fx**2 + fy**2)
    color, saturation = angle/np.pi*0.5+0.5, magnitude / magnitude.max()
    rgb = cm.hsv(color)

    # compute the colored flow field
    if rgba:
        rgb[...,3] = saturation
    else:
        rgb = 1.0 - saturation[...,None]*(1.0 - rgb[...,:3])
    return rgb


def compose(image, overlay, opacity=0.7, alpha=None):
    """Compose an image with an overlay

    Args:
        image: The RGB image
        overlay: The RGB image to overlay. eg. A PDE map or flow field.

    Keyword Args:
        opacity: The mixing opacity of the overlay (default: 0.7)
        alpha: The alpha mask [0.0 1.0] for the overlay (default: None)
    """

    # rescale the inputs so they can be merged without overflow
    M,N = image.shape[:2]
    image = rescale(image).reshape(M,N,-1)
    overlay = rescale(overlay).reshape(M,N,-1)

    # compute the global alpha channel
    alpha = np.ones((M,N)) if alpha is None else alpha

    # separate the RGB and alpha channels
    rgb = np.ones((3,)).reshape(1,1,3)
    rgbi,ai = image[...,:3]*rgb, image[...,3:]
    rgbo,ao = overlay[...,:3]*rgb, overlay[...,3:]

    # compute the mixing
    return (1.0-opacity*alpha[...,None])*rgbi + (opacity*alpha[...,None])*rgbo


def compose_pde(image, pde, opacity=0.7, colormap=plt.get_cmap('jet')):
    """Compose an image with a single channel PDE

    Args:
        image: The RGB image
        pde: The PDE

    Keyword Args:
        opacity: Maximum opacity of the PDE (default: 0.7)
        colormap: The colormap instance used to color the PDE (default: jet)
    """
    pde = rescale(pde)
    alpha = rescale(pde, 0.2)
    pde = colormap(pde)[...,:3]
    return compose(image, pde, opacity, alpha)


def compose_flow(image, fx, fy, opacity=1.0, zero_mean=True):
    """Compose an image with a color field representation of the estimated flow"""

    rgba = colorflow(fx,fy, zero_mean=zero_mean, rgba=True)
    flow, alpha = rgba[...,:3], rgba[...,3]
    return compose(image, flow, opacity, alpha)


def synthesize_image(I1, I2, fx, fy, forwards=False, alpha=None):
    """Synthesize the matched image from the flow vectors

    Args:
        I1,I2: The image pair
        fx,fy: The flow vectors from the first to the second image

    Keywrod Args:
        forwards: If True, (partially) reconstruct the second image from the
           first. If False, (fully) reconstruct the first image from the
           second (default: False).
    """
    M1,N1 = I1.shape[:2]
    M2,N2 = I2.shape[:2]

    # compute the grid locations
    x1,y1 = np.meshgrid(np.arange(N1), np.arange(M1))
    x2,y2 = np.meshgrid(np.linspace(0,N2-1,N1), np.linspace(0,M2-1,M1))
    x2,y2 = np.round(x2+fx).astype(int), np.round(y2+fy).astype(int)

    # mask the feasible points
    mask = (x2 >= 0) * (x2 < N2) * (y2 >= 0) * (y2 < M2)

    if forwards:
        # (partially) reconstruct the second image
        Ir = np.zeros((M2,N2,4), dtype=I1.dtype)
        Ir[y2[mask], x2[mask], :3] = I1[y1[mask], x1[mask], :3]

        # compute the alpha mask
        scale = 255 if I2.dtype == np.uint8 else 1
        alpha = np.ones((M1,N1), dtype=I1.dtype) if alpha is None else alpha
        Ir[y2[mask], x2[mask], 3] = alpha[mask]*scale
    else:
        # (fully) reconstruct the first image
        Ir = np.zeros((M1,N1,4), dtype=I2.dtype)
        Ir[y1[mask], x1[mask], :3] = I2[y2[mask], x2[mask], ...]

        # compute the alpha mask
        scale = 255 if I2.dtype == np.uint8 else 1
        alpha = np.ones((M1,N1), dtype=I2.dtype) if alpha is None else alpha
        Ir[y1[mask], x1[mask], 3] = alpha[mask]*scale
    return Ir


# ----------------------------------------------------------------------------
# Basic Visualization
# ----------------------------------------------------------------------------
def display(image):
    """Display an image"""
    image = rescale(image)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_axis_off()
    fig.show()
    return fig


# ----------------------------------------------------------------------------
# Base Corresponder
# ----------------------------------------------------------------------------
class CorresponderBase(object):

    def __init__(self, I1, I2, subplots=(1,2), interactive=True):
        """Base class for visualizing correspondences

        Args:
            I1: The first image
            I2: The second image
        """

        # initialize the figure
        self.fig  = plt.figure(figsize=(16,6))
        self.axes = []
        gs = gridspec.GridSpec(*subplots)

        # initialize the axes
        for n,subplot in enumerate(gs):
            axis = self.fig.add_subplot(subplot)
            axis.imshow(I2 if n else I1)
            axis.set_autoscalex_on(False)
            axis.set_autoscaley_on(False)
            plt.axis('off')
            self.axes.append(axis)
            setattr(self, 'ax{n}'.format(n=n+1), axis)
        plt.subplots_adjust(wspace=0.05)

        # create the colormap
        self.cmap = cycle([cm.jet(i) for i in reversed(range(0,cm.jet.N,16))])
        self.color = self.cmap.next()

        # bind the events
        if interactive:
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey)
            self.fig.show()

    def onkey(self, event):
        """Base method for handling deletion of lines"""
        if event.key in ('escape',):
            self.clear()
        if event.key in ('delete', 'backspace'):
            self.clearlast()

    def clear(self):
        """Clear all points and lines from the figure"""
        del self.fig.lines[:]
        del self.ax1.lines[:]
        del self.ax2.lines[:]
        plt.draw()

    def clearlast(self):
        """Clear the last plotted point"""
        lines   = (self.fig.lines, self.ax1.lines, self.ax2.lines)
        lengths = map(len, lines)
        max_length = max(max(lengths), 1)
        for line,length in zip(lines,lengths):
            if length == max_length:
                del line[-1]
        plt.draw()

    def onclick(self, event):
        """Route the click into the left or right image"""
        try:
            {
                self.ax1: self.onleftclick,
                self.ax2: self.onrightclick,
            }[event.inaxes](event)
        except KeyError:
            pass

    def onleftclick(self, event):
        """Click event occurring in the left axes"""
        pass

    def onrightclick(self, event):
        """Click event occurring in the left axes"""
        pass

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
        if xy_left:
            self.ax1.plot(xy_left[0],xy_left[1],'o',color=self.color)
        if xy_right:
            self.ax2.plot(xy_right[0],xy_right[1],'o',color=self.color)
        if xy_left and xy_right:
            self.color = self.cmap.next()
        plt.draw()

    def savefig(self, filename):
        savefig(self.fig, filename)


# ----------------------------------------------------------------------------
# Manual Corresponder
# ----------------------------------------------------------------------------
class Labeller(CorresponderBase):
    def __init__(self, I1, I2, correspondences=None):
        """A visualizer for labelling image pairs

        The labelled correspondences are stored in the correspondences attribute
        as a list of [ ((x1,y1), (x2,y2)), ... ]

        Args:
            I1: The first image
            I2: The second image

        Keyword Args:
            correspondences: The initial set of correspondences from a previous
                labelling session. This allows one to either view the labels,
                or continue labelling
        """
        super(Labeller, self).__init__(I1, I2)
        self.correspondences = []
        if correspondences is not None:
            for left,right in correspondences:
                left,right = tuple(left), tuple(right)
                self.correspondences.append((left,right))
                self.plot(left,right)

    def clear(self):
        super(Labeller, self).clear()
        self.correspondences = []

    def clearlast(self):
        super(Labeller, self).clearlast()

    def onclick(self, event):
        xy = event.xdata, event.ydata
        new_left, new_right = (xy,None) if event.inaxes == self.ax1 else (None,xy)
        if not self.correspondences or all(self.correspondences[-1]):
            left,right = new_left,new_right
            self.correspondences.append((new_left, new_right))
        else:
            left,right = self.correspondences[-1]
            left  = new_left if new_left else left
            right = new_right if new_right else right
            self.correspondences[-1] = (left,right)
            self.clearlast()
        self.plot(left, right)


class PNLabeller(CorresponderBase):
    def __init__(self, I1, I2, positive=None, negative=None):
        """A visualizer for labelling positive and negative regions

        Args:
            I1, I2: The images to label

        Keyword Args:
            positive: The initial set of positive correspondences
            negative: The initial set of background examples
        """
        super(PNLabeller, self).__init__(I1, I2)
        self.positive = []
        self.negative = []
        self.mode = 'P'

        # initialize the colormaps
        self.pmap = cycle([cm.cool(i) for i in range(0, cm.cool.N, 16)])
        self.nmap = cycle([cm.autumn(i) for i in range(0, cm.autumn.N, 16)])
        self.cmap = self.pmap
        self.color = self.pmap.next()

        # redraw the initial set
        if positive is not None:
            for left,right in positive:
                left,right = tuple(left), tuple(right)
                self.positive.append((left,right))
                self.plot(left,right)

        if negative is not None:
            for left, self.color in zip(negative, self.nmap):
                left = tuple(left)
                self.negative.append(left)
                self.plot(left)

    def onkey(self, event):
        """Switch modes from labelling positive to negative examples"""
        super(PNLabeller, self).onkey(event)
        if event.key in ('p', 'P'):
            self.mode = 'P'
            self.color = self.pmap.next()
        if event.key in ('n', 'N'):
            self.mode = 'N'

    def onclick(self, event):
        xy = event.xdata, event.ydata
        new_left, new_right = (xy,None) if event.inaxes == self.ax1 else (None,xy)
        if self.mode == 'P':
            if not self.positive or all(self.positive[-1]):
                left,right = new_left,new_right
                self.positive.append((left,right))
            else:
                left,right = self.positive[-1]
                left  = new_left if new_left else left
                right = new_right if new_right else right
                self.positive[-1] = (left,right)
                self.clearlast()
        if self.mode == 'N':
            self.color = self.nmap.next()
            left,right = new_left,None
            if left:
                self.negative.append(left)
        self.plot(left,right)


# ----------------------------------------------------------------------------
# Dense Visualizer
# ----------------------------------------------------------------------------
class CorrespondenceVisualizer(CorresponderBase):
    def __init__(self, I1, I2, fx, fy, **kwargs):
        """A visualizer for displaying densely computed correspondences

        Correspondences should be computed from the first image to the second.
        Thus correspondences will only be displayed when clicking on the first
        image

        Args:
            I1: The first image
            I2: The second image
            fx,fy: The dense flow vectors from the first image to the second,
                where I1.shape[:2] == fx.shape == fy.shape
        """
        super(CorrespondenceVisualizer, self).__init__(I1, I2, **kwargs)
        self.ux,self.uy = stretchgrid(I1,I2)
        self.fx,self.fy = fx,fy

    def onleftclick(self, event):
        x1,y1 = event.xdata, event.ydata
        i,j   = int(round(y1)), int(round(x1))
        x2,y2 = self.ux[i,j]+self.fx[i,j], self.uy[i,j]+self.fy[i,j]
        if np.isfinite(x2) and np.isfinite(y2):
            self.plot((x1,y1), (x2,y2))


# ----------------------------------------------------------------------------
# Raw Probability Density Visualizer
# ----------------------------------------------------------------------------
def manhattan(I1,I2,x,y):
    """L1-distance in feature space"""
    return -np.linalg.norm(I2-I1[y,x,...], ord=1, axis=2)

def euclidean(I1,I2,x,y):
    """L2-distance in feature space"""
    return -np.linalg.norm(I2-I1[y,x,...], ord=2, axis=2)

class PDEVisualizer(CorresponderBase):
    def __init__(self, I1, I2, F1=None, F2=None, transform=None,
            loss_function=manhattan, loss_functions=None):
        """A visualizer for displaying the full PDE for a point

        Args:
            I1,I2: The images
            loss_function: The function to compute the PDE. Takes three
                arguments, loss_function(I1, I2, (x,y)) where (x,y)
                is the point in the first image where the kernel should be
                extracted

        Keyword Args:
            scale: Scale the PDEs to cover the full color range, but loses
                absolute interpretability (default: True)
            feature_transform: The transform from image pixels to
                image features, which are actually passed to the loss_function
                    OR
            F1,F2: The feature image representation of the pixel images
        """
        self.loss = loss_functions if loss_functions else (loss_function,)
        super(PDEVisualizer, self).__init__(I1, I2, subplots=(1,1+len(self.loss)))
        self.I1 = I1
        self.I2 = I2
        self.F1 = transform(I1) if F1 is None else F1
        self.F2 = transform(I2) if F2 is None else F2
        self.ds = I1.shape[0]/self.F1.shape[0], I1.shape[1]/self.F1.shape[1]

    def onleftclick(self, event):
        # get the click location
        x,y = event.xdata, event.ydata
        xd,yd = int(round(x/self.ds[1])), int(round(y/self.ds[0]))
        self.x, self.y = xd,yd

        # compute and display each of the PDEs
        for loss,axis in zip(self.loss, self.axes[1:]):
            pde = loss(self.F1, self.F2, xd, yd)
            pde = imresize(pde, self.I2.shape[:2])
            im_pde = compose_pde(self.I2, pde)
            self.clearlast()
            axis.imshow(im_pde)
        self.plot((x,y))


# ----------------------------------------------------------------------------
# Graph Visualization
# ----------------------------------------------------------------------------
class GraphVisualizer(object):
    def __init__(self, G, pos=None, draw_images=False, threshold=1e-4):

        # import networkx
        import networkx as nx
        self.nx = nx
        self.G  = G
        self.threshold = threshold

        # create the network graph
        self.fig, self.ax = plt.subplots(figsize=(14,12))

        # layout the nodes and plot them in order of decreasing distance
        if not pos: pos = nx.spring_layout(G)
        self.pos = pos
        maximum_edge_weight = self.maximum_edge_weight()
        nodelist = sorted(G.nodes(), key=lambda k: maximum_edge_weight[k])

        # visualize the graph
        if draw_images:
            self.draw_image_nodes(G, pos, nodelist)
        else:
            self.draw_point_nodes(G, pos, nodelist).set_picker(True)
        edges = nx.draw_networkx_edges(G, pos, nodelist=nodelist, ax=self.ax, color='#CDCDCD', alpha=0.2)
        self.ax.set_axis_off()
        #self.fig.show()

        # fix the limits
        self.ax.set_autoscaley_on(False)
        self.ax.set_autoscalex_on(False)

        # try to register an interactive rescaler
        if draw_images:
            try:
                self.current = None
                self.fig.canvas.draw()
                self.raster = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self.fig.canvas.mpl_connect('pick_event', self.onpick)
            except AttributeError:
                pass

    def maximum_edge_weight(self):
        G = self.G
        return { n:max(d['weight'] for u,v,d in G.edges(n, data=True)) for n in G.nodes() }

    def draw_image_nodes(self, G, pos, nodelist):
        N = len(nodelist)
        maxshape = 1.0/np.sqrt(N)
        for n in nodelist:
            image  = G.node[n]['image']
            shape  = image.shape[:2]
            shape  = np.array(shape) / max(shape) * maxshape
            dx,dy  = shape[1]/2.0, shape[0]/2.0
            posn   = pos[n]
            extent = (posn[0]-dx,posn[0]+dx,posn[1]-dy,posn[1]+dy)
            self.ax.imshow(image, extent=extent, zorder=100, picker=True)

    def draw_point_nodes(self, G, pos, nodelist):
        maximum_edge_weight = self.maximum_edge_weight()
        node_color = [maximum_edge_weight[k] for k in nodelist]
        return self.nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=nodelist, node_color=node_color)

    def onpick(self, event):

        # restore the current image
        if self.current:
            image,extent = self.current
            image.set_extent(extent)
            self.current = None
        else:
            image = None

        # get the image that was clicked on
        if event.artist is not image:

            # scale the image up
            image = event.artist
            old_extent = image.get_extent()
            left,right,top,bottom = old_extent
            x,y = (left+right)/2, (top+bottom)/2
            new_extent = (x-0.1, x+0.1, y-0.1, y+0.1)
            image.set_extent(new_extent)
            self.current = (image, old_extent)

        # redraw the image
        self.fig.canvas.restore_region(self.raster)
        self.ax.draw_artist(image)
        self.fig.canvas.blit(self.ax.bbox)
        time.sleep(0.5)
