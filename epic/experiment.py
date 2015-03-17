import json
import itertools
import gridengine
import numpy as np
from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')
from epic import solver, vis

I1 = imread('/home/n8016518/code/epic/data/safari1.jpg')
I2 = imread('/home/n8016518/code/epic/data/safari2.jpg')

class Event:
    def __init__(self, x, y, axes):
        self.xdata = x
        self.ydata = y
        self.inaxes = axes

def lions(kwargs):
    corr = solver.EPICSolver(**kwargs)
    kwargs['image_size'] = kwargs['image_size'][0]
    name = '_'.join(map(str, kwargs.values())).replace(' ','')
    corr.precompute()
    fx,fy,obj,confidence = corr.solve(I1,I2)
    viscorr = vis.CorrespondenceVisualizer(I1,I2,fx,fy,interactive=False)
    left = vis.keypoints(I1)
    for y,x in zip(*left):
        viscorr.onclick(Event(x,y,viscorr.ax1))
    viscorr.savefig('/home/n8016518/code/epic/results/trials/correspondences_tvl1_'+name+'.png')


def sweep_parameters():
    meanshift_window = [25, 15, 25, 31, 45]
    meanshift_sigma_mul = [1.0, 0.98, 0.95, 0.9, 0.85]
    prox_rho = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    prox_max_iters = [20, 30, 50, 100]
    image_size = [(125,125)]#[(100,100), (125,125), (150,150)]
    values = itertools.product(meanshift_window,meanshift_sigma,prox_rho,prox_max_iters,image_size)
    keys = ['meanshift_window','meanshift_sigma','prox_rho','prox_max_iters','image_size']
    args = [dict(zip(keys,value)) for value in values]
    print args

    y = gridengine.map(lions, args)
    return y
