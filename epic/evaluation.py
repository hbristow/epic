from __future__ import division
import os
import glob
import random
import itertools
import numpy as np
from matplotlib import colors, cm, pyplot as plt
from scipy.misc import imread, imsave

from epic import color, dataset, flow, io, optimize, vis

EPIC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(EPIC_DIR, 'groundtruth')
RESULT_DIR = os.path.join(EPIC_DIR, 'results')

# ----------------------------------------------------------------------------
# Compose subject variance over groundtruth
# ----------------------------------------------------------------------------
def visualize_gmm(colormap=color.rose, opacity=0.1):
    sources = sorted(glob.glob(os.path.join(DATA_DIR, '*_source.jpg')))
    targets = sorted(glob.glob(os.path.join(DATA_DIR, '*_target.jpg')))
    for n, (source, target) in enumerate(zip(sources,targets)):
        # read the groundtruth data
        source = imread(source).astype(float) / (255.0)
        target = imread(target).astype(float) / (255.0)
        M,N,K  = target.shape
        subjects = glob.glob(os.path.join(DATA_DIR, '{:03d}_correspondences*.npy'.format(n+1)))

        # compute the subject variance
        left,right,mean,cov = dataset.groundtruth_variance(subjects)
        gmm = dataset.gmm(mean,cov,target.shape)

        # visualize the true correspondences
        colors = colormap(np.linspace(0.5,1,len(right)))
        tinted = colormap(np.zeros((M,N)))
        fig,ax = plt.subplots()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(tinted)
        ax.set_autoscale_on(False)
        for m, (x,y) in enumerate(right.transpose(0,2,1)):
            ax.plot(x,y,'.',color=colors[m])
        fig.set_size_inches(N/80.0, M/80.0)
        fig.savefig(os.path.join(RESULT_DIR, '{:03}_subject_variance_points.pdf'.format(n+1)),
            dpi=80, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)

        # visualize the true correspondences
        colors = colormap(np.linspace(0.5,1,len(right)))
        tinted = colormap(np.zeros((M,N)))
        fig,axes = plt.subplots(1,3)
        for im,ax in zip([source,target,tinted],axes):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(im)
            ax.set_autoscale_on(False)
        for m, (x,y) in enumerate(right.transpose(0,2,1)):
            ax.plot(x,y,'.',color=colors[m])
        fig.subplots_adjust(hspace=0.001)
        fig.set_size_inches(3*N/80.0, M/80.0)
        fig.savefig(os.path.join(RESULT_DIR, '{:03}_composite.pdf'.format(n+1)),
            dpi=80, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)

        # write the composites to file
        colored = colormap(vis.rescale(gmm))
        composite = vis.compose(colored, target, opacity=opacity, rescale_inputs=False)
        imsave(os.path.join(RESULT_DIR, '{:03d}_subject_variance_distribution.png'.format(n+1)), colored)
        imsave(os.path.join(RESULT_DIR, '{:03d}_subject_variance_composite.png'.format(n+1)), composite)


# ----------------------------------------------------------------------------
# Estimate the CDF of a method
# ----------------------------------------------------------------------------
def visualize_cdf(colormap=color.rose, order=None, **methods):
    """Compare the CDF of different methods on (some) groundtruth data

    Keyword Args:
        methods: A list of correspondence files
    """
    fig,ax = plt.subplots()
    groundtruth, names, x, y, areas = {}, [], [], [], []
    for name, paths in methods.iteritems():
        points, means, covs = [], [], []
        for path in paths:
            index = int(os.path.basename(path).split('_')[0])

            # compute the groundtruth distributions
            subjects = glob.glob(os.path.join(DATA_DIR, '{:03d}_correspondences*.npy'.format(index)))
            left, right, mean, cov = dataset.groundtruth_variance(subjects)

            # load the computed points
            reference, point = np.load(path).transpose(1,0,2)
            points.extend(point), means.extend(mean), covs.extend(cov)

        # compute the CDF for the method
        xn,yn = dataset.cdf(points, means, covs)
        x.append(xn)
        y.append(yn)
        names.append(name)
        areas.append(np.trapz(yn,xn)/xn[-1])

    # sort the methods by area under curve
    comparator = lambda x:order.index(x[3]) if order else x[2]
    x,y,areas,names = zip(*sorted(zip(x,y,areas,names), key=comparator, reverse=True))

    # plot
    colors = colormap(np.linspace(0,0.8,len(methods)))
    for n, (xn,yn) in enumerate(zip(x,y)):
        ax.plot(xn,yn,color=colors[n])

    # add axis properties
    ax.set_xlabel('Mahalanobis Distance ($\sigma$)')
    ax.set_ylabel('Cumulative Density')
    ax.legend(names)
    return fig


# ----------------------------------------------------------------------------
# Apply a solver to the dataset
# ----------------------------------------------------------------------------
def apply(solver, sweep=True, **parameters):
    solver = solver()
    domains, ranges = [],[]

    sources = sorted(glob.glob(os.path.join(DATA_DIR, '*_source.jpg')))
    targets = sorted(glob.glob(os.path.join(DATA_DIR, '*_target.jpg')))
    for n, (source, target) in enumerate(zip(sources, targets)):

        # get the data sources
        source = imread(source)
        target = imread(target)
        subjects = glob.glob(os.path.join(DATA_DIR, '{:03d}_correspondences*.npy'.format(n+1)))

        # compute the result
        x0,y0 = flow.stretchgrid(source, target)
        fx,fy = solver.solve(source, target, **parameters)
        dx,dy = x0+fx,y0+fy

        # sample at the correspondences
        left,right,means,covs = dataset.groundtruth_variance(subjects)
        x,y = np.round(left).astype(int).transpose()
        points = np.column_stack((dx[y,x], dy[y,x]))

        # write the estimates
        if not sweep:
            filename = '{:03d}_{solver}.npy'.format(n+1,solver=solver.__class__.__name__)
            correspondences = np.column_stack((left,points)).reshape(-1,2,2)
            np.save(os.path.join(RESULT_DIR, filename), correspondences)

        # compute the CDF
        domain,range = dataset.cdf(points, means, covs)
        domains.append(domain)
        ranges.append(range)

    # robust mean
    median = np.median([d for domain in domains for d in domain])

    if sweep:
        filename = '_'.join('{}_{:n}'.format(*items) for items in parameters.iteritems())
        filename = '{}_{:f}_{}.npz'.format(solver.__class__.__name__, median, filename)
        np.savez(os.path.join(RESULT_DIR, filename),
            domains=domains, ranges=ranges, median=median)

    # return the statistics
    return median, domains, ranges


# ----------------------------------------------------------------------------
# Parameter Sweepers
# ----------------------------------------------------------------------------
class DenseCRF(object):
    @staticmethod
    def apply(parameters):
        return apply(optimize.DenseCRF, **parameters)[0]

    @staticmethod
    def sweep_parameters():
        import gridengine

        # the arguments to sweep
        sigma_space = [1.0, 3.0, 5.0, 8.0, 12.0]
        sigma_color = [5.0, 10.0, 20.0, 30.0, 50.0]
        sigma_label = [1.0, 3.0, 5.0, 8.0, 12.0]
        bilateral_weight = [1e3, 2e3, 5e3, 1e4, 5e4, 1e5]
        smooth_weight = [1e4]#, 1e5, 1e6, 1e7, 1e8]

        # iterate all combinations of the args
        names = ['sigma_space', 'sigma_color', 'sigma_label', 'bilateral_weight', 'smooth_weight']
        args  = [dict(zip(names,args)) for args in itertools.product(
                    sigma_space, sigma_color, sigma_label,
                    bilateral_weight, smooth_weight)]

        # gridengine
        random.shuffle(args)
        scheduler = gridengine.GridEngineScheduler(mem='8G', walltime='3:00:00')
        distances = gridengine.map(DenseCRF.apply, args, scheduler=scheduler)
        return zip(args,distances)


class LDAFlow(object):
    @staticmethod
    def apply(parameters):
        return apply(optimize.LDAFlow, **parameters)[0]

    @staticmethod
    def sweep_parameters():
        import gridengine

        # the arguments to sweep
        alpha  = [300,500,700]
        d      = [10000, 100000]
        gamma  = [0, 0.01, 0.1, 1.125]
        levels = [2]
        domain = [3,5,7]
        coarse_domain = [15]

        # iterate all combinations of the args
        names = ['alpha', 'd', 'gamma', 'levels', 'domain', 'coarse_domain']
        args  = [dict(zip(names,args)) for args in itertools.product(
                    alpha, d, gamma, levels, domain, coarse_domain)]

        # gridengine
        random.shuffle(args)
        scheduler = gridengine.GridEngineScheduler(mem='8G', walltime='3:00:00')
        distances = gridengine.map(SIFTFlow.apply, args, scheduler=scheduler)
        return zip(args,distances)


class SIFTFlow(object):
    @staticmethod
    def apply(parameters):
        return apply(optimize.SIFTFlow, **parameters)[0]

    @staticmethod
    def sweep_parameters():
        import gridengine

        # the arguments to sweep
        alpha  = [300,500,700]
        d      = [10000, 100000]
        gamma  = [0, 0.01, 0.1, 1.125]
        levels = [2]
        domain = [3,5,7]
        coarse_domain = [15]

        # iterate all combinations of the args
        names = ['alpha', 'd', 'gamma', 'levels', 'domain', 'coarse_domain']
        args  = [dict(zip(names,args)) for args in itertools.product(
                    alpha, d, gamma, levels, domain, coarse_domain)]

        # gridengine
        random.shuffle(args)
        scheduler = gridengine.GridEngineScheduler(mem='8G', walltime='3:00:00')
        distances = gridengine.map(SIFTFlow.apply, args, scheduler=scheduler)
        return zip(args,distances)
