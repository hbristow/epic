import numpy as np
from scipy.misc import imread, imresize
from scipy.ndimage.filters import gaussian_filter

from epic import color, crf, flow, io, solver, vis

# ----------------------------------------------------------------------------
# Optimization Interface
# ----------------------------------------------------------------------------
class Baseline(object):
    """A baseline inference method which simply returns zero flow"""

    def solve(self, I1, I2):
        M,N = I1.shape[:2]
        return np.zeros((M,N)), np.zeros((M,N))

class Argmax(object):
    def __init__(self):
        self.solver = solver.EPICSolver(image_size=(125,125))

    def solve(self, I1, I2):
        # compute the features
        F1 = self.solver.transform(I1)
        F2 = self.solver.transform(I2)

        # compute the unary
        unary = self.solver.probability(self.solver.likelihood(F1,F2))
        M,N,P,Q = unary.shape

        # get the argmax
        argmax = unary.reshape(M,N,P*Q).argmax(axis=-1)
        dy,dx  = np.unravel_index(argmax, (P,Q))
        x0,y0  = flow.stretchgrid(F1,F2)
        fx,fy  = flow.resize_flow(I1,I2,dx-x0,dy-y0)
        return fx,fy


class CostVolume(object):
    def __init__(self):
        self.solver = solver.EPICSolver(image_size=(125,125))

    def solve(self, I1, I2, sigma=10):
        # compute the features
        F1 = self.solver.transform(I1)
        F2 = self.solver.transform(I2)

        # compute the unary
        unary = self.solver.probability(self.solver.likelihood(F1,F2))
        M,N,P,Q = unary.shape

        # filter
        filtered = gaussian_filter(unary, sigma)
        argmax = filtered.reshape(M,N,P*Q).argmax(axis=-1)
        dy,dx  = np.unravel_index(argmax, (P,Q))
        x0,y0  = flow.stretchgrid(F1,F2)
        fx,fy  = flow.resize_flow(I1,I2,dx-x0,dy-y0)
        return fx,fy


class DenseCRF(object):
    def __init__(self):
        self.solver = solver.EPICSolver(image_size=(125,125))

    def solve(self, I1, I2, sigma_space=5.0, sigma_color=20.0, sigma_label=5.0,
            bilateral_weight=5e3, smooth_weight=1e6, iterations=20):

        # compute the features
        F1 = self.solver.transform(I1)
        F2 = self.solver.transform(I2)

        # compute the bilateral reference
        lab = imresize(color.rgb2lab(I1), F1.shape[:2])
        lab[...,0] = vis.rescale(lab[...,0], 0, 255)
        lab[...,1] = vis.rescale(lab[...,1], 0, 255)
        lab[...,2] = vis.rescale(lab[...,2], 0, 255)

        # compute the unary
        #unary = self.solver.probability(self.solver.likelihood(F1,F2))
        unary = self.solver.likelihood(F1,F2)
        unary = flow.center_unary(unary)
        M,N,P,Q = unary.shape

        # solve the CRF
        densecrf = crf.DenseCRF(unary)
        densecrf.add_pairwise(bilateral_weight,
            crf.GaussianCompatibility((P,Q), sigma_label),
            crf.BilateralKernel(lab, sigma_space, sigma_color))
        #densecrf.add_pairwise(smooth_weight,
        #    crf.IdentityCompatibility(),
        #    crf.HigherOrderSmoothness())
        fy,fx = densecrf.map(iterations)

        # extract the flow
        Pc = np.floor(P/2)
        Qc = np.floor(Q/2)
        fx,fy = flow.resize_flow(I1,I2,fx-Qc,fy-Pc)
        return fx,fy


class LDAFlow(object):
    def __init__(self):
        import siftflow
        self.siftflow = siftflow
        self.solver = solver.EPICSolver(image_size=(125,125))

    def solve(self, I1, I2, alpha=300, d=100000, gamma=0, levels=2, domain=7, coarse_domain=15):

        #compute the features
        F1 = self.solver.transform(I1).astype(np.uint8)
        F2 = self.solver.transform(I2).astype(np.uint8)

        fx,fy,obj = self.siftflow.flow_pyramid(F1,F2,
            alpha=alpha, d=d, levels=levels, domain=domain, coarse_domain=coarse_domain,
            mean=self.solver.mean, cov=self.solver.covinv, sigma=self.solver.logistic_sigma,
            bias=self.solver.logistic_prior)
        fx,fy = flow.resize_flow(I1,I2,fx,fy)
        return fx,fy


class SIFTFlow(object):
    def __init__(self):
        import siftflow
        self.siftflow = siftflow
        self.solver = solver.EPICSolver(image_size=(125,125))

    def solve(self, I1, I2, alpha=500, d=100000, gamma=1.125, levels=2, domain=3, coarse_domain=15):

        #compute the features
        F1 = self.solver.transform(I1).astype(np.uint8)
        F2 = self.solver.transform(I2).astype(np.uint8)

        fx,fy,obj = self.siftflow.flow_pyramid(F1,F2,
            alpha=alpha, d=d, levels=levels, domain=domain, coarse_domain=coarse_domain)
        fx,fy = flow.resize_flow(I1,I2,fx,fy)
        return fx,fy


class DeepFlow(object):
    def solve(self, I1, I2):
        from scipy.misc import imsave
        import subprocess
        import uuid
        import os
        identifier = uuid.uuid4()
        source = '{}_source.jpg'.format(identifier)
        target = '{}_target.jpg'.format(identifier)
        flow   = '{}_flow.flo'.format(identifier)

        # write the source files
        imsave(source.format(identifier), I1)
        imsave(target.format(identifier), I2)

        try:
            # call the deepflow binary
            p1 = subprocess.Popen(
                ['deepmatching/deepmatching', source, target, '-iccv_settings'],
                stdout = subprocess.PIPE)
            p2 = subprocess.Popen(
                ['python', 'deepmatching/rescore.py', source, target],
                stdin = p1.stdout, stdout = subprocess.PIPE)
            p1.stdout.close()
            p3 = subprocess.Popen(
                ['deepflow/deepflow', source, target, flow, '-sintel', '-matchf'],
                stdin = p2.stdout)
            p2.stdout.close()
            p3.communicate()
        except:
            raise
        finally:
            os.remove(source)
            os.remove(target)

        # load the flow into memory
        with io.openflow(flow) as f:
            fx = f[0].copy()
            fy = f[1].copy()
        os.remove(flow)
        return fx,fy
