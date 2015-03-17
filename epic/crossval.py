from __future__ import division
from matplotlib import cm, pyplot as plt
from matplotlib.widgets import Slider
from scipy.misc import imresize
from scipy.ndimage.filters import maximum_filter
import itertools
import numpy as np
from epic import vis

class Event:
    def __init__(self, x, y, axes):
        self.xdata = x
        self.ydata = y
        self.inaxes = axes


def score_distance(pnfiles, features, loss):

    # scores and distances
    scoredist = []

    for pnfile in pnfiles:

        # load the annotations from file (annotated with PNLabeller)
        with np.load(pnfile) as f:
            I1 = f['left']
            I2 = f['right']
            positive = f['positive']
            negative = f['negative']

        # compute the features
        F1 = features(I1)
        F2 = features(I2)
        M,N = F2.shape[:2]

        # compute the scaling
        S1 = np.array(F1.shape[1::-1]) / np.array(I1.shape[1::-1])
        S2 = np.array(F2.shape[1::-1]) / np.array(I2.shape[1::-1])

        # compute the raw responses
        for left,right in positive:
            lx,ly = np.round(left * S1).astype(int)
            rx,ry = np.round(right * S2).astype(int)
            try:
                score = loss(F1,F2,(lx,ly))
                scoremax = maximum_filter(score, 11)
                y,x = (score == scoremax).nonzero()
                score = score[y,x]
                d = np.sqrt((rx-x)**2 + (ry-y)**2).reshape(-1)
                scoredist = scoredist + [(sn,dn) for sn,dn in zip(score,d)]
            except ValueError:
                pass

        for left in negative:
            lx,ly = np.round(left * S1).astype(int)
            try:
                score = loss(F1,F2,(lx,ly))
                scoremax = maximum_filter(score, 11)
                y,x = (score == scoremax).nonzero()
                score = score[y,x]
                scoredist = scoredist + [(sn,np.inf) for sn in score]
            except ValueError:
                pass

    return sorted(scoredist, key=lambda pair: pair[0], reverse=True)


def roc(methods, legend, thresholds=(5,8,10,12,15,20)):

    fig,axes = plt.subplots()
    colormaps = (cm.RdPu, cm.PuBuGn)
    indices  = range(200, 50, -int(150/len(thresholds)))
    lines = []

    for method,colormap in zip(methods, colormaps):
        colormap = itertools.cycle([colormap(i) for i in indices])
        for m, threshold in enumerate(thresholds):
            precision = []
            recall = []
            scores,distances = zip(*method)
            labels = np.array(distances) < threshold
            scores = np.array(scores)
            N = len(labels)
            for n in range(0, N, max(1,int(N/1000))):
                tp = (labels[:n] == 1).sum()
                fp = (labels[n:] == 1).sum()
                tn = (labels[n:] == 0).sum()
                fn = (labels[:n] == 0).sum()
                precision.append(tp/(tp+fp))
                recall.append(tp/(tp+fn))
            l, = axes.plot(recall[1:],precision[1:], color=colormap.next())
            if m == 0: lines.append(l)

    axes.legend(lines, legend)
    axes.set_ylabel('Precision')
    axes.set_xlabel('Recall')
    axes.set_xlim(0.0,1.0)
    axes.set_ylim(0.0,1.0)
    fig.show()
    return fig,precision,recall


def estimate_logistic_parameters(pnfiles, features, loss):

    # raw responses
    pos = []
    neg = []

    for pnfile in pnfiles:

        # load the annotations from file (annotated with PNLabeller)
        with np.load(pnfile) as f:
            I1 = f['left']
            I2 = f['right']
            positive = f['positive']
            negative = f['negative']

        # compute the features
        F1 = features(I1)
        F2 = features(I2)

        # compute the scaling
        S1 = np.array(F1.shape[1::-1]) / np.array(I1.shape[1::-1])
        S2 = np.array(F2.shape[1::-1]) / np.array(I2.shape[1::-1])

        # compute the raw positive responses at the true location
        for left,right in positive:
            lx,ly = np.round(left * S1).astype(int)
            rx,ry = np.round(right * S2).astype(int)
            try:
                y = loss(F1,F2,(lx,ly))
                pos.append(y.max())
            except ValueError:
                pass

        # compute the maximum negative response across the image
        for left in negative:
            lx,ly = np.round(left * S1).astype(int)
            try:
                y = loss(F1,F2,(lx,ly))
                neg.append(y.max())
            except ValueError:
                pass

    return np.array(pos), np.array(neg)


class LogisticSlider(object):
    def __init__(self, pos,neg):

        # setup the plot
        self.fig, self.ax = plt.subplots()
        axsigma = plt.axes([0.25, 0.12, 0.5, 0.03])
        axprior  = plt.axes([0.25, 0.17, 0.5, 0.03])
        axsigma.set_xscale('log')
        self.sigma = Slider(axsigma, 'Sigma', 1e-4, 1e-1, valinit=0.001)
        self.prior = Slider(axprior, 'Prior', -10.0, 10.0, valinit=0.0)

        # plot the data
        self.pos = np.array(pos)
        self.neg = np.array(neg)
        self.lpos, = self.ax.plot(np.linspace(0,1,len(pos)), self.logistic(pos), 'b.')
        self.lneg, = self.ax.plot(np.linspace(0,1,len(neg)), self.logistic(neg), 'r.')
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(-0.2, 1.0)

        # hook the callbacks
        self.sigma.on_changed(self.update)
        self.prior.on_changed(self.update)

        self.fig.show()

    def update(self, slider):
        self.lpos.set_ydata(self.logistic(self.pos))
        self.lneg.set_ydata(self.logistic(self.neg))
        self.fig.canvas.draw_idle()

    def logistic(self, X):
        log = 1.0 / (1.0 + np.exp(-(self.prior.val + self.sigma.val*X)))
        return log
