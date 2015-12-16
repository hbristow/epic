EPIC
====

Dense Semantic Correspondence where Every Pixel is a Classifier

    Hilton Bristow, Jack Valmadre and Simon Lucey,
    "Dense Semantic Correspondence where Every Pixel is a Classifier",
    International Conference on Computer Vision (ICCV), 2015

EPiC solves the dense semantic correspondence problem by constructing an LDA
classifier around every pixel in the source image, and convolving it with
every point in the target image to produce a probability likelihood estimate.

The best correspondence is then estimated by regularizing the likelihood
with spatial constraints.

Instalation
-----------

Using `pip`, the repository can be cloned and built automatically:

    pip install git+https://github.com/hbristow/epic

The requirements are pure-Python, and will be retrieved automatically


NOTES
-----

The initial public release of this research only contains code to build and
apply detectors on image pairs. It does not contain functionality to perform
regularization. We are working to provide wrappers to SIFT Flow.
