from setuptools import setup, Extension
import numpy

# ----------------------------------------------------------------------------
# EPiC
# ----------------------------------------------------------------------------
setup(name = 'EPiC',
    version = '0.1',
    description = 'Dense Semantic Correspondence where Every Pixel is a Classifier',
    long_description = open('README.md').read(),
    keywords = 'semantic, correspondences, LDA, SIFT, mean-field, CRF',
    url = 'https://github.com/hbristow/epic/',
    author = 'Hilton Bristow',
    author_email = 'hilton.bristow+epic@gmail.com',
    license = 'GPL',
    packages = ['epic'],
    package_data = {
        'epic': ['statistics_sift_imagenet.npz']
    },
    install_requires = [
        'numpy >= 1.9',
        'pillow',
        'scipy',
        'matplotlib'
    ],
    extras_require = {
        'train': ['scikit-learn'],
        'fftw': ['pyfftw'],
    },
    zip_safe=False)
