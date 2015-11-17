"""Perceptually improved colormaps

This module contains a number of colormaps and color space manipulation
functions that aid the visualization of data in Matplotlib. The colormaps
are taken from palettes at colorbrewer.org.

Sequential, diverging and qualitative (categorical) colormaps are provided.
They are, respectively:

    - blue, green, purple, orange, pink, rose, ocean, winter, earth, fire
    - terrain, pebble, charcoal, spectrum
    - category
"""
import sys as _sys
from matplotlib.colors import hex2color, LinearSegmentedColormap
try:
    from skimage.color import convert_colorspace,\
        gray2rgb, hed2rgb, hsv2rgb, lab2lch, lab2rgb, lch2lab,\
        rgb2gray, rgb2hed, rgb2hsv, rgb2lab, rgb2xyz, xyz2rgb, xyz2lab
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Sequential Colormaps
# ----------------------------------------------------------------------------
gold    = ['#543005','#f7e9c3','#ffffff']
blue    = ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#ece7f2', '#fff7fb']
green   = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6', '#e5f5f9', '#f7fcfd']
purple  = ['#4d004b', '#810f7c', '#88419d', '#8c6bb1', '#8c96c6', '#9ebcda', '#bfd3e6', '#e0ecf4', '#f7fcfd']
orange  = ['#7f0000', '#b30000', '#d7301f', '#ef6548', '#fc8d59', '#fdbb84', '#fdd49e', '#fee8c8', '#fff7ec']
pink    = ['#67001f', '#980043', '#ce1256', '#e7298a', '#df65b0', '#c994c7', '#d4b9da', '#e7e1ef', '#f7f4f9']
rose    = ['#49006a', '#7a0177', '#ae017e', '#dd3497', '#f768a1', '#fa9fb5', '#fcc5c0', '#fde0dd', '#fff7f3']
ocean   = ['#084081', '#0868ac', '#2b8cbe', '#4eb3d3', '#7bccc4', '#a8ddb5', '#ccebc5', '#e0f3db', '#f7fcf0']
winter  = ['#081d58', '#253494', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1', '#ffffd9']
earth   = ['#662506', '#993404', '#cc4c02', '#ec7014', '#fe9929', '#fec44f', '#fee391', '#fff7bc', '#ffffe5']
fire    = ['#800026', '#bd0026', '#e31a1c', '#fc4e2a', '#fd8d3c', '#feb24c', '#fed976', '#ffeda0', '#ffffcc']

# ----------------------------------------------------------------------------
# Diverging Colormaps
# ----------------------------------------------------------------------------
terrain  = ['#003c30','#01665e','#35978f','#80cdc1','#c7eae5','#f6e8c3','#dfc27d','#bf812d','#8c510a','#543005']
pebble   = ['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f']
charcoal = ['#1a1a1a','#4d4d4d','#878787','#bababa','#e0e0e0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f']
spectrum = ['#5e4fa2','#3288bd','#66c2a5','#abdda4','#e6f598','#fee08b','#fdae61','#f46d43','#d53e4f','#9e0142']

# ----------------------------------------------------------------------------
# Build the colormaps
# ----------------------------------------------------------------------------
_module = _sys.modules[__name__]
for _colormap in dir(_module):
    try:
        _colors = map(hex2color, getattr(_module, _colormap))
        setattr(_module, _colormap, LinearSegmentedColormap.from_list(_colormap, _colors))
    except:
        pass
