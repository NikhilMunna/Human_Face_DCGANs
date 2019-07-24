url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
filename = "lfw.tgz"
directory = "imgs"
new_dir = "new_imgs"
import urllib
import tarfile
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.misc import imresize, imsave
import tensorflow as tf

if not os.path.isdir(directory):
    if not os.path.isfile(filename):
        urllib.urlretrieve (url, filename)
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(path=directory)
    tar.close()