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

filepaths = []
for dir_, _, files in os.walk(directory):
    for fileName in files:
        relDir = os.path.relpath(dir_, directory)
        relFile = os.path.join(relDir, fileName)
        filepaths.append(directory + "/" + relFile)
        
for i, fp in enumerate(filepaths):
    img = imread(fp) #/ 255.0
    img = imresize(img, (40, 40))
    imsave(new_dir + "/" + str(i) + ".png", img)