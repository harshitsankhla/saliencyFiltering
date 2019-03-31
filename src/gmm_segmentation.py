import numpy as np
from sys import argv
from skimage import io,color
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab
from skimage.util import img_as_float

def display_image(rgb):
	fig = plt.figure("gmm_segmented")
	ax = fig.add_subplot(1,1,1)
	ax.imshow(rgb)
	plt.axis("off")
	plt.show()

def display_superpixels(rgb, superpixels):
	fig = plt.figure("gmm_segmented")
	ax = fig.add_subplot(1,1,1)
	ax.imshow(mark_boundaries(rgb, superpixels))
	plt.axis("off")
	plt.show()

def to_rgbxy(rgb):
	row, col = rgb.shape[:2]
	output = np.zeros((row*col, 5))
	
	for i in range(row):
	    for j in range(col):
	        r,g,b =  rgb[i,j]
	        output[i*col + j,:] = r,g,b,i,j

	return output

def to_rgb(rgb):
	row, col = rgb.shape[:2]
	output = np.zeros((row*col, 3))
	
	for i in range(row):
	    for j in range(col):
	        output[i*col + j,:] = rgb[i, j]

	return output

def to_labxy(rgb):
	lab = rgb2lab(rgb)
	row, col = lab.shape[:2]
	output = np.zeros((row*col, 5))
	
	for i in range(row):
	    for j in range(col):
	        l, a, b =  lab[i,j]
	        output[i*col + j,:] = l,a,b,i,j

	return output	

def labels_to_image(labels, rgb):
	row, col = rgb.shape[:2]
	output = np.zeros([row, col])

	for i in range(row):
	    for j in range(col):
	    	output[i, j] = labels[i*col + j]

	return output

def apply_gmm(n_components, rgb):
	image = img_as_float(rgb)
	# flat_image = to_rgbxy(image)
	flat_image = to_labxy(image)

	gmm = GaussianMixture(n_components).fit(flat_image)
	labels = gmm.predict(flat_image)

	superpixels = labels_to_image(labels, image)	
	superpixels = np.int8(superpixels)

	return superpixels


if __name__ == '__main__':
	path = str(argv[1])
	rgb = io.imread(path)

	n_components = 100
	superpixels = apply_gmm(n_components, rgb)
	
	display_superpixels(rgb, superpixels)