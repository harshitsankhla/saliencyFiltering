# Contrast Based Saliency Filtering

1. Download dataset from [here(109 MB)](http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip)

2. *Directory structure:*
	1. `src/main.py:` Calculates the final saliency. Its major components are:
		1. `apply_slic` or `apply_gmm`
		2. `apply_uniqueness`
		3. `apply_distribution`
		4. `apply_saliency`

	2. `src/slic_segmentation.py:` Apply SLIC superpixels based on *LAB color space* and *Euclidean distance*  using *K-means* clustering.
	3. `src/gmm_segmentation.py:` Fit GMM onto image pixels based on *RGB*, *RGBXY* and *LABXY* space. And divide pixels into `n_components` clusters to form *Superpixels.* 

3. *Usage:*
	1. `python src/main.py <path_to_image>`
	2. `python src/gmm_segmentation.py <path_to_image>`
	3. *Superpixels* count can be varied by changing `n_components` in the code.