import numpy as np
from PIL.Image import Image
from skimage.feature import local_binary_pattern, graycomatrix


def lbp(img: Image, method="uniform", radius=1):
    """Create Local Binary Pattern for an image
    @param img: image to convert
    @param method: which method, accepts 'default', 'ror', 'uniform', 'nri_uniform' or 'var'.
    Read skimage.feature.local_binary_pattern for more information
    @param radius: how many pixels adjacent to center pixel to calculate from.
    @return: (n, m) array as image
    """
    n_points = 8 * radius
    if img.mode != "L":
        img = img.convert("L")
    lbps = np.array(local_binary_pattern(img, n_points, radius, method))
    print(lbps.shape)
    return lbps


def rlbp(img: Image, method="uniform", radius=1):
    """Create RGB Local Binary Pattern for an image. Instead of greyscale, creates LBP for each RGB channel
    @param img: image to convert
    @param method: which method, accepts 'default', 'ror', 'uniform', 'nri_uniform' or 'var'.
    Read skimage.feature.local_binary_pattern for more information
    @param radius: how many pixels adjacent to center pixel to calculate from.
    @return: (n, m, 3) array as image
    """
    n_points = 8 * radius
    channels = [img.getchannel("R"), img.getchannel("G"), img.getchannel("B")]
    return [local_binary_pattern(ch, n_points, radius, method) for ch in channels]


def glcm(img: Image, distance:list, angles:list):
    if angles is None:
        angles = range(0, 361, 45)
    if distance is None:
        distance = range(0, 5)
    if img.mode != "L":
        img = img.convert("L")
    return graycomatrix(img, distance, angles)
