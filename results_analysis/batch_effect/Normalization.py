import os
from PIL import Image
import numpy as np
import cv2
from scipy.stats.mstats import mquantiles as quantiles

import matplotlib.pyplot as plt

# reference:
#  https://github.com/jnkather/ColorDeconvolutionMatlab/blob/master/ColorDeconvolutionDemo.m
# https://github.com/scikit-image/scikit-image/blob/e37fc660aeb96d53db27c9e2f1a0a88920bc2b65/skimage/color/colorconv.py#L629


# transformation matrix in QuPath
H = [0.651, 0.701, 0.269]
# H = [0.269, 0.701, 0.651] # BGR
E = [0.216, 0.801, 0.558]
Res = [0.316, -0.598, 0.737]

a = np.array([H/np.linalg.norm(H), E/np.linalg.norm(E), Res/np.linalg.norm(Res)])
HDABtoRGB = a.conj().T
RGBtoHDAB = np.linalg.inv(HDABtoRGB)

thumbnail_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/thumbnails"
out_dir = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/pipeline/color_normalization"
case_list = os.listdir(thumbnail_dir)



def SeparateStains(imageRGB, Matrix):
    # convert input image to double precision float
    # add 2 to avoid artifacts of log transformation
    imageRGB = imageRGB.astype(np.float) + 2
    # perform color deconvolution
    imageOut = np.reshape(-np.log(imageRGB), [], order="F") * Matrix
    imageOut = np.reshape(imageOut, imageRGB.shape)
    # post-processing
    imageOut = normalizeImage(imageOut, 'stretch')
    return imageOut

def normalizeImage(imageIn, opt):
    imageOut = np.copy(imageIn)
    for i in range (imageIn.shape[2]):
        Ch = imageIn[:,:, i]
        imageOut[:,:, i] = (imageIn[:,:, i] - np.min(Ch)) / (np.max(Ch) - np.min(Ch))
        imageOut[:,:, i] = 1 - imageOut[:,:, i]
        if opt == 'stretch':
            imageOut[:,:, i] = imadjust(imageOut[:,:, i], stretchlim(imageOut[:,:, i]), [])
    return imageOut


def stretchlim(im, bottom=0.001, top=None, mask=None, in_place=False):
    """Stretch the image so new image range corresponds to given quantiles.
    Parameters
    ----------
    im : array, shape (M, N, [...,] P)
        The input image.
    bottom : float, optional
        The lower quantile.
    top : float, optional
        The upper quantile. If not provided, it is set to 1 - `bottom`.
    mask : array of bool, shape (M, N, [...,] P), optional
        Only consider intensity values where `mask` is ``True``.
    in_place : bool, optional
        If True, modify the input image in-place (only possible if
        it is a float image).
    Returns
    -------
    out : np.ndarray of float
        The stretched image.
    """
    if in_place and np.issubdtype(im.dtype, np.float):
        out = im
    else:
        out = np.empty(im.shape, np.float32)
        out[:] = im
    if mask is None:
        mask = np.ones(im.shape, dtype=bool)
    if top is None:
        top = 1 - bottom
    q0, q1 = quantiles(im[mask], [bottom, top])
    out -= q0
    out /= q1 - q0
    out = np.clip(out, 0, 1, out=out)
    return out


for cl in case_list:
    case_thumb_dir = os.path.join(thumbnail_dir, cl)
    thumb_list = os.listdir(case_thumb_dir)
    for tl in thumb_list:
        if "thumb" in tl and "7111256" in tl:
            thumb_fn = os.path.join(thumbnail_dir, cl, tl)
            img = cv2.imread(thumb_fn)
            # img = Image.open(thumb_fn, 'r')
            img_arr = np.array(img)

            imageHDAB = SeparateStains(img_arr, RGBtoHDAB)

            # H_img = np.array(img_arr * H).astype(np.uint8)
            # save_to = os.path.join(out_dir, cl, tl.replace(".png", "_H.png"))
            # if not os.path.exists(os.path.join(out_dir, cl)):
            #     os.makedirs(os.path.join(out_dir, cl))
            # # Image.fromarray(H_img).save(save_to)
            # cv2.imwrite(save_to, H_img)







