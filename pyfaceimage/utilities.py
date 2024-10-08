"""
Utilities for pyfaceimage.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pyfaceimage.im import image

def radial_gaussian(imsize=256, opaque=.5, kernal=10):
    """Make a radial gaussian mask ('L') image (with some opaque region).

    Parameters
    ----------
    imsize : int tuple, optional. 
        Image size (width, height). {imsize} will be used as both width and height if only one value is given. Default to 256.
    opaque : float, optional. 
        Ratio of the regions to be "opaqued". Larger value means larger opaque region (from center to periphery). Default to .5.
    kernal : int, optional
        kernal for the gaussian filter, by default 10

    Returns
    -------
    PIL.Image instance
        A PIL.Image instance of the radial gaussian mask.
    """    
    
    # make circle if imsize is not tuple
    if type(imsize) is not tuple or len(imsize) == 1:
        imsize = (imsize, imsize) # a and b in ellipse formula
        
    # set the opaque area (roughly)
    if type(opaque) is not tuple or len(opaque) == 1:
        opaque = (opaque, opaque)
    if not isinstance(opaque, int):
        opaque = (int(opaque[0] * imsize[0]), int(opaque[1] * imsize[1]))
    
    # make a 'L' mask in PIL
    mask_im = Image.new('L', imsize, 0)
    draw = ImageDraw.Draw(mask_im)
    centeroval = (int(imsize[0]/2-opaque[0]/2), int(imsize[1]/2-opaque[0]/2),
                  int(imsize[0]/2+opaque[0]/2), int(imsize[1]/2+opaque[0]/2))
    draw.ellipse(centeroval, fill=255)
    # apply gaussian blur
    mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(kernal))
    
    return mask_im_blur


def radial_gradient(imsize=256, opaque=.5, power=2):
    """Make a radial gradient RGBA image (with some opaque region). [There are some issues in the generated images (a ring appears). Please use radial_gaussian() instead.]

    Parameters
    ----------
    imsize : int tuple, optional. 
        Image size (width, height). {imsize} will be used as both width and height if only one value is given. Default to 256.
    opaque : float, optional. 
        Ratio of the regions to be "opaqued". Larger value means larger opaque region (from center to periphery). Default to .5.
    power : int, optional. 
        "Power" of the nonlinearity of the gradient. Larger value means "slower" gradient change from opaque to transparent. Default to 2.

    Returns
    -------
    im.image() instance
        A radial gradient RGBA image.
    np.array
        A radial gradient RGBA image matrix.
        
    Examples
    --------
    >>> im=radial_gradient() # make a default radial gradient image
    >>> im=radial_gradient((512, 256), .8, 3) # make a radial gradient image with size (512, 256), 80% opaque region, and power 3
    
    Notes
    -----
    This code is inspired by https://github.com/python-pillow/Pillow/blob/3ab83f22ca36dcd21fc4a28d1c93066c2da75eb9/src/libImaging/Fill.c#L105
    
    Created on 2023-June-1 by Haiyang Jin (https://haiyangjin.github.io/en/about/)
    """
        
    # make circle if imsize is not tuple
    if type(imsize) is not tuple or len(imsize) == 1:
        imsize = (imsize, imsize) # a and b in ellipse formula
    im_mat = np.ones((imsize[1], imsize[0], 4), dtype=np.uint8) * 255
    
    # set the opaque area
    if type(opaque) is not tuple or len(opaque) == 1:
        opaque = (opaque, opaque)
    if not isinstance(opaque, int):
        opaque = (int(opaque[0] * imsize[0]), int(opaque[1] * imsize[1]))
    
    # The main idea here is to identify regions for opaque, gradient, and transparent
    # critical value for transparent region (outside is the transparent area)
    d0 = np.sqrt(((imsize[0]/2-imsize[0]/2)/(opaque[0]/2))**2 + ((imsize[1]/2)/(opaque[1]/2))**2)
    # whether a pixel is in the opaque area
    gradients = np.ones((imsize[1], imsize[0])) * 255
    for x in range(im_mat.shape[1]):
        for y in range(im_mat.shape[0]):
            # inner oval/circle (inside is the opaque area)
            d1 = np.sqrt(((x - imsize[0]/2)/(opaque[0]/2))**2 + ((y - imsize[1]/2)/(opaque[1]/2))**2)
            # outer oval/circle (outside is the transparent area)
            # d2 = np.sqrt(((x - imsize[0]/2)/opaque[0])**2 + ((y - imsize[1]/2)/opaque[1])**2)
            
            # negate the value if it is within the two area
            # clip the region outside the outer oval/circle with d0
            if d1 > 1: gradients[y,x] = -min(d1, d0)**power
            
    # idenify the second largest value as the max and the min value
    tmpmax = np.unique(gradients)[-2]
    tmpmin = np.min(gradients)
    assert tmpmin==-d0**power, 'Something unknown went wrong.'
    # update the alpha channel as gradient transparent
    im_mat[:,:,3] = np.clip((gradients - tmpmin) * 254 / (tmpmax - tmpmin), 0, 255).astype(np.uint8)
    
    # save it to an image instance
    im = image('')
    im.remat(im_mat)
    
    return im, im_mat


    
    