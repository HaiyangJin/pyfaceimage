"""
Utilities for pyfaceimage.
"""

import itertools
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pyfaceimage.im import image


def exp_design_builder(exp_conditions, rand_block=None, sort_block=None, is_rand=True):
    """Generate a full factorial design matrix based on the input conditions.

    Parameters
    ----------
    exp_conditions : list
        A list of tuples, each containing a condition name and a list of levels for that condition, e.g., `[("IV1", [1, 2, 3]), ("IV2", [1, 2, 3, 4])]`.
    rand_block : str list, optional
        A list of condition names to randomize by block. The latter columns/blocks will be randomized within the former columns/blocks. by default None
    sort_block : str list, optional
        A list of condition names to sort by block, by default None
    is_rand : bool, optional
        Whether to randomize the design matrix, by default True

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the full factorial design matrix. Each row represents a trial (Usage: `design.iloc[trial_index]['IV1']`).
    int
        The number of trials in the design matrix.
    int
        The number of blocks in the design matrix.

    Raises
    ------
    ValueError
        `rand_block` and `sort_block` must be available in the condition names in `exp`_conditions`.
    ValueError
        `rand_block` and `sort_block` must not overlap.
        
    Examples
    --------
    >>> exp_conditions = [("IV1", [1, 2, 3]),
                          ("IV2", [1, 2, 3, 4]),
                          ("IV3", [1, 2, 3, 4, 5]),
                          ("blockNumber", [1, 2])]
    >>> rand_block = ["IV3", "IV2"]
    >>> sort_block = ["blockNumber"]
    >>> design = exp_design_builder(exp_conditions, rand_block, sort_block)
    >>> print(design)
    >>> print(f"Total trials: {n_trials} \nTotal blocks: {n_blocks}")
    """
    # Validate input conditions
    cond_names = [cond[0] for cond in exp_conditions]
    levels_per_cond = [len(cond[1]) for cond in exp_conditions]

    # Ensure rand_block and sort_block are lists of indices
    def _process_block(block, var_name):
        if isinstance(block, list):
            block_indices = [cond_names.index(name) for name in block]
        elif block is None:
            block_indices = []
        else:
            raise ValueError(f"{var_name} must be a list of condition names or None.")
        return block_indices

    rand_block_indices = _process_block(rand_block, "rand_block")
    sort_block_indices = _process_block(sort_block, "sort_block")

    if set(rand_block_indices) & set(sort_block_indices):
        raise ValueError("rand_block and sort_block must not overlap.")

    # Generate full factorial design
    design_matrix = np.array(list(itertools.product(*[range(1, levels + 1) for levels in levels_per_cond])))
    # Randomize design if requested
    if is_rand:
        np.random.shuffle(design_matrix)
    
    # Randomize design by blocks (rand_block)
    rand_block_indices = sort_block_indices + rand_block_indices
    if rand_block_indices:
        for bloidx in range(len(rand_block_indices),0,-1):
            
            thisidx = rand_block_indices[0:bloidx]
            
            # Identify unique blocks
            tmp, block_indices = np.unique(design_matrix[:, thisidx], axis=0, return_inverse=True)

            # Shuffle the block order
            final_indices = np.argsort(np.random.permutation(np.unique(block_indices)))
            design_matrix = design_matrix[np.concatenate([np.where(block_indices == i)[0] for i in final_indices])]    
     
    # Sort the design matrix by sort_block
    if sort_block_indices:
        design_matrix = design_matrix[np.lexsort([design_matrix[:, idx] for idx in reversed(sort_block_indices)])]

    # Replace indices with actual levels
    design = pd.DataFrame({
        cond_names[i]: [exp_conditions[i][1][level - 1] for level in design_matrix[:, i]]
        for i in range(len(exp_conditions))
    })

    # n_trials = len(design)
    # n_blocks = np.prod([len(exp_conditions[idx][1]) for idx in rand_block_indices])

    return design
    

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


    
    