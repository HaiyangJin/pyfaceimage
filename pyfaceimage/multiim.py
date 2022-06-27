"""Functions to play with more than one images.
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw


def mkcf(im1, im2, nwline=3, misali=0, width_cf=None, topis1=True):
    
    alistrs = ['ali', 'mis']
    bboxes = [(0, 0, im1.w, im1.h/2), (0, im1.h/2, im1.w, im1.h)] # top, bottom
    fns_12 = [im1.fnonly, im2.fnonly]
    fn_cf = os.path.join(im1.dir, fns_12[1-topis1]+'_'+fns_12[topis1]+'_'+alistrs[misali!=0]+im1.ext)
    
    # make a ellipse/oval mask
    pil_mask = Image.new('L', im1.pil.size)
    draw = ImageDraw.Draw(pil_mask)
    draw.rectangle(bboxes[1-topis1], fill=255)
    
    # make the aligned composite image
    pil_cf = Image.composite(im1.pil, im2.pil, pil_mask)
    
    im_cf = im1.deepcopy()
    im_cf._repil(pil_cf)
    im_cf.refile(fn_cf)
    
    # pad 
    if width_cf is None:
        width_cf = im1.w*3
    im_cf.pad(trgw=width_cf)
    
    return im_cf
    
    

