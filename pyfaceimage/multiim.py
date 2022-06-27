"""Functions to play with more than one images.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def mkcf(im1, im2, **kwargs):
    
    
    defaultKwargs = {'misali':0, 'topis1': True,
                     'lineh':3, 'width_cf': im1.w*3, 'lineclr':(255,255,255)} # white line
    kwargs = {**defaultKwargs, **kwargs}
        
    alistrs = ['ali', 'mis']
    bboxes = [(0, 0, im1.w, im1.h/2), (0, im1.h/2, im1.w, im1.h)] # top, bottom
    dests = [((kwargs['width_cf']-im1.w)//2, 0), 
             ((kwargs['width_cf']-im1.w)//2+kwargs['misali'], im1.h//2+kwargs['lineh'])]
    fns_12 = [im1.fnonly, im2.fnonly]
    
    fn_cf = os.path.join(im1.dir, fns_12[1-kwargs['topis1']]+'_'+fns_12[kwargs['topis1']]+'_'+alistrs[kwargs['misali']!=0]+im1.ext)
    
    # top and bottom pil
    im1_half = im1.deepcopy()
    im2_half = im2.deepcopy()
    im1_half.croprect(bboxes[1-kwargs['topis1']])
    im2_half.croprect(bboxes[kwargs['topis1']])
    
    # create a new canvas and paste the image
    dist = Image.new(im1.pil.mode, (kwargs['width_cf'], im1.h+kwargs['lineh']))
    # white line
    draw = ImageDraw.Draw(dist)
    draw.rectangle((0,im1.h//2,kwargs['width_cf'],im1.h//2+kwargs['lineh']-1),fill=kwargs['lineclr'])
    # top and bottom
    dist.paste(im1_half.pil, dests[1-kwargs['topis1']])
    dist.paste(im2_half.pil, dests[kwargs['topis1']])
    
    im_cf = im1.deepcopy()
    im_cf._repil(dist)
    im_cf.refile(fn_cf)
        
    return im_cf
    
    

