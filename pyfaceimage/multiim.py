"""Functions to play with more than one images.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def mkcf(im1, im2, **kwargs):
    
    
    defaultKwargs = {'misali':0, 'topis1': True,
                     'lineh':3, 'width_cf': im1.w*3, 'lineclr':(255,255,255),
                     'cueposi': None, 'cuethick': 4, 'cuew': int(im1.w*1.1), 'cueh': int(im1.h*.05), 'cuedist': None} # white line
    kwargs = {**defaultKwargs, **kwargs}
    
    w_cf = kwargs['width_cf']
        
    alistrs = ['ali', 'mis']
    bboxes = [(0, 0, im1.w, im1.h/2), (0, im1.h/2, im1.w, im1.h)] # top, bottom
    dests = [((w_cf-im1.w)//2, 0), 
             ((w_cf-im1.w)//2+kwargs['misali'], im1.h//2+kwargs['lineh'])]
    fns_12 = [im1.fnonly, im2.fnonly]
    
    fn_cf = os.path.join(im1.dir, fns_12[1-kwargs['topis1']]+'_'+fns_12[kwargs['topis1']]+'_'+alistrs[kwargs['misali']!=0]+im1.ext)
    
    # top and bottom pil
    im1_half = im1.deepcopy()
    im2_half = im2.deepcopy()
    im1_half.croprect(bboxes[1-kwargs['topis1']])
    im2_half.croprect(bboxes[kwargs['topis1']])
    
    # create a new canvas and paste the image
    dist_cf = Image.new(im1.pil.mode, (w_cf, im1.h+kwargs['lineh']))
    # white line
    drawl = ImageDraw.Draw(dist_cf)
    drawl.rectangle((0,im1.h//2,w_cf,im1.h//2+kwargs['lineh']-1),fill=kwargs['lineclr'])
    # top and bottom
    dist_cf.paste(im1_half.pil, dests[1-kwargs['topis1']])
    dist_cf.paste(im2_half.pil, dests[kwargs['topis1']])
    
    # cue positions
    cueposi = [kwargs['cueposi']==posi for posi in ['top', 'bottom']]
    
    if np.any(cueposi):
        if kwargs['cuedist'] is None:
            kwargs['cuedist']= kwargs['cueh']*2
        # cues
        cue1 = Image.new(im1.pil.mode, (w_cf, kwargs['cuedist']))
        cue2 = cue1.copy()
        drawc = ImageDraw.Draw(cue1)
        drawc.rectangle(((w_cf-kwargs['cuew'])//2,0,(w_cf+kwargs['cuew'])//2-1,kwargs['cuethick']-1),fill=kwargs['lineclr'])
        drawc.rectangle(((w_cf-kwargs['cuew'])//2,kwargs['cuethick'],
                        (w_cf-kwargs['cuew'])//2+kwargs['cuethick']-1,kwargs['cueh']+kwargs['cuethick']-1),
                        fill=kwargs['lineclr'])
        drawc.rectangle(((w_cf+kwargs['cuew'])//2-kwargs['cuethick'],kwargs['cuethick'],
                        (w_cf+kwargs['cuew'])//2-1,kwargs['cueh']+kwargs['cuethick']-1),
                        fill=kwargs['lineclr'])
        cues = [cue1, cue2]
        
        # concatenate CF and cues
        dist = Image.new(im1.pil.mode, (w_cf, cue1.size[1]*2+dist_cf.size[1]))
        dist.paste(cues[np.where(cueposi)[0][0]], (0, 0))
        dist.paste(dist_cf, (0, cue1.size[1]))
        dist.paste(ImageOps.flip(cues[1-np.where(cueposi)[0][0]]), (0, cue1.size[1]+dist_cf.size[1]))
    
    else:
        dist = dist_cf
    
    im_cf = im1.deepcopy()
    im_cf._repil(dist)
    im_cf.refile(fn_cf)
        
    return im_cf
    
    

