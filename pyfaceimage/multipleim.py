"""Functions to play with more than one images.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def mkcf(im1, im2, **kwargs):
    """Make composite faces with im1 and im2.

    Args:
        im1 (fim.image instance): image one (ready by fim.image) for creating composite faces.
        im2 (fim.image instance): image two (ready by fim.image) for creating composite faces.
        misali (int, num): misalignment. Positive int will shift to the right and negative int will shift to the left. If misali is int, it refers to pixels. If misali is decimal, it misali*face width is the amount of misalignment. Defaults to 0.
        topis1 (bool): whether im1 (or im2) is used as the top of the composite face. Defaults to True. 
        cueistop (bool): whether the top half is cued. Defaults to True.
        lineh (int): the height (in pixels) of the line between the top and bottom facial havles. Defaults to 3.
        width_cf (int): the width of the composite face/the line (also the width of the output composite face image). Defaults to three times of the face width.
        lineclr (int tuple): the color of the line. Defaults to (255,255,255), i.e., white.
        showcue (bool): whether to display cue in the image. Defaults to False.
        cuethick (int): the thickness (in pixels) of the cue. Defaults to 4.
        cuew (int): the width (in pixels) of the cue. Defaults to 110% of the face width.
        cueh (int): the height (in pixels) of the very left and right parts of the cue. Defaults to about 5% of the face height.
        cuedist (int): distance from the main part of the cue to edge of the face. Defaults to twice of cueh.
        
    Returns:
        im_cf: the composite face stimuli as a fim.image instance.
    """
    
    defaultKwargs = {'misali':0, 'topis1': True, 'cueistop': True,
                     'lineh':3, 'width_cf': im1.w*3, 'lineclr':(255,255,255),  # white line
                     'showcue':False, 'cuethick': 4, 'cuew': int(im1.w*1.1), 'cueh': int(im1.h*.05), 'cuedist': None}
    kwargs = {**defaultKwargs, **kwargs}
    
    w_cf = kwargs['width_cf']
    if isinstance(kwargs['misali'], int):
        misali = kwargs['misali']
    else:
        misali = int(kwargs['misali']*im1.w)
    assert (misali+im1.w/2)<=w_cf/2, f'Please make sure the width of the composite face ({w_cf}) is set appropriately to fit the misalignment ({misali}).'
        
    alistrs = ['ali', 'mis']
    bboxes = [(0, 0, im1.w, im1.h/2), (0, im1.h/2, im1.w, im1.h)] # top, bottom
    dests = [((w_cf-im1.w)//2+misali*(1-kwargs['cueistop']), 0), # top position
             ((w_cf-im1.w)//2+misali*kwargs['cueistop'], im1.h//2+kwargs['lineh'])] # bottom position
    fns_12 = [im1.fnonly, im2.fnonly]
    
    fn_cf = os.path.join(im1.dir, fns_12[1-kwargs['topis1']]+'_'+fns_12[kwargs['topis1']]+'_'+alistrs[misali!=0])
    
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
        
    if kwargs['showcue']:
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
        dist.paste(cues[1-kwargs['cueistop']], (0, 0))
        dist.paste(dist_cf, (0, cue1.size[1]))
        dist.paste(ImageOps.flip(cues[kwargs['cueistop']]), (0, cue1.size[1]+dist_cf.size[1]))
        
        # add the cue str to filename
        fn_cf = fn_cf+'_'+['top', 'bot'][1-kwargs['cueistop']]
    
    else:
        dist = dist_cf
    
    # update/save the cf image information
    im_cf = im1.deepcopy()
    im_cf._repil(dist)
    im_cf.refile(fn_cf+im1.ext)
        
    return im_cf
    
    

