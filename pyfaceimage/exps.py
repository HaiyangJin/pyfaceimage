
import pyfaceimage

def mkcf_prf(dict_cf, dict_bg, imsize_bg=(500,500), opaque=.75):
    """Make composite face stimuli for the pRF experiment.

    Parameters
    ----------
    dict_cf : dict
        a dictionary of composite faces.
    dict_bg : dict
        a dictionary of background images.
    imsize_bg : tuple, optional
        the output image size, by default (500,500)
    opaque : float, optional
        ratio of the opaque region, by default .75

    Returns
    -------
    dict
        a dictionary of composite face stimuli to be used for the pRF experiment.
    """
    
    # Initialize the output dict
    imdict_out = {}
    
    for k,im_fore in dict_cf.items():
        
        # this fore and background images
        k_back = k.replace('mis', 'ali').replace('_l', '').replace('_r', '')+'_pscr' # only use aligend background
        im_back = dict_bg[k_back]

        # phase scrambed background
        im_back.resize(trgw=imsize_bg[0], trgh=imsize_bg[1])
        
        # mask
        mask = pyfaceimage.utilities.radial_gaussian(imsize_bg, opaque)

        # add fore to back
        im_back_tmp = im_back.deepcopy()
        pyfaceimage.multim.composite(im_back_tmp, im_fore)

        # apply mask
        im_back_tmp.pil.putalpha(mask)
        
        # update the filename
        im_back_tmp.filename = im_fore.filename.replace('_pad', '')
        im_back_tmp._updatefromfilename()
        
        # save the output dict
        imdict_out[k] = im_back_tmp
        del im_back_tmp

    return imdict_out

