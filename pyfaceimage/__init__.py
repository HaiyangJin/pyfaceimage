"""
Tools for dealing with images in dictionary.
"""

from .im import (image)
from .multim import (mkcf, concatenate)
from .utilities import (radial_gaussian)
from .exps import (mkcf_prf)

__all__ = ['image', 
           'mkcf',
           'concatenate',
           'radial_gaussian',
           'mkcf_prf']

import os, copy, random, glob, warnings
from itertools import permutations, combinations_with_replacement, combinations, product

def dir(path=os.getcwd(), imwc='*.png', read=True, sep='/'):
    """List all images in the path and subdirs. Please make sure none of the subdirectories is named as 'path'.

    Parameters
    ----------
    path : str, optional
        path to the directory where the images are stored, by default os.getcwd()
    imwc : str, optional
        wildcard to identify images (but not the subdirectories), by default '*.png'
    read : bool, optional
        whether read the images (more see im.image()), by default True
    sep: str, optional
        a string to be used to concatnate the subdirectory and image name, which is used as the key for flattening the dictionary, by default '/'

    Returns
    -------
    dict
        A dictionary of image() instances in the path and subdirs.
    """
    
    # list all files in the path
    imdict_root = _dir(path, imwc, read)
    [im._setgroup('path') for im in imdict_root.values()] # update group info as 'path'    
    
    # list all dirs in the subdirectories
    subdirs = [d for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path,d))] # is a dir
    assert 'path' not in subdirs, f'"path" is not a valid subdir name...'
    imdict = {sub:_dir(path=os.path.join(path, sub), imwc=imwc, read=read) 
                  for sub in subdirs}
    
    # add images in path to the dictionary
    if bool(imdict_root):
        imdict['path'] = imdict_root 
        
    # flatten the dictionary
    if bool(sep): imdict = _flatten(imdict, sep=sep)      
    
    # set the global path
    [im._setgpath(path) for im in imdict.values()]
    
    return imdict


def _dir(path, imwc, read):
    """List all images (only) in the path.

    Parameters
    ----------
    path : str, optional
        path to the directory where the images are stored.
    imwc : str, optional
        wildcard to identify images.
    read : bool, optional
        whether read the images (more see im.image()).

    Returns
    -------
    dict
        A dictionary of images in the path.
    """
    
    # list all files in the path
    ".".join(os.path.basename("cau.jpg").split('.')[0:-1])
    imdict = {".".join(os.path.basename(f).split('.')[0:-1]):image(f, read) 
              for f in glob.glob(os.path.join(path,imwc))}
    print(f'Found {len(imdict)} files in {path}...')
    
    return imdict
    
    
def _flatten(imdict_nested, sep='/'):
    """Flatten the nested dictionary.

    Parameters
    ----------
    imdict_nested : dict
        A nested dictionary of images in the path and subdirs, e.g., the output of dir(flatten=False).
    sep : str, optional
        a string to be used to concatnate the subdirectory and image name, which is used as the key for flattening the dictionary, by default '/'

    Returns
    -------
    dict
        A flatten dictionary of images in the path and subdirs.
    """
    
    assert not _isflatten(imdict_nested), 'It seems that the dictionary is already flatten...'
    
    # initialize the flatten dictionary
    imdict_flat = {}
    for k,v in imdict_nested.items():

        if sep in k:
            warnings.warn(f'A subdir "{k}" already contains the sep "{sep}". This may cause problems. Please consider renaming this subdir.')
        
        # apply the sep to all keys except for images in path
        k = k+sep if k != 'path' else ''
        
        # save one item in the flatten dictionary
        imdict_tmp = {(k+kk):vv for kk,vv in v.items()}
        imdict_flat.update(imdict_tmp)
    
    # import pandas as pd (not necessary)
    # imdict_list = pd.json_normalize(imdict, sep=sep)
    # imdict = imdict_list.to_dict(orient='records')[0]

    return imdict_flat


def _nested(imdict_flat, sep='/'):
    """Convert the flatten dictionary to nested dictionary.

    Parameters
    ----------
    imdict_flat : dict
        A flatten dictionary of images in the path and subdirs.
    sep : str, optional
        a string to be used to separate the key in flatten dictionary into the subdirectory and image name, by default '/'

    Returns
    -------
    dict
        A nested dictionary of images in the path and subdirs.
    """
    
    assert _isflatten(imdict_flat), 'It seems that the dictionary is already in nested...'
    
    # get all group names
    groups = set([im.group for im in imdict_flat.values()])
    # initialize the nested dictionary
    imdict_nested = {g:{} for g in groups}
    
    for k,v in imdict_flat.items():
        
        if sep in k:
            # the string before the first sep is the group name
            group, fnonly = k.split(sep,1)
        else:
            group, fnonly = 'path', k
        
        # save the value in the nested dictionary
        imdict_nested[group][fnonly] = v
        
    return imdict_nested


def _isflatten(imdict):
    """Check if the dictionary is flatten.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.

    Returns
    -------
    bool
        whether the dictionary is flatten.
    """
    # check if the dictionary is flatten
    isflat = not any([isinstance(v, dict) for v in imdict.values()])
    return isflat


def sample(imdict, n=1, valueonly=True):
    """Randomly sample n images from the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    n : int, optional
        number of images to be sampled, by default 1
    valueonly : bool, optional
        whether return the value (i.e., the image instance only), by default True. If False, the key-value pair will be returned.

    Returns
    -------
    instance or list of tuples
        randomly sampled value or key-value pair.
    """
    
    # make sure the dictionary is flatten
    if not _isflatten(imdict):
        imdict = _flatten(imdict)
    
    # random select one image from dictionary to check the output
    assert n<=len(imdict)
    samples = random.sample(sorted(imdict.items()), n)
    
    if valueonly:
        # return the value (i.e., the image instance only)
        out = [v[1] for v in samples]
        if n==1:
            out = out[0]
    else:
        # return the key-value pair/tuple
        out = samples
    
    return out


def deepcopy(imdict):
    """Deep copy the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.

    Returns
    -------
    dict
        A deep copied dictionary of images.
    """
    return copy.deepcopy(imdict)


def mkcfs(imdict, sep='/', **kwargs):
    """Make composite faces for all possible combinations of images in the dictionary.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    sep : str, optional
        a string to be used to concatnate the subdirectory and image name, which is used as the key for flattening the dictionary, by default '/'. If sep is None, the dictionary will not be flatten.
   
    Other Parameters
    ----------------
    misali : int, num
        misalignment. Positive int will shift to the right and negative int will shift to the left. If misali is int, it refers to pixels. If misali is decimal, it misali*face width is the amount of misalignment. Defaults to 0.
    topis1 : bool
        whether im1 (or im2) is used as the top of the composite face. Defaults to True.
    cueistop : bool
        whether the top half is cued. Defaults to True.
    lineh : int
        the height (in pixels) of the line between the top and bottom facial havles. Defaults to 3.
    width_cf : int
        the width of the composite face/the line (also the width of the output composite face image). Defaults to three times of the face width.
    lineclr : int tuple
        the color of the line. Defaults to (255,255,255), i.e., white.
    showcue : bool
        whether to display cue in the image. Defaults to False.
    cueclr : int tuple
        the color of the cue. Defaults to the same as lineclr.
    cuethick : int
        the thickness (in pixels) of the cue. Defaults to 4.
    cuew : int
        the width (in pixels) of the cue. Defaults to 110% of the face width.
    cueh : int
        the height (in pixels) of the very left and right parts of the cue. Defaults to about 5% of the face height.
    cuedist : int
        distance from the main part of the cue to edge of the face. Defaults to twice of cueh.

    Returns
    -------
    A dictionary of im.image() instance
        the composite face stimuli as a im.image() instance.
    """
    
    # make sure the dictionary is flatten
    if len(set([im.group for im in imdict.values()])) > 1 and bool(sep):
        imdict_nested = _nested(imdict, sep=sep)
        
        cfdict_nested = {}
        for k,v in imdict_nested.items():
            cfdict_nested[k] = _mkcfs(v, **kwargs)
        
        cfdict = _flatten(cfdict_nested, sep=sep)
            
    else:
        cfdict = _mkcfs(imdict, **kwargs)
            
    return cfdict
    
    
def _mkcfs(imdict, **kwargs):
    """Please refer to mkcfs() for more details.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    misali : int, num
        misalignment. Positive int will shift to the right and negative int will shift to the left. If misali is int, it refers to pixels. If misali is decimal, it misali*face width is the amount of misalignment. Defaults to [0,0.5].
    showcue : bool
        whether to display cue in the image. Defaults to False.
    cueistop : bool
        whether the top half is cued. Defaults to True.
    pairstyle : str
        the style of the pairs. Defaults to "perm". Options are "perm" (`itertools.permutation`), "comb" (`itertools.combination`), "comb_repl" (`itertools.combinations_with_replacement`), "prod" (`itertools.product`) and "itself" (concatenate itself).
    
    Returns
    """
    defaultKwargs = {'misali':[0,0.5], 'showcue':False, 'cueistop': True, 'pairstyle':'perm'}
    kwargs = {**defaultKwargs, **kwargs}
    
    # make sure the dictionary is flatten
    if not _isflatten(imdict):
        imdict = _flatten(imdict)
    
    # check the number of im
    nim = len(imdict)
    assert nim>1, f'There should be more than one im in imdict... (Now {nim})'
    
    # generate ali and mis CF by default
    misali = kwargs['misali']
    if not(isinstance(misali, list) | isinstance(misali, tuple)):
        misali=[misali]
        
    # generate for both top and bottom if showcue
    if kwargs['showcue']:
        cueistop = [1, 0]
    else:
        cueistop = kwargs['cueistop']
    if not(isinstance(cueistop, list) | isinstance(cueistop, tuple)):
        cueistop=[cueistop]
    
    # generate all possible combinations of composite face images
    cfdict = {}
    for (k1, k2) in _mkpair(imdict, pairstyle=kwargs['pairstyle']):
         
        for (mis, cue) in product(misali, cueistop): # both aligned and misaligned
            kwargs['misali']=mis
            kwargs['cueistop']=cue
            tmpcf, cf_fn = mkcf(imdict[k1], imdict[k2], **kwargs)
            cfdict[cf_fn]=tmpcf
    
    return cfdict

def concateims(imdict, pairstyle="perm", **kwargs):
    """Concatenate images in the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    pairstyle : str
        the style of the pairs. Defaults to "perm". Options are "perm" (`itertools.permutation`), "comb" (`itertools.combination`), "comb_repl" (`itertools.combinations_with_replacement`), "prod" (`itertools.product`) and "itself" (concatenate itself).
        
    Other Parameters
    ----------------
    axis : int
        the axis along which the images are concatenated. Defaults to 1, i.e., horizontally.
    sep : str
        the separator between the two images. Defaults to "-".
    padvalue : int
        padding value. Defaults to 0 (show as transparent if alpha channel exists).
    
    Returns
    -------
    im.image() instance
        the concatenated image.
    """
    
    # make sure the dictionary is flatten
    if not _isflatten(imdict):
        imdict = _flatten(imdict)
    
    # check the number of im
    nim = len(imdict)
    assert nim>1, f'There should be more than one im in imdict... (Now {nim})'
    
    # generate all possible combinations of concatenated images
    concatedict = {}
    for (k1, k2) in _mkpair(imdict, pairstyle=pairstyle):
        tmpconcate, concate_fn = concatenate(imdict[k1], imdict[k2], **kwargs)
        concatedict[concate_fn]=tmpconcate
    
    return concatedict

def _mkpair(imdict, pairstyle="perm"):
    """Make all possible pairs of images in the dictionary (based on pairstyle).

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    pairstyle : str, optional
        the style of the pairs. Defaults to "perm". Options are "perm" (`itertools.permutation`), "comb" (`itertools.combination`), "comb_repl" (`itertools.combinations_with_replacement`), "prod" (`itertools.product`) and "itself" (concatenate itself). Examples are as follows for a list (dictionary) of [1,2,3]:
        - perm: all possible permutations of the images. [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]
        - comb: all possible combinations of the images. [(1,2), (1,3), (2,3)]
        - comb_repl: all possible combinations of the images with replacement. [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        - prod: all possible products of the images. [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
        - itself: all possible pairs of the same image. [(1,1), (2,2), (3,3)]

    Raises
    ------
    ValueError
        if pairstyle is not supported.
        
    Returns
    -------
    generator
        all possible pairs of images in the dictionary.
    """
    # make sure the dictionary is flatten
    if not _isflatten(imdict):
        imdict = _flatten(imdict)
    
    # generate image pairs
    if pairstyle=="perm":
        pairs = permutations(sorted(imdict), 2)
    elif pairstyle=="comb_repl":
        pairs = combinations_with_replacement(sorted(imdict), 2)
    elif pairstyle=="comb":
        pairs = combinations(sorted(imdict), 2)
    elif pairstyle=="prod":
        pairs = product(sorted(imdict), repeat=2)
    elif pairstyle=="itself":
        pairs= [(k,k) for k in sorted(imdict)]
    else:
        raise ValueError(f'pairstyle "{pairstyle}" is not supported...')
    
    return pairs

def read(imdict):
    """read the images if read was False in dir()
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    """
    [v.read() for v in imdict.values()]

def save(imdict, **kwargs):
    """Save the image PIL.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.

    Other Parameters
    ----------------
    newfname : str
            strings to be added before the extension, by default ''
    newfolder : str
            folder name to replace the global directory or the last directory level, by default ''
    addfn : bool
            whether to add the newfname to the the original fname (instead of replacing it), by default True
    """
    [v.save(**kwargs) for v in imdict.values()]
    
def updateext(imdict, **kwargs):
    """Update the filename information.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    
    Other Parameters
    ----------------
    ext : str
        the new extension.
    """
    [v.updateext(**kwargs) for v in imdict.values()]
    
def torgba(imdict, **kwargs):
    """Convert the image to RGBA.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    """
    [v.torgba(**kwargs) for v in imdict.values()]
    
def grayscale(imdict, **kwargs):
    """Convert the image to gray-scale.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    """
    [v.grayscale(**kwargs) for v in imdict.values()] 
    
def rotate(imdict, **kwargs):
    """Rotate the image unclockwise.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    
    Other Parameters
    -----------------
    angle : float
        the angle to rotate the image. Defaults to 180.
    """
    [v.rotate(**kwargs) for v in imdict.values()]

def adjust(imdict, **kwargs):
    """Adjust the luminance and contrast of the image.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    lum : float
        the desired mean of the image. Defaults to None.
    rms : float
        the desired standard deviation of the image. Defaults to None.
    mask : np.array
        the mask for the image. Defaults to None.
    """
    [v.adjust(**kwargs) for v in imdict.values()]
    
def cropoval(imdict, **kwargs):
    """Crop the image with an oval shape.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.

    Other Parameters
    ----------------
    radius : tuple
        the radius of the oval. Defaults to (100,128).
    bgcolor : tuple
        the background color. Defaults to None.
    """
    [v.cropoval(**kwargs) for v in imdict.values()]
    
def croprect(imdict, **kwargs):
    """Crop the image with a rectangle box.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.

    Other Parameters
    ----------------
    box : tuple
        the box to crop the image. Defaults to None.
    """
    [v.croprect(**kwargs) for v in imdict.values()]

def resize(imdict, **kwargs):
    """Resize the image.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    trgw : int
        the width of the target/desired stimuli.
    trgh : int
        the height of the target/desired stimuli.
    ratio : float
        the ratio to resize the image. Defaults to 0.
    newfolder : str
        the folder to save the resized image. Defaults to None.
    """
    [v.resize(**kwargs) for v in imdict.values()]
    
def pad(imdict, **kwargs):
    """
    Add padding to the image/stimuli.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    
    Other Parameters
    ----------------
    trgw : int
        the width of the target/desired stimuli. 
    trgh : int
        the height of the target/desired stimuli.
    padvalue : int
        padding value. Defaults to 0 (show as transparent if alpha channel exists).
    top : bool
        padding more to top if needed. Defaults to True.
    left : bool
        padding more to left if needed. Defaults to True.
    padalpha : int
        the transparent color. Defaults to -1, i.e., not to force it to transparent.
    extrafn : str
        the string to be added to the filename. Defaults to '_pad'.
    """
    [v.pad(**kwargs) for v in imdict.values()]

def mkboxscr(imdict, **kwargs):
    """Make box scrambled stimuli.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    nBoxW : int
        the number of boxes in width. Defaults to 10.
    nBoxH : int
        the number of boxes in height. Defaults to 16.
    pBoxW : int
        the width of a box. Defaults to 0.
    pBoxH : int
        the height of a box. Defaults to 0.
    pad : bool
        whether to add padding to the image. Defaults to False.
    padcolor : int
        the padding color. Defaults to 0.
    padalpha : int
        the padding alpha. Defaults to -1.
    """
    [v.mkboxscr(**kwargs) for v in imdict.values()]
    
def mkphasescr(imdict, **kwargs):
    """Make phase scrambled stimuli.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    rms : float
        the desired RMS of the image. Defaults to 0.3.
    """
    [v.mkphasescr(**kwargs) for v in imdict.values()]
    
def sffilter(imdict, **kwargs):
    """Apply spatial frequency filter to the image.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    rms : float
        the desired RMS of the image. Defaults to 0.3.
    maxvalue : int
        the maximum value of the image. Defaults to 255.
    sffilter : str
        the spatial frequency filter. Defaults to 'low'.
    cutoff : float
        the cutoff frequency. Defaults to 0.05.
    n : int
        the order of the filter. Defaults to 10.
    """
    [v.sffilter(**kwargs) for v in imdict.values()]

