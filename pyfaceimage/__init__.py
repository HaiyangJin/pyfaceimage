"""
Tools for dealing with images in dictionary.
"""

from .im import (image)
from .multim import (mkcf, concatenate)
from .utilities import (exp_design_builder, radial_gaussian)
from .exps import (mk_cf_design, mkcf_prf)

__all__ = ['image', 
           'mkcf',
           'concatenate',
           'exp_design_builder',
           'radial_gaussian',
           'mk_cf_design',
           'mkcf_prf']

import os, copy, random, glob, warnings
import numpy as np
import pandas as pd
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

def stimlist(imdict, sep='/'):
    """Get the list of image names (keys) in the dictionary with a randomize assigned integer key.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.  
    sep : str, optional
        a string to be used to separate the key in flatten dictionary into the subdirectory and image name, by default '/'

    Returns
    -------
    dict
        A dictionary of image names (keys) in the dictionary with a randomize assigned integer key.
    """
       
    # make sure the dictionary is not flatten
    if _isflatten(imdict):
        imdict = _nested(imdict)
    
    imgdict = {}
    for k0,v0 in imdict.items():
        
        shuffled_keys = random.sample(list(range(len(v0))), len(v0))
        thiskeydict = {shuffled_keys[i]: k for i, k in enumerate(v0.keys())}
        imgdict.update({k0:thiskeydict})
    
    shuffled_groups = random.sample(list(range(len(imdict))), len(imdict))
    groupdict = {shuffled_groups[i]: k for i, k in enumerate(imdict.keys())}
    
    return groupdict, imgdict

def updatekey(imdict, key='fnonly'):
    """Update the key of the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    key : str, optional
        which attributes to be used as the key, by default 'fnonly'

    Returns
    -------
    dict
        A dictionary of image() instances with updated key.
    """
    return {getattr(v, key):v for v in imdict.values()}
    
    
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


def standardize(imdict, clip=2, matkey='mat'):
    """Standardize the luminance of the images in the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    clip : int, optional
        the number of standard deviations to clip, by default 2
    matkey : str, optional
        the key to be used for the matrix, by default 'mat'
    """
    # standardize each image separately
    [v.stdmat(clip=clip, matkey=matkey) for v in imdict.values()]

    # grand min and max
    grand_min = min([np.min(getattr(v, matkey)) for v in imdict.values()])
    grand_max = max([np.max(getattr(v, matkey)) for v in imdict.values()])
        
    # Compute the grand normalized images
    
    [v.rescale(range=[grand_min,grand_max], matkey=matkey) for v in imdict.values()]
    
    
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

def touch(path, touchfolder="touch"):
    """Touch empty files for the directory.

    Parameters
    ----------
    path : str or dict
        path to be touched.
        A dictionary of image() instances.
    touchfolder : str, optional
        the new folder name for saving the empty files.
    """
    # if path is not a dictionary, it will be treated as a path
    if not isinstance(path, dict):
        imdict = dir(path=path)
    else:
        imdict = path
        
    # make sure the dictionary is flatten
    if not _isflatten(imdict):
        imdict = _flatten(imdict)
    
    # touch the directory
    [v.touch(touchfolder) for v in imdict.values()]
            

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
    -------
    A dictionary of im.image() instance
        the composite face stimuli as a im.image() instance.
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


def mk_cf_designs(imdict, **kwargs):
    """Make composite face task design matrix for each face group in `imdict`.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
        
    Other Parameters
    ----------------
    isTopCued : list
        whether the top half is cued. Defaults to `[1]`.
    isCongruent : list
        whether the top and bottom halves are congruent. Defaults to `[0, 1]`.
    isAligned : list
        whether the top and bottom halves are aligned. Defaults to `[0, 1]`.
    isCuedSame : list
        whether the cued half is the same as the target half. Defaults to `[0, 1]`.
    studyIsAligned : int
        whether the study faces are always aligned. Defaults to `1`.
    faceselector : list
        the face selector for the composite faces. Defaults to the default selector (`default_selector`).
    cue_str : list
        the cue strings. Defaults to `['bot', 'top']`.
    con_str : list
        the congruency strings. Defaults to `['inc', 'con']`.
    ali_str : list
        the alignment strings. Defaults to `['mis', 'ali']`.
    ca_str : list
        the correct answer strings. Defaults to `['dif', 'sam']`.
    task : str
        the task name. Defaults to `'CCF'`.
    cf_sep : str
        the separator within the composite face names. Defaults to `'_'`.
    cfs_sep : str
        the separator between the study and test faces. Defaults to `''`.
    showlevels : bool
        whether to show the levels of the independent variables. Defaults to `False`.

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame containing the composite face task design matrix.
    """
    
    defaultKwargs = {'is_rand': False,
                     'showlevels': True}
    kwargs = {**defaultKwargs, **kwargs}
    
    # make sure the dictionary is nested
    if _isflatten(imdict):
        imdict = _nested(imdict)

    # make composite face list for each face group
    cfdict = dict()
    for k,v in imdict.items():
        kwargs['cf_group'] = k
        cfdict[k] = mk_cf_design(v, **kwargs)
    
    # concatenate the composite face list    
    cf_designs = pd.concat(cfdict.values(), ignore_index=True)
    
    # randomize the design
    if kwargs['is_rand']:
        cf_designs = cf_designs.sample(frac=1).reset_index(drop=True)
    
    # add trial numbers
    cf_designs['trial'] = [x+1 for x in range(len(cf_designs))]
    columns = ['trial'] + [col for col in cf_designs.columns if col != 'trial']
    cf_designs = cf_designs[columns]

    return cf_designs


def concateims(imdict, pairstyle="perm", sep='/', **kwargs):
    """Concatenate images in the dictionary.

    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    pairstyle : str, optional
        the style of the pairs. Defaults to "perm". Options are "perm" (`itertools.permutation`), "comb" (`itertools.combination`), "comb_repl" (`itertools.combinations_with_replacement`), "prod" (`itertools.product`) and "itself" (concatenate itself).
    sep : str, optional
        a string to be used to concatnate the subdirectory and image name, which is used as the key for flattening the dictionary, by default '/'. If sep is None, the dictionary will not be flatten.
        
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
    A dictionary of im.image() instance
        the concatenated image.
    """
    # make sure the dictionary is flatten
    if len(set([im.group for im in imdict.values()])) > 1 and bool(sep):
        imdict_nested = _nested(imdict, sep=sep)
        
        concatedict_nested = {}
        for k,v in imdict_nested.items():
            concatedict_nested[k] = _concateims(v, pairstyle=pairstyle, **kwargs)
        
        concatedict = _flatten(concatedict_nested, sep=sep)
            
    else:
        concatedict = _concateims(imdict, pairstyle=pairstyle, **kwargs)
            
    return concatedict


def _concateims(imdict, pairstyle="perm", **kwargs):
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
    
def imsave(imdict, **kwargs):
    """Save the image mat.

    Parameters
    ----------
    newfname : str, optional
        strings to be added before the extension, by default ''
    newfolder : str, optional
        folder name to replace the global directory or the last directory level, by default ''
    addfn : bool, optional
        whether to add the newfname to the the original fname (instead of replacing it), by default True
    kwargs : dict, optional
        keyword arguments for matplotlib.pyplot.imsave(), by default {}
    """
    [v.imsave(**kwargs) for v in imdict.values()]

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
    
def addbg(imdict, **kwargs):
    """Add background to the image.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    
    Other Parameters
    ----------------
    bgcolor : tuple
        the background color. Defaults to (255,255,255).
    """
    [v.addbg(**kwargs) for v in imdict.values()]
    
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
    radius : tuple, optional
            the radius of the oval. Defaults to (100,128).
        position : tuple, optional
            the position of the oval. Defaults to None.
        bgcolor : tuple, optional
            the background color. Defaults to None. For instance, if bgcolor is (255, 255, 255, 255), the output image is not transparent; if bgcolor is (255, 255, 255, 0), the output image is transparent.
        crop : bool, optional
            whether to crop the image to the oval (true) or keep its original dimension. Defaults to True.
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
    
def filter(imdict, **kwargs):
    """Apply spatial frequency filter to the image. It is suggested to use this function to generate the filtered images and then standarize them with `standardize(imdict, clip=2)`.
    
    Parameters
    ----------
    imdict : dict
        A dictionary of image() instances.
    
    Other Parameters
    ----------------
    filter : str
            the filter type. Defaults to 'low'. Other option is 'high'.
    vapi : int
        visual angle per image. Defaults to 5.
    cutoff : int
        cycles per image (width) or cycles per degree if vapi>0. Defaults to 8.
    clip : int
        the clip value. Defaults to 0, i.e., no clipping or normalization will be applied. If not 0, standardization will be applied to each image (i.e., its LSF, HSF, FS versions) separately.
    """
    # apply the filter to each image (but without standardization)  
    [v.filter(**kwargs) for v in imdict.values()]

