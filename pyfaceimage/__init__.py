"""
Tools for dealing with images in dictionary.
"""

from .im import (image)
from .multim import (mkcf)
from .utilities import (radial_gradient)

__all__ = ['image', 
           'mkcf',
           'radial_gradient']

import os, copy, random, glob, warnings
from itertools import permutations, product

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
        A dictionary of images in the path and subdirs.
    """
    
    # list all files in the path
    imdict_root = _dir(path, imwc, read)
    [im.setgroup('path') for im in imdict_root.values()] # update group info as 'path'    
    
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
    imdict = {os.path.basename(f).split('.')[0]:image(f, read) 
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


def _tonested(imdict_flat, sep='/'):
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


def deepcopy(imdict):
    return copy.deepcopy(imdict)

def sample(imdict, n=1, return_value=True):
    
    # random select one image from dictionary to check the output
    assert n<=len(imdict)
    samples = random.sample(sorted(imdict.items()), n)
    
    if return_value:
        out = [v[1] for v in samples]
        if n==1:
            out = out[0]
    else:
        out = samples
    
    return out

def mkcfs(imdict,  **kwargs):
    
    defaultKwargs = {'misali':[0,0.5], 'showcue':False, 'cueistop': True}
    kwargs = {**defaultKwargs, **kwargs}
    
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
    
    cfdict = {}
    for (k1, k2) in permutations(sorted(imdict), 2):
         
        for (mis, cue) in product(misali, cueistop): # both aligned and misaligned
            kwargs['misali']=mis
            kwargs['cueistop']=cue
            tmpcf = mkcf(imdict[k1], imdict[k2], **kwargs)
            cfdict[tmpcf.fnonly]=tmpcf
    
    return cfdict
    
def adjust(imdict, **kwargs):
    [v.adjust(**kwargs) for v in imdict.values()]

def read(imdict):
    # read the images if read was False in dir()
    [v.read() for v in imdict.values()]

def save(imdict, **kwargs):
    [v.save(**kwargs) for v in imdict.values()]
    
def torgba(imdict, **kwargs):
    [v.torgba(**kwargs) for v in imdict.values()]
    
def grayscale(imdict, **kwargs):
    [v.grayscale(**kwargs) for v in imdict.values()] 
    
def cropoval(imdict, **kwargs):
    [v.cropoval(**kwargs) for v in imdict.values()]
    
def croprect(imdict, **kwargs):
    [v.croprect(**kwargs) for v in imdict.values()]

def resize(imdict, **kwargs):
    [v.resize(**kwargs) for v in imdict.values()]
    
def pad(imdict, **kwargs):
    [v.pad(**kwargs) for v in imdict.values()]

def mkboxscr(imdict, **kwargs):
    [v.mkboxscr(**kwargs) for v in imdict.values()]
    
def sffilter(imdict, **kwargs):
    [v.sffilter(**kwargs) for v in imdict.values()]
    
def mkphasescr(imdict, **kwargs):
    [v.mkphasescr(**kwargs) for v in imdict.values()]
    
    

