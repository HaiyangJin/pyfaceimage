import os
import random
import copy
from itertools import permutations, product

from pyfaceimage import im, multipleim


def dir(path='.', imgtype='.png', read=True):
    
    imdict = {os.path.splitext(f)[0]:im.image(f, path, read) for f in os.listdir(path) if f.endswith(imgtype)}
    print(f'Found {len(imdict)} files...')
    
    return imdict

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
            tmpcf = multipleim.mkcf(imdict[k1], imdict[k2], **kwargs)
            cfdict[tmpcf.fnonly]=tmpcf
    
    return cfdict
    


def read(imdict):
    # read the images if read was False in dir()
    [v.read() for v in imdict.values()]

def save(imdict, **kwargs):
    [v.save(**kwargs) for v in imdict.values()]
    
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
    
    

    
    

