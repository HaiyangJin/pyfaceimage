import os
import random
import copy
from itertools import permutations

from pyfaceimage import im, multipleim


def dir(path='.', imgtype='.png', read=True):
    
    imdict = {os.path.splitext(f)[0]:im.image(f, path, read) for f in os.listdir(path) if f.endswith(imgtype)}
    print(f'Find {len(imdict)} files...')
    
    return imdict

def deepcopy(imdict):
    return copy.deepcopy(imdict)

def checksample(imdict, n=1, return_value=True):
    
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
    
    defaultKwargs = {'misali':[0,0.5]}
    kwargs = {**defaultKwargs, **kwargs}
    misali = kwargs['misali']
    if not(isinstance(misali, list) | isinstance(misali, tuple)):
        misali=[misali]
    
    nim = len(imdict)
    assert nim>1, f'There should be more than one im in imdict... (Now {nim})'
    
    cfdict = {}
    for (k1, k2) in permutations(sorted(imdict), 2):
         
        for mis in misali: # both aligned and misaligned
            kwargs['misali']=mis
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
    
    

    
    

