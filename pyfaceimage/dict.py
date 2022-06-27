import os
import random
import copy

from pyfaceimage import im


def dir(path='.', imgtype='.png', read=True):
    
    stimdict = {os.path.splitext(f)[0]:im.image(f, path, read) for f in os.listdir(path) if f.endswith(imgtype)}
    
    return stimdict

def read(stimdict):
    # read the images if read was False in dir()
    [v.read() for v in stimdict.values()]
    
    
def deepcopy(stimdict):
    return copy.deepcopy(stimdict)

def checksample(stimdict, n=1, return_value=True):
    
    # random select one image from dictionary to check the output
    assert n<=len(stimdict)
    
    samples = random.sample(sorted(stimdict.items()), n)
    
    if return_value:
        out = [v[1] for v in samples]
        if n==1:
            out = out[0]
    else:
        out = samples
    
    return out


def mkboxscr(stimdict, **kwargs):
    
    defaultKwargs = {'nBoxW':10, 'nBoxH':16, 
                     'pBoxW':0, 'pBoxH':0, 
                     'pad': False, 'padcolor':0, 'padalpha': -1}
    kwargs = {**defaultKwargs, **kwargs}
    
    [v.mkboxscr(**kwargs) for v in stimdict.values()]
    
    
def save(stimdict, extrafn='', extrafolder='.', **kwargs):
    [v.save(extrafn, extrafolder, **kwargs) for v in stimdict.values()]
    
    
def resize(stimdict, **kwargs):
    
    [v.resize(**kwargs) for v in stimdict.values()]
