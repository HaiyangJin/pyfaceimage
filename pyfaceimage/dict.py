import os
import random

import pyfaceimage.im

def dir(path='.', imgtype='.png', read=True):
    
    stimdict = {os.path.splitext(f)[0]:pyfaceimage.im.image(f, path, read) for f in os.listdir(path) if f.endswith(imgtype)}
    
    return stimdict

def read(stimdict):
    # read the images if read was False in dir()
    [v.read() for v in stimdict.values()]
    
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
    
    defaultKwargs = {'nBoxX':10, 'nBoxY':16, 
                     'pBoxX':0, 'pBoxY':0, 
                     'makeup': False, 'mkcolor':0, 'mkalpha': None}
    kwargs = {**defaultKwargs, **kwargs}
    
    [v.mkboxscr(**kwargs) for v in stimdict.values()]
    
    
def save(stimdict, which='mat', extrastr='', **kwargs):
    [v.save(which, extrastr, **kwargs) for v in stimdict.values()]