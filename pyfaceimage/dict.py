import os
import random
import copy

from pyfaceimage import im


def dir(path='.', imgtype='.png', read=True):
    
    stimdict = {os.path.splitext(f)[0]:im.image(f, path, read) for f in os.listdir(path) if f.endswith(imgtype)}
    print(f'Find {len(stimdict)} files...')
    
    return stimdict

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



def read(stimdict):
    # read the images if read was False in dir()
    [v.read() for v in stimdict.values()]

def save(stimdict, **kwargs):
    [v.save(**kwargs) for v in stimdict.values()]
    
def grayscale(stimdict, **kwargs):
    [v.grayscale(**kwargs) for v in stimdict.values()] 
    
def cropoval(stimdict, **kwargs):
    [v.cropoval(**kwargs) for v in stimdict.values()]
    
def croprect(stimdict, **kwargs):
    [v.croprect(**kwargs) for v in stimdict.values()]

def resize(stimdict, **kwargs):
    [v.resize(**kwargs) for v in stimdict.values()]
    
def pad(stimdict, **kwargs):
    [v.pad(**kwargs) for v in stimdict.values()]

def mkboxscr(stimdict, **kwargs):
    [v.mkboxscr(**kwargs) for v in stimdict.values()]
    
def sffilter(stimdict, **kwargs):
    [v.sffilter(**kwargs) for v in stimdict.values()]
    
def mkphasescr(stimdict, **kwargs):
    [v.mkphasescr(**kwargs) for v in stimdict.values()]
    
    

    
    

