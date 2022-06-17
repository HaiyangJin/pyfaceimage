"""
Tools used to read or write images.
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


# Get the stimulus list
def stimdir(stimpath, imtype='png', subdir=True):
    """
    Get the stimlus list in <stimpath>.

    Args:
        stimpath (str): path where the stimuli are;
        imtype (str, optional): image type. Defaults to 'png'.
        subdir (bool, optional): whether check the subdirectory within <stimpath>. Defaults to True.

    Returns:
        stimlist (list): list of stimulus list in each folder or sub-folder.
        paths (list): list of folder names. 
    """
    
    # in the main folder
    stimlist = [f for f in os.listdir(stimpath) if f.endswith(imtype)]
    folders = ['.']
    
    if (subdir):
        subfolders = [g for g in os.listdir(stimpath) if os.path.isdir(os.path.join(stimpath, g))]
        
        sublist=[]
        for isub in range(len(subfolders)):
            sublist.append([f for f in os.listdir(os.path.join(stimpath, subfolders[isub])) if f.endswith(imtype)])
        
        if len(stimlist) > 0:
            folders = folders + subfolders
            stimlist = [stimlist] + sublist
        else:
            folders = subfolders
            stimlist = sublist
    
    # the full path information
    paths = [os.path.join(stimpath,f) for f in folders]
        
    return [stimlist, paths]

# read the files
def readdir(stimlist, paths=None, stimdirout=True):
    
    if (len(stimlist) == 2) & stimdirout:
        # assume it is the output of stimdir()
        paths = stimlist[1]
        stimlist = stimlist[0]
    
    stimmat = []
    for i in range(len(paths)):
        tmpmat = ([mpimg.imread(os.path.join(paths[i], img)) for img in stimlist[i]])
        stimmat.append([np.moveaxis(np.tile(im,(3,1,1)),0,2) for im in tmpmat if len(im.shape)==2])
    
    return stimmat

# write/save the files
def writedir(stimmat, stimlist, outpath=None, extrastr=''):
    
    # make new directories
    
    if outpath is None:
        outpath = ''
    paths = stimlist[1]
    stimlist = stimlist[0]
        
    for i in range(len(stimlist)):
        fnlist=[os.path.join(outpath, paths[i], stimlist[i][im][:-4]+extrastr+stimlist[i][im][-4:]) for im in range(len(stimmat[i]))]
        [_imwrite(stimmat[i][im].copy(order='C'), fnlist[im]) for im in range(len(stimmat[i]))]
        
    return fnlist

def _imwrite(stimmat, fn):
    
    thedir = os.path.dirname(fn)
    if not os.path.isdir(thedir):
        os.makedirs(thedir)
    
    # save the image file
    print(stimmat.shape)
    plt.imsave(fn, stimmat)
    return fn

# make composite faces
def mkcf(stimlist, paths=None, outx=500, wlinex=3):
    cfnames = None
    return cfnames

# make box-scrambled faces
def mkboxscr(stimlist, *args, **kwargs):
    defaultKwargs = {'paths':None, 
                     'nBoxX':10, 'nBoxY':16, 
                     'pBoxX':0, 'pBoxY':0, 
                     'makeup': False, 'mkcolor':0}
    kwargs = {**defaultKwargs, **kwargs}
    
    stimmat = readdir(stimlist, kwargs['paths'])
    
    bsmats = []
    for ifolder in range(len(stimmat)):
        bsmats.append([_boxscrambled(orig, **kwargs) for orig in stimmat[ifolder]])
    
    return bsmats

# make box scrambled faces for each matrix
def _boxscrambled(stimmat, *args, **kwargs):
        
    [y, x] = stimmat.shape[0], stimmat.shape[1]
    
    # x and y pixels for each box
    _pBoxX = x/kwargs['nBoxX']
    _pBoxY = y/kwargs['nBoxY']
    
    if not (_pBoxX.is_integer() & _pBoxY.is_integer()):
        if kwargs['makeup']:
            # add complementary parts to top and right
            xnew = np.ceil(_pBoxX) * kwargs['nBoxX']
            ynew = np.ceil(_pBoxY) * kwargs['nBoxY']
            
            stimmat = np.hstack((np.vstack((np.ones((int(ynew-y),stimmat.shape[1],stimmat.shape[2]),dtype=np.uint8)*kwargs['mkcolor'], stimmat)),
                                 np.ones((int(ynew), int(xnew-x),stimmat.shape[2]),dtype=np.uint8)*kwargs['mkcolor']))
            
            _pBoxX = xnew/kwargs['nBoxX']
            _pBoxY = ynew/kwargs['nBoxY']
            
        else:
            raise Exception('Please input valid nBoxX and nBoxY. Or set "makeup" to True.')
        
    if kwargs['pBoxX']==0 | kwargs['pBoxY']==0:
        kwargs['pBoxX'] = int(np.ceil(_pBoxX))
        kwargs['pBoxY'] = int(np.ceil(_pBoxY))
    else:
        _nBoxX = x/kwargs['pBoxX']
        _nBoxY = x/kwargs['pBoxY']
        
        if not (_nBoxX.is_integer() & _nBoxY.is_integer()):
            if kwargs['makeup']:
                # add complementary parts to top and right
                xnew = np.ceil(_nBoxX) * kwargs['pBoxX']
                ynew = np.ceil(_nBoxY) * kwargs['pBoxY']
                
                stimmat = np.hstack((np.vstack((np.ones(ynew-y,stimmat.shape[1],stimmat.shape[2],dtype=np.uint8)*kwargs['mkcolor'], stimmat)),
                                    np.ones(stimmat.shape[0], xnew-x, stimmat.shape[2],dtype=np.uint8)*kwargs['mkcolor']))
                
                kwargs['nBoxX'] = x/kwargs['pBoxX']
                kwargs['nBoxY'] = x/kwargs['pBoxY']
                
            else:
                raise Exception('Please input valid pBoxX and pBoxY. Or set "makeup" to True.')
    
    [y, x] = stimmat.shape[0], stimmat.shape[1]
    assert(kwargs['nBoxX']*kwargs['pBoxX']==x)
    assert(kwargs['nBoxY']*kwargs['pBoxY']==y)

    # x and y for all boxes
    xys = list(product(range(0,x,kwargs['pBoxX']), range(0,y,kwargs['pBoxY'])))
    boxes = [stimmat[i[1]:(i[1]+kwargs['pBoxY']), i[0]:(i[0]+kwargs['pBoxX'])] for i in xys]
    # randomize the boxes
    bsboxes = np.random.permutation(boxes)
    # save as np.array
    bslist = [bsboxes[i:i+kwargs['nBoxX']] for i in range(0,len(bsboxes),kwargs['nBoxX'])]
    # bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
    bsmatm = np.asarray(bslist)
    if len(bsmatm.shape)==4:
        bsmatm = bsmatm[..., np.newaxis]
    print(bsmatm.shape)
    bsmat = np.moveaxis(bsmatm, [-1, 1], [0, -2]).reshape(-1, y, x)
    
    return np.squeeze(np.moveaxis(bsmat,0,2))




