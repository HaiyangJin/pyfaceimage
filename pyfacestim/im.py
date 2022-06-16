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
        stimmat.append([mpimg.imread(os.path.join(paths[i], img)) for img in stimlist[i]])
        
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
    plt.imsave(fn, stimmat)
    return fn

# make composite faces
def mkcf(stimlist, paths=None, outx=500, wlinex=3):
    cfnames = None
    return cfnames

# make box-scrambled faces
def mkboxscr(stimlist, paths=None, nxBox=10, nyBox=16):
    
    stimmat = readdir(stimlist, paths)
    
    bsmats = []
    for ifolder in range(len(stimmat)):
        bsmats.append([_boxscrambled(orig, nxBox, nyBox) for orig in stimmat[ifolder]])
    
    return bsmats

# make box scrambled faces for each matrix
def _boxscrambled(stimmat, nxBox, nyBox):
        
    [y, x] = stimmat.shape[0], stimmat.shape[1]
    
    # x and y pixels for each box
    boxXPer = int(x/nxBox)
    boxYPer = int(y/nyBox)
    assert(boxXPer==x/nxBox)
    assert(boxYPer==y/nyBox)
    
    # x and y for all boxes
    xys = list(product(range(0,x,boxXPer), range(0,y,boxYPer)))
    boxes = [stimmat[i[1]:(i[1]+boxYPer), i[0]:(i[0]+boxXPer)] for i in xys]
    # randomize the boxes
    bsboxes = np.random.permutation(boxes)
    # save as np.array
    bslist = [bsboxes[i:i+nxBox] for i in range(0,len(bsboxes),nxBox)]
    bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
    
    return np.moveaxis(bsmat,[-2,-1], [0,1])




