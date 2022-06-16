import os
from pyfacestim import im
import matplotlib.pyplot as plt

stimlist1 = im.stimdir(os.path.join('demos', 'test1'), 'png')
stimlist10 = im.stimdir(os.path.join('demos', 'test1'), 'png', 0)

stimlist2 = im.stimdir(os.path.join('demos', 'test2'), 'png')

stimlist3 = im.stimdir(os.path.join('demos', 'test3'), 'png')

# read
stimmat1 = im.readdir(stimlist1)

bsstim = im.mkboxscr(stimlist1, paths=None, nxBox=10, nyBox=16)

# tmp = im.writedir(stimmat1, stimlist1, None, extrastr='scr')
tmp = im.writedir(bsstim, stimlist1, 'new', extrastr='scr1')

