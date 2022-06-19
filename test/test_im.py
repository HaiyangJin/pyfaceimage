
import os
import pyfacestim as fim
import numpy as np

# create instances
stim1 = fim.stim('test/teststim/test_stim/l1.png')
stim1_ = fim.stim('l1.png', 'test/teststim/test_stim')

# read images
stim1.read()
stim1_.read()
assert np.array_equal(stim1.mat, stim1_.mat)

#######################

# save another file
stim1.imsave('_tmp1')
stim1_.save('mat', '_tmp2')
os.remove(stim1.outfile)
os.remove(stim1_.outfile)

stim1.imshow()
stim1_.show()

# tests
stim1.file
stim1.fn
stim1.fnonly
stim1.ext
stim1.dir
stim1.group
stim1.isfile

assert(stim1.mat.shape[2]==4)
assert(stim1.rgbmat.shape[2]==3)
assert(len(stim1.amat.shape)==2)

stim1.dims

# pad
stim1.pad(500, 500)
stim1.imshow()
stim1.show()

# box-scrambled
stim1_.mkboxscr(nBoxX=20, nBoxY=20, makeup=1)
stim1_.imsave('_bs1', 'bs')
os.remove(stim1_.outfile)
stim1_.imshow('bs')
stim1_.show('bs')

stim1.mkboxscr(nBoxX=20, nBoxY=20)
stim1.imsave('_bs1', 'bs')
os.remove(stim1.outfile)
stim1.imshow('bs')
stim1.show('bs')

# resize
stim1_.resize(width=600)
stim1_.imshow('re')
stim1_.imshow('re')


