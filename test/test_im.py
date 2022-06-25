
import os
import pyfaceimage as fim
import numpy as np

###############

# create instances
im1 = fim.image('test/teststim/test_stim/l1.png')
im1_ = fim.image('l1.png', 'test/teststim/test_stim')

# read images
im1.read()
im1_.read()
assert np.array_equal(im1.mat, im1_.mat)

#######################

# save another file
im1.imsave('_tmp1')
im1_.save('mat', '_tmp2')
os.remove(im1.outfile)
os.remove(im1_.outfile)

im1.imshow()
im1_.show()

# tests
im1.file
im1.fn
im1.fnonly
im1.ext
im1.dir
im1.group
im1.isfile

assert(im1.mat.shape[2]==4)
assert(im1.rgbmat.shape[2]==3)
assert(len(im1.amat.shape)==2)

im1.dims

# pad
im1.pad(500, 500)
im1.imshow()
im1.show()

# box-scrambled
im1_.mkboxscr(nBoxX=20, nBoxY=20, makeup=1)
im1_.imsave('_bs1', 'bs')
os.remove(im1_.outfile)
im1_.imshow('bs')
im1_.show('bs')

im1.mkboxscr(nBoxX=20, nBoxY=20)
im1.imsave('_bs1', 'bs')
os.remove(im1.outfile)
im1.imshow('bs')
im1.show('bs')

# resize
im1_.resize(width=600)
im1_.imshow('re')
im1_.imshow('re')

##### spatial frequency
face = fim.image('test/me.jpeg')
face.read()

face.sffilter()
face.imshow('c', face.flmat)

face.sffilter(sffilter='high')
face.imshow('c', face.flmat)


### phase-scrambled
face.mkphasescr()
face.imshow('ps')
face.imshow('c', face.psmat)
