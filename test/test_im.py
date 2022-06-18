
import os
import pyfacestim as fim
import numpy as np

# create instances
stim1 = fim.stim('test/teststim/test_stim/l1.png')
stim1_ = fim.stim('l1.png', 'test/teststim/test_stim')

# read images
stim1.imread()
stim1_.imread()
assert np.array_equal(stim1.mat, stim1_.mat)

# save another file
stim1.imsave('_tmp1')
stim1_.imsave('_tmp2')
os.remove(stim1.outfile)
os.remove(stim1_.outfile)

# tests
stim1.file
stim1.fn
stim1.fnonly
stim1.ext
stim1.dir
stim1.group
stim1.isfile

stim1.dims

# reduce and add alyers
stim1.updatelayer(1)

stim1.imsave('_onelayer', cmap='gray')
os.remove(stim1.outfile)

stim1.updatelayer()
assert (np.array_equal(stim1.mat, stim1_.mat[:,:,0:3]))

# alphpa
stim1.addalpha()

# pad
stim1.pad(500, 500)

# box-scrambled
stim1_.mkboxscr(nBoxX=20, nBoxY=20, makeup=1)
stim1_.imsave('_bs1', 'bsmat')
os.remove(stim1_.outfile)

stim1.mkboxscr(nBoxX=20, nBoxY=20)
stim1.imsave('_bs2', 'bsmat')
os.remove(stim1.outfile)

