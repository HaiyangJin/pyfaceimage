
import os
import pyfaceimage as fim
import numpy as np

###############

# create instances
color = fim.image('test/im/color.png', read=True)
rgba = fim.image('test/im/rgba.png', read=True)
rgb = fim.image('test/im/rgb.png', read=True)
ga = fim.image('test/im/ga.png', read=True)
g = fim.image('test/im/g.png', read=True)

#######################
# tests
assert rgba.file=='./test/im/rgba.png'
assert rgba.fn=='rgba.png'
assert rgba.fnonly=='rgba'
assert rgba.ext=='.png'
assert rgba.dir=='./test/im'
assert rgba.group=='im'
assert rgba.isfile==1

assert rgba.nlayer==4
assert rgb.nlayer==3
assert ga.nlayer==2
assert g.nlayer==0

########################
# pad
rgba_pad = rgba.deepcopy()
rgba_pad.pad(trgw=600, trgh=350)
rgba_pad.show()

rgb_pad = rgb.deepcopy()
rgb_pad.pad(trgw=600, trgh=350)
rgb_pad.show()

rgb_pad = rgb.deepcopy()
rgb_pad.pad(trgw=600, trgh=350, padvalue=255)
rgb_pad.show()

ga_pad = ga.deepcopy()
ga_pad.pad(trgw=600, trgh=350)
ga_pad.show()

g_pad = g.deepcopy()
g_pad.pad(trgw=600, trgh=350)
g_pad.show()

g_pad = g.deepcopy()
g_pad.pad(trgw=600, trgh=350, padvalue=255)
g_pad.show()

# force transparent
rgba_pad = rgba.deepcopy()
rgba_pad.pad(trgw=600, trgh=350, alphavalue=255)
rgba_pad.show()

rgb_pad = rgb.deepcopy()
rgb_pad.pad(trgw=600, trgh=350, alphavalue=255)
rgb_pad.show()

ga_pad = ga.deepcopy()
ga_pad.pad(trgw=600, trgh=350, alphavalue=255)
ga_pad.show()

g_pad = g.deepcopy()
g_pad.pad(trgw=600, trgh=350, alphavalue=255)
g_pad.show()

########################
# box-scrambled (n)
rgba_bs = rgba.deepcopy()
rgba_bs.mkboxscr()
rgba_bs.show()

rgb_bs = rgb.deepcopy()
rgb_bs.mkboxscr()
rgb_bs.show()

ga_bs = ga.deepcopy()
ga_bs.mkboxscr()
ga_bs.show()

g_bs = g.deepcopy()
g_bs.mkboxscr()
g_bs.show()

# box-scrambled (p)
rgba_bs = rgba.deepcopy()
rgba_bs.mkboxscr(pBoxW=10, pBoxH=8)
rgba_bs.show()

rgb_bs = rgb.deepcopy()
rgb_bs.mkboxscr(pBoxW=10, pBoxH=8)
rgb_bs.show()

ga_bs = ga.deepcopy()
ga_bs.mkboxscr(pBoxW=10, pBoxH=8)
ga_bs.show()

g_bs = g.deepcopy()
g_bs.mkboxscr(pBoxW=10, pBoxH=8)
g_bs.show()

# box-scrambled (n with makeup)
nBoxW, nBoxH = 21, 20

rgba_bs = rgba.deepcopy()
rgba_bs.mkboxscr(nBoxW=nBoxW, nBoxH=nBoxH, pad=True)
rgba_bs.show()
assert rgba_bs.h == np.ceil(rgba.h/nBoxH)*nBoxH
assert rgba_bs.w == np.ceil(rgba.w/nBoxW)*nBoxW

rgb_bs = rgb.deepcopy()
rgb_bs.mkboxscr(nBoxW=nBoxW, nBoxH=nBoxH, pad=True)
rgb_bs.show()

ga_bs = ga.deepcopy()
ga_bs.mkboxscr(nBoxW=nBoxW, nBoxH=nBoxH, pad=True)
ga_bs.show()

g_bs = g.deepcopy()
g_bs.mkboxscr(nBoxW=nBoxW, nBoxH=nBoxH, pad=True)
g_bs.show()

# box-scrambled (p with makeup)
pBoxW, pBoxH = 9, 13
rgba_bs = rgba.deepcopy()
rgba_bs.mkboxscr(pBoxW=pBoxW, pBoxH=pBoxH, pad=True)
rgba_bs.show()
assert rgba_bs.h == np.ceil(rgba.h/pBoxH)*pBoxH
assert rgba_bs.w == np.ceil(rgba.w/pBoxW)*pBoxW

########################
# resize
rgba_resize = rgba.deepcopy()
rgba_resize.resize(width=300)
rgba_resize.show()
assert rgba_resize.h==384
assert rgba_resize.nlayer==4

rgb_resize = rgb.deepcopy()
rgb_resize.resize(height=384)
rgb_resize.show()
assert rgb_resize.w==300
assert rgb_resize.nlayer==3

ga_resize = ga.deepcopy()
ga_resize.resize(ratio=1.5)
ga_resize.show()
assert ga_resize.w==300
assert ga_resize.nlayer==2

g_resize = g.deepcopy()
g_resize.resize(ratio=1.5)
g_resize.show()
assert g_resize.w==300
assert g_resize.nlayer==0


########################
##### spatial frequency
color.show()

colorl = color.deepcopy()
colorl.sffilter()
colorl.show()

colorh = color.deepcopy()
colorh.sffilter(sffilter='high')
colorh.show()

########################
### phase-scrambled
color_ps = color.deepcopy()
color_ps.mkphasescr()
color_ps.show()

rgba_ps = rgba.deepcopy()
rgba_ps.mkphasescr()
rgba_ps.show()

rgb_ps = rgb.deepcopy()
rgb_ps.mkphasescr()
rgb_ps.show()

ga_ps = ga.deepcopy()
ga_ps.mkphasescr()
ga_ps.show()

g_ps = g.deepcopy()
g_ps.mkphasescr()
g_ps.show()
