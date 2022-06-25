"""
The class to process single images.    
"""
    
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.image as mpimg
from itertools import product


class image:
    def __init__(self, file, dir='.', read=False):
        # make sure .file exists  os.getcwd()
        file = os.path.join(dir, file)
        assert os.path.isfile(file), f'Cannot find {file}...'
        self.file = file
        self._info()
         # default settings (for scrambled images)
        self.bsmat = None
        self.bsfile = None
        self.bspil = None
        self.remat = None
        self.refile = None
        self.repil = None
        self.psmat = None
        self.psfile = None
        self.pspil = None
        self.cmat = None
        self.cfile = None
        self.cpil = None
        
        if read:
            self.read()
            
    def _info(self):
        self.fn = os.path.basename(self.file)
        self.fnonly = os.path.splitext(self.fn)[0]
        self.ext = os.path.splitext(self.fn)[1]
        self.dir = os.path.dirname(self.file)
        self.group = os.path.split(self.dir)[1] # the upper-level folder name
        self.isfile = os.path.isfile(self.file)
        
    def read(self):
        self.pil = Image.open(self.file) # PIL.Image.open() 
        # potential useful functions
        # .filename .format, .mode, .split()
        self.mat = np.asarray(self.pil)
        self.update2pil()
        self.grayscale()        
        
    def imshow(self, which='mat', cmat=None):
        # for debugging purpose (check the mat)
        # it seems that .show() is more useful
        if which=='c':
            themat = cmat
        else:
            outs = self._whichmat()
            themat = outs[which]['mat']
        Image.fromarray(themat).show()
        
    def show(self, which='mat'):
        # for debugging purpose (check the PIL)
        outs = self._whichmat()
        outs[which]['pil'].show()

    def imsave(self, extrastr='', which='mat', **kwargs):
        # use matplotlib.pyplot.imsave() to save .mat
        outs = self._whichmat()
        self.outfile = os.path.splitext(outs[which]['file'])[0]+extrastr+os.path.splitext(outs[which]['file'])[1]
        
        if self.nlayer==1:
            outmat = outs[which]['mat'][:,:,0]
        else:
            outmat = outs[which]['mat']
            
        mpimg.imsave(self.outfile,outmat.copy(order='C'),**kwargs)
        
    def save(self, which='mat', extrastr='', **kwargs):
        # use PIL.Image.save() to save .pil
        outs = self._whichmat()
        im = outs[which]
        if not os.path.isdir(os.path.dirname(im['file'])):
            os.makedirs(os.path.dirname(im['file']))
        
        self.outfile = os.path.splitext(im['file'])[0]+extrastr+os.path.splitext(im['file'])[1]
        
        im['pil'].save(self.outfile, format=None, **kwargs)
        
    def _whichmat(self):
        outs = {'mat': {'mat':self.mat, 'file':self.file, 'pil':self.pil},
                'gs': {'mat':self.gsmat, 'file':self.gsfile, 'pil':self.gspil}, # gray scale
                're': {'mat':self.remat, 'file':self.refile, 'pil':self.repil}, # resize
                'bs': {'mat':self.bsmat, 'file':self.bsfile, 'pil':self.bspil}, # box-scrambled
                'ps': {'mat':self.psmat, 'file':self.psfile, 'pil':self.pspil}} # phase-scrambled
        return outs
    
    def update2pil(self):
        self.mat = np.asarray(self.pil)
        self.update()
        
    def update(self):
        self.dims = self.mat.shape
        self.ndim = len(self.dims)
        self.h, self.w, self.nlayer = self.dims
        self.torgba()
        
    def torgba(self):
        # convert pil to RGBA and save in .mat
        if len(self.mat.shape)==2: # Gray
            self.mat = self.mat[..., np.newaxis]
        assert len(self.mat.shape)==3
        
        nlayer = self.mat.shape[2]
        if nlayer==1: # Gray 'L'
            rgbmat = np.repeat(self.mat, 3, axis=2) # RGB
            amat = np.ones_like(rgbmat[:,:,0],dtype=np.uint8)*rgbmat.max() # alpha
        elif nlayer==2: # Gray 'LA'
            rgbmat = np.repeat(self.mat[:,:,0:1], 3, axis=2) # RGB
            amat = self.mat[:,:,-1] # alpha
        elif nlayer==3: # RGB
            rgbmat = self.mat # RGB
            amat = np.ones_like(rgbmat[:,:,0],dtype=np.uint8)*rgbmat.max() # alpha
        elif nlayer==4: # RGBA
            rgbmat = self.mat[:,:,0:3]
            amat = self.mat[:,:,-1]
            
        self.rgbmat = rgbmat
        self.amat = amat
        self.mat = np.concatenate((rgbmat, amat[..., np.newaxis]), axis=2) # RGBA
        
    def grayscale(self):
        self.gspil = ImageOps.grayscale(self.pil)
        self.gsmat = np.asarray(self.gspil)
        self.gsfile = os.path.join('.', self.dir, self.fnonly+'gray'+self.ext)
    
    def resize(self, **kwargs):
        # resize the image
        defaultKwargs = {'width': None, 'height': None, 'ratio':0}
        kwargs = {**defaultKwargs, **kwargs}
        
        if (kwargs['width'] is not None) & (kwargs['height'] is not None):
            (w, h) = (kwargs['width'], kwargs['height'])
        elif (kwargs['width'] is not None) & (kwargs['height'] is None):
            w = kwargs['width']
            h = int(w*self.h/self.w)
        elif (kwargs['width'] is None) & (kwargs['height'] is not None): 
            h = kwargs['height']
            w = int(h*self.w/self.h)
        elif kwargs['ratio']>0:
            w = int(self.w*kwargs['ratio'])
            h = int(self.h*kwargs['ratio'])
        else:
            raise 'Cannot determine the desired dimentions...'
        
        kwargs['size'] = (w,h)
        [kwargs.pop(k) for k in defaultKwargs.keys()] # remove unused keys
        
        # save re-sized images (information)
        self.repil = self.pil.resize(**kwargs)
        self.refile = os.path.join(str(w)+'_'+str(h), self.file)
        self.remat = np.asarray(self.repil)
        
    def pad(self, trgw, trgh, padvalue=0, top=True, left=True):
        """Add padding to the image/stimuli.

        Args:
            trgw (int): the width of the target/desired stimuli. 
            trgh (int): the height of the target/desired stimuli.
            padvalue (int, optional): padding value. Defaults to 0.
            top (bool, optional): padding more to top if needed. Defaults to True.
            left (bool, optional): padding more to left if needed. Defaults to True.
        """
        assert(trgw>=self.w)
        assert(trgh>=self.h)
        
        x1 = int(np.ceil(trgw-self.w)/2)
        x2 = trgw-self.w-x1
        y1 = int(np.ceil(trgh-self.h)/2)
        y2 = trgh-self.h-y1
        
        if top:
            htop, hbot = y1,y2
        else:
            htop, hbot = y2,y1
        if left:
            wleft, wright = x1,x2
        else:
            wleft, wright = x2,x1
            
        self.mat = np.hstack((
            np.ones((trgh,wleft,self.nlayer),dtype=np.uint8)*padvalue, 
            np.vstack((
                np.ones((htop,self.w,self.nlayer),dtype=np.uint8)*padvalue, 
                self.mat,
                np.ones((hbot,self.w,self.nlayer),dtype=np.uint8)*padvalue
            )),
            np.ones((trgh,wright,self.nlayer),dtype=np.uint8)*padvalue, 
        ))
        self.update()
        
    def mkboxscr(self, **kwargs):
        """Make box scrambled stimuli.
        """
        defaultKwargs = {'nBoxX':10, 'nBoxY':16, 
                     'pBoxX':0, 'pBoxY':0, 
                     'makeup': False, 'mkcolor':0, 'mkalpha': None}
        kwargs = {**defaultKwargs, **kwargs}
            
        # x and y pixels for each box
        _pBoxX = self.w/kwargs['nBoxX']
        _pBoxY = self.h/kwargs['nBoxY']
    
        if not ((_pBoxX.is_integer()) & (_pBoxY.is_integer())):
            if kwargs['makeup']:
                # add complementary parts (top, right, bottom, left)
                xnew = int(np.ceil(_pBoxX) * kwargs['nBoxX'])
                ynew = int(np.ceil(_pBoxY) * kwargs['nBoxY'])
                self.pad(trgw=xnew, trgh=ynew, padvalue=kwargs['mkcolor'])
            
                if kwargs['mkalpha'] is not None:
                    self.addalpha(amat=None, avalue=kwargs['mkalpha'])

                _pBoxX = xnew/kwargs['nBoxX']
                _pBoxY = ynew/kwargs['nBoxY']
                
            else:
                raise Exception('Please input valid nBoxX and nBoxY. Or set "makeup" to True.')
        
        if kwargs['pBoxX']==0 | kwargs['pBoxY']==0:
            kwargs['pBoxX'] = int(np.ceil(_pBoxX))
            kwargs['pBoxY'] = int(np.ceil(_pBoxY))
        
        _nBoxX = self.w/kwargs['pBoxX']
        _nBoxY = self.h/kwargs['pBoxY']
            
        if not ((_nBoxX.is_integer()) & (_nBoxY.is_integer())):
            if kwargs['makeup']:
                # add complementary parts to top and right
                xnew = int(np.ceil(_nBoxX) * kwargs['pBoxX'])
                ynew = int(np.ceil(_nBoxY) * kwargs['pBoxY'])
            
                self.pad(trgw=xnew, trgh=ynew, padvalue=kwargs['mkcolor'])
                            
                if kwargs['mkalpha']:
                    self.addalpha(amat=None, avalue=kwargs['mkalpha'])

                kwargs['nBoxX'] = self.w/kwargs['pBoxX']
                kwargs['nBoxY'] = self.h/kwargs['pBoxY']
                
            else:
                raise Exception('Please input valid pBoxX and pBoxY. Or set "makeup" to True.')
        
        self.update()
        assert(kwargs['nBoxX']*kwargs['pBoxX']==self.w)
        assert(kwargs['nBoxY']*kwargs['pBoxY']==self.h)

        # x and y for all boxes
        xys = list(product(range(0,self.w,kwargs['pBoxX']), range(0,self.h,kwargs['pBoxY'])))
        boxes = [self.mat[i[1]:(i[1]+kwargs['pBoxY']), i[0]:(i[0]+kwargs['pBoxX'])] for i in xys]
        # randomize the boxes
        bsboxes = np.random.permutation(boxes)
        # save as np.array
        bslist = [bsboxes[i:i+kwargs['nBoxX']] for i in range(0,len(bsboxes),kwargs['nBoxX'])]
        # bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
        bsmatm = np.asarray(bslist)
        if len(bsmatm.shape)==4:
            bsmatm = bsmatm[..., np.newaxis]
        bsmat = np.moveaxis(bsmatm, [-1, 1], [0, -2]).reshape(-1, self.h, self.w)
        
        # save box-scrambled images (and information)
        self.bsmat = np.squeeze(np.moveaxis(bsmat,0,2))
        self.bsfile = os.path.join('.', self.dir, self.fnonly+'_bscr'+self.ext)
        self.bspil = Image.fromarray(self.bsmat)
    
    def sffilter(self, **kwargs):
        # https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        import psychopy.filters
        
        defaultKwargs = {'rms':0.3, 'maxvalue':255, 'sffilter':'low',
                         'cutoff': 0.05, 'n': 10}
        kwargs = {**defaultKwargs, **kwargs}
        
        stdmat = self._stdim(self.gsmat, kwargs['rms'])
        img_freq = np.fft.fft2(stdmat)

        # calculate amplitude spectrum
        # img_amp = np.fft.fftshift(np.abs(img_freq))

        # # for display, take the logarithm
        # img_amp_disp = np.log(img_amp + 0.0001)

        # # rescale to -1:+1 for display
        # img_amp_disp = (
        #     ((img_amp_disp - np.min(img_amp_disp)) * 2) / 
        #     np.ptp(img_amp_disp)  # 'ptp' = range
        # ) - 1
        
        if kwargs['sffilter']=='low':
            # for generating blury images
            fsfilt = psychopy.filters.butter2d_lp(
                size=(self.w, self.h),
                cutoff=kwargs['cutoff'],
                n=kwargs['n']
            )
        elif kwargs['sffilter']=='high':      
             # for gernerating sharp images
            fsfilt = psychopy.filters.butter2d_hp(
                size=(self.w, self.h),
                cutoff=kwargs['cutoff'],
                n=kwargs['n']
            )
        else:
            raise 'Cannot identify the "sffilter" value...'
        
        img_filt = np.fft.fftshift(img_freq) * fsfilt.transpose()
        # convert back to an image
        img_new = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))
        # standardize the image to [-1, 1]
        img_new = self._stdim(img_new, kwargs['rms'])
        # convert the range to [0, 255]
        img_new = (img_new+1)/2*kwargs['maxvalue']
        self.flmat = img_new
    
    
    def _stdim(self, mat, rms=0.3):
        # standardize the image (the range of output should be -1,1)
        # make the standard deviation to be the desired RMS
        mat = (mat - np.mean(mat))/np.std(mat) * rms
        
        # there may be some stray values outside of the presentable range; convert < -1
        # to -1 and > 1 to 1
        mat = np.clip(mat, a_min=-1.0, a_max=1.0)
        return mat
    
    
    def mkphasescr(self, **kwargs):
        defaultKwargs = {'rms':0.3}
        kwargs = {**defaultKwargs, **kwargs}
        
        # make a random phase
        randphase = np.angle(np.fft.fft2(np.random.rand(self.h, self.w)))
        
        tmpmat = np.empty(self.rgbmat.shape)
        tmpmat[:] = np.NaN
        
        for i in range(3):
            img_freq = np.fft.fft2(self._stdim(self.rgbmat[:,:,i], kwargs['rms']))
            amp = np.abs(img_freq)
            phase = np.angle(img_freq) + randphase
            outimg = np.real(np.fft.ifft2(amp * np.exp(np.sqrt(-1+0j)*phase)))
            stdimg1= self._stdim(outimg, kwargs['rms'])
            tmpmat[:,:,i] = (stdimg1+1)/2*255
        
        self.psmat = np.uint8(tmpmat)
        self.psfile = os.path.join('.', self.dir, self.fnonly+'pscr'+self.ext)
        self.pspil = Image.fromarray(self.psmat)
        