"""
The class to process single images.    
"""
    
import os
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.image as mpimg
from itertools import product
import warnings
import copy

class image:
    def __init__(self, file, dir=os.getcwd(), read=False):
        # make sure .file exists  os.getcwd()
        if not file.startswith(os.sep):
            file = os.path.join(dir, file)
        assert os.path.isfile(file), f'Cannot find {file}...'
        self.file = file
        self._updatefromfile() # update filename information
        if read:
            self.read()
        
    def read(self):
        self._repil(Image.open(self.file)) # PIL.Image.open() 
        # potential useful functions
        # .filename .format, .mode, .split()
        
    def imshow(self):
        # for debugging purpose (check the mat)
        # it seems that .show() is more useful
        Image.fromarray(self.mat).show()
        
    def show(self):
        # for debugging purpose (check the PIL)
        self.pil.show()

    def imsave(self, extrafn='', extrafolder='.', **kwargs):
        # use matplotlib.pyplot.imsave() to save .mat
        self.outfile = self._updatefile(extrafn, extrafolder)
        
        # make dir(s)
        if not os.path.isdir(os.path.dirname(self.outfile)):
            os.makedirs(os.path.dirname(self.outfile))
            
        if self.nlayer==1:
            outmat = self.mat[:,:,0]
        else:
            outmat = self.mat
            
        mpimg.imsave(self.outfile,outmat.copy(order='C'),**kwargs)
        
    def save(self, extrafn='', extrafolder='.', **kwargs):
        # use PIL.Image.save() to save .pil        
        self.outfile = self._updatefile(extrafn, extrafolder)
        
        # make dir(s)
        if not os.path.isdir(os.path.dirname(self.outfile)):
            os.makedirs(os.path.dirname(self.outfile))
        
        self.pil.save(self.outfile, format=None, **kwargs)
        
    def _updatefile(self, extrafn='', extrafolder='.'):
        # update file with extra fn or extra (sub)folder
        file = os.path.splitext(self.file)[0]+extrafn+os.path.splitext(self.file)[1]
        file = os.path.join(os.path.dirname(file), extrafolder, os.path.basename(file))
        return file
        
    def deepcopy(self):
        # make a deep copy of the instance
        return copy.deepcopy(self)
    
    def refile(self, newfilename):
        # rename the file and update the related information
        self.file = newfilename
        self._updatefromfile()
        if self.isfile:
            warnings.warn(f"The file named '{self.file}' already exists...")
            
    def remat(self, mat):
        # re-assign value to .mat and update related information
        self.mat = mat
        self.pil = Image.fromarray(mat)
        self._updatefrommat()
        
    def _repil(self, pil):
        self.pil = pil
        self.mat = np.asarray(self.pil)
        self._updatefrommat()
    
    def _updatefromfile(self):
        self.fn = os.path.basename(self.file)
        self.fnonly = os.path.splitext(self.fn)[0]
        self.ext = os.path.splitext(self.fn)[1]
        self.dir = os.path.dirname(self.file)
        self.group = os.path.split(self.dir)[1] # the upper-level folder name
        self.isfile = os.path.isfile(self.file)
        
    def _updatefrommat(self):
        self.dims = self.mat.shape
        self.ndim = len(self.dims)
        if self.ndim==2:
            self.h, self.w = self.dims
            self.nlayer = 0
        elif self.ndim==3:
            self.h, self.w, self.nlayer = self.dims
        
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
        self._updatefrommat()
        
    def grayscale(self):
        # convert image to gray-scale
        self._repil(ImageOps.grayscale(self.pil))
        self.refile(self._updatefile(extrafn='_gray'))
        
    def cropoval(self, radius=(100,128), bgcolor=None):
        
        # for instance, if bgcolor is (255, 255, 255, 255), the output image is not transparent; if bgcolor is (255, 255, 255, 0), the output image is transparent
        
        # for circle
        if type(radius) is not tuple:
            radius = (radius, radius) # a and b in ellipse formula
        bbox = (self.w/2-radius[0], self.h/2-radius[1], self.w/2+radius[0], self.h/2+radius[1])
        
        # make a ellipse/oval mask
        pil_a = Image.new("L", self.pil.size, 0)
        draw = ImageDraw.Draw(pil_a)
        draw.ellipse(bbox, fill=255)
        
        if bgcolor is not None:
            if type(bgcolor) is not tuple:
                bgcolor = ((bgcolor),)*len(self.pil.mode)
            pil_2 = Image.new(self.pil.mode, self.pil.size, bgcolor)
            draw = ImageDraw.Draw(pil_2)
            draw.ellipse(bbox, fill=255)
            self.pil = Image.composite(self.pil, pil_2, pil_a)
        else:
            # only apply cropping to the alpha layer
            self.pil.putalpha(pil_a)
        # update to pil        
        self._repil(self.pil)
        # crop the rectangle
        self.croprect(bbox)
        
    def croprect(self, box=None):
        # crop the image with a rectangle box
        self._repil(self.pil.crop(box))
    
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
        self._repil(self.pil.resize(**kwargs))
        self.refile(self._updatefile(extrafolder=str(w)+'_'+str(h)))
        
    def pad(self, **kwargs):
        """Add padding to the image/stimuli.

        Args:
            trgw (int): the width of the target/desired stimuli. 
            trgh (int): the height of the target/desired stimuli.
            padvalue (int, optional): padding value. Defaults to 0 (show as transparent if alpha layer exists).
            top (bool, optional): padding more to top if needed. Defaults to True.
            left (bool, optional): padding more to left if needed. Defaults to True.
            padalpha (int, optional): the transparent color. Defaults to -1, i.e., not to force it to transparent.
        """
        
        defaultKwargs = {'trgw':self.w, 'trgh':self.h, 'padvalue': 0, 'top': True, 'left':True, 'padalpha':-1}
        kwargs = {**defaultKwargs, **kwargs}
        
        trgw,trgh = kwargs['trgw'], kwargs['trgh']
        
        assert(trgw>=self.w)
        assert(trgh>=self.h)
        
        x1 = int(np.ceil(trgw-self.w)/2)
        x2 = trgw-self.w-x1
        y1 = int(np.ceil(trgh-self.h)/2)
        y2 = trgh-self.h-y1
        
        if kwargs['top']:
            htop, hbot = y1,y2
        else:
            htop, hbot = y2,y1
        if kwargs['left']:
            wleft, wright = x1,x2
        else:
            wleft, wright = x2,x1
            
        if self.nlayer==0:
            nlayer = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nlayer = self.nlayer
            mat = self.mat
            
        if (kwargs['padalpha']>=0) & (nlayer==1 | nlayer==3):
            mat = np.concatenate((mat, np.ones((self.h, self.w, 1), dtype=np.uint8)*kwargs['padalpha']),axis=2)
            nlayer = nlayer + 1
        
        padmat = np.hstack((
            np.ones((trgh,wleft,nlayer),dtype=np.uint8)*kwargs['padvalue'], 
            np.vstack((
                np.ones((htop,self.w,nlayer),dtype=np.uint8)*kwargs['padvalue'], 
                mat,
                np.ones((hbot,self.w,nlayer),dtype=np.uint8)*kwargs['padvalue']
            )),
            np.ones((trgh,wright,nlayer),dtype=np.uint8)*kwargs['padvalue'], 
        ))
        
        if (self.nlayer==0) & (kwargs['padalpha']<0):
            padmat = padmat[:,:,0]
        
        self.remat(padmat)
        self.refile(self._updatefile(extrafn='_pad'))
        
    def mkboxscr(self, **kwargs):
        """Make box scrambled stimuli.
        """
        defaultKwargs = {'nBoxW':10, 'nBoxH':16, 
                     'pBoxW':0, 'pBoxH':0, 
                     'pad': False, 'padcolor':0, 'padalpha': -1}
        kwargs = {**defaultKwargs, **kwargs}
            
        if (kwargs['pBoxW']!=0) & (kwargs['pBoxH']!=0):
        
            _nBoxW = self.w/kwargs['pBoxW']
            _nBoxH = self.h/kwargs['pBoxH']
            
            if not ((_nBoxW.is_integer()) & (_nBoxH.is_integer())):
                assert kwargs['pad'], 'Please input valid pBoxW and pBoxH. Or set "pad" to True.'
                
                # add complementary parts to top and right
                xnew = int(np.ceil(_nBoxW) * kwargs['pBoxW'])
                ynew = int(np.ceil(_nBoxH) * kwargs['pBoxH'])
            
                self.pad(trgw=xnew, trgh=ynew, padvalue=kwargs['padcolor'], padalpha=kwargs['padalpha'])

            kwargs['nBoxW'] = int(self.w/kwargs['pBoxW'])
            kwargs['nBoxH'] = int(self.h/kwargs['pBoxH'])
                
        elif (kwargs['nBoxW']!=0) & (kwargs['nBoxH']!=0):
            
            # x and y pixels for each box
            _pBoxW = self.w/kwargs['nBoxW']
            _pBoxH = self.h/kwargs['nBoxH']
    
            if not ((_pBoxW.is_integer()) & (_pBoxH.is_integer())):
                assert kwargs['pad'], 'Please input valid nBoxW and nBoxH. Or set "pad" to True.'
                
                # add padding (top, right, bottom, left)
                newW = int(np.ceil(_pBoxW) * kwargs['nBoxW'])
                newY = int(np.ceil(_pBoxH) * kwargs['nBoxH'])
                self.pad(trgw=newW, trgh=newY, padvalue=kwargs['padcolor'], padalpha=kwargs['padalpha'])

            kwargs['pBoxW'] = int(self.w/kwargs['nBoxW'])
            kwargs['pBoxH'] = int(self.h/kwargs['nBoxH'])
        else:
            raise 'Please set valid nBoxW and nBoxH (or pBoxW and pBoxH).'
        
        assert kwargs['nBoxW']*kwargs['pBoxW']==self.w, f"'nBoxW': {kwargs['nBoxW']}   'pBoxW': {kwargs['pBoxW']}    'self.w': {self.w}"
        assert kwargs['nBoxH']*kwargs['pBoxH']==self.h, f"'nBoxH': {kwargs['nBoxH']}   'pBoxH': {kwargs['pBoxH']}    'self.h': {self.h}"
        
        # x and y for all boxes
        xys = list(product(range(0,self.w,kwargs['pBoxW']), range(0,self.h,kwargs['pBoxH'])))
        boxes = [self.mat[i[1]:(i[1]+kwargs['pBoxH']), i[0]:(i[0]+kwargs['pBoxW'])] for i in xys]
        # randomize the boxes
        bsboxes = np.random.permutation(boxes)
        # save as np.array
        bslist = [bsboxes[i:i+kwargs['nBoxW']] for i in range(0,len(bsboxes),kwargs['nBoxW'])]
        # bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
        bsmatm = np.asarray(bslist)
        if len(bsmatm.shape)==4:
            bsmatm = bsmatm[..., np.newaxis]
        bsmat = np.moveaxis(bsmatm, [-1, 1], [0, -2]).reshape(-1, self.h, self.w)
        
        # save box-scrambled images (and information)
        self.remat(np.squeeze(np.moveaxis(bsmat,0,2)))
        self.refile(self._updatefile(extrafn='_bscr'))
    
    def sffilter(self, **kwargs):
        # https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        import psychopy.filters
        
        defaultKwargs = {'rms':0.3, 'maxvalue':255, 'sffilter':'low',
                         'cutoff': 0.05, 'n': 10}
        kwargs = {**defaultKwargs, **kwargs}
        
        self.grayscale()
        stdmat = self._stdim(self.mat, kwargs['rms'])
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
        self.remat(img_new)
        self.refile(self._updatefile(extrafn='_'+kwargs['sffilter']+'_filtered'))
    
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

        if self.nlayer==0:
            nlayer = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nlayer = self.nlayer
            mat = self.mat
            
        outmat = np.empty(mat.shape)
        outmat[:] = np.NaN
        
        for i in range(nlayer):
            img_freq = np.fft.fft2(self._stdim(mat[:,:,i], kwargs['rms']))
            amp = np.abs(img_freq)
            phase = np.angle(img_freq) + randphase
            outimg = np.real(np.fft.ifft2(amp * np.exp(np.sqrt(-1+0j)*phase)))
            stdimg1= self._stdim(outimg, kwargs['rms'])
            outmat[:,:,i] = (stdimg1+1)/2*255
            
        if self.nlayer==0:
            outmat = outmat[:,:,0]
        
        self.remat(np.uint8(outmat))
        self.refile(self._updatefile(extrafn='_pscr'))
        