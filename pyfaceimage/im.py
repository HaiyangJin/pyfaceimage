"""
A class to process a single image.    
"""
    
import os, warnings, copy
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.image as mpimg
from itertools import product

class image:
    def __init__(self, filename, read=False):
        """Create an image instance.

        Parameters
        ----------
        filename : str
            path and image filename.
        read : bool, optional
            Whether to read the image via PIL, by default False
        """
        # make sure the file exists 
        assert os.path.isfile(filename) | (not bool(filename)), f'Cannot find {filename}...'
        self.filename = filename
        self._updatefromfilename() # update filename information
        self._setgroup() 
        self._setgpath()
        if read: self.read()
        
    def _updatefromfilename(self):
        """Update information from the filename.
        """
        self.fname = os.path.basename(self.filename)
        self.fnonly = os.path.splitext(self.fname)[0]
        self.ext = os.path.splitext(self.fname)[1]
        self.dirname = os.path.dirname(self.filename)
        self.isfile = os.path.isfile(self.filename)
        
    def read(self):
        """Read the image via PIL.
        """
        self._repil(Image.open(self.filename)) # PIL.Image.open() 
        # potential useful functions
        # .filename .format, .mode, .split()
        
    def _setgroup(self, gname=''):
        """Set the group name of the image.

        Parameters
        ----------
        gname : str, optional
            the group name, by default the upper directory name of the image.
        """
        # update group information
        if not bool(gname):
            gname = os.path.split(os.path.dirname(self.filename))[1]
        self.group = gname
        
    def _setgpath(self, gpath=''):
        """Set the global path of the image (among other images). [This atribute matters only when mutiltple images are processed together. more see dir().]

        Parameters
        ----------
        gpath : str, optional
            global path, i.e. path in dir(), by default ''
        """
        self.gpath = gpath
        
    def imshow(self):
        """Show the image matrix.
        """
        # for debugging purpose (check the mat)
        # it seems that .show() is more useful
        Image.fromarray(self.mat).show()
        
    def show(self):
        """Show the image PIL.
        """
        # for debugging purpose (check the PIL)
        self.pil.show()

    def imsave(self, newfname='', newfolder='', addfn=True, **kwargs):
        """Save the image mat.

        Parameters
        ----------
        newfname : str, optional
            strings to be added before the extension, by default ''
        newfolder : str, optional
            folder name to replace the global directory or the last directory level, by default ''
        addfn : bool, optional
            whether to add the newfname to the the original fname (instead of replacing it), by default True
        kwargs : dict, optional
            keyword arguments for matplotlib.pyplot.imsave(), by default {}
        """
        # update the filename
        self._newfilename(newfname, newfolder, addfn=addfn)
        
        # make dir(s)
        if not os.path.isdir(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        if self.nchan==1:
            outmat = self.mat[:,:,0]
        else:
            outmat = self.mat
        
        # use matplotlib.pyplot.imsave() to save .mat
        mpimg.imsave(self.filename,outmat.copy(order='C'),**kwargs)
        self.isfile = os.path.isfile(self.filename)
        
        
    def save(self, newfname='', newfolder='', addfn=True, **kwargs):
        """Save the image PIL.

        Parameters
        ----------
        newfname : str, optional
            strings to be added before the extension, by default ''
        newfolder : str, optional
            folder name to replace the global directory or the last directory level, by default ''
        addfn : bool, optional
            whether to add the newfname to the the original fname (instead of replacing it), by default True
        kwargs : dict, optional
            keyword arguments for matplotlib.pyplot.imsave(), by default {}
        """
        # update the filename
        self._newfilename(newfname, newfolder, addfn=addfn)
        
        # make dir(s)
        if not os.path.isdir(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        
        # use PIL.Image.save() to save .pil    
        self.pil.save(self.filename, format=None, **kwargs)
        self.isfile = os.path.isfile(self.filename)
        
        
    def _newfilename(self, newfname='', newfolder='', addfn=True):
        """Update the filename with newfname and newfolder.

        Parameters
        ----------
        newfname : str, optional
            strings to be added before the extension, by default ''
        newfolder : str, optional
            folder name to replace the global directory or the last directory level, by default ''
        addfn : bool, optional
            whether to add the newfname to the the original fname (instead of replacing it), by default True
        """
        # apply newfname to the old one
        oldfname = os.path.splitext(self.fname)[0] if addfn else ''
        fname = oldfname+newfname+os.path.splitext(self.fname)[1]
        
        # replace the path folder with newfolder
        if bool(self.gpath):
            # rename the global path and filename
            foldername = newfolder if bool(newfolder) else os.path.basename(self.gpath) # apply newfolder to the old one if needed
            group = self.group if self.group != 'path' else ''
            self.filename = os.path.join(os.path.dirname(self.gpath), foldername, group, fname)
            self.gpath = os.path.join(os.path.dirname(self.gpath), foldername)
        else:
            foldername = newfolder if bool(newfolder) else os.path.basename(self.dirname) # apply newfolder to the old one if needed
            self.filename = os.path.join(os.path.dirname(self.filename), foldername, fname)
        
        self._updatefromfilename()
        if self.isfile:
            warnings.warn("The file named '%s' already exists..." % {self.filename})
        
    def deepcopy(self):
        """make a deep copy of the instance
        """
        return copy.deepcopy(self)
            
    def remat(self, mat):
        # re-assign value to .mat and update related information
        self.mat = mat
        self.pil = Image.fromarray(mat)
        self._updatefrommat()
        
    def _repil(self, pil):
        self.pil = pil
        self.mat = np.asarray(self.pil)
        self._updatefrommat()
            
    def _updatefrommat(self):
        self.dims = self.mat.shape
        self.ndim = len(self.dims)
        if self.ndim==2:
            self.h, self.w = self.dims
            self.nchan = 0
        elif self.ndim==3:
            self.h, self.w, self.nchan = self.dims
        
    def torgba(self):
        """Convert the image to RGBA.
        """
        # convert pil to RGBA and save in .mat
        if len(self.mat.shape)==2: # Gray
            self.mat = self.mat[..., np.newaxis]
        assert len(self.mat.shape)==3
        
        nchan = self.mat.shape[2]
        if nchan==1: # Gray 'L'
            rgbmat = np.repeat(self.mat, 3, axis=2) # RGB
            amat = np.ones_like(rgbmat[:,:,0],dtype=np.uint8)*rgbmat.max() # alpha
        elif nchan==2: # Gray 'LA'
            rgbmat = np.repeat(self.mat[:,:,0:1], 3, axis=2) # RGB
            amat = self.mat[:,:,-1] # alpha
        elif nchan==3: # RGB
            rgbmat = self.mat # RGB
            amat = np.ones_like(rgbmat[:,:,0],dtype=np.uint8)*rgbmat.max() # alpha
        elif nchan==4: # RGBA
            rgbmat = self.mat[:,:,0:3]
            amat = self.mat[:,:,-1]
            
        self.rgbmat = rgbmat
        self.amat = amat
        self.mat = np.concatenate((rgbmat, amat[..., np.newaxis]), axis=2) # RGBA
        self.remat(self.mat)
        
    def grayscale(self):
        """Convert the image to gray-scale.
        """
        # convert image to gray-scale
        self._repil(ImageOps.grayscale(self.pil))
        
    def _logit(self, ratio=None, correction=0.00001):
        """Convert the ratio to log odds.

        Parameters
        ----------
        ratio : np.array, optional
            the ratio of the image. Defaults to None.
        correction : float, optional
            the correction value. Defaults to 0.00001.

        Returns
        -------
        np.array
            the log odds of the image.
        """
        
        if ratio is None:
            self.grayscale()
            ratio = self.mat/255
        elif type(ratio) is not np.ndarray:
            ratio = np.array(ratio)
            
        ratio[ratio==1] = (255-correction)/255
        ratio[ratio==0] = correction/255
        
        return np.log(ratio/(1-ratio))
        
    def _sigmoid(self, logodds, correction=0.00001):
                       
        ratio = np.exp(logodds)/(np.exp(logodds)+1)
        ratio[ratio>=(1-correction)] = 1
        ratio[ratio<=(correction)] = 0
        
        gray = ratio * 255
        
        return gray.astype(dtype=np.uint8)
    
    def rotate(self, angle=180):
        """Rotate the image unclockwise.

        Parameters
        ----------
        angle : float
            the angle to rotate the image. Defaults to 180.
        """
        # rotate the image
        self._repil(self.pil.rotate(angle))
        
    def adjust(self, lum=None, rms=None, mask=None):
        """Adjust the luminance and contrast of the image.
        
        Parameters
        ----------
        lum : float, optional
            the desired mean of the image. Defaults to None.
        rms : float, optional
            the desired standard deviation of the image. Defaults to None.
        mask : np.array, optional
            the mask for the image. Defaults to None.
        """
        # adjust the luminance (mean) or/and contrast (standard deviations) of the gray-scaled image. Default is do nothing.
        
        # default for mask
        if self.nchan in (2,4):
            alpha = self.mat[...,-1]
            isalpha = True
        else:
            alpha = np.ones((self.h, self.w))*255
            isalpha = False
        if mask is None:
            mask = (alpha/255).astype(dtype=bool)
        
        # force the image to be gray-scaled
        self.grayscale()
        ratio = self.mat/255
        logodds = self._logit(ratio)
        
        # in ratio
        rmsratio = np.std(ratio[mask]-np.mean(ratio[mask]))
        
        # in log odds
        meanlo = np.mean(logodds[mask])
        stdlo = np.std(logodds[mask]-meanlo)
        
        if lum is not None:
            lumlo = self._logit(lum)
        else:
            lumlo = meanlo
            
        if rms is not None:
            rmslo = rms * stdlo / rmsratio
        else:
            rmslo = stdlo
           
        newlo = logodds
        # apply the new luminance (in logodds)
        newlo[mask] = (logodds[mask] - meanlo)/stdlo * rmslo + lumlo
        newmat = self._sigmoid(newlo)
        
        if isalpha:
            newmat = np.concatenate((newmat[...,np.newaxis], alpha[...,np.newaxis]), axis=2)
        
        self.remat(newmat.astype(dtype=np.uint8))

        
    def cropoval(self, radius=(100,128), bgcolor=None):
        """Crop the image with an oval shape.
        
        Parameters
        ----------
        radius : tuple, optional
            the radius of the oval. Defaults to (100,128).
        bgcolor : tuple, optional
            the background color. Defaults to None.
        """
        
        # for instance, if bgcolor is (255, 255, 255, 255), the output image is not transparent; if bgcolor is (255, 255, 255, 0), the output image is transparent
        
        # to make circle
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
            # only apply cropping to the alpha channel
            self.pil.putalpha(pil_a)
        # update to pil        
        self._repil(self.pil)
        # crop the rectangle
        self.croprect(bbox)
        
    def croprect(self, box=None):
        """Crop the image with a rectangle box.

        Parameters
        ----------
        box : tuple, optional
            the box to crop the image. Defaults to None.
        """
        # crop the image with a rectangle box
        self._repil(self.pil.crop(box))
    
    def resize(self, **kwargs):
        """Resize the image.
        
        Kwargs
        ----------
        trgw: int, optional
            the width of the target/desired stimuli.
        trgh: int, optional
            the height of the target/desired stimuli.
        ratio: float, optional
            the ratio to resize the image. Defaults to 0.
        newfolder: str, optional
            the folder to save the resized image. Defaults to None.
        """
        # resize the image
        defaultKwargs = {'trgw': None, 'trgh': None, 'ratio':0, 'newfolder':None}
        kwargs = {**defaultKwargs, **kwargs}
        
        if (kwargs['trgw'] is not None) & (kwargs['trgh'] is not None):
            (w, h) = (kwargs['trgw'], kwargs['trgh'])
        elif (kwargs['trgw'] is not None) & (kwargs['trgh'] is None):
            w = kwargs['trgw']
            h = int(w*self.h/self.w)
        elif (kwargs['trgw'] is None) & (kwargs['trgh'] is not None): 
            h = kwargs['trgh']
            w = int(h*self.w/self.h)
        elif kwargs['ratio']>0:
            w = int(self.w*kwargs['ratio'])
            h = int(self.h*kwargs['ratio'])
        else:
            raise 'Cannot determine the desired dimentions...'
        
        kwargs['size'] = (w,h)
        if (kwargs['newfolder'] is None):
            newfolder = str(w)+'_'+str(h)
        else:
            newfolder = kwargs['newfolder']
            
        [kwargs.pop(k) for k in defaultKwargs.keys()] # remove unused keys
                
        # save re-sized images (information)
        self._repil(self.pil.resize(**kwargs))
        self._newfilename(newfolder=newfolder)
        
    def pad(self, **kwargs):  
        """
        Add padding to the image/stimuli.
        
        Kwargs
        ----------
        trgw: int, optional
            the width of the target/desired stimuli. 
        trgh: int, optional
            the height of the target/desired stimuli.
        padvalue: int, optional
            padding value. Defaults to 0 (show as transparent if alpha channel exists).
        top: bool, optional
            padding more to top if needed. Defaults to True.
        left: bool, optional 
            padding more to left if needed. Defaults to True.
        padalpha: int, optional
            the transparent color. Defaults to -1, i.e., not to force it to transparent.
        extrafn: str, optional
            the string to be added to the filename. Defaults to '_pad'.
        """
        
        defaultKwargs = {'trgw':self.w, 'trgh':self.h, 'padvalue': 0, 'top': True, 'left':True, 'padalpha':-1, 'extrafn':'_pad'}
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
            
        if self.nchan==0:
            nchan = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nchan = self.nchan
            mat = self.mat
            
        if (kwargs['padalpha']>=0) & (nchan==1 | nchan==3):
            mat = np.concatenate((mat, np.ones((self.h, self.w, 1), dtype=np.uint8)*kwargs['padalpha']),axis=2)
            nchan = nchan + 1
        
        padmat = np.hstack((
            np.ones((trgh,wleft,nchan),dtype=np.uint8)*kwargs['padvalue'], 
            np.vstack((
                np.ones((htop,self.w,nchan),dtype=np.uint8)*kwargs['padvalue'], 
                mat,
                np.ones((hbot,self.w,nchan),dtype=np.uint8)*kwargs['padvalue']
            )),
            np.ones((trgh,wright,nchan),dtype=np.uint8)*kwargs['padvalue'], 
        ))
        
        if (self.nchan==0) & (kwargs['padalpha']<0):
            padmat = padmat[:,:,0]
        
        self.remat(padmat)
        if kwargs['extrafn']!='':
            self._newfilename(newfname=kwargs['extrafn'])
        
    def mkboxscr(self, **kwargs):
        """Make box scrambled stimuli.
        
        Kwargs
        ----------
        nBoxW: int, optional
            the number of boxes in width. Defaults to 10.
        nBoxH: int, optional
            the number of boxes in height. Defaults to 16.
        pBoxW: int, optional
            the width of a box. Defaults to 0.
        pBoxH: int, optional
            the height of a box. Defaults to 0.
        pad: bool, optional
            whether to add padding to the image. Defaults to False.
        padcolor: int, optional
            the padding color. Defaults to 0.
        padalpha: int, optional
            the padding alpha. Defaults to -1.
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
        self._newfilename(newfname='_bscr')
        
    def mkphasescr(self, **kwargs):
        """Make phase scrambled stimuli.
        
        Kwargs
        ----------
        rms: float, optional
            the desired RMS of the image. Defaults to 0.3.
        """
        defaultKwargs = {'rms':0.3}
        kwargs = {**defaultKwargs, **kwargs}
        
        # make a random phase
        randphase = np.angle(np.fft.fft2(np.random.rand(self.h, self.w)))

        if self.nchan==0:
            nchan = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nchan = self.nchan
            mat = self.mat
            
        outmat = np.empty(mat.shape)
        outmat[:] = np.NaN
        
        for i in range(nchan):
            img_freq = np.fft.fft2(self._stdim(mat[:,:,i], kwargs['rms']))
            amp = np.abs(img_freq)
            phase = np.angle(img_freq) + randphase
            outimg = np.real(np.fft.ifft2(amp * np.exp(np.sqrt(-1+0j)*phase)))
            stdimg1= self._stdim(outimg, kwargs['rms'])
            outmat[:,:,i] = (stdimg1+1)/2*255
            
        if self.nchan==0:
            outmat = outmat[:,:,0]
        
        self.remat(np.uint8(outmat))
        self._newfilename(newfname='_pscr')
    
    def sffilter(self, **kwargs):
        """Apply spatial frequency filter to the image.
        
        Kwargs
        ----------
        rms: float, optional
            the desired RMS of the image. Defaults to 0.3.
        maxvalue: int, optional
            the maximum value of the image. Defaults to 255.
        sffilter: str, optional
            the spatial frequency filter. Defaults to 'low'.
        cutoff: float, optional
            the cutoff frequency. Defaults to 0.05.
        n: int, optional
            the order of the filter. Defaults to 10.
        """
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
        self._newfilename(newfname='_'+kwargs['sffilter']+'_filtered')
    
    def _stdim(self, mat, rms=0.3):
        """Standardize the image.

        Parameters
        ----------
        mat : np.array
            the image matrix.
        rms : float, optional
            the desired RMS of the image. Defaults to 0.3.

        Returns
        -------
        np.array
            the standardized image matrix.
        """
        # standardize the image (the range of output should be -1,1)
        # make the standard deviation to be the desired RMS
        mat = (mat - np.mean(mat))/np.std(mat) * rms
        
        # there may be some stray values outside of the presentable range; convert < -1
        # to -1 and > 1 to 1
        mat = np.clip(mat, a_min=-1.0, a_max=1.0)
        return mat

 
        
    
        