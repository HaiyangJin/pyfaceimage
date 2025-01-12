"""
A class to process a single image.    
"""
    
import os, warnings, copy
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops
import numpy as np
import matplotlib.image as mpimg
from itertools import product
from pathlib import Path

class image:
    """A class to process a single image.
    
    Parameters
    ----------
    filename : str
        path and image filename.  
    read : bool, optional
        Whether to read the image via PIL, by default False.
            
    Attributes
    ----------
    filename : str
        path and image filename.
    fname : str
        the image filename.
    fnonly : str
        the image filename without extension.
    ext : str
        the image extension.
    dirname : str
        the image directory.
    isfile : bool
        whether the image file exists.
    group : str
        the group name of the image.
    gpath : str
        the global path of the image.
    pil : PIL.Image
        the PIL image.
    mat : np.array
        the image matrix.
    dims : tuple
        the image dimensions.
    ndim : int
        the number of dimensions.
    h : int
        the height of the image.
    w : int
        the width of the image.
    nchan : int
        the number of channels.
    rgbmat : np.array
        the RGB matrix.
    amat : np.array
        the alpha matrix.
        
    Properties
    ----------
    filename : str
        Return the image filename.
    fname : str
        Return the image filename.
    fnonly : str
        Return the image filename without extension.
    ext : str
        Return the image extension.
    dirname : str
        Return the image directory.
    isfile : bool
        Return whether the image file exists.
    dims : tuple
        Return the image dimensions.
    ndim : int
        Return the number of dimensions.
    h : int
        Return the height of the image.
    w : int
        Return the width of the image.
    nchan : int
        Return the number of channels.
    maxlum : float
        Return the maximum luminance of the image.
    minlum : float
        Return the minimum luminance of the image.
    meanlum : float
        Return the mean luminance of the image.
    rms : float
        Return the RMS of the image.
    luminfo : None
        Print the luminance information of the image.
    
    Methods
    -------
    update_fninfor()
        Update the image filename related information.
    updateext(ext)
        Update the filename information.
    read()
        Read the image via PIL.
    imshow()
        Show the image matrix.
    show()
        Show the image PIL.
    imsave(newfname='', newfolder='', addfn=True, **kwargs)
        Save the image mat.
    save(newfname='', newfolder='', addfn=True, **kwargs)
        Save the image PIL.
    deepcopy()
        Make a deep copy of the instance.
    remat(mat)
        Re-assign value to .mat and update related information.
    torgba()
        Convert the image to RGBA.
    grayscale()
        Convert the image to gray-scale.
    addbg(bgcolor=(255,255,255))
        Add background to the RGBA image.
    rotate(angle=180)
        Rotate the image unclockwise.
    adjust(lum=None, rms=None, mask=None)
        Adjust the luminance and contrast of the image.
    cropoval(radius=(100,128), bgcolor=None)
        Crop the image with an oval shape.
    croprect(box=None)
        Crop the image with a rectangle box.
    resize(**kwargs)
        Resize the image.
    pad(**kwargs)
        Add padding to the image/stimuli.
    mkboxscr(**kwargs)
        Make box scrambled stimuli.
    mkphasescr(**kwargs)
        Make phase scrambled stimuli.
    filter(**kwargs)
        Apply spatial frequency filter to the image.
    psychopy_sffilter(**kwargs)
        Apply spatial frequency filter to the image.
    stdmat(rms=1, outrange=[0,255])
        Standardize the image matrix.
    _logit(ratio=None, correction=0.00001)
        Convert the ratio to log odds.
    _sigmoid(logodds, correction=0.00001)
        Convert the log odds to the ratio.
    _repil(pil)
        Update the image PIL.
    _updatefrommat()
        Update information from the image matrix.
    _newfilename(newfname='', newfolder='', addfn=True)
        Update the filename with newfname and newfolder.
    _setgroup(gname='')
        Set the group name of the image.
    _setgpath(gpath='')
        Set the global path of the image.
    """
    
    def __init__(self, filename, read=False):
        """Create an image instance.

        Parameters
        ----------
        filename : str
            path and image filename.  
        read : bool, optional
            Whether to read the image via PIL, by default False.
            
        Raises
        ------
        AssertionError
            If the file does not exist.
        """
        # make sure the file exists 
        self._filename = filename
        self.update_fninfo(filename, newfn=False) 
        self._setgroup() 
        self._setgpath()
        if read: self.read()
    
    @property
    def filename(self):
        """Return the image filename.

        Returns
        -------
        str
            the image filename.
        """
        return self._filename
    
    def update_fninfo(self, filename, newfn=True):
        """Set the image filename.

        Parameters
        ----------
        filename : str
            path and image filename.
        newfn : bool, optional
            whether `filename` is a new file name, by default True. So it will not check whether the file exists.
        """
        if newfn & os.path.isfile(filename):
            # throw warning if filename is newly created and exists
            warnings.warn(f"The file named '{filename}' already exists...")
        elif not newfn:
            # throw error if filename is not new and does not exist
            assert os.path.isfile(filename) | (not bool(filename)), f'Cannot find {filename}...'
            
        self._filename = filename
        self._fname = os.path.basename(filename)
        self._fnonly = ".".join(os.path.splitext(self._fname)[0:-1])
        self._ext = os.path.splitext(self._fname)[-1]
        self._dirname = os.path.dirname(filename)
        self._isfile = os.path.isfile(filename)
        
    @property
    def fname(self):
        """Return the image filename.

        Returns
        -------
        str
            the image filename.
        """
        return self._fname
    
    @property
    def fnonly(self):
        """Return the image filename without extension.

        Returns
        -------
        str
            the image filename without extension.
        """
        return self._fnonly
    
    @property
    def ext(self):
        """Return the image extension.

        Returns
        -------
        str
            the image extension.
        """
        return self._ext
    
    @property
    def dirname(self):
        """Return the image directory.

        Returns
        -------
        str
            the image directory.
        """
        return self._dirname
    
    @property
    def isfile(self):
        """Return whether the image file exists.

        Returns
        -------
        bool
            whether the image file exists.
        """
        return self._isfile
    
    def updateext(self, ext):
        """Update the filename information.
        
        Parameters
        ----------
        ext : str
            the new extension.
        """
        if ext[0] != '.':
            ext = '.'+ext
        self.update_fninfo(self._dirname + os.sep + self._fnonly + ext)
        
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
        if self._nchan==1:
            outmat = self.mat[:,:,0]
        else:
            outmat = self.mat
        
        # use matplotlib.pyplot.imsave() to save .mat
        mpimg.imsave(self.filename,outmat.copy(order='C'),**kwargs)
        self._isfile = os.path.isfile(self.filename)
        
        
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
        self._isfile = os.path.isfile(self.filename)
        
        
    def touch(self, touchfolder=''):
        """Touch a new empty file.

        Parameters
        ----------
        touchfolder : str, optional
            folder name to replace the global directory or the last directory level, by default ''
        """
        # update the directory
        self._newfilename(newfolder=touchfolder)
        
        if self.isfile:
            # throw warnings if file exists
            warnings.warn("The file named '%s' already exists..." % self.filename)
        else:
            # make dir if needed
            if not os.path.isdir(self._dirname):
                os.makedirs(self._dirname)
            # touch the file
            Path(self.filename).touch()
        
    
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
        oldfname = os.path.splitext(self._fname)[0] if addfn else ''
        fname = oldfname+newfname+os.path.splitext(self._fname)[1]
        
        # replace the path folder with newfolder
        if bool(self.gpath):
            # rename the global path and filename
            foldername = newfolder if bool(newfolder) else os.path.basename(self.gpath) # apply newfolder to the old one if needed
            group = self.group if self.group != 'path' else ''
            self.update_fninfo(os.path.join(os.path.dirname(self.gpath), foldername, group, fname))
            self.gpath = os.path.join(os.path.dirname(self.gpath), foldername)
        else:
            foldername = newfolder if bool(newfolder) else os.path.basename(self._dirname) # apply newfolder to the old one if needed
            self.update_fninfo(os.path.join(os.path.dirname(self.filename), foldername, fname))
        
    def deepcopy(self):
        """make a deep copy of the instance
        """
        return copy.deepcopy(self)
            
    def remat(self, mat):
        """Re-assign value to .mat and update related information.

        Parameters
        ----------
        mat : np.array
            the new image matrix.
        """
        # re-assign value to .mat and update related information
        self.mat = mat
        self.pil = Image.fromarray(mat)
        self._updatefrommat()
        
    def _repil(self, pil):
        self.pil = pil
        self.mat = np.asarray(self.pil)
        self._updatefrommat()
            
    def _updatefrommat(self):
        self._dims = self.mat.shape
        self._ndim = len(self._dims)
        self._h = self._dims[0]
        self._w = self._dims[1]
        self._nchan = self._dims[2] if self._ndim==3 else 0
    
    @property
    def dims(self):
        """Return the image dimensions.

        Returns
        -------
        tuple
            the image dimensions.
        """
        return self._dims
    
    @property
    def ndim(self):
        """Return the number of dimensions.

        Returns
        -------
        int
            the number of dimensions.
        """
        return self._ndim
    
    @property
    def h(self):
        """Return the height of the image.

        Returns
        -------
        int
            the height of the image.
        """
        return self._h
    
    @property
    def w(self):
        """Return the width of the image.

        Returns
        -------
        int
            the width of the image.
        """
        return self._w
    
    @property
    def nchan(self):
        """Return the number of channels.

        Returns
        -------
        int
            the number of channels.
        """
        return self._nchan
    
    @property
    def maxlum(self):
        """Return the maximum luminance of the image.

        Returns
        -------
        float
            the maximum luminance of the image.
        """
        return np.max(self.mat)
    
    @property
    def minlum(self):
        """Return the minimum luminance of the image.

        Returns
        -------
        float
            the minimum luminance of the image.
        """
        return np.min(self.mat)
    
    @property
    def meanlum(self):
        """Return the mean luminance of the image.

        Returns
        -------
        float
            the mean luminance of the image.
        """
        return np.mean(self.mat)
    
    @property
    def rms(self):
        """Return the RMS of the image.

        Returns
        -------
        float
            the RMS of the image.
        """
        return np.std(self.mat)
    
    @property
    def luminfo(self):
        """Print the luminance information of the image.
        """
        print(f'Maximum luminance: {self.maxlum}\n'+
              f'Minimum luminance: {self.minlum}\n'+
              f'Mean luminance: {self.meanlum}\n'+
              f'RMS: {self.rms}')
    
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
        
    def addbg(self, bgcolor=(255,255,255)):
        """Add background to the RGBA image.

        Parameters
        ----------
        bgcolor : list, optional
            the background color. Defaults to (255,255,255).
        """
        
        if len(bgcolor)>4:
            bgcolor = bgcolor[:4]
                
        # make sure the image is in RGBA
        if self._ndim<4:
            self.torgba()
            
        # make a new image with the same size and the background color
        bg = Image.new('RGBA', (self._w, self._h), tuple(bgcolor))
        # paste self to the background image
        bg.paste(self.pil, (0,0), self.pil)
        # apply pil
        self._repil(bg)
        
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
        
    def stdmat(self, clip=2, rms=1, lum=[0,255]):
        """Standardize the image with the desired Root-Mean-Square contrast or `outrange`.
        
        For the algorithm, see Appendix B in 
        Loschky, L. C., Sethi, A., Simons, D. J., Pydimarri, T. N., Ochs, D., & Corbeille, J. L. (2007). The importance of information localization in scene gist recognition. Journal of Experimental Psychology: Human Perception and Performance, 33(6), 1431-1450. https://doi.org/10.1037/0096-1523.33.6.1431

        Parameters
        ----------
        clip : float, optional
            the desired clip value. Defaults to 2.
        rms : float, optional
            the desired Root-Mean-Square of the image. Defaults to 1.
        lum : num or list, optional
            if `lum` is number, it will be treated as the mean luminance. If `lum` is list, it refers to the output luminance range of the image. Defaults to [0,255]. When `rms` is not 1, `outrange` is forced to be [0, 255].  
        """
        
        # standardize the image (the range of output should be -1,1)
        mat_std = (self.mat - np.mean(self.mat))/np.std(self.mat)

        # clip the image
        clip = np.abs(clip)
        if clip > 0:
            Nclipped = np.round(np.mean(np.abs(mat_std)>clip) * 100, 2)
            mat_std[mat_std>clip] = clip
            mat_std[mat_std<-clip] = -clip
            print(f'Clipped {Nclipped}% pixels...')
            
            # rescale with std due to clipping
            mat_std = mat_std / np.std(mat_std)

        # make the standard deviation to be the desired RMS
        mat_std_rms = mat_std * rms
        
        mat_min = np.min(mat_std_rms)
        mat_max = np.max(mat_std_rms)
        
        # if lum is a single value, it will be treated as mean luminance
        if type(lum) is not list:
            lum = [lum - min(lum, 255-lum), lum + min(lum, 255-lum)]
        # force lum to [0, 255] if rms is not 1
        if rms != 1:
            lum = [0,255]
            
        mat_out = (mat_std_rms - mat_min) * (lum[1] - lum[0]) / (mat_max - mat_min) + lum[0]
        
        # update the image matrix to self
        self.remat(mat_out.astype(dtype=np.uint8))
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
        if self._nchan in (2,4):
            alpha = self.mat[...,-1]
            isalpha = True
        else:
            alpha = np.ones((self._h, self._w))*255
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
           
        newlo = mat_std
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
        bbox = (self._w/2-radius[0], self._h/2-radius[1], self._w/2+radius[0], self._h/2+radius[1])
        
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
        
        Other Parameters
        ----------------
        trgw : int
            the width of the target/desired stimuli.
        trgh : int
            the height of the target/desired stimuli.
        ratio : float
            the ratio to resize the image. Defaults to 0.
        newfolder : str
            the folder to save the resized image. Defaults to None.
        """
        # resize the image
        defaultKwargs = {'trgw': None, 'trgh': None, 'ratio':0, 'newfolder':None}
        kwargs = {**defaultKwargs, **kwargs}
        
        if (kwargs['trgw'] is not None) & (kwargs['trgh'] is not None):
            (w, h) = (kwargs['trgw'], kwargs['trgh'])
        elif (kwargs['trgw'] is not None) & (kwargs['trgh'] is None):
            w = kwargs['trgw']
            h = int(w*self._h/self._w)
        elif (kwargs['trgw'] is None) & (kwargs['trgh'] is not None): 
            h = kwargs['trgh']
            w = int(h*self._w/self._h)
        elif kwargs['ratio']>0:
            w = int(self._w*kwargs['ratio'])
            h = int(self._h*kwargs['ratio'])
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
        
        Other Parameters
        ----------------
        trgw : int
            the width of the target/desired stimuli. 
        trgh : int
            the height of the target/desired stimuli.
        padvalue : int
            padding value. Defaults to 0 (show as transparent if alpha channel exists).
        top : bool
            padding more to top if needed. Defaults to True.
        left : bool 
            padding more to left if needed. Defaults to True.
        padalpha : int
            the transparent color. Defaults to -1, i.e., not to force it to transparent.
        extrafn : str
            the string to be added to the filename. Defaults to '_pad'.
        """
        
        defaultKwargs = {'trgw':self._w, 'trgh':self._h, 'padvalue': 0, 'top': True, 'left':True, 'padalpha':-1, 'extrafn':'_pad'}
        kwargs = {**defaultKwargs, **kwargs}
        
        trgw,trgh = kwargs['trgw'], kwargs['trgh']
        
        assert(trgw>=self._w)
        assert(trgh>=self._h)
        
        x1 = int(np.ceil(trgw-self._w)/2)
        x2 = trgw-self._w-x1
        y1 = int(np.ceil(trgh-self._h)/2)
        y2 = trgh-self._h-y1
        
        if kwargs['top']:
            htop, hbot = y1,y2
        else:
            htop, hbot = y2,y1
        if kwargs['left']:
            wleft, wright = x1,x2
        else:
            wleft, wright = x2,x1
            
        if self._nchan==0:
            nchan = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nchan = self._nchan
            mat = self.mat
            
        if (kwargs['padalpha']>=0) & (nchan==1 | nchan==3):
            mat = np.concatenate((mat, np.ones((self._h, self._w, 1), dtype=np.uint8)*kwargs['padalpha']),axis=2)
            nchan = nchan + 1
        
        padmat = np.hstack((
            np.ones((trgh,wleft,nchan),dtype=np.uint8)*kwargs['padvalue'], 
            np.vstack((
                np.ones((htop,self._w,nchan),dtype=np.uint8)*kwargs['padvalue'], 
                mat,
                np.ones((hbot,self._w,nchan),dtype=np.uint8)*kwargs['padvalue']
            )),
            np.ones((trgh,wright,nchan),dtype=np.uint8)*kwargs['padvalue'], 
        ))
        
        if (self._nchan==0) & (kwargs['padalpha']<0):
            padmat = padmat[:,:,0]
        
        self.remat(padmat)
        if kwargs['extrafn']!='':
            self._newfilename(newfname=kwargs['extrafn'])
        
    def mkboxscr(self, **kwargs):
        """Make box scrambled stimuli.
        
        Other Parameters
        ----------------
        nBoxW : int
            the number of boxes in width. Defaults to 10.
        nBoxH : int
            the number of boxes in height. Defaults to 16.
        pBoxW : int
            the width of a box. Defaults to 0.
        pBoxH : int
            the height of a box. Defaults to 0.
        pad : bool
            whether to add padding to the image. Defaults to False.
        padcolor : int
            the padding color. Defaults to 0.
        padalpha : int
            the padding alpha. Defaults to -1.
        """
        defaultKwargs = {'nBoxW':10, 'nBoxH':16, 
                     'pBoxW':0, 'pBoxH':0, 
                     'pad': False, 'padcolor':0, 'padalpha': -1}
        kwargs = {**defaultKwargs, **kwargs}
            
        if (kwargs['pBoxW']!=0) & (kwargs['pBoxH']!=0):
        
            _nBoxW = self._w/kwargs['pBoxW']
            _nBoxH = self._h/kwargs['pBoxH']
            
            if not ((_nBoxW.is_integer()) & (_nBoxH.is_integer())):
                assert kwargs['pad'], 'Please input valid pBoxW and pBoxH. Or set "pad" to True.'
                
                # add complementary parts to top and right
                xnew = int(np.ceil(_nBoxW) * kwargs['pBoxW'])
                ynew = int(np.ceil(_nBoxH) * kwargs['pBoxH'])
            
                self.pad(trgw=xnew, trgh=ynew, padvalue=kwargs['padcolor'], padalpha=kwargs['padalpha'])

            kwargs['nBoxW'] = int(self._w/kwargs['pBoxW'])
            kwargs['nBoxH'] = int(self._h/kwargs['pBoxH'])
                
        elif (kwargs['nBoxW']!=0) & (kwargs['nBoxH']!=0):
            
            # x and y pixels for each box
            _pBoxW = self._w/kwargs['nBoxW']
            _pBoxH = self._h/kwargs['nBoxH']
    
            if not ((_pBoxW.is_integer()) & (_pBoxH.is_integer())):
                assert kwargs['pad'], 'Please input valid nBoxW and nBoxH. Or set "pad" to True.'
                
                # add padding (top, right, bottom, left)
                newW = int(np.ceil(_pBoxW) * kwargs['nBoxW'])
                newY = int(np.ceil(_pBoxH) * kwargs['nBoxH'])
                self.pad(trgw=newW, trgh=newY, padvalue=kwargs['padcolor'], padalpha=kwargs['padalpha'])

            kwargs['pBoxW'] = int(self._w/kwargs['nBoxW'])
            kwargs['pBoxH'] = int(self._h/kwargs['nBoxH'])
        else:
            raise 'Please set valid nBoxW and nBoxH (or pBoxW and pBoxH).'
        
        assert kwargs['nBoxW']*kwargs['pBoxW']==self._w, f"'nBoxW': {kwargs['nBoxW']}   'pBoxW': {kwargs['pBoxW']}    'self._w': {self._w}"
        assert kwargs['nBoxH']*kwargs['pBoxH']==self._h, f"'nBoxH': {kwargs['nBoxH']}   'pBoxH': {kwargs['pBoxH']}    'self._h': {self._h}"
        
        # x and y for all boxes
        xys = list(product(range(0,self._w,kwargs['pBoxW']), range(0,self._h,kwargs['pBoxH'])))
        boxes = [self.mat[i[1]:(i[1]+kwargs['pBoxH']), i[0]:(i[0]+kwargs['pBoxW'])] for i in xys]
        # randomize the boxes
        bsboxes = np.random.permutation(boxes)
        # save as np.array
        bslist = [bsboxes[i:i+kwargs['nBoxW']] for i in range(0,len(bsboxes),kwargs['nBoxW'])]
        # bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
        bsmatm = np.asarray(bslist)
        if len(bsmatm.shape)==4:
            bsmatm = bsmatm[..., np.newaxis]
        bsmat = np.moveaxis(bsmatm, [-1, 1], [0, -2]).reshape(-1, self._h, self._w)
        
        # save box-scrambled images (and information)
        self.remat(np.squeeze(np.moveaxis(bsmat,0,2)))
        self._newfilename(newfname='_bscr')
        
    def mkphasescr(self, **kwargs):
        """Make phase scrambled stimuli.
        
        Other Parameters
        ----------------
        rms : float
            the desired RMS of the image. Defaults to 0.3.
        """
        defaultKwargs = {'rms':0.3}
        kwargs = {**defaultKwargs, **kwargs}
        
        # make a random phase
        randphase = np.angle(np.fft.fft2(np.random.rand(self._h, self._w)))

        if self._nchan==0:
            nchan = 1
            mat = self.mat[..., np.newaxis] # add one more axis
        else:
            nchan = self._nchan
            mat = self.mat
            
        outmat = np.empty(mat.shape)
        outmat[:] = np.NaN
        
        for i in range(nchan):
            thismat = self.mat[:,:,i]
            stdmat = (thismat - np.mean(thismat))/np.std(thismat)
            img_freq = np.fft.fft2(stdmat)
            amp = np.abs(img_freq)
            phase = np.angle(img_freq) + randphase
            outimg = np.real(np.fft.ifft2(amp * np.exp(np.sqrt(-1+0j)*phase)))
            stdout= (outimg - np.mean(outimg))/np.std(outimg)
            outmat[:,:,i] = (stdout - np.min(stdout)) * 255 / (np.max(stdout) - np.min(stdout))
            
        if self._nchan==0:
            outmat = outmat[:,:,0]
        
        self.remat(np.uint8(outmat))
        self._newfilename(newfname='_pscr')
        
        
    def filter(self, **kwargs):
        
        
        
        defaultKwargs = {'rms':0.3, 'maxvalue':255, 'filter':'low',
                         'cpi': 5,   # cycles per image (width) or cycles per degree if vapi>0
                         'vapi': 1,  # visual angle per image
                         'radius': 5, }
        kwargs = {**defaultKwargs, **kwargs}
        
        self.grayscale()
        # self.stdmat(kwargs['rms'])
        
        if kwargs['cpi'] > 0:
            # cycles per image (width) OR cycles per degree (along the width)
            radius = int(self._w / (2 * np.pi * kwargs['cpi'] * kwargs['vapi']))
        else:
            radius = kwargs['radius']
            
        # Apply low-pass filter
        low_pass_image = self.pil.filter(ImageFilter.GaussianBlur(radius=radius))

        # Create high-pass filter by subtracting low-pass image
        high_pass_image = ImageChops.subtract(self.pil, low_pass_image)
        # Enhance the high-pass image (optional, to increase contrast)
        high_pass_image = high_pass_image.point(lambda x: x * 2)  # Amplify high frequencies
        
        if kwargs['filter']=='low':
            self._repil(low_pass_image)
        elif kwargs['filter']=='high':
            self._repil(high_pass_image)
        
        self.stdmat(kwargs['rms'])
        self._newfilename(newfname='_'+kwargs['filter']+'_filtered')
            
            
    def psychopy_sffilter(self, **kwargs):
        """Apply spatial frequency filter to the image.
        
        Other Parameters
        ----------------
        rms : float
            the desired RMS of the image. Defaults to 0.3.
        maxvalue : int
            the maximum value of the image. Defaults to 255.
        filter : str
            the spatial frequency filter. Defaults to 'low'.
        cutoff : float
            the cutoff frequency. Defaults to 0.05.
        n : int
            the order of the filter. Defaults to 10.
        """
        # https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        import psychopy.filters
        
        defaultKwargs = {'rms':0.3, 'maxvalue':255, 'filter':'low',
                         'cutoff': 0.05, 'n': 10}
        kwargs = {**defaultKwargs, **kwargs}
        
        self.grayscale()
        self.stdmat(kwargs['rms'])
        img_freq = np.fft.fft2(self.mat)

        # calculate amplitude spectrum
        # img_amp = np.fft.fftshift(np.abs(img_freq))

        # # for display, take the logarithm
        # img_amp_disp = np.log(img_amp + 0.0001)

        # # rescale to -1:+1 for display
        # img_amp_disp = (
        #     ((img_amp_disp - np.min(img_amp_disp)) * 2) / 
        #     np.ptp(img_amp_disp)  # 'ptp' = range
        # ) - 1
        
        if kwargs['filter']=='low':
            # for generating blury images
            fsfilt = psychopy.filters.butter2d_lp(
                size=(self._w, self._h),
                cutoff=kwargs['cutoff'],
                n=kwargs['n']
            )
        elif kwargs['filter']=='high':      
             # for gernerating sharp images
            fsfilt = psychopy.filters.butter2d_hp(
                size=(self._w, self._h),
                cutoff=kwargs['cutoff'],
                n=kwargs['n']
            )
        else:
            raise 'Cannot identify the "filter" value...'
        
        img_filt = np.fft.fftshift(img_freq) * fsfilt.transpose()
        # convert back to an image
        img_new = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))
        self.remat(img_new)
        # standardize the image 
        img_new = self.stdim(kwargs['rms'])
        self._newfilename(newfname='_'+kwargs['filter']+'_filtered')
    
    
    