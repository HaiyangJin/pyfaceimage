"""
The class to process single images.    
"""
    
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


class stim:
    def __init__(self, file, dir='.'):
        # make sure .file exists  os.getcwd()
        file = os.path.join(dir, file)
        assert os.path.isfile(file), f'Cannot find {file}...'
        self.info(file)
         # default settings
        self.bsmat = None
        self.psmat = None
        self.cmat = None
        self.bsfile = None
        self.psfile = None
        self.cfile = None
            
    def info(self, file):
        self.file = file
        self.fn = os.path.basename(file)
        self.fnonly = os.path.splitext(self.fn)[0]
        self.ext = os.path.splitext(self.fn)[1]
        self.dir = os.path.dirname(file)
        self.group = os.path.split(self.dir)[1] # the upper-level folder name
        self.isfile = os.path.isfile(file)
        
    def imread(self, **kwargs):
        self.mat = plt.imread(self.file, **kwargs)
        self.update()
        
    def imsave(self, extrastr='', out='mat',**kwargs):
        outmats = {'mat': self.mat, 
                   'bsmat': self.bsmat,
                   'psmat': self.psmat,
                   'cmat': self.cmat} # custom
        outfiles = {'mat': self.file, 
                   'bsmat': self.bsfile,
                   'psmat': self.psfile,
                   'cmat': self.cfile}
        
        self.outfile = os.path.splitext(outfiles[out])[0]+extrastr+os.path.splitext(outfiles[out])[1]
        
        if self.nlayer==1:
            theout = outmats[out][:,:,0]
        else:
            theout = outmats[out]
            
        plt.imsave(self.outfile,theout.copy(order='C'),**kwargs)
        
    def update(self):
        if len(self.mat.shape)<3:
            self.mat = self.mat[..., np.newaxis]
        
        self.dims = self.mat.shape
        self.ndim = len(self.dims)
        self.y, self.x, self.nlayer = self.dims
        
    def updatelayer(self, layer=3):
        if (self.nlayer==1) & (layer==3):
            # self.mat = np.stack((self.mat,)*3, axis=-1) # when .ndim==2
            self.mat = np.repeat(self.mat, 3, axis=2)

        if (self.nlayer>2) & (layer==1):
            self.mat = self.mat[:,:,0]
            
        self.update()
                
    def addalpha(self, amat=None, avalue=1):
        if amat is None:
            amat = np.ones_like(self.mat[:,:,0:1],dtype=np.uint8)
        self.mat = np.concatenate((self.mat, amat*avalue), axis=2)
        self.update()
        
    def pad(self, trgx, trgy, padvalue=0, top=True, left=True):
        """Add padding to the image/stimuli.

        Args:
            trgx (int): the x of the target/desired stimuli. 
            trgy (int): the y of the target/desired stimuli.
            padvalue (int, optional): padding value. Defaults to 0.
            top (bool, optional): padding more to top if needed. Defaults to True.
            left (bool, optional): padding more to left if needed. Defaults to True.
        """
        assert(trgx>=self.x)
        assert(trgy>=self.y)
        
        x1 = int(np.ceil(trgx-self.x)/2)
        x2 = trgx-self.x-x1
        y1 = int(np.ceil(trgy-self.y)/2)
        y2 = trgy-self.y-y1
        
        if top:
            ytop, ybot = y1,y2
        else:
            ytop, ybot = y2,y1
            
        if left:
            xleft, xright = x1,x2
        else:
            xleft, xright = x2,x1
            
        self.mat = np.hstack((
            np.ones((trgy,xleft,self.nlayer),dtype=np.uint8)*padvalue, 
            np.vstack((
                np.ones((ytop,self.x,self.nlayer),dtype=np.uint8)*padvalue, 
                self.mat,
                np.ones((ybot,self.x,self.nlayer),dtype=np.uint8)*padvalue
            )),
            np.ones((trgy,xright,self.nlayer),dtype=np.uint8)*padvalue, 
        ))
        self.update()
        
    def mkboxscr(self, *args, **kwargs):
        """Make box scrambled stimuli.
        """
        defaultKwargs = {'nBoxX':10, 'nBoxY':16, 
                     'pBoxX':0, 'pBoxY':0, 
                     'makeup': False, 'mkcolor':0, 'mkalpha': None}
        kwargs = {**defaultKwargs, **kwargs}
            
        # x and y pixels for each box
        _pBoxX = self.x/kwargs['nBoxX']
        _pBoxY = self.y/kwargs['nBoxY']
    
        if not ((_pBoxX.is_integer()) & (_pBoxY.is_integer())):
            if kwargs['makeup']:
                # add complementary parts (top, right, bottom, left)
                xnew = int(np.ceil(_pBoxX) * kwargs['nBoxX'])
                ynew = int(np.ceil(_pBoxY) * kwargs['nBoxY'])
                self.pad(trgx=xnew, trgy=ynew, padvalue=kwargs['mkcolor'])
            
                if kwargs['mkalpha'] is not None:
                    self.addalpha(amat=None, avalue=kwargs['mkalpha'])

                _pBoxX = xnew/kwargs['nBoxX']
                _pBoxY = ynew/kwargs['nBoxY']
                
            else:
                raise Exception('Please input valid nBoxX and nBoxY. Or set "makeup" to True.')
        
        if kwargs['pBoxX']==0 | kwargs['pBoxY']==0:
            kwargs['pBoxX'] = int(np.ceil(_pBoxX))
            kwargs['pBoxY'] = int(np.ceil(_pBoxY))
        
        _nBoxX = self.x/kwargs['pBoxX']
        _nBoxY = self.y/kwargs['pBoxY']
            
        if not ((_nBoxX.is_integer()) & (_nBoxY.is_integer())):
            if kwargs['makeup']:
                # add complementary parts to top and right
                xnew = int(np.ceil(_nBoxX) * kwargs['pBoxX'])
                ynew = int(np.ceil(_nBoxY) * kwargs['pBoxY'])
            
                self.pad(trgx=xnew, trgy=ynew, padvalue=kwargs['mkcolor'])
                            
                if kwargs['mkalpha']:
                    self.addalpha(amat=None, avalue=kwargs['mkalpha'])

                kwargs['nBoxX'] = self.x/kwargs['pBoxX']
                kwargs['nBoxY'] = self.y/kwargs['pBoxY']
                
            else:
                raise Exception('Please input valid pBoxX and pBoxY. Or set "makeup" to True.')
        
        self.update()
        assert(kwargs['nBoxX']*kwargs['pBoxX']==self.x)
        assert(kwargs['nBoxY']*kwargs['pBoxY']==self.y)

        # x and y for all boxes
        xys = list(product(range(0,self.x,kwargs['pBoxX']), range(0,self.y,kwargs['pBoxY'])))
        boxes = [self.mat[i[1]:(i[1]+kwargs['pBoxY']), i[0]:(i[0]+kwargs['pBoxX'])] for i in xys]
        # randomize the boxes
        bsboxes = np.random.permutation(boxes)
        # save as np.array
        bslist = [bsboxes[i:i+kwargs['nBoxX']] for i in range(0,len(bsboxes),kwargs['nBoxX'])]
        # bsmat = np.moveaxis(np.asarray(bslist), [-1, 1], [0, -2]).reshape(-1, y, x)
        bsmatm = np.asarray(bslist)
        if len(bsmatm.shape)==4:
            bsmatm = bsmatm[..., np.newaxis]
        bsmat = np.moveaxis(bsmatm, [-1, 1], [0, -2]).reshape(-1, self.y, self.x)
        
        self.bsmat = np.squeeze(np.moveaxis(bsmat,0,2))
        self.bsfile = os.path.join('.', self.dir, self.fnonly+'bscr'+self.ext)
        
    # def mkphasescr(self):