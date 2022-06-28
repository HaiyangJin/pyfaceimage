"""
Tools used during surface-based analysis with FreeSurfer.
"""

from .im import (image)
        
from .dictim import (dir, deepcopy,
                   checksample, 
                   read, save, grayscale, cropoval,
                   croprect, resize, pad, mkboxscr, 
                   sffilter, mkphasescr)

from .multipleim import (mkcf)

__all__ = ['image', 
           'dir', 'deepcopy',
           'checksample', 
           'read', 'save', 'grayscale', 'cropoval',
           'croprect', 'resize', 'pad', 'mkboxscr', 
           'sffilter', 'mkphasescr',
           'mkcf']
