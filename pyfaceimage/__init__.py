"""
Tools used during surface-based analysis with FreeSurfer.
"""

from .im import (image)
        
from .dictim import (dir, deepcopy,
                   sample, 
                   read, save, grayscale, cropoval,
                   croprect, resize, pad, mkboxscr, 
                   sffilter, mkphasescr,
                   mkcfs)

from .multipleim import (mkcf)

__all__ = ['image', 
           'dir', 'deepcopy',
           'sample', 
           'read', 'save', 'grayscale', 'cropoval',
           'croprect', 'resize', 'pad', 'mkboxscr', 
           'sffilter', 'mkphasescr',
           'mkcfs',
           'mkcf']
