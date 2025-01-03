
import warnings, string
import numpy as np
import pyfaceimage


def mk_cf_design(cfs, **kwargs):
    """Make a list, which should work in PychoPy or other software, for the composite face task.

    Parameters
    ----------
    cfs : int, dict, list
        the number of composite faces (e.g., `4`), a dictionary of composite faces (e.g., `imdict`), or a list of composite face names (e.g., `['AB', 'CD', 'EF', 'GH']`).
        
    Other Parameters
    ----------------
    isTopCued : list
        whether the top half is cued. Defaults to `[1]`.
    isCongruent : list
        whether the top and bottom halves are congruent. Defaults to `[0, 1]`.
    isAligned : list
        whether the top and bottom halves are aligned. Defaults to `[0, 1]`.
    isCuedSame : list
        whether the cued half is the same as the target half. Defaults to `[0, 1]`.
    studyIsAligned : int
        whether the study faces are always aligned. Defaults to `1`.
    faceselector : list
        the face selector for the composite faces. Defaults to the default selector (`default_selector`).
    cue_str : list
        the cue strings. Defaults to `['bot', 'top']`.
    con_str : list
        the congruency strings. Defaults to `['inc', 'con']`.
    ali_str : list
        the alignment strings. Defaults to `['mis', 'ali']`.
    ca_str : list
        the correct answer strings. Defaults to `['dif', 'sam']`.
    task : str
        the task name. Defaults to `'CF'`.
    cf_group : str
        the composite face group name. Defaults to `''`.
    cf_sep : str
        the separator within the composite face names. Defaults to `'_'`.
    cfs_sep : str
        the separator between the study and test faces. Defaults to `''`.
    showlevels : bool
        whether to show the levels of the independent variables. Defaults to `False`.
    is_rand : bool
        whether to randomize the design matrix. Defaults to `True`.
    face_order : str
        Whether to randomize or order the original face names. Defaults to `'sort'`. Other options are `'rand'`.

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame containing the composite face task design matrix.
        
    Examples
    --------
    >>> design = mkcflist(4)
    >>> design = mkcflist(['AB', 'CD', 'EF', 'GH'])
    >>> design.to_csv('cf_design.csv', index=False) # save the design matrix to a csv file
    """
    
    default_selector = [
        [0, 1, 0, 1],  # TCS
        [0, 1, 2, 3],  # TCD
        [0, 1, 0, 2],  # TIS
        [0, 1, 3, 1],  # TID
        [0, 1, 0, 1],  # BCS
        [0, 1, 2, 3],  # BCD
        [0, 1, 3, 1],  # BIS
        [0, 1, 0, 2]   # BID
    ]
    defaultKwargs = {'isTopCued': [1], 'isCongruent': [0, 1], 
                     'isAligned': [0, 1], 'isCuedSame': [0, 1], 
                     'studyIsAligned': 1,
                     'faceselector': default_selector,
                     'cue_str': ['bot', 'top'],
                     'con_str': ['inc', 'con'],
                     'ali_str': ['mis', 'ali'],
                     'ca_str': ['dif', 'sam'],
                     'task': 'CCF',
                     'cf_group': '',
                     'cf_sep': '_',
                     'cfs_sep': '',
                     'showlevels': False,
                     'is_rand': True,
                     'face_order': 'sort'}
    kwargs = {**defaultKwargs, **kwargs}
    
    # deal with input cfs
    if isinstance(cfs, int):
        ncfs = cfs
        # use alphabets as the default cf names
        alphabets = list(string.ascii_uppercase)
        cfs = alphabets[:ncfs]
    elif isinstance(cfs, dict):
        ncfs = len(cfs)
        cfs = list(cfs.keys())
    elif isinstance(cfs, list):
        ncfs = len(cfs)
                
    # throw warnings if the number of cfs is less than 4
    if ncfs < 4:
        warnings.warn(f'The number of composite faces shoul not be less than 4 (ncfs: {ncfs}).')
    
    # whether randomize or sort the original face names
    if kwargs['face_order'] == 'rand':
        np.random.shuffle(cfs)
    elif kwargs['face_order'] == 'sort':
        cfs.sort()
    
    # make the exp design matrix
    exp_conditions = [("isTopCued", kwargs['isTopCued']), 
                      ("isCongruent", kwargs['isCongruent']),
                      ("isCuedSame", kwargs['isCuedSame']),
                      ("isAligned", kwargs['isAligned']),
                      ("basefaceIndex", list(range(ncfs)))]
    design = pyfaceimage.utilities.exp_design_builder(exp_conditions, is_rand=kwargs['is_rand'])
    print(f'Total trials: {len(design)}')
    
    # generate the composite face indices
    design['trialType'] = 4*(1-design['isTopCued']) + 2*(1-design['isCongruent']) + (1-design['isCuedSame'])
    design['base_selector'] = design['trialType'].apply(lambda x: kwargs['faceselector'][x]) # base selector
    design['thisFaceSet'] = design.apply(lambda row: [np.mod(x + row['basefaceIndex'], ncfs) for x in row["base_selector"]], axis=1)
    
    # make the composite face names
    design['studyFace'] = design.apply(lambda row: cfs[row['thisFaceSet'][0]]+kwargs['cf_sep']+cfs[row['thisFaceSet'][1]]+'_'+kwargs['ali_str'][max(row['isAligned'], kwargs['studyIsAligned'])], axis=1)
    design['testFace'] = design.apply(lambda row: cfs[row['thisFaceSet'][2]]+kwargs['cf_sep']+cfs[row['thisFaceSet'][3]]+'_'+kwargs['ali_str'][row['isAligned']], axis=1)
    
    # add task column and move it to the front
    design['Task'] = kwargs['task']
    design = design[["Task"] + [col for col in design.columns if col != "Task"]]
    
    # combine the study and test faces
    if kwargs['cfs_sep']:
        design['cfPair'] = design['studyFace'] + kwargs['cfs_sep'] + design['testFace']
        
    # add the stim/CF group information
    if kwargs['cf_group']:
        design['cfGroup'] = kwargs['cf_group']
        
    # update IV levels
    if kwargs['showlevels']:
        design['Cue'] = design.apply(lambda row: kwargs['cue_str'][row['isTopCued']], axis=1)
        design['Congruency'] = design.apply(lambda row: kwargs['con_str'][row['isCongruent']], axis=1)
        design['Alignment'] = design.apply(lambda row: kwargs['ali_str'][row['isAligned']], axis=1)
        design['CorrectAnswer'] = design.apply(lambda row: kwargs['ca_str'][row['isCuedSame']], axis=1)
        columns_to_reorder = ['Cue', 'Congruency', 'Alignment', 'CorrectAnswer', 'studyFace', 'testFace']
        new_order = columns_to_reorder + [col for col in design.columns if col not in columns_to_reorder]
        design = design[new_order]
        
    # remove unused columns
    design = design.drop(columns=['trialType', 'base_selector', 'thisFaceSet'])
    
    return design


def mkcf_prf(dict_cf, dict_bg, imsize_bg=(500,500), opaque=.75):
    """Make composite face stimuli for the pRF experiment.

    Parameters
    ----------
    dict_cf : dict
        a dictionary of composite faces.
    dict_bg : dict
        a dictionary of background images.
    imsize_bg : tuple, optional
        the output image size, by default (500,500)
    opaque : float, optional
        ratio of the opaque region, by default .75

    Returns
    -------
    dict
        a dictionary of composite face stimuli to be used for the pRF experiment.
    """
    
    # Initialize the output dict
    imdict_out = {}
    
    for k,im_fore in dict_cf.items():
        
        # this fore and background images
        k_back = k.replace('mis', 'ali').replace('_l', '').replace('_r', '')+'_pscr' # only use aligend background
        im_back = dict_bg[k_back]

        # phase scrambed background
        im_back.resize(trgw=imsize_bg[0], trgh=imsize_bg[1])
        
        # mask
        mask = pyfaceimage.utilities.radial_gaussian(imsize_bg, opaque)

        # add fore to back
        im_back_tmp = im_back.deepcopy()
        pyfaceimage.multim.composite(im_back_tmp, im_fore)

        # apply mask
        im_back_tmp.pil.putalpha(mask)
        
        # update the filename
        im_back_tmp.filename = im_fore.filename.replace('_pad', '')
        im_back_tmp._updatefromfilename()
        
        # save the output dict
        imdict_out[k] = im_back_tmp
        del im_back_tmp

    return imdict_out

