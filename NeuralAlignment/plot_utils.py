from pathlib import Path
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import pickle
import os
# from sklearn.decomposition import PCA
# import plotly.graph_objects as go
# import plotly.express as px
import datetime
import sys
import copy
from tqdm import tqdm
from os.path import join
import getpass
import typing
import matplotlib
import glob
from textwrap import wrap
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

from resources import *
def shorten_savestr(savestr: str):
    """
    Replace True and False with 1 and 0.
    Replace None with 0.
    Replace 'anatglasser' with 'ag'
    """
    savestr = savestr.replace('True', '1').replace('False', '0').replace('None', '0').replace('anatglasser', 'ag')

    # Also shorten 848-853-865-875-876 to 5T and 797-841-880-837-856 to 5D
    savestr = savestr.replace('848-853-865-875-876', '5T').replace('797-841-880-837-856', '5D')

    # Shorten the feature stimset from 'beta-control-neural_stimset_D-S_light_compiled-feats' to 'bcn_D-S_feats
    savestr = savestr.replace('beta-control-neural_stimset_D-S_light_compiled-feats', 'bcn_D-S_feats')

    # Shorten bert-large-cased to bert-lc
    savestr = savestr.replace('bert-large-cased', 'bert-lc')

    # Shorten surprisal-gpt2-xl_raw_mean to log_prob-gpt2-xl
    savestr = savestr.replace('surprisal-gpt2-xl_raw_mean', 'log_prob-gpt2-xl')

    # And also shorten the other features
    savestr = savestr.replace('rating_gram_mean', 'gram')
    savestr = savestr.replace('rating_sense_mean', 'sense')
    savestr = savestr.replace('rating_others_thoughts_mean', 'others')
    savestr = savestr.replace('rating_physical_mean', 'physical')
    savestr = savestr.replace('rating_places_mean', 'places')
    savestr = savestr.replace('rating_valence_mean', 'valence')
    savestr = savestr.replace('rating_arousal_mean', 'arousal')
    savestr = savestr.replace('rating_imageability_mean', 'imageability')
    savestr = savestr.replace('rating_frequency_mean', 'frequency')
    savestr = savestr.replace('rating_conversational_mean', 'conversational')

    # For pretransformer predictions
    # pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean to pred-surp-gpt2-xl_mean
    savestr = savestr.replace('pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean', 'pred-surp-gpt2-xl_mean')
    savestr = savestr.replace('pretransformer_pred-surprisal-5gram-surprisal-5gram_mean', 'pred-surp-5gram_mean')
    savestr = savestr.replace('pretransformer_pred-surprisal-pcfg-surprisal-pcfg_mean', 'pred-surp-pcfg_mean')

    return savestr


def separate_savestr_prefix_specific_target(savestr_prefix_AND_specific_target: str):
    """Separate the savestr_prefix from the specific target.

    Args
        savestr_prefix_AND_specific_target: str, e.g., '62-rois-None'

    Returns
        savestr_prefix: str, e.g., '62-rois'
        specific_target: str, e.g., 'None'
    """

    # If more than one hyphen, the string after the last hypen is the specific target
    if savestr_prefix_AND_specific_target.count('-') > 1:
        savestr_prefix = '-'.join(savestr_prefix_AND_specific_target.split('-')[:-1])
        specific_target = savestr_prefix_AND_specific_target.split('-')[-1]
    else:
        savestr_prefix = savestr_prefix_AND_specific_target.split('-')[0]
        specific_target = savestr_prefix_AND_specific_target.split('-')[1]

    return savestr_prefix, specific_target


def separate_savestr_modeltype(savestr: str):
    """Obtain modeltype from the savestr (filename)

    Args
        savestr: str, e.g., 'CV-k-5-folds_SOURCE-act1_TARGET-20221214-None_d-swr-5-0.05-bySessVoxZ_MAPPING-pca800-True-False.pkl'

    Returns
        modeltype: str, e.g., '62-rois'
    """
    modeltype = savestr.split('_')[3].split('-')[0]
    if modeltype == 'MAPPING':  # bug when there are other hyphens etc
        modeltype = savestr.split('_')[4].split('-')[0]
    return modeltype


def get_roi_list_name(rois_of_interest: str):
	"""
	If roi exists in d_roi_lists_names, use the name in the dictionary otherwise use the string itself.
	E.g. if we pass rois_of_interest = 'lang_LH_ROIs', we get back the list of ROIs in the language network.
	We do retain the 'lang_LH_ROIs' as the name (for use in save string, etc.)


	:param rois_of_interest:
	:param d_roi_lists_names:
	:return:
	"""

	rois_of_interest_name = rois_of_interest
	if rois_of_interest in d_roi_lists_names.keys():
		rois_of_interest = d_roi_lists_names[rois_of_interest]
	else:
		rois_of_interest = None # I.e. just use the string as the name, and pass a None such that we use all ROIs

	return rois_of_interest, rois_of_interest_name
