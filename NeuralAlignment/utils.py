import pandas as pd
import getpass
import warnings
import os
from os.path import join
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import typing
from matplotlib import pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pickle
from datetime import datetime
import sys
import re
import json
from scipy.spatial.distance import cosine

torch.set_default_dtype(torch.double)

user = getpass.getuser()
date = datetime.now().strftime("%Y%m%d-%T")
now = datetime.now()
date_tag = now.strftime("%Y%m%d")

from resources import *


#### MISC FUNCTIONS ####
def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        print(f'Already a boolean: {v}')
        return v
    if v.lower() in ('true', 't'):
        print(f'String arg - True: {v}')
        return True
    elif v.lower() in ('false', 'f'):
        print(f'String arg - False: {v}')
        return False
    else:
        print(f'String arg - {v}')
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    """If string is 'None', return None. Else, return the string"""
    if v is None:
        print(f'Already None: {v}')
        return v
    if v.lower() in ('none'):
        print(f'String arg - None: {v}')
        return None
    else:
        return v


def nansem(x, axis=0):
    """Compute standard error of the mean ignoring NaNs.
    Ddof is by default 1 in pandas, so the division is by sqrt(n-1) where n is the number of non-NaN values.

    Parameters
    ----------
    x : array (2D)
        Input array.

    Returns
    -------
    array
        Standard error of the mean of the non-NaN values.

    """
    df = pd.DataFrame(x)
    sem = df.sem(axis=axis).values
    return sem


def format_argparse(args):
    """Make args into a list of format ['--argument1', args.argument1, '--argument2', args.argument2, ...] """
    formatted_args = []
    for i in vars(args):
        # Append --argument to the argument name
        formatted_args.append(f'--{i}')
        formatted_args.append(str(getattr(args, i)))  # arguments have to be passed as strings

    return formatted_args


def cos_sim_matrix(A, B):
    """Compute the cosine similarity matrix between two matrices A and B.
    1 means the two vectors are identical. 0 means they are orthogonal. -1 means they are opposite."""
    return (A * B).sum(axis=1) / (A * A).sum(axis=1) ** .5 / (B * B).sum(axis=1) ** .5


def obtain_sess_id(UID: str,
                   sess_num: str,
                   d_UID_to_session: dict, ) -> str:
    """ Given a UID and a session number, return the session ID (FED_XXXXX_3T1).
    Sessions for the same UID are joined with a dash, while sessions across UIDs are joined with an underscore.

    Args:
        UID: str (e.g., '853', '848-853' if the aggregate of two subjects)
        sess_num: str (e.g., '1-2')
        d_UID_to_session: dict (e.g., '848':{'1':'FED_20220420b_3T1',
                       '2':'FED_20220427a_3T1',
                       '1-2':'FED_20220420b_3T1-FED_20220427a_3T1',})
    Returns:
        str (e.g., 'FED_20220420b_3T1-FED_20220427a_3T1')

    """
    # If UID consists of multiple subjects, split it
    if '-' in UID:
        UIDs = UID.split('-')
        sess_ids = []
        for UID in UIDs:
            sess_ids.append(d_UID_to_session[UID][sess_num])
        sess_id = '_'.join(sess_ids)
    else:
        sess_id = d_UID_to_session[UID][sess_num]

    return sess_id


def get_corpus_base_name(neural_data: pd.DataFrame,
                         stimset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Given two dataframes with a corpus name separated with periods, get rid of everything besides the term before the
    first period and after the last period.
    E.g., beta-control-neural-T.848-853.1 will become beta-control-neural-T.1

    Args:
        neural_data: pd.DataFrame (rows: items, columns: neuroids)
        stimset: pd.DataFrame (rows: items, columns: stim metadata)

    Returns:
        neural_data: pd.DataFrame (rows: items, columns: neuroids) with modified index
        stimset: pd.DataFrame (rows: items, columns: stim metadata) with modified index
    """

    # Get rid of the UID string notation in the row index. Strip the term between punctuations
    assert (neural_data.index == stimset.index).all()

    stripped_neural_data_index = [x.split('.')[0] + '.' + x.split('.')[-1] for x in neural_data.index]
    neural_data.index = stripped_neural_data_index

    stripped_stimset_index = [x.split('.')[0] + '.' + x.split('.')[-1] for x in stimset.index]
    stimset.index = stripped_stimset_index

    assert (stripped_neural_data_index == stripped_stimset_index)

    return neural_data, stimset


def drop_sentences_for_uid_853(stimset: pd.DataFrame,
                               neural_data: pd.DataFrame,
                               stimset_T_path: str, ):
    """
    We want to drop the neural values from the 12 sentences that were exchanged, but leave the 8 sentences with minor edits.
    We want to switch in the updated 20 sentences from the df_feats dataframe (because those align across participants).
    """

    # Assert that we only perform this for 853
    assert (stimset.index.str.split('_').str[0].unique() == neural_data.index.str.split('_').str[0].unique()).all()
    UID = stimset.index.str.split('_').str[0].unique()[0]
    assert (UID == '853')

    # 1. Get the "correct", i.e. exchanged sentences from another participant's stimset and exchange them with the old ones
    # for 853.
    stimset_T = pd.read_csv(stimset_T_path)

    # Check that indeed 20 sentences are different
    assert ((stimset_T['sentence'].values == stimset['sentence'].values).sum() == 1000 - 20)
    assert (stimset_T.item_id.values == stimset.item_id.values).all()

    # Let's iterate through the stimset from 853, and if drop_bool or change_bool == 1, we exchange the sentence with the one from stimset_T
    stimset_orig = stimset.copy(deep=True)

    for sent in stimset.iterrows():
        if sent[1].drop_bool == 1:
            # Get the sentence from stimset_T
            sent_T = stimset_T.loc[stimset_T.item_id == sent[1].item_id, 'sentence'].values[0]

            print(f'Exchanging sentence {sent[1].sentence} with {sent_T}')

            # Exchange the sentence
            stimset.loc[stimset.item_id == sent[1].item_id, 'sentence'] = sent_T
        elif sent[1].change_bool == 1:
            # Get the sentence from stimset_T
            sent_T = stimset_T.loc[stimset_T.item_id == sent[1].item_id, 'sentence'].values[0]

            print(f'Exchanging sentence {sent[1].sentence} with {sent_T}')

            # Exchange the sentence
            stimset.loc[stimset.item_id == sent[1].item_id, 'sentence'] = sent_T
        else:
            continue

    # Check how many sentences were exchanged
    assert ((stimset['sentence'].values == stimset_orig['sentence'].values).sum() == 1000 - 20)

    # 2. Set neural_data rows to NaN for the 12 exchanged sentences
    neural_data_orig = neural_data.copy(deep=True)
    neural_data.loc[stimset_orig['drop_bool'] == 1, :] = np.nan

    print(f'Dropping {len(stimset_orig[stimset_orig["drop_bool"] == 1])} sentences for UID {UID}')

    return stimset, neural_data


def collapse_session_indexing_to_UID(neural_data: pd.DataFrame,
                                     stimset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Given two dataframes, neural and stimset, (outputted from load_voxel_data.py), with UID and session indexing, e.g.:
    '841_FED_20220803a_3T1-FED_20220809a_3T1-FED_20220812a_3T1.1'

    Collapse the sessions into a single UID, e.g.:
    '841.1'

    Args:
        neural_data: pd.DataFrame (rows: items, columns: neuroids)
        stimset: pd.DataFrame (rows: items, columns: stim metadata)

    Returns:
        neural_data: pd.DataFrame (rows: items, columns: neuroids) with modified index
        stimset: pd.DataFrame (rows: items, columns: stim metadata) with modified index
        UID: the unique UID for the supplied neural data and stimset
    """
    neural_data_to_return = neural_data.copy(deep=True)
    stimset_to_return = stimset.copy(deep=True)

    # Get rid of the UID string notation in the row index. Strip the term between punctuations
    assert (neural_data_to_return.index == stimset_to_return.index).all()

    stripped_neural_data_UID = [x.split('_')[0] for x in neural_data_to_return.index]  # Obtain UID, e.g. 841
    stripped_neural_data_UID_index = [x.split('_')[0] + '.' + x.split('.')[-1] for x in neural_data_to_return.index]

    assert len(np.unique(stripped_neural_data_UID)) == 1  # Make sure all UIDs are the same

    neural_data_to_return.index = stripped_neural_data_UID_index

    stripped_stimset_UID = [x.split('_')[0] for x in stimset_to_return.index]  # Obtain UID, e.g. 841
    stripped_stimset_UID_index = [x.split('_')[0] + '.' + x.split('.')[-1] for x in stimset_to_return.index]

    assert len(np.unique(stripped_stimset_UID)) == 1  # Make sure all UIDs are the same

    stimset_to_return.index = stripped_stimset_UID_index

    assert (stripped_stimset_UID_index == stripped_neural_data_UID_index)

    return neural_data_to_return, stimset_to_return, stripped_neural_data_UID[
        0]  # Return the unique UID for the supplied neural data and stimset


def add_nan_rows_to_df(df: pd.DataFrame,
                       rois_to_check_for: typing.Union[list, np.ndarray],
                       data_col: str,
                       ) -> pd.DataFrame:
    """Given a dataframe where each row is a stimuli x ROI prediction (such as df_pred_external),
    if a given ROI is missing from the dataframe, add it and fill with NaNs (pretending it is just an ordinary
    ROI, but with no data (remove the 'real data' as specified by "data_col")

    Args:
        df (pd.DataFrame): Dataframe where each row is a stimuli x ROI prediction (such as df_pred_external)
        rois_to_check_for (typing.Union[list, np.ndarray]): List of ROIs to check for (mapping_target_mismatch_cols)
        data_col (str): Name of the column that contains the actual data (e.g. 'pred-full_from-848-853-865-875-876')

    Returns:
        pd.DataFrame: Dataframe with NaNs added for missing ROIs (concatenated with the original dataframe)

    """

    # In df_pred_external, each row is a stimuli x ROI prediction, hence we need to add the missing rows
    for col in rois_to_check_for:  # mapping_target_mismatch_cols
        if col not in df.roi.unique():
            # Take the first ROI in the df, copy, make the data_col (pred-full_from-{mapping UID}) nan, and add the new ROI name
            # This retains the stimuli metadata, which still persists despite there being no prediction being made on the ROI
            mock_roi = df.roi.unique()[0]
            df_mock = df.loc[df.roi == mock_roi, :].copy(deep=True)
            # Make all columns values nan
            df_mock[data_col] = np.nan
            df_mock['roi'] = col
            df = pd.concat([df, df_mock], axis=0)
            print(f'{col} not in df_pred_external. Adding it and filling with NaN')

    return df


### ENCODER FUNCTIONS ###

def pick_matching_token_ixs(
        batchencoding: "transformers.tokenization_utils_base.BatchEncoding",
        char_span_of_interest: slice,
        verbose: bool = False) -> slice:
    """Picks token indices in a tokenized encoded sequence that best correspond to
        a substring of interest in the original sequence, given by a char span (slice)

    Args:
        batchencoding (transformers.tokenization_utils_base.BatchEncoding): the output of a
            `tokenizer(text)` call on a single text instance (not a batch, i.e. `tokenizer([text])`).
        char_span_of_interest (slice): a `slice` object denoting the character indices in the
            original `text` string we want to extract the corresponding tokens for

    Returns:
        slice: the start and stop indices within an encoded sequence that
            best match the `char_span_of_interest`
    """
    from transformers import tokenization_utils_base

    start_token = 0
    end_token = batchencoding.input_ids.shape[-1]
    for i, _ in enumerate(batchencoding.input_ids.reshape(-1)):
        span = batchencoding[0].token_to_chars(
            i
        )  # batchencoding 0 gives access to the encoded string

        if span is None:  # for [CLS], no span is returned
            if verbose:
                print(f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"')
            continue
        else:
            span = tokenization_utils_base.CharSpan(*span)

        if span.start <= char_span_of_interest.start:
            start_token = i
        if span.end >= char_span_of_interest.stop:
            end_token = i + 1
            break

    assert (
            end_token - start_token <= batchencoding.input_ids.shape[-1]
    ), f"Extracted span is larger than original span"

    return slice(start_token, end_token)


def package_top_bottom_results_as_df(df: pd.DataFrame,
                                     stimset: pd.DataFrame,
                                     pred_full: typing.Union[dict, None],
                                     n_stim: int, ):
    """
    Package the top and bottom predicted stimuli into a a dataframe with stimuli x ROI as rows, predictions in the pred column,
    index as {stimuli identifier}-{roi}, and the corresponding stimuli metadata as columns.

    If pred_full is not None, we compute stretch metrics.

    Args:
        df: dataframe with predictions. Assumes stimuli as rows and predictions as columns (named as ROIs)
        stimset: dataframe with stimuli metadata (assumes same indices as the predictions dataframe, df)
        pred_full: dictionary with keys = 'y' and 'y_pred-full' and values = dataframes with stimuli as rows and
            predictions for ROIs as columns
        n_stim: number of stimuli to include in the top and bottom

    Returns:
        df: dataframe with top and bottom stimuli as rows (for each ROI), predictions in the pred column

    """
    stimset = stimset.copy(deep=True)

    # Package into a df
    df_lst = []
    for roi in df.columns.values:
        # Obtain the min/max stimuli for each ROI separately!
        df_preds_roi = df.copy(deep=True)[[roi]].sort_values(by=roi, ascending=False).rename(
            columns={roi: f'pred'})

        df_preds_roi_top = df_preds_roi.head(n_stim)
        df_preds_roi_bottom = df_preds_roi.tail(n_stim)

        # Obtain indices for the top and bottom stimuli and merge with the stimset
        indices_top = df_preds_roi_top.index
        indices_bottom = df_preds_roi_bottom.index
        stimset_roi_top = stimset.reindex(indices_top)
        stimset_roi_bottom = stimset.reindex(indices_bottom)

        # Merge the top and bottom stimuli with the predictions
        assert (df_preds_roi_top.index.values == stimset_roi_top.index.values).all()
        df_preds_roi_top = pd.merge(df_preds_roi_top, stimset_roi_top, left_index=True, right_index=True)
        df_preds_roi_top['search_type'] = 'top'
        assert (df_preds_roi_bottom.index.values == stimset_roi_bottom.index.values).all()
        df_preds_roi_bottom = pd.merge(df_preds_roi_bottom, stimset_roi_bottom, left_index=True, right_index=True)
        df_preds_roi_bottom['search_type'] = 'bottom'

        # Merge top and bottom
        df_preds_roi_top_bottom = pd.concat([df_preds_roi_top, df_preds_roi_bottom], axis=0)
        df_preds_roi_top_bottom['roi'] = roi

        # Count num stimuli in df_preds_roi
        df_preds_roi_top_bottom['num_stim_in_corpus'] = len(df_preds_roi)
        df_preds_roi_top_bottom['num_unique_stim_in_corpus'] = len(stimset.sentence.unique())

        # Append .roi to the index
        df_preds_roi_top_bottom.index = df_preds_roi_top_bottom.index.str.cat(df_preds_roi_top_bottom.roi, sep='-')

        if pred_full is not None:  # Compute stretch metrics
            df_preds_roi_top_bottom = compute_stretch_metrics(df=df_preds_roi_top_bottom,
                                                              pred_full=pred_full,
                                                              roi=roi)
        df_lst.append(df_preds_roi_top_bottom)

    df_full = pd.concat(df_lst, axis=0)

    return df_full


def package_top_bottom_results_as_dict(df: pd.DataFrame,
                                       stimset: pd.DataFrame,
                                       pred_full: typing.Union[dict, None],
                                       n_stim: int, ):
    """
    Package the top and bottom predicted stimuli into a dictionary where key = ROI and value = dataframe with
    stimuli as rows, predictions in the {roi}_pred column, and the corresponding stimuli metadata as columns.

    If pred_full is not None, we compute stretch metrics.

    Args:
        df: dataframe with predictions. Assumes stimuli as rows and predictions as columns (named as ROIs)
        stimset: dataframe with stimuli metadata (assumes same indices as the predictions dataframe, df)
        pred_full: dictionary with keys = 'y' and 'y_pred-full' and values = dataframes with stimuli as rows and
            predictions for ROIs as columns
        n_stim: number of stimuli to include in the top and bottom

    Returns:
        d: dictionary with key = ROI and value = dataframe with stimuli as rows

    """
    stimset = stimset.copy(deep=True)

    # Package into a dict where the key = ROI and the value = a dataframe with stimuli as rows and predictions+metadata as columns
    d = {}
    for roi in df.columns.values:
        # Obtain the min/max stimuli for each ROI separately!
        df_preds_roi = df.copy(deep=True)[[roi]].sort_values(by=roi, ascending=False).rename(
            columns={roi: f'{roi}_pred'})

        df_preds_roi_top = df_preds_roi.head(n_stim)
        df_preds_roi_bottom = df_preds_roi.tail(n_stim)

        # Obtain indices for the top and bottom stimuli and merge with the stimset
        indices_top = df_preds_roi_top.index
        indices_bottom = df_preds_roi_bottom.index
        stimset_roi_top = stimset.reindex(indices_top)
        stimset_roi_bottom = stimset.reindex(indices_bottom)

        # Merge the top and bottom stimuli with the predictions
        assert (df_preds_roi_top.index.values == stimset_roi_top.index.values).all()
        df_preds_roi_top = pd.merge(df_preds_roi_top, stimset_roi_top, left_index=True, right_index=True)
        df_preds_roi_top['search_type'] = 'top'
        assert (df_preds_roi_bottom.index.values == stimset_roi_bottom.index.values).all()
        df_preds_roi_bottom = pd.merge(df_preds_roi_bottom, stimset_roi_bottom, left_index=True, right_index=True)
        df_preds_roi_bottom['search_type'] = 'bottom'

        # Merge top and bottom
        df_preds_roi_top_bottom = pd.concat([df_preds_roi_top, df_preds_roi_bottom], axis=0)
        df_preds_roi_top_bottom['roi'] = roi

        if pred_full is not None:  # Compute stretch metrics
            df_preds_roi_top_bottom = compute_stretch_metrics(df=df_preds_roi_top_bottom,
                                                              pred_full=pred_full,
                                                              roi=roi)
        d[roi] = df_preds_roi_top_bottom

    return d


def package_pred_results_as_df(df: pd.DataFrame,
                               stimset: pd.DataFrame,
                               stimset_cols_to_drop: typing.Union[list, np.ndarray, None],
                               pred_full: typing.Union[dict, None],
                               df_target: typing.Union[pd.DataFrame, None] = None, ):
    """
    Package predicted stimuli into a dataframe with stimuli x ROI as rows, predictions in the pred column,
    index as {stimuli identifier}-{roi}, and the corresponding stimuli metadata as columns.
    Similar packaging to package_top_bottom_results_as_df.

    If pred_full is not None, we compute stretch metrics.

    Args:
        df: dataframe with predictions. Assumes stimuli as rows and predictions as columns (named as ROIs)
        stimset: dataframe with stimuli metadata (assumes same indices as the predictions dataframe, df)
        stimset_cols_to_drop: list of columns to drop from the stimset
        pred_full: dictionary with keys = 'y' and 'y_pred' and values = dataframes with stimuli as rows and
            predictions for ROIs as columns (based on the mapping, i.e. training set, so obtaining information on the
            how these new, external predictions are relative to the actual values and prediction (not held-out) made on the
            training set)
        df_target: dataframe with true neural response values for the stimuli that were predicted (in df).
            Assumes stimuli as rows and response as columns (named as their respective ROI name).
            If not None, add the true neural response values as a column in the returned dataframe (as

    Returns:
        df: dataframe with stimuli predictions for each ROI as rows, predictions in the pred column

    """
    stimset = stimset.copy(deep=True)

    if df_target is not None:
        # Assert that the indices of the predicted stimuli and the true neural response values are the same
        assert (df.index.values == df_target.index.values).all()
        # Assert that all ROIs that exist in the predicted stimuli dataframe also exist in the true neural response values dataframe
        assert (df.columns.isin(df_target.columns).all())

    # Drop columns from stimset
    if stimset_cols_to_drop is not None:
        # Drop the columns in stimset_cols_to_drop if they exist
        stimset_cols_to_drop = [col for col in stimset_cols_to_drop if col in stimset.columns.values]
        stimset = stimset.drop(columns=stimset_cols_to_drop, inplace=False)

    # Package into a df
    df_lst = []
    for roi in df.columns.values:
        df_preds_roi = df.copy(deep=True)[[roi]].rename(columns={roi: f'pred'})
        df_preds_roi['roi'] = roi

        # Add in the true neural response that matches the prediction stimset index (rows)
        if df_target is not None:
            df_preds_roi['response_target'] = df_target[roi].values

        # Merge with the stimset
        assert (df_preds_roi.index.values == stimset.index.values).all()
        df_preds_roi = pd.concat([df_preds_roi, stimset], axis=1)

        # Append .roi to the index
        df_preds_roi.index = df_preds_roi.index.str.cat(df_preds_roi.roi, sep='-')

        if pred_full is not None:  # Compute stretch metrics
            df_preds_roi = compute_stretch_metrics(df=df_preds_roi,
                                                   pred_full=pred_full,
                                                   roi=roi)
        df_lst.append(df_preds_roi)

    df_full = pd.concat(df_lst, axis=0)

    return df_full


def compute_stretch_metrics(df: pd.DataFrame,
                            pred_full: typing.Union[dict, None],
                            roi: str, ):
    """
    Compute stretch metrics for a given ROI.

    Args:
        df: dataframe with predictions. Assumes stimuli as rows and predictions as in the column {roi}_pred
        pred_full: dictionary with keys = 'y' and 'y_pred-full' and values = dataframes with stimuli as rows and
        predictions for ROIs as columns
        roi: ROI to compute stretch metrics for (has to match the {roi}_pred column in df)

    Returns:
        df_stretch: dataframe with the stretch metrics for the given ROI
    """
    df = df.copy(deep=True)

    pred_col_name = f'pred'  # change to f'{roi}_pred' if packaging search stimuli as dict

    # Obtain the min and max values for y and y_pred for this ROI
    y = pred_full['y'][[roi]]
    y_pred = pred_full['y_pred-full'][[roi]]

    # Add the min/max values to the dataframe
    df['y_max'] = y.max().values[0]
    df['y_min'] = y.min().values[0]  # add y_train_min to name?
    df['y_pred_max'] = y_pred.max().values[0]
    df['y_pred_min'] = y_pred.min().values[0]

    # Compute stretch metrics using predicted max and min values
    df['stretch_pred_max'] = (df[pred_col_name]) / (df['y_pred_max'])
    df['stretch_pred_min'] = (df[pred_col_name]) / (df['y_pred_min'])

    df['stretch_pred_max_minmax'] = (df[pred_col_name] - df['y_pred_min']) / (
            df['y_pred_max'] - df['y_pred_min'])
    df['stretch_pred_min_minmax'] = -df['stretch_pred_max_minmax'] + 1

    # Compute stretch metrics using actual max and min values
    df['stretch_max'] = (df[pred_col_name]) / (df['y_max'])
    df['stretch_min'] = (df[pred_col_name]) / (df['y_min'])

    df['stretch_max_minmax'] = (df[pred_col_name] - df['y_min']) / (
            df['y_max'] - df['y_min'])
    df['stretch_min_minmax'] = -df['stretch_max_minmax'] + 1

    return df


def add_CV_results_to_df(df_preds_packaged: pd.DataFrame,
                         df_cv_scores: pd.DataFrame,
                         col_prefix: str = 'mapping_') -> pd.DataFrame:
    """
    Input a dataframe of predicted values (df_preds_packaged). Rows are stimuli x ROI, i.e., each row contains the
    predicted response for that particular stimuli and ROI (index is the {stimuli_identifier-ROI}. Packaged using
    either package_pred_results_as_df() or package_top_bottom_results_as_df().

    The goal is to append the CV scores associated with each ROI. Here, we only have one value per ROI.

    Output a dataframe with the same rows as df_preds_packaged, but with extra columns containing the CV score
    result for that ROI. Prefix each column name in df_cv_scores with "mapping_" as the CV scores were derived
    based on the mapping data.

    Args
        df_preds_packaged (pd.DataFrame): Dataframe of predicted values (stimuli x ROI) and colummns are "pred" and metadata.
        df_cv_scores (pd.DataFrame): Dataframe of CV scores for each ROI (index is ROI), columns are CV score results.
        col_prefix (str): Prefix to add to the column names in df_cv_scores.

    Returns
        pd.DataFrame: Dataframe with the same rows as df_preds_packaged, but with extra columns containing the CV scores,
                      prefixed with col_prefix.

    """
    # Add col_prefix to the column names in df_cv_scores
    df_cv_scores.columns = [f'{col_prefix}{col}' for col in df_cv_scores.columns]

    # In df_cv_scores, each row is results for an ROI.
    # In df_preds_packaged, each row is a stimuli x ROI prediction.
    # So we want to add in the columns from df_cv_scores to df_preds_packaged by repeating each row in df_cv_scores
    df_lst = []
    for roi in df_cv_scores.index.values:
        df_preds_packaged_roi = df_preds_packaged.loc[df_preds_packaged.roi == roi, :]
        df_cv_scores_roi = df_cv_scores.loc[roi, :]

        # Add in all the values from df_cv_scores_roi to df_preds_packaged_roi as new columns
        df_preds_packaged_roi = df_preds_packaged_roi.assign(**df_cv_scores_roi.to_dict())
        df_lst.append(df_preds_packaged_roi)

    df_preds_packaged = pd.concat(df_lst)

    return df_preds_packaged


### LOAD DATA FUNCTIONS ###
def _reorder_itemids(neural_data: pd.DataFrame = None,
                     stimset: pd.DataFrame = None):
    # Stimset has the itemid column
    stimset = stimset.sort_values(by='itemid')

    # Check whether the itemid matches with the index of stim meta
    index_itemid = [int(x.split('.')[-1]) for x in stimset.index]
    assert (index_itemid == stimset.itemid.values).all()

    # Now reindex the neural data
    neural_data = neural_data.reindex(index=stimset.index)
    assert (neural_data.index.values == stimset.index.values).all()

    return neural_data, stimset


def load_pereira_spm(fname: str = None,
                     UID: str = None,
                     reorder_itemids: bool = True):
    """Load Pereira SPM data (from Ben) and return stimset and neural data.
    If reorder_itemsids is True, the items are reordered according to ascending order of item ID.
    """
    d_target = pd.read_csv(fname)
    d_target.set_index('stim_id', inplace=True)
    # Obtain UID of interest
    d_target = d_target.query(f'UID == {UID}')
    # All columns that do not start with 'lang' or 'md' are considered as stimset and reorder according to itemid
    stimset = d_target[
        d_target.columns[~d_target.columns.str.startswith('lang') & ~d_target.columns.str.startswith('md')]]
    # Get the target
    neural_data = d_target[
        d_target.columns[d_target.columns.str.startswith('lang') | d_target.columns.str.startswith('md')]]

    if reorder_itemids:
        neural_data, stimset = _reorder_itemids(neural_data=neural_data,
                                                stimset=stimset)

    return neural_data, stimset


def load_neural_data(args: dict,
                     regression_dict_type: str,
                     DATAROOT: str = None,
                     key_prefix: str = '',
                     savestr_prefix: str = '', ):
    """
    Function for loading neural data.

    If args.manual_target is not None, load the data from a manual target file.
        Available options are string arguments:
            'pereira_spm'
            'control_pilot3_SPM'
        OR
            A filepath that points to a pickle file that can be loaded as:
                neural_data = df['df_rois_normalized']
                stimset = df['stimset']

    If args.manual_target is None, load data from DATAROOT/dict_neural_stimuli

    If scriptname is use_mapping, prefix all the arguments with "target_

    Args:
        args (dict): dictionary of arguments
        regression_dict_type (str): type of regression dictionary to use. Options are: "rois" or "voxs" (or none,
            for compatability with initial packaging of train participants)
        DATAROOT (str): root directory of the data
        key_prefix (str): prefix to add to the keys of the dictionary
        savestr_prefix (str): prefix which was added to the savestr in package_data.py in fMRI prep and propagated to load_voxel_data.py.

    Returns:
        neural_data (pd.DataFrame): the neural data
        neural_meta (pd.DataFrame or None): the metadata for the neural data (if available)
        stimset (pd.DataFrame): the stimulus set
        modified_args (dict): the modified arguments (if any), to be added to args_logger

    """
    if key_prefix != '':
        print(f'\nLoading data using a key prefix! {key_prefix}\n')

    modified_args_dict = {}  # Dictionary of modified arguments to add to args_logger

    #### LOAD DATA IF MANUAL TARGET IS SPECIFIED ####
    if args[f'{key_prefix}manual_target'] is not None:
        print(f' ###### Loading manual target data from {args[f"{key_prefix}manual_target"]} ######')

        # Pre-specified manual data loaders
        if args[f'{key_prefix}manual_target'] == 'pereira_SPM':
            fname_to_load = join(DATAROOT, 'external',
                                 'Pereira_FirstSession_TrialEffectSizes_20220223_fixed-col-names_wstimid.csv')
            neural_data, stimset = load_pereira_spm(fname=fname_to_load,
                                                    UID=args[f'{key_prefix}UID'])

        elif args[f'{key_prefix}manual_target'] == 'control_pilot3_SPM':
            fname_to_load = '/Users/gt/Documents/GitHub/control-neural/data/dict_neural_stimuli/control/pilot3/dict_UID-853_SESSION-FED_20211008a_3T1_PL2017-FED_20211013b_3T1_PL2017_FL-control_tr1_20220109.pkl'
            d_target = pd.read_pickle(fname_to_load)
            neural_data = d_target['df_rois_normalized']
            stimset = d_target['stimset']

        else:
            raise ValueError(f'Manual target {args[f"{key_prefix}manual_target"]} not found')

    #### LOAD DATA FROM DATAROOT/dict_neural_stimuli ####
    else:
        # If regression_dict_type is not '' we should append '_' to it
        if regression_dict_type != '':
            regression_dict_type = regression_dict_type + '_'

        if savestr_prefix != '' and savestr_prefix[-1] != '_':
            savestr_prefix = savestr_prefix + '_'

        ## GLM SINGLE: Load file with all gs params ##
        if args[f'{key_prefix}FL'] == 'gs':
            fname_to_load = join(DATAROOT,
                                 'dict_neural_stimuli',
                                 f'dict_{regression_dict_type}'
                                 f'{savestr_prefix}'
                                 f'{args[f"{key_prefix}UID"]}_'
                                 f'{args[f"{key_prefix}sess_num"]}_'
                                 f'{args[f"{key_prefix}FL"]}_'
                                 f'thresh-{args[f"{key_prefix}func_thresh"]}_'
                                 f'type-{args[f"{key_prefix}modeltype"]}_'
                                 f'preproc-{args[f"{key_prefix}preproc"]}_'
                                 f'pcstop-{args[f"{key_prefix}pcstop"]}_'
                                 f'fracs-{args[f"{key_prefix}fracs"]}.pkl')

            d_target = pd.read_pickle(fname_to_load)

        ## SPM: Load file with spm params ##
        elif args[f'{key_prefix}FL'] == 'spm':
            modified_args_dict['modeltype'] = 'spm'
            modified_args_dict['preproc'] = 'nan'
            modified_args_dict['pcstop'] = '0'
            modified_args_dict['fracs'] = '0'

            fname_to_load = join(DATAROOT,
                                 'dict_neural_stimuli',
                                 f'dict_{regression_dict_type}'
                                 f'{savestr_prefix}'
                                 f'{args[f"{key_prefix}UID"]}_'
                                 f'{args[f"{key_prefix}sess_num"]}_'
                                 f'{args[f"{key_prefix}FL"]}_'
                                 f'thresh-{args[f"{key_prefix}func_thresh"]}_'
                                 f'type-{modified_args_dict[f"{key_prefix}modeltype"]}_'
                                 f'preproc-{modified_args_dict[f"{key_prefix}preproc"]}_'
                                 f'pcstop-{modified_args_dict[f"{key_prefix}pcstop"]}_'
                                 f'fracs-{modified_args_dict[f"{key_prefix}fracs"]}.pkl')

            d_target = pd.read_pickle(fname_to_load)

        else:
            raise ValueError(f'FL {args[f"{key_prefix}FL"]} not recognized')

        if regression_dict_type == '':  # Old packaging version, we have to use "rois" to load the actual neural data
            regression_dict_type = 'rois_'

        neural_data = d_target[f'df_{regression_dict_type}norm-{args[f"{key_prefix}norm"]}']
        stimset = d_target['stimset']

        # Check if neural meta exists
        if 'neural_meta' in d_target.keys():
            neural_meta = d_target['neural_meta']
        else:
            neural_meta = None

    print(f'\nLoaded {fname_to_load} with shape {neural_data.shape}')

    return neural_data, neural_meta, stimset, modified_args_dict


def get_voxel_ROI_subset(neural_data: pd.DataFrame,
                         neural_meta: pd.DataFrame,
                         roi: str, ):
    """
    Given a neural_data df (n_stimuli x n_voxels) and a neural_meta df (n_voxels x n_meta), return a subset of the
    neural_data df that only contains voxels from the specified ROI.
    Voxels are indexed according to their "roi" meta column. The only exception is "lang_LH_netw" which
    consists of 5 ROIs: lang_LH_IFGorb, lang_LH_IFG, 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp'.

    Args
        neural_data (pd.DataFrame): n_stimuli x n_voxels
        neural_meta (pd.DataFrame): n_voxels x n_meta
        roi (str): ROI to extract
    """
    assert (
                neural_data.columns.values == neural_meta.index.values).all(), 'neural_data and neural_meta must have the same columns/index'

    neural_data_to_return = neural_data.copy(deep=True)

    if roi == 'lang_LH_netw':
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('lang_LH') &
                                                             ~neural_meta['roi'].str.endswith('AngG')]
        # Print out the ROIs that are included in lang_LH_netw in neural_meta
        print(
            f'\nROIs included in lang_LH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("lang_LH") & ~neural_meta["roi"].str.endswith("AngG"), "roi"].unique()}')

    elif roi == 'lang_RH_netw':
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('lang_RH') &
                                                             ~neural_meta['roi'].str.endswith('AngG')]
        # Print out the ROIs that are included in lang_RH_netw in neural_meta
        print(
            f'\nROIs included in lang_RH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("lang_RH") & ~neural_meta["roi"].str.endswith("AngG"), "roi"].unique()}')

    elif roi == 'md_LH_netw':  # Get all ROIs that start with md_LH
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('md_LH')]
        print(
            f'\nROIs included in md_LH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("md_LH"), "roi"].unique()}')

    elif roi == 'md_RH_netw':  # Get all ROIs that start with md_RH
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('md_RH')]
        print(
            f'\nROIs included in md_RH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("md_RH"), "roi"].unique()}')

    elif roi == 'dmn_LH_netw':  # Get all ROIs that start with dmn_LH
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('dmn_LH')]

        print(
            f'\nROIs included in dmn_LH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("dmn_LH"), "roi"].unique()}')

    elif roi == 'dmn_RH_netw':  # Get all ROIs that start with dmn_RH
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'].str.startswith('dmn_RH')]

        print(
            f'\nROIs included in dmn_RH_netw: {neural_meta.loc[neural_meta["roi"].str.startswith("dmn_RH"), "roi"].unique()}')

    else:
        neural_data_to_return = neural_data_to_return.loc[:, neural_meta['roi'] == roi]

        print(f'\nROIs included in {roi}: {neural_meta.loc[neural_meta["roi"] == roi, "roi"].unique()}')

    # Print number of voxels and ROI name
    print(f'\nExtracted {neural_data_to_return.shape[1]} voxels from {roi}')

    return neural_data_to_return


def check_constant_y_ypred(y_pred: typing.Union[np.ndarray, pd.DataFrame],
                           y: typing.Union[np.ndarray, pd.DataFrame],
                           score: typing.Union[list, np.ndarray],
                           p: typing.Union[list, np.ndarray], ):
    """
    Checks if the y (real neural data) and y_pred (predicted neural data) (for each neuroid) are constant across stimuli.
    If so, the correlation coefficient is not defined, and will be nan. We want it to be 0.
    Also set the p-value to 1 if the score is nan.

    This function checks whether the y_test/y_pred arrays are indeed constant (zero variance):
    Figure out where y_pred or y_test is constant as defined by
    (This is because that dependent on which compute node the job is run on, different precision might be used --
    this is an attempt to unify what we consider a constant array across compute nodes)


     (x == x[0]).all() as in scipy stats pearsonr (ie. pearsonr not defined).

    When this is run without CV (full mapping), y is the full neural data, and y_pred is y_pred_full.
    When this is run with CV, y is the y_test from each fold, and y_pred is the y_pred from each fold.


    Args
         y_pred (np.ndarray or pd.DataFrame): The predicted neural data, matrix of size [n_stimuli, n_neuroids]
         y (np.ndarray or pd.DataFrame): The real neural data, matrix of size [n_stimuli, n_neuroids]
         score (list or np.ndarray): The score (e.g., correlation coefficient) for each neuroid, size [n_neuroids]


    Returns
         score_no_nan (list or np.ndarray): The score (e.g., correlation coefficient) for each neuroid, size [n_neuroids],
             with the scores for constant y_test/y_pred neuroids set to 0.
         p_no_nan (list or np.ndarray): The p-value for each neuroid, size [n_neuroids],
             with the p-values for constant y_test/y_pred neuroids set to 1.
    """

    # Define a constant array as np.std(y) < threshold (whether all values in each column are the same)
    threshold = 1e-7  # Neural data are float32, so we do not have more precision than 1e-8 https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points)

    # Find constant y_pred
    y_pred_constant = np.std(y_pred, axis=0) < threshold
    y_pred_constant = np.where(y_pred_constant)[0]
    # Find constant y
    y_constant = np.std(y, axis=0) < threshold
    y_constant = np.where(y_constant)[0]

    if len(y_pred_constant) > 0:
        print(
            f'Constant predictions ({threshold}) for neuroids ({len(y_pred_constant)} neuroids) {y_pred.columns.values[y_pred_constant]}')
    if len(y_constant) > 0:
        print(f'Constant y_test ({threshold}) for neuroids ({len(y_constant)} neuroids) {y.columns.values[y_constant]}')

    # Join the neuroids that are constant in y_pred and y_test
    constant_neuroids = np.unique(np.concatenate((y_pred_constant, y_constant)))

    # Check which index the fold_score has nan values
    score_nan = np.argwhere(np.isnan(score)).flatten()

    # Check whether all the values that are nan in the score_nan array are also in the constant_neuroids array
    assert np.all(np.isin(score_nan,
                          constant_neuroids)), f'Not all the nan values in the score array are also in the constant_neuroids array'

    # Now we can set the score to 0 for the constant neuroids
    score_no_nan = np.array(score)
    score_no_nan[constant_neuroids] = 0

    # Also set the p-value to 1 if the score is nan
    # First check whether the indices of nan in p are also the constant neuroids
    p_nan = np.argwhere(np.isnan(p)).flatten()
    assert np.all(np.isin(p_nan,
                          constant_neuroids)), f'Not all the nan values in the p array are also in the constant_neuroids array'
    # Now we can set the fold p-value to 1 for the constant neuroids
    p_no_nan = np.array(p)
    p_no_nan[constant_neuroids] = 1

    assert (np.isnan(score_no_nan).sum() == 0), "There are still nan values in the score"
    assert (np.isnan(p_no_nan).sum() == 0), "There are still nan values in the p-value"

    return score_no_nan, p_no_nan


def format_source_layer(source_model: str,
                        source_layer: typing.Union[int, str],
                        ):
    """Figure out whether the source layer should be an int or str. We want to support both.
    We want int for GPT and BERT-based models (their activations are stored indexed by int layer number)
    We want str for models that use str layer names (e.g. ResNet, layer 'act1') or if we provide a true neural source (e.g. layer '848-853')
    """

    if source_model.startswith('gpt') or source_model.startswith('bert') or source_model.startswith(
            'vq') or source_model.startswith('distil'):
        if not type(source_layer) == int:
            try:
                source_layer = int(source_layer)
                print(
                    f'Converted source layer {source_layer} (type {type(source_layer)}) to int {source_layer} (type {type(source_layer)})')
            except:
                raise ValueError(
                    f'Could not convert source layer {source_layer} (type {type(source_layer)}) to int')

    else:
        if not type(source_layer) == str:
            try:
                source_layer = str(source_layer)
                print(
                    f'Converted source layer {source_layer} (type {type(source_layer)}) to str {source_layer} (type {type(source_layer)})')
            except:
                raise ValueError(
                    f'Could not convert source layer {source_layer} (type {type(source_layer)}) to str')

    print(f'Using source layer {source_layer} (type {type(source_layer)}) for {source_model} model')
    return source_layer
