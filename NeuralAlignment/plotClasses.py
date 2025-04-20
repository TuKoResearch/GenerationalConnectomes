import getpass
import os
import typing
from os.path import join

user = getpass.getuser()

ROOTDIR = os.path.abspath(join(os.path.dirname(__file__)))  # '/Users/gt/Documents/GitHub/beta-neural-control/'
RESULTROOT = join(ROOTDIR, 'results')

from plot_utils import *

# Set random seed for plot jitters
np.random.seed(0)

class MappingOutput:
    """
    Represent the df output from fit_mapping.py, [ADD MORE]

    """

    def __init__(self, source_models: typing.Union[list, np.ndarray],
                 source_layers: typing.Union[list, np.ndarray],
                 sent_embeds: typing.Union[list, np.ndarray],

                 regression_dict_types: typing.Union[list, np.ndarray],
                 savestr_prefixes: typing.Union[list, np.ndarray],
                 UIDs: typing.Union[list, np.ndarray],
                 sess_nums: typing.Union[list, np.ndarray],
                 FLs: typing.Union[list, np.ndarray],
                 modeltypes: typing.Union[list, np.ndarray],
                 func_threshs: typing.Union[list, np.ndarray],
                 norms: typing.Union[list, np.ndarray],

                 mapping_classes: typing.Union[list, np.ndarray],
                 metrics: typing.Union[list, np.ndarray],
                 preprocessors: typing.Union[list, np.ndarray],
                 preprocess_xs: typing.Union[list, np.ndarray],
                 preprocess_ys: typing.Union[list, np.ndarray], ):
        """
        """
        # SOURCE
        self.source_models = source_models
        self.source_layers = source_layers
        self.sent_embeds = sent_embeds
        # TARGET
        self.regression_dict_type = regression_dict_types  # How the neural data was stored ("rois" or "voxs")
        self.savestr_prefix = savestr_prefixes  # Which savestr was used to package the neural data
        self.UIDs = UIDs
        self.sess_nums = sess_nums
        self.FLs = FLs
        self.modeltypes = modeltypes
        self.func_threshs = func_threshs
        self.norms = norms
        # MAPPING
        self.mapping_classes = mapping_classes
        self.metrics = metrics
        self.preprocessor = preprocessors  # Which preprocessor was used to preprocess the activations OR neural data prior regression
        self.preprocess_x = preprocess_xs  # Whether the activations were preprocessed prior to regression
        self.preprocess_y = preprocess_ys  # Whether the neural data was preprocessed prior to regression

        self.base_savestr = self._generate_base_savestr()  # To store plots and csvs

    def _generate_base_savestr(self):
        """
        Generate a savestr that contain most of the parameters that we do not iterate over.
        Important parameters left:
            - source_layers
            - UIDs
            - (roi)
        """
        base_savestr = 'SOURCE-' + '-'.join(self.source_models) + '_' + \
                       '-'.join(self.sent_embeds) + '_' + \
                       'TARGET-' + \
                       '-'.join(self.regression_dict_type) + '_' + \
                       '-'.join(self.savestr_prefix) + '_' + \
                       '-'.join(self.sess_nums) + '_' + \
                       '-'.join(self.FLs) + '_' + \
                       '-'.join(self.modeltypes) + '_' + \
                       '-'.join(self.func_threshs) + '_' + \
                       '-'.join(self.norms) + '_' + \
                       'MAPPING-' + '-'.join(self.mapping_classes) + '_' + \
                       '-'.join(self.metrics) + '_' + \
                       '-'.join(self.preprocessor) + '-' + \
                       '-'.join(self.preprocess_x) + '-' + \
                       '-'.join(self.preprocess_y)

        # If we have many source-models, let's replace with the number of source models
        if len(self.source_models) > 2:
            base_savestr = base_savestr.replace("-".join(self.source_models), f'n{len(self.source_models)}')

        return base_savestr

    def find_results_folders(self,
                             subfolder: str = 'fit_mapping'):
        """Find the results folders based on the source models, layers, sent embeds, UIDs, sess_nums, FLs, func_threshs,
        mapping_classes, and metrics specified in the class.

        Args:
            subfolder: str

        Returns:
            folders: list of str
        """
        all_folders = os.listdir(join(RESULTROOT, subfolder))
        all_folders = [f for f in all_folders if not f.startswith('.')]

        # Only keep folders that match the self.parameters
        folders = []
        for folder in all_folders:
            source_model = folder.split('SOURCE-')[1].split(f'_TARGET')[0]
            if 'last' in source_model: # let's just strip _last in the end
                source_model = source_model.split('_last')[0]
            # sent_embed = folder.split('_')[1]
            UID = '-'.join(folder.split('TARGET-')[1].split('_')[0].split('-')[1:])
            # sess_num = folder.split('_')[3]
            # FL = folder.split('_')[4]
            # func_thresh = folder.split('_')[5]
            mapping = folder.split('MAPPING-')[1].split('-')[0]
            metric = folder.split('MAPPING-')[1].split('-')[1]
            regression_dict_type = folder.split('TARGET-')[1].split('-')[0]

            if source_model in self.source_models and UID in self.UIDs and mapping in self.mapping_classes and metric in self.metrics and regression_dict_type in self.regression_dict_type:
                folders.append(folder)

        self.results_folders = folders

        return folders

    def load_results(self,
                     subfolder='fit_mapping',
                     prefix_str: str = '',
                     suffix_str: str = '',
                     roi: typing.Union[str, None] = None,
                     truncate: typing.Union[int, None] = None, ):
        """Load results from the results_folders.

        Only loads the layers specified in the class. Only loads the norm specified in the class.

        Possible to supply a prefix_str and suffix_str to add to the filename. Will only load the file if it contains the prefix_str and suffix_str.

        Args:
            subfolder: str
            prefix_str: str, any string that the filename must contain in the beginning
            suffix_str: str, any string that the filename must contain in the end
            roi: str, if not None, will only load the results for the specified roi (relevant for use_mapping_external
                where we have item-level predictions for each roi)
            truncate: int, if not None, will truncate loading truncate number of files

        """
        dfs = []
        load_counter = 0

        if subfolder == 'fit_mapping':
            for folder in self.results_folders:
                print(f'Loading {folder}')
                files = os.listdir(join(RESULTROOT, subfolder, folder))
                for file in tqdm(files):  # Sort based on pre/suffix and layer
                    layer = (file.split('SOURCE-')[1].split('_')[0])
                    try:
                        layer = int(layer)
                    except:
                        pass
                    norm = file.split('_MAPPING')[0].split('-')[-1]
                    # modeltype = separate_savestr_modeltype(file)
                    savestr_prefix, _ = separate_savestr_prefix_specific_target(
                        savestr_prefix_AND_specific_target=file.split('TARGET-')[1].split('_')[
                            0])  # we don't use specific target
                    preprocess_specs = file.split('MAPPING-')[-1].split('.')[0]
                    try:
                        preprocessor, preprocess_x, preprocess_y = preprocess_specs.split('-')
                    except:
                        preprocessor = preprocess_specs
                        preprocess_x = 'False'
                        preprocess_y = 'False'

                    if file.startswith(prefix_str) and file.endswith(
                            suffix_str) and layer in self.source_layers and norm in self.norms and savestr_prefix in self.savestr_prefix and preprocessor in self.preprocessor and preprocess_x in self.preprocess_x and preprocess_y in self.preprocess_y:
                        fname = join(RESULTROOT, subfolder, folder, file)
                        df = pd.read_pickle(fname)
                        dfs.append(df)
                        print(f'Loaded {fname}')

                        load_counter += 1
                        if truncate is not None and load_counter >= truncate:
                            break

        elif subfolder.startswith('use_mapping'):
            for folder in self.results_folders:
                print(f'Loading {folder}')
                subfolders_mapping = os.listdir(join(RESULTROOT, subfolder, folder))

                for subfolder_mapping in subfolders_mapping:
                    # Sort which subfolders we want to load
                    layer = int(subfolder_mapping.split('SOURCE-')[1].split('_')[0])
                    norm = subfolder_mapping.split('_MAPPING')[0].split('-')[-1]
                    modeltype = subfolder_mapping.split('_')[2].split('-')[0]
                    savestr_prefix, _ = separate_savestr_prefix_specific_target(
                        savestr_prefix_AND_specific_target=subfolder_mapping.split('TARGET-')[1].split('_')[
                            0])  # we don't use specific target
                    preprocess_specs = subfolder_mapping.split('MAPPING-')[-1].split('.')[0]
                    preprocessor, preprocess_x, preprocess_y = preprocess_specs.split('-')

                    if layer in self.source_layers and norm in self.norms and modeltype in self.modeltypes and savestr_prefix in self.savestr_prefix and preprocessor in self.preprocessor and preprocess_x in self.preprocess_x and preprocess_y in self.preprocess_y:
                        files = os.listdir(join(RESULTROOT, subfolder, folder, subfolder_mapping))
                        for file in tqdm(files):
                            if file.startswith(prefix_str) and file.endswith(f'{suffix_str}.pkl'):
                                df = pd.read_pickle(join(RESULTROOT, subfolder, folder, subfolder_mapping, file))
                                if roi is not None:
                                    print(f'Only subselecting the part of {file} that is in the {roi} ROI')
                                    df = df[df['roi'] == roi]
                                dfs.append(df)
                                load_counter += 1
                                if truncate is not None and load_counter >= truncate:
                                    break

        else:
            raise ValueError(f'{subfolder} is not a valid subfolder')

        df = pd.concat(dfs)
        self.df = df
        self.df['base_savestr'] = self.base_savestr
        self.df_as_loaded = df.copy(deep=True)  # To ensure no inplace changes are made later on

    def load_dict_results(self,
                          subfolder='fit_mapping',
                          prefix_str: str = '',
                          suffix_str: str = '',
                          source_layer: int = 22,
                          truncate: typing.Union[int, None] = None, ):
        """Load dictionary results from the results_folders.
        We want to only load one instantiation (it gets messy to concatenate several dictionaries, hence the source_layer argument).

        Possible to supply a prefix_str and suffix_str to add to the filename. Will only load the file if it contains the prefix_str and suffix_str.

        Args:
            subfolder: str
            prefix_str: str, any string that the filename must contain in the beginning
            suffix_str: str, any string that the filename must contain in the end
            source_layer: int, the source layer we want to load
            truncate: int, if not None, will truncate loading truncate number of files

        """
        dfs = []
        load_counter = 0

        if subfolder == 'fit_mapping':
            for folder in self.results_folders:
                print(f'Loading {folder}')
                files = os.listdir(join(RESULTROOT, subfolder, folder))
                for file in tqdm(files):  # Sort based on pre/suffix and layer
                    layer = int(file.split('SOURCE-')[1].split('_')[0])
                    norm = file.split('_MAPPING')[0].split('-')[-1]
                    modeltype = file.split('_')[3].split('-')[0]
                    savestr_prefix, _ = separate_savestr_prefix_specific_target(
                        savestr_prefix_AND_specific_target=file.split('TARGET-')[1].split('_')[
                            0])  # we don't use specific target
                    preprocess_specs = file.split('MAPPING-')[-1].split('.')[0]
                    preprocessor, preprocess_x, preprocess_y = preprocess_specs.split('-')

                    if layer == source_layer and file.startswith(prefix_str) and file.endswith(
                            suffix_str) and norm in self.norms and modeltype in self.modeltypes and savestr_prefix in self.savestr_prefix and preprocessor in self.preprocessor and preprocess_x in self.preprocess_x and preprocess_y in self.preprocess_y:
                        df = pd.read_pickle(join(RESULTROOT, subfolder, folder, file))
                        dfs.append(df)
                        load_counter += 1
                        if truncate is not None and load_counter >= truncate:
                            break

        elif subfolder.startswith('use_mapping'):
            for folder in self.results_folders:
                print(f'Loading {folder}')
                subfolders_mapping = os.listdir(join(RESULTROOT, subfolder, folder))

                for subfolder_mapping in subfolders_mapping:
                    # Sort which subfolders we want to load
                    layer = int(subfolder_mapping.split('SOURCE-')[1].split('_')[0])
                    norm = subfolder_mapping.split('_MAPPING')[0].split('-')[-1]
                    modeltype = subfolder_mapping.split('_')[2].split('-')[0]
                    savestr_prefix, _ = separate_savestr_prefix_specific_target(
                        savestr_prefix_AND_specific_target=file.split('TARGET-')[1].split('_')[
                            0])  # we don't use specific target
                    preprocess_specs = file.split('MAPPING-')[-1].split('.')[0]
                    preprocessor, preprocess_x, preprocess_y = preprocess_specs.split('-')

                    if layer == source_layer and file.startswith(prefix_str) and file.endswith(
                            suffix_str) and norm in self.norms and modeltype in self.modeltypes and savestr_prefix in self.savestr_prefix and preprocessor in self.preprocessor and preprocess_x in self.preprocess_x and preprocess_y in self.preprocess_y:
                        files = os.listdir(join(RESULTROOT, subfolder, folder, subfolder_mapping))
                        for file in tqdm(files):
                            if file.startswith(prefix_str) and file.endswith(suffix_str):
                                df = pd.read_pickle(join(RESULTROOT, subfolder, folder, subfolder_mapping, file))
                                dfs.append(df)
                                load_counter += 1
                                if truncate is not None and load_counter >= truncate:
                                    break

        else:
            raise ValueError(f'{subfolder} is not a valid subfolder')

        self.df = pd.concat(dfs)

    def get_avail_pklfiles(self,
                           path: str = '', ):
        pkl_files = [i for i in os.listdir(path) if i.endswith('.pkl')]

        return pkl_files

    def loop_across_heatmaps(self, rois: list = ['lang_LH_netw'],
                             modeltypes: list = ['d'],
                             values: list = ['CV_score_mean'],
                             indices=['fracs'],
                             columns=['pcstop'],
                             **kwargs):
        for roi in rois:
            for index in indices:
                for column in columns:
                    for modeltype in modeltypes:
                        for value in values:
                            df = self.df.loc[roi].query(f'modeltype == "{modeltype}"')
                            self.heatmap(df=df,
                                         value=value,
                                         index=index,
                                         column=column,
                                         **kwargs)

    def cols_str_to_int(self,
                        cols: list):
        for col in cols:
            try:
                self.df[col] = self.df[col].astype(int)
                print(f'Converted {col} to int in self.df')
            except:
                raise ValueError(f'Could not convert {col} to int')

    def cols_bool_and_none_to_str(self,
                                  cols: list):
        """Convert columns to from bool or None to str, e.g. False -> 'False' and None -> 'None'"""
        for col in cols:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(str)
                    print(f'Converted {col} to str in self.df')
                except:
                    raise ValueError(f'Could not convert {col} to str')

    def merge_with_NC(self,
                      df_nc: pd.DataFrame,
                      inplace: bool = True):
        """
        Supply a noiseceling (NC) dataframe and merge it with self.df if inplace is True, else just return the merged df.

        Args:
            df_nc (pd.DataFrame): NC dataframe, index are ROIs and columns are the NC values.
            inplace (bool): If True, merge with self.df. Else, just return the merged df.

        Returns:
            pd.DataFrame: Merged df with NC values.
        """
        df = self.df.copy(deep=True)
        # First check that the df_nc has the same ROIs as self.df
        if not set(df_nc.index) == set(df.index):
            print('df_nc and self.df do not have the same ROIs')

        df_merged = df.merge(df_nc, left_index=True, right_index=True, how='left')

        # For duplicated columns (_x and _y), check that they are the same, and then drop the _y columns and rename the _x columns to the original column name
        for col in df_merged.columns:
            col_x = col.replace('_y', '_x')

            if col.endswith('_y') and col not in ['preprocess_x', 'preprocess_y']:
                # If col startswith "savestr" then change _x to _cv and _y to _nc (it is ok they are different)
                if col.startswith('savestr'):
                    col_x = col.replace('_y', '_cv')
                    col = col.replace('_y', '_nc')
                    df_merged = df_merged.rename(columns={col_x: col_x.replace('_cv', '')}, inplace=False)
                    df_merged = df_merged.rename(columns={col: col.replace('_nc', '')}, inplace=False)

                else:

                    if not (df_merged[col_x] == df_merged[col]).all():
                        print(f'Columns {col_x} and {col} are not the same')

                    # Drop the _y columns and rename the _x columns to the original column name
                    df_merged = df_merged.drop(columns=[col], inplace=False)
                    df_merged = df_merged.rename(columns={col_x: col_x.replace('_x', '')}, inplace=False)

        if inplace:
            self.df = df_merged

        return df_merged

    def CV_vs_NC(self,
                 df: pd.DataFrame,
                 cv_col: str = 'CV_score_mean',
                 cv_err_col: str = 'CV_score_sem',
                 nc_col: str = 'nc',
                 nc_err_col: str = 'split_half_se',
                 rois_of_interest: str = 'rois_func_lang_md_dmn',
                 source_layer: typing.Union[str, int, None] = None,
                 xlim: typing.Union[list, np.ndarray] = None,
                 ylim: typing.Union[list, np.ndarray] = None,
                 save: bool = False,
                 add_savestr: str = '',
                 PLOTDIR: str = '',
                 CSVDIR: str = '', ):
        """
        Plot CV score vs NC for all ROIs in rois.

        Args:
            df (pd.DataFrame): Dataframe with CV and NC values. Rois are index.
            cv_col (str): Column name for CV values.
            cv_err_col (str): Column name for CV error values.
            nc_col (str): Column name for NC values.
            nc_err_col (str): Column name for NC error values.
            rois_of_interest (list): List of ROIs to plot.
            source_layer (str, int, None): If not None, plot only for this source layer.
            xlim (list): List of x-axis limits.
            ylim (list): List of y-axis limits.
            save (bool): If True, save the plot.
            PLOTDIR (str): Path to directory where to save the plot.
            CSVDIR (str): Path to directory where to save the csv file.
        """
        ###### GET ROI NAMES ######
        rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)
        if rois_of_interest is None:
            # Just use all the ROIs in the dataframe
            rois_of_interest = df.index.unique().tolist()

        ###### CHECK THAT DF HAS THE EXPECTED ENTRIES ######
        self.check_df_against_class_attributes(df=df)

        add_savestr = [f'_{add_savestr}' if add_savestr else ''][0]

        ###### ITERATE & PLOT ######
        if xlim is not None:
            xlimstr = f'{xlim[0]}-{xlim[1]}'
        else:
            xlimstr = 'None'
        if ylim is not None:
            ylimstr = f'{ylim[0]}-{ylim[1]}'
        else:
            ylimstr = 'None'

        # Get num unique source layers
        if source_layer is not None:
            df_q = df.query(f'source_layer == "{source_layer}"')
        else:
            df_q = df.copy(deep=True)
        n_source_layers = len(df_q.source_layer.unique())

        # Get ROIs
        df_q = df_q.loc[df_q.index.isin(rois_of_interest)]
        n_rois = len(df_q.index.unique())

        # Define savestr
        savestr = f'CV_vs_NC_' \
                  f'CV={cv_col}_ERR={cv_err_col}_' \
                  f'NC={nc_col}_ERR={nc_err_col}_' \
                  f'xl={xlimstr}_yl={ylimstr}_' \
                  f'layers-n={n_source_layers}_' \
                  f'{rois_of_interest_name}_' \
                  f'rois-n={n_rois}_' \
                  f'{self.base_savestr}' \
                  f'{add_savestr}'

        # Same plot but color according to ROI. use d_netw_colors when available, otherwise gray
        fig, ax = plt.subplots(figsize=(10, 10))
        for roi in rois_of_interest:
            df_q_roi = df.query(f'roi == "{roi}"')
            if roi in d_netw_colors.keys():
                color = d_netw_colors[roi]
            else:
                color = 'grey'
            if len(df_q_roi) == 0:
                print(f'No data for {roi}')
            ax.errorbar(x=df_q_roi[cv_col],
                        y=df_q_roi[nc_col],
                        xerr=df_q_roi[cv_err_col],
                        yerr=df_q_roi[nc_err_col],
                        ls='none',
                        ecolor=color,
                        elinewidth=0.4,
                        capsize=0,
                        capthick=0,
                        label=f'{roi}',
                        color=color,
                        markersize=10,
                        alpha=0.5)  # default 0.5. for a lot of anatomical ROIs, 0.2 is better
            ax.scatter(x=df_q_roi[cv_col],
                       y=df_q_roi[nc_col],
                       color=color,
                       label=f'{roi}',
                       s=100,
                       alpha=0.7,  # default 0.7. for a lot of anatomical ROIs, 0.2 is better
                       edgecolors='none')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # add x=y line
        plot_identity(ax, color='black', linestyle='-', alpha=0.6)
        ax.set_xlabel(d_axes_legend[cv_col])
        ax.set_ylabel(d_axes_legend[nc_col])
        ax.set_title('\n'.join(wrap(savestr, 80)))
        # ax.legend()
        # make ticks bigger and x and y lims
        ax.tick_params(axis='both', which='major', labelsize=20)
        # Make x and y labels bigger
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        plt.tight_layout()
        if save:
            savestr = shorten_savestr(savestr=savestr)
            os.chdir(PLOTDIR)
            plt.savefig(f'{savestr}.pdf', dpi=180)
            os.chdir(CSVDIR)
            df_q.to_csv(f'{savestr}.csv')
        plt.show()

    def check_df_against_class_attributes(self,
                                          df: pd.DataFrame,
                                          check_source_models: bool = False,
                                          check_source_layers: bool = False,
                                          check_sent_embeds: bool = True,
                                          check_UIDs: bool = True,
                                          check_FLs: bool = True,
                                          check_modeltypes: bool = True,
                                          check_norms: bool = True,
                                          check_preprocessors: bool = True,
                                          check_preprocess_x: bool = True,
                                          check_preprocess_y: bool = True):
        """
        Given a df (usually, a copy of self.df) with a set of results, check that the df has the same attributes as was specified
        in the class instantiation. E.g. if UIDs was specified as ['848', '853'] in the class instantiation, then
        assert that the df contains only these UIDs.

        Note that the default for check_source_layers is False because all available layers might not be present in the df.
        If so, print warning.

        Also note that for check_preprocessors, check_preprocess_x, and check_preprocess_y, the default is True,
        but we additionally assert that the df contains only one value for each of these columns.
        (we do not want to mix analyses that used different preprocessors, preprocess_x, or preprocess_y)

        """

        ###### CHECK THAT WE HAVE ONE ENCODING ANALYSIS INSTANTIATION ######
        source_models = df.source_model.unique()
        if check_source_models:
            assert (np.sort(source_models) == np.sort(self.source_models)).all()
        else:
            if not (np.sort(source_models) == np.sort(self.source_models)).all():
                print(
                    f'Warning: source_models in df do not match those specified in class instantiation')  # This can be totally ok

        source_layers = df.source_layer.unique()
        if check_source_layers:
            assert (np.sort(source_layers) == np.sort(self.source_layers)).all()
        else:
            if not (np.sort(source_layers) == np.sort(self.source_layers)).all():
                print(f'Warning: source_layers in df do not match those specified in class instantiation')

        sent_embeds = df.sent_embed.unique()
        if check_sent_embeds:
            assert (np.sort(sent_embeds) == np.sort(self.sent_embeds)).all()
        else:
            if not (np.sort(sent_embeds) == np.sort(self.sent_embeds)).all():
                print(f'Warning: sent_embeds in df do not match those specified in class instantiation')

        try:
            UIDs = df.UID.unique()
        except:
            UIDs = df.mapping_UID.unique()  # for pred-full analyses
        # Ensure that they are either list or np.array
        if isinstance(UIDs, np.ndarray):
            UIDs = UIDs.tolist()
        if check_UIDs:
            assert (np.sort(UIDs) == np.sort(self.UIDs)).all()
        else:
            try:
                if not (np.sort(UIDs) == np.sort(self.UIDs)).all():
                    print(f'Warning: UIDs in df do not match those specified in class instantiation')
            except:
                print(f'Warning: UIDs in df do not match those specified in class instantiation')
                print(f'np.sort(UIDs): {np.sort(UIDs)}')
                print(f'np.sort(self.UIDs): {np.sort(self.UIDs)}')
                print(f'Types of UIDs and self.UIDs: {type(UIDs)}, {type(self.UIDs)}')

        try:
            FLs = df.FL.unique()
        except:
            FLs = df.mapping_FL.unique()
        if check_FLs:
            assert (np.sort(FLs) == np.sort(self.FLs)).all()
        else:
            if not (np.sort(FLs) == np.sort(self.FLs)).all():
                print(f'Warning: FLs in df do not match those specified in class instantiation')

        try:
            modeltypes = df.modeltype.unique()
        except:
            modeltypes = df.mapping_modeltype.unique()
        if check_modeltypes:
            assert (np.sort(modeltypes) == np.sort(self.modeltypes)).all()
        else:
            if not (np.sort(modeltypes) == np.sort(self.modeltypes)).all():
                print(f'Warning: modeltypes in df do not match those specified in class instantiation')

        try:
            norms = df.norm.unique()
        except:
            norms = df.mapping_norm.unique()
        if check_norms:
            assert (np.sort(norms) == np.sort(self.norms)).all()
        else:
            if not (np.sort(norms) == np.sort(self.norms)).all():
                print(f'Warning: norms in df do not match those specified in class instantiation')

        preprocessors = df.preprocessor.unique()
        if check_preprocessors:
            assert (np.sort(preprocessors) == np.sort(self.preprocessor)).all()
        else:
            if not (np.sort(preprocessors) == np.sort(self.preprocessor)).all():
                print(f'Warning: preprocessors in df do not match those specified in class instantiation')

        preprocess_x = df.preprocess_X.unique()
        if check_preprocess_x:
            assert (np.sort(preprocess_x) == np.sort(self.preprocess_x)).all()
        else:
            if not (np.sort(preprocess_x) == np.sort(self.preprocess_x)).all():
                print(f'Warning: preprocess_x in df do not match those specified in class instantiation')

        preprocess_y = df.preprocess_y.unique()
        if check_preprocess_y:
            assert (np.sort(preprocess_y) == np.sort(self.preprocess_y)).all()
        else:
            if not (np.sort(preprocess_y) == np.sort(self.preprocess_y)).all():
                print(f'Warning: preprocess_y in df do not match those specified in class instantiation')

        # Check that preprocessors, preprocess_x, preprocess_y only have one value, because we don't iterate over it
        assert len(preprocessors) == 1
        assert len(preprocess_x) == 1
        assert len(preprocess_y) == 1

        print(f'== Checked that df has the same attributes as specified in class instantiation')

    def heatmap(self, df: pd.DataFrame = None,
                value: str = 'CV_score_mean',
                index: str = 'fracs',
                column: str = 'source_layer',
                PLOTDIR: str = '',
                CSVDIR: str = '',
                **kwargs):

        piv = df.pivot_table(values=value, index=index, columns=column, aggfunc=kwargs.get('aggfunc'))
        # Make sure that pcstop is sorted correctly (1, 2, 3, ...) and not (1, 10, 2, )
        # piv = piv.reindex([str(i) for i in range(1, len(piv.index) + 1)])
        roi = df.index.unique()
        assert (len(roi) == 1)
        roi = roi[0]

        # Values for title
        d_title = {}
        for val in ['source_model', 'sent_embed', 'UID', 'sess_num', 'FL', 'func_thresh', 'mapping_class', 'modeltype']:
            extracted_val = df[val].unique()
            if len(extracted_val) == 1:
                d_title[val] = extracted_val[0]
            else:
                d_title[val] = '-'.join(extracted_val)
                print(f'{val} has multiple values: {extracted_val}')

        if piv.max().max() > 1:
            fmt = '.3g'
        else:
            fmt = '.2f'

        # Make heatmap
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(piv, annot=kwargs.get('annot'), fmt=fmt, ax=ax,
                    cmap='RdBu_r', square=True,
                    cbar_kws={'label': d_axes_legend[value],
                              'shrink': 0.6, })
        plt.title(f'SOURCE: {d_title["source_model"]}-{self.layer_str} {d_title["sent_embed"]}\n'
                  f'TARGET: {roi} {d_title["UID"]}, sessions: {d_title["sess_num"]}\n'
                  f'First-level: {d_title["FL"]}, func thresh: {d_title["func_thresh"]}\n'
                  f'Mapping: {d_title["mapping_class"]}, modeltype: {d_title["modeltype"]}')
        plt.tight_layout(pad=1.5)
        if kwargs.get('save', True):
            # Get additional savestr if provided:
            try:
                savestr_add = kwargs['savestr_add']
            except:
                savestr_add = ''
            plt.savefig(join(PLOTDIR, f'heatmap_{value}_{index}_{column}_{roi}_{self.savestr}{savestr_add}.png'),
                        dpi=180)
            plt.savefig(join(PLOTDIR, f'heatmap_{value}_{index}_{column}_{roi}_{self.savestr}{savestr_add}.svg'),
                        dpi=180)

            # Save dataframe
            piv.to_csv(join(CSVDIR, f'heatmap_{value}_{index}_{column}_{roi}_{self.savestr}{savestr_add}.csv'))

        plt.show()

    def val_across_layers(self,
                          val: str = 'CV_score_mean',
                          val_err: typing.Union[str, None] = 'CV_score_sem',
                          add_NC: bool = True,
                          rois_of_interest: str = 'lang_LH_ROIs',
                          pcstop: int = 5,
                          fracs: float = 0.05,
                          ylim: typing.Union[tuple, None] = (-0.15, 0.3),
                          alpha: float = 0.7,
                          save: bool = True,
                          add_savestr: str = '',
                          PLOTDIR: str = '',
                          CSVDIR: str = '',
                          combine_source_models: bool = True,
                          **kwargs):
        """
        Obtain one instantiation of an encoding analysis. The goal is to obtain a dataframe that has rows according to
        number of source layers (one model) and plot the results (val) across layers for rois of interest.

        Args:
            val: str, which column (value) to plot
            val_err: str, which column (value) to use for errorbars. 'CV_score_sem' is the default which is SEM across
                CV folds.
            add_NC: bool, whether to add NC to the plot
            rois_of_interest: str, which ROIs to plot (uses the dictionary d_roi_lists_names)
            pcstop: int, which PCstop model to plot (GLMsingle)
            fracs: float, which frac model to plot (GLMsingle)
            ylim: tuple, y-axis limits
            alpha: float, transparency of the plot
            save: bool, whether to save the plot
            add_savestr: str, additional string to add to savestr
            PLOTDIR: str, path to plot directory
            CSVDIR: str, path to csv directory
            combine_source_models: bool, whether to include all unique source models on the same plot

        """
        df = self.df.copy(deep=True)

        ###### GET ROI NAMES ######
        rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

        ###### CHECK THAT DF HAS THE EXPECTED ENTRIES ######
        self.check_df_against_class_attributes(df=df)

        add_savestr = [f'_{add_savestr}' if add_savestr else ''][0]

        ###### ITERATE & PLOT ######
        for UID in df.UID.unique():
            for FL in df.FL.unique():
                if FL == 'gs':  # Only obtain one parameter combination!
                    df = df.query(f'pcstop == {pcstop} & fracs == {fracs}')
                else:
                    print(f'FL: {FL}, not indexing by pcstop and fracs')

                for norm in df.norm.unique():
                    for modeltype in df.modeltype.unique():
                        for sent_embed in df.sent_embed.unique():

                            # Get df for this instantiation
                            df_q = df.query(f'UID == "{UID}" and '
                                            f'FL == "{FL}" and '
                                            f'norm == "{norm}" and '
                                            f'modeltype == "{modeltype}" and '
                                            f'sent_embed == "{sent_embed}"')

                            # If df is empty, skip
                            if df_q.empty:
                                print(f'Empty df for {UID} {FL} {modeltype} {norm} {sent_embed}')
                                continue

                            # Constrain to ROIs of interest
                            df_q = df_q.loc[df_q.index.isin(rois_of_interest)]

                            # Sort by layers
                            if combine_source_models:
                                assert ((df_q.source_layer.unique().shape[0] * len(rois_of_interest)) * len(df_q.source_model.unique()) == df_q.shape[0])
                            else:
                                assert (df_q.source_layer.unique().shape[0] * len(rois_of_interest) == df_q.shape[0])
                            df_q = df_q.sort_values('source_layer')

                            # Define savestr
                            savestr = f'val-across-layers_' \
                                      f'VAL={val}_ERR={val_err}_' \
                                      f'yl={ylim}_' \
                                      f'NC={add_NC}_' \
                                      f'layers-n={len(df_q.source_layer.unique())}_' \
                                      f'{UID}_{rois_of_interest_name}_' \
                                      f'{self.base_savestr}' \
                                      f'{add_savestr}'

                            # Use savestr as titlestr, but wrap at 50 characters
                            titlestr = '\n'.join(wrap(savestr, 60))

                            print(f'Plotting {savestr}\n'
                                  f'Unique layers: {len(df_q.source_layer.unique())} and '
                                  f'unique ROIs: {len(df_q.index.unique())}\n'
                                  f'Yields {len(df_q.index)} rows\n')

                            # Plot CV score across layers for this UID as points and lines for each ROI
                            fig, ax = plt.subplots(figsize=(10, 7))
                            ax.set_box_aspect(0.6)
                            ax.set_xlabel('Source layer')
                            if ylim is not None:
                                ax.set_ylim(ylim)
                            ax.set_ylabel(d_axes_legend[val])

                            if combine_source_models:
                                for source_model in df_q.source_model.unique():
                                    df_source = df_q.query(f'source_model == "{source_model}"')

                                    for roi in df_source.index.unique():
                                        df_roi = df_source.loc[df_source.index == roi]
                                        try:
                                            color = d_model_colors[source_model]
                                        except:
                                            color = 'grey'
                                        if val_err:
                                            ax.errorbar(df_roi.source_layer, df_roi[val], yerr=df_roi[val_err],
                                                        label=f'{roi} ({source_model})',
                                                        alpha=alpha - 0.2, linewidth=2, elinewidth=1.5, color=color)
                                        else:
                                            ax.plot(df_roi.source_layer, df_roi[val], '-', label=f'{roi} ({source_model})',
                                                    alpha=alpha, linewidth=1.5)
                            else:
                                for roi in df_q.index.unique():
                                    df_roi = df_q.loc[df_q.index == roi]
                                    try:
                                        color = d_roi_colors[roi]
                                    except:
                                        color = 'grey'
                                    if val_err:
                                        ax.errorbar(df_roi.source_layer, df_roi[val], yerr=df_roi[val_err], color=color,
                                                    alpha=alpha - 0.2, linewidth=2, elinewidth=1.5)
                                    ax.plot(df_roi.source_layer, df_roi[val], '-', label=roi, color=color, alpha=alpha,
                                            linewidth=1.5)

                            if add_NC:
                                ax.plot(df_roi.source_layer, df_roi['nc'], '--', color='gray',
                                        alpha=alpha - 0.2, linewidth=2)
                                ax.fill_between(df_roi.source_layer, df_roi['nc'] - df_roi['split_half_se'],
                                                df_roi['nc'] + df_roi['split_half_se'], color='gray',
                                                edgecolor='none', alpha=0.15)

                            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
                            plt.title(titlestr, fontsize=12)
                            # add horizontal gridlines
                            ax.yaxis.grid(True, alpha=0.4)
                            plt.tight_layout(pad=2)
                            plt.xticks(fontsize=18)
                            plt.yticks(fontsize=18)
                            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                         ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(18)
                            if save:
                                savestr = shorten_savestr(savestr=savestr)
                                os.chdir(PLOTDIR)
                                plt.savefig(f'{savestr}.pdf', dpi=180)
                                os.chdir(CSVDIR)
                                df_q.to_csv(f'{savestr}.csv')
                            plt.tight_layout()
                            plt.show()

                            print()

    def val_across_layers_category(self,
                          val: str = 'CV_score_mean',
                          val_err: typing.Union[str, None] = 'CV_score_sem',
                          add_NC: bool = True,
                          rois_of_interest: str = 'lang_LH_ROIs',
                          pcstop: int = 5,
                          fracs: float = 0.05,
                          ylim: typing.Union[tuple, None] = (-0.15, 0.3),
                          alpha: float = 0.7,
                          save: bool = True,
                          add_savestr: str = '',
                          PLOTDIR: str = '',
                          CSVDIR: str = '',
                          **kwargs):
        """
        Obtain one instantiation of an encoding analysis. The goal here is to
        *aggregate over 'category'* and plot the mean + SEM across layers for
        ROIs of interest.

        Args:
            val: str, which column (value) to plot
            val_err: str, which column (value) to use for errorbars
            add_NC: bool, whether to add NC to the plot
            rois_of_interest: str, which ROIs to plot (uses the dictionary d_roi_lists_names)
            pcstop: int, which PCstop model to plot (GLMsingle)
            fracs: float, which frac model to plot (GLMsingle)
            ylim: tuple, y-axis limits
            alpha: float, transparency of the plot
            save: bool, whether to save the plot
            add_savestr: str, additional string to add to savestr
            PLOTDIR: str, path to plot directory
            CSVDIR: str, path to csv directory
            combine_source_models: bool, whether to combine all unique source models in the same plot
                                   (still relevant for filtering but not for per-model lines)
            **kwargs: other keyword arguments
        """

        df = self.df.copy(deep=True)

        # --- GET ROI NAMES ---
        rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

        # --- CHECK DF ---
        self.check_df_against_class_attributes(df=df)

        add_savestr = [f'_{add_savestr}' if add_savestr else ''][0]

        # --- ITERATE & PLOT ---
        for UID in df.UID.unique():
            for FL in df.FL.unique():
                # Only constrain pcstop & fracs for GLMsingle
                if FL == 'gs':
                    df = df.query(f'pcstop == {pcstop} & fracs == {fracs}')
                else:
                    print(f'FL: {FL}, not indexing by pcstop and fracs')

                for norm in df.norm.unique():
                    for modeltype in df.modeltype.unique():
                        for sent_embed in df.sent_embed.unique():

                            # Subset the dataframe
                            df_q = df.query(f'UID == "{UID}" and '
                                            f'FL == "{FL}" and '
                                            f'norm == "{norm}" and '
                                            f'modeltype == "{modeltype}" and '
                                            f'sent_embed == "{sent_embed}"')
                            if df_q.empty:
                                print(f'Empty df for {UID} {FL} {modeltype} {norm} {sent_embed}')
                                continue

                            # Constrain to ROIs of interest
                            df_q = df_q.loc[df_q.index.isin(rois_of_interest)]

                            # Sort by layers
                            df_q = df_q.sort_values('source_layer')

                            # Sanity check, take all the pruned source model, if startswith that
                            df_q_sanity = df_q.query('source_model.str.startswith("pruned")', engine='python')
                            # sort by source layer
                            df_q_sanity = df_q_sanity.sort_values('source_layer')

                            # --------------------------------------------------
                            #  NEW PART:
                            #  We now want to aggregate over "category".
                            #  We'll compute the mean and SEM of `val`
                            #  for each (category, layer).
                            # --------------------------------------------------
                            df_cat = (
                                df_q.groupby(['category', 'source_layer'])[val]
                                .agg(['mean', 'sem'])  # Could also use .agg(['mean','sem'])
                                .reset_index()
                            )

                            # Rename columns for convenience
                            df_cat.columns = ['category', 'source_layer', val, val_err]

                            # Define savestr
                            savestr = (f'val-across-layers-category_'
                                       f'VAL={val}_ERR={val_err}_'
                                       f'yl={ylim}_'
                                       f'NC={add_NC}_'
                                       f'layers-n={df_q.source_layer.nunique()}_'
                                       f'{UID}_{rois_of_interest_name}_'
                                       f'{self.base_savestr}'
                                       f'{add_savestr}')

                            titlestr = '\n'.join(wrap(savestr, 60))

                            print(f'Plotting {savestr}\n'
                                  f'Unique layers: {df_q.source_layer.nunique()} | '
                                  f'Unique categories: {df_q.category.nunique()}\n'
                                  f'Subset size: {df_q.shape[0]} rows\n')

                            # Create the plot
                            fig, ax = plt.subplots(figsize=(10, 7))
                            ax.set_box_aspect(0.6)
                            ax.set_xlabel('Source layer')
                            if ylim is not None:
                                ax.set_ylim(ylim)
                            ax.set_ylabel('Cross-validated predictivity (mean ± model seed SE)')

                            # Plot a separate line for each category
                            for cat in df_cat['category'].unique():
                                df_sub = df_cat.query('category == @cat').sort_values('source_layer')
                                color = d_category_colors[cat]
                                label = f'Category: {cat}'

                                if val_err and val_err in df_sub.columns:
                                    ax.errorbar(
                                        df_sub['source_layer'], df_sub[val],
                                        yerr=df_sub[val_err],
                                        label=label, alpha=alpha, linewidth=2.5, elinewidth=2, color=color
                                    )
                                else:
                                    ax.plot(
                                        df_sub['source_layer'], df_sub[val],
                                        '-', label=label, alpha=alpha, linewidth=2, color=color
                                    )

                            # Optionally plot NC
                            if add_NC:
                                # If we want a single NC line, we can average across categories or pick one
                                # Below, we simply pick the average across the subset for demonstration:
                                df_nc = df_q.groupby('source_layer')['nc', 'split_half_se'].mean()
                                # ^ Note: if you have multiple entries for NC across categories,
                                #   you may need a different aggregator or approach.

                                ax.plot(df_nc.index, df_nc['nc'], '--', color='gray',
                                        alpha=alpha - 0.2, linewidth=2, label='Noise Ceiling')
                                ax.fill_between(
                                    df_nc.index,
                                    df_nc['nc'] - df_nc['split_half_se'],
                                    df_nc['nc'] + df_nc['split_half_se'],
                                    color='gray', edgecolor='none', alpha=0.15
                                )

                            # Formatting
                            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
                            plt.title(titlestr, fontsize=12)
                            plt.tight_layout(pad=2)
                            # Add a tick for each layer
                            ax.set_xticks(df_q.source_layer.unique())
                            plt.xticks(fontsize=18)
                            plt.yticks(fontsize=18)
                            # Add horizontal gridlines
                            ax.yaxis.grid(True, alpha=0.4)
                            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                         ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(18)

                            # Save plot & CSV if requested
                            if save:
                                savestr_short = shorten_savestr(savestr=savestr)
                                os.chdir(PLOTDIR)
                                plt.savefig(f'{savestr_short}.pdf', dpi=180)
                                os.chdir(CSVDIR)
                                df_cat.to_csv(f'{savestr_short}.csv', index=False)

                            plt.tight_layout()
                            plt.show()
                            print()