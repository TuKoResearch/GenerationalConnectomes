from plotClasses import *

script_name = os.path.basename(__file__).split('.')[0]

"""Plot CV outputs from fit_mapping.py"""

PLOTDIR = f'{ROOTDIR}/plots/{script_name}/'
CSVDIR = f'{ROOTDIR}/csvs/{script_name}/'
if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)
if not os.path.exists(CSVDIR):
    os.makedirs(CSVDIR)

##### GENERAL SETTINGS #####
NC_5T = True  # Load the NC + reliability computed based on the 5-avg model
k_fold_cv_across_layers = True  # Plot k-fold CV results across layers (not langloc)
k_fold_langloc = True # Plot the langloc results

save = False

modeltypes = ['d']
func_threshs = ['90']
sess_nums = ['1-2']
FLs = ['gs']
pcstop = 5
fracs = 0.05
mapping_classes = ['ridgeCV']
metrics = ['pearsonr']
norms = ['bySessVoxZ']
regression_dict_types = ['rois']
savestr_prefixes = ['20221214a']
preprocessors = ["None"]
preprocess_xs = ["False"]  # fit_mapping preprocessing
preprocess_ys = ["False"]  # fit_mapping preprocessing

##### LOAD THE NOISECEILING (NC) + SPLIT-HALF RELIABILITY COMPUTED BASED ON THE 5-AVG MODEL #####
if NC_5T:
    """
	Load the NC + reliability computed based on the 5-avg model. Can be used for downstream analyses (flags).
	"""

    ## Settings ##
    BETACONTROL_ROOTDIR = '/Users/gt/Documents/GitHub/beta-neural-control'
    NCDIR = f'{BETACONTROL_ROOTDIR}/data_packaging_visualization/csvs/compute_NSD_noiseceiling/'
    UIDs = "848-853-865-875-876"
    NC_n = 5
    # NC_rois_20221214a_gs_848-853-865-875-876_pcstop-5_fracs-0.05_norm-bySessVoxZ_n-rois-488_NC_n-5_with-split-half.csv
    fname_nc = f'NC_{regression_dict_types[0]}_{"-".join(savestr_prefixes)}_{FLs[0]}_{UIDs}_pcstop-{pcstop}_fracs-{fracs}_norm-{norms[0]}_n-rois-488_NC_n-{NC_n}_with-split-half'

    ## Load ##
    df_nc_all = pd.read_csv(f'{NCDIR}{fname_nc}.csv', index_col=0)

    # All that have nc_split_half_se_zero = False
    # df_nc = df_nc_all[df_nc_all['nc_split_half_se_zero_overlap'] == False]

##### PLOT K-FOLD (WITHIN-AVG SUBJECT) RESULTS ACROSS LAYERS #####
if k_fold_cv_across_layers:
    """
	Flag for plotting k-fold (within-subject) fit_mapping.py scores across all layers for source models:

	UIDs: 848-853-865-875-876

	"""

    ## Settings ##
    source_models = [
        'pruned_seed0_embeddings',
        'pruned_seed1_embeddings',
        'pruned_seed2_embeddings',
        'pruned_seed3_embeddings',

        'random_seed0_embeddings',
        'random_seed1_embeddings',
        'random_seed2_embeddings',
        'random_seed3_embeddings',

        'dense_seed0_embeddings',
        'dense_seed1_embeddings',
        'dense_seed2_embeddings',
        'dense_seed3_embeddings',
    ]

    sent_embeds = ['last', ]
    UIDs = "848-853-865-875-876"
    UIDs = UIDs.split()

    ## Define output df and unpack data ##
    o = MappingOutput(source_models=source_models,
                      source_layers=d_source_layers[source_models[0]],
                      sent_embeds=sent_embeds,
                      regression_dict_types=regression_dict_types,
                      savestr_prefixes=savestr_prefixes,
                      UIDs=UIDs,
                      sess_nums=sess_nums,
                      FLs=FLs,
                      modeltypes=modeltypes,
                      func_threshs=func_threshs,
                      norms=norms,
                      mapping_classes=mapping_classes,
                      metrics=metrics,
                      preprocessors=preprocessors,
                      preprocess_xs=preprocess_xs,
                      preprocess_ys=preprocess_ys, )

    o.find_results_folders(subfolder='fit_mapping')
    o.load_results(subfolder='fit_mapping',
                   prefix_str=f'CV-k-5_')

    ## Format dataframe ##
    o.cols_str_to_int(cols=['source_layer'])
    o.cols_bool_and_none_to_str(cols=['manual_target', 'specific_target',
                                      'preprocessor', 'preprocess_X', 'preprocess_y',
                                      'run_cv', 'store_pred_per_fold', 'store_full_mapping',
                                      'store_full_pred', 'run_RSA', 'verbose',
                                      'permute_X'])

    ## Store full df before indexing and generating plots ##
    if save:
        o.df.to_csv(
            f'{CSVDIR}k-fold-cv_{o.base_savestr}_layers-{len(np.unique(o.df.source_layer))}_UIDs-{UIDs}_rois-{len(np.unique(o.df.index.values))}.csv')

    ## If NC_5T is True and UIDs = '848-853-865-875-876', then we can add the NC values to the df ##
    if NC_5T and UIDs == ['848-853-865-875-876']:
        # Merge on index (both have the ROI as index)
        df_with_nc = o.merge_with_NC(df_nc=df_nc_all,
                                     inplace=True)

    o.df['category'] = o.df['source_model'].str.split('_').str[0]

    ### FIT MAPPING ###
    ## Plot score across layers ##
    for ylim in [(0, 0.4)]:
        add_savestr = ''

        if ylim == (0, 0.7):
            add_NC = False
        else:
            add_NC = False

        ## Same y-lim for all plots ##
        o.val_across_layers(val='CV_score_mean',
                            rois_of_interest='lang_LH_netw',
                            add_NC=add_NC,
                            pcstop=pcstop,
                            fracs=fracs,
                            save=save,
                            add_savestr=add_savestr,
                            ylim=ylim,
                            PLOTDIR=PLOTDIR, CSVDIR=CSVDIR, )

    # Plot one with the categories as mean and SEM over random seeds
    o.val_across_layers_category(val='CV_score_mean',
                                 rois_of_interest='lang_LH_netw',
                                 add_NC=False,
                                 pcstop=pcstop,
                                 fracs=fracs,
                                 save=save,
                                 add_savestr='',
                                 ylim=(0, 0.40),
                                 PLOTDIR=PLOTDIR, CSVDIR=CSVDIR, )

##### PLOT K-FOLD (WITHIN-AVG SUBJECT) RESULTS FOR LANGLOC RESULTS (JUST ONE VAL PER MODEL) #####
if k_fold_langloc:
    """
    Flag for plotting k-fold (within-subject) fit_mapping.py scores on the langloc localized embeddings.

    UIDs: 848-853-865-875-876
    """

    ## Settings ##
    source_models = [
        'pruned_seed0_embeddingslanglocperc=1',
        'pruned_seed1_embeddingslanglocperc=1',
        'pruned_seed2_embeddingslanglocperc=1',
        'pruned_seed3_embeddingslanglocperc=1',

        'random_seed0_embeddingslanglocperc=1',
        'random_seed1_embeddingslanglocperc=1',
        'random_seed2_embeddingslanglocperc=1',
        'random_seed3_embeddingslanglocperc=1',

        'dense_seed0_embeddingslanglocperc=1',
        'dense_seed1_embeddingslanglocperc=1',
        'dense_seed2_embeddingslanglocperc=1',
        'dense_seed3_embeddingslanglocperc=1',
    ]

    # Append 0 to the source models
    perc = 10  # or 1
    if perc == 10:
        source_models = [f'{model}0' for model in source_models]

    sent_embeds = ['last', ]
    UIDs = "848-853-865-875-876"
    UIDs = UIDs.split()

    ## Define output df and unpack data ##
    o = MappingOutput(source_models=source_models,
                      source_layers=[0],
                      sent_embeds=sent_embeds,
                      regression_dict_types=regression_dict_types,
                      savestr_prefixes=savestr_prefixes,
                      UIDs=UIDs,
                      sess_nums=sess_nums,
                      FLs=FLs,
                      modeltypes=modeltypes,
                      func_threshs=func_threshs,
                      norms=norms,
                      mapping_classes=mapping_classes,
                      metrics=metrics,
                      preprocessors=preprocessors,
                      preprocess_xs=preprocess_xs,
                      preprocess_ys=preprocess_ys, )

    o.find_results_folders(subfolder='fit_mapping')
    o.load_results(subfolder='fit_mapping',
                   prefix_str=f'CV-k-5_')

    ## Format dataframe ##
    o.cols_str_to_int(cols=['source_layer'])
    o.cols_bool_and_none_to_str(cols=['manual_target', 'specific_target',
                                      'preprocessor', 'preprocess_X', 'preprocess_y',
                                      'run_cv', 'store_pred_per_fold', 'store_full_mapping',
                                      'store_full_pred', 'run_RSA', 'verbose',
                                      'permute_X'])

    ## Store full df before indexing and generating plots ##
    if save:
        o.df.to_csv(
            f'{CSVDIR}k-fold-cv_{o.base_savestr}_layers-{len(np.unique(o.df.source_layer))}_UIDs-{UIDs}_rois-{len(np.unique(o.df.index.values))}.csv')

    ## If NC_5T is True and UIDs = '848-853-865-875-876', then we can add the NC values to the df ##
    if NC_5T and UIDs == ['848-853-865-875-876']:
        # Merge on index (both have the ROI as index)
        df_with_nc = o.merge_with_NC(df_nc=df_nc_all,
                                     inplace=True)

    ### FIT MAPPING ###
    ## We only have one value per model ##
    df_lang = o.df.loc['lang_LH_netw']
    assert (df_lang.shape[0] == len(source_models))

    # Make a category col which has the first part pre _ in the source_model col
    df_lang['category'] = df_lang['source_model'].str.split('_').str[0]

    # Now make a barplot per category, with individual points for each model, and each bar as the category. Plot SEM
    # as error bars
    # Calculate the mean and SEM for each category
    category_stats = df_lang.groupby('category').agg(
        CV_score_mean=('CV_score_mean', 'mean'),
        CV_score_sem=('CV_score_mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()
    # Sort categories by pruned, random, dense
    category_stats['category'] = pd.Categorical(category_stats['category'], categories=['pruned', 'random', 'dense'],
                                                ordered=True)
    category_stats = category_stats.sort_values('category')

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 6))
    # Barplot with SEM as error bars
    sns.barplot(data=category_stats, x='category', y='CV_score_mean', ax=ax,
                order=['pruned', 'random', 'dense'],
                yerr=category_stats['CV_score_sem'].values,
                # add the colors
                palette=[d_category_colors[cat] for cat in category_stats['category'].values],
                alpha=0.8, zorder=2)
    # Swarmplot with individual points, colored by model
    # sns.swarmplot(data=df_lang, x='category', y='CV_score_mean', ax=ax,
    #               hue='source_model', palette=d_model_colors, size=10, dodge=True)
    # Adjusting plot aesthetics
    ax.set_ylim(0.1, 0.3)
    # Make ticks larger
    ax.tick_params(axis='both', which='major', labelsize=14)
    # add very light horizontal lines
    ax.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax.set_ylabel('CV score')
    ax.set_xlabel('Model')
    ax.set_title('CV score on langloc LH network')
    plt.tight_layout()
    if save:
        plt.savefig(f'{PLOTDIR}category-barplot_k-fold-cv_{o.base_savestr}_langloc{perc}perc_lang_LH_netw.pdf', dpi=300)
    plt.show()

    print()

    # import glob
    #
    # # In this folder /Users/gt/Documents/GitHub/ConnectomePruning/NeuralAlignment/results/fit_mapping/
    # # if the fname contains langloc, load the pickle
    # for fname in os.listdir(f'{ROOTDIR}/results/fit_mapping/'):
    #     if 'langloc' in fname:
    #         # find the pkl fname in that folder
    #         pkl_fname = glob.glob(f'{ROOTDIR}/results/fit_mapping/{fname}/*.pkl')[0]
    #
    #         # Load the pkl
    #         df = pd.read_pickle(pkl_fname)
    #
    #         lang = df.loc['lang_LH_netw']
    #
    #         print(f'{fname} lang_LH_netw: {lang["CV_score_mean"]}')
    #
