from BrainClasses import *
from paths import ROOTDIR, RESULTROOT, WEIGHTROOT, ACTVROOT, LOGROOT

script_name = os.path.basename(__file__).split('.')[0]

"""
Script to fit a mapping based on a given brain target. This script loads the source model from a specified csv.
Estimate cross-validated prediction performance for the brain target, using k-fold ridge regression.
We evaluate the lang_LH_netw region which consists of the voxels from the five left-hemisphere language fROIs.
"""


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='')
    # ANN (source) specific
    parser.add_argument('--source_layer', default=10, help='Which layer to use for prediction. '
                                                          '0 is the embedding layer')
    parser.add_argument('--sent_embed', default='last', type=str,
                        help='How to obtain sentence embeddings.', )
    parser.add_argument('--source_CSVDIR', default='/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings',
                        type=str, help='Which directory to load the source CSV from')
    parser.add_argument('--source_CSV_fname',
                        default='pruned_seed0_embeddings',
                        type=str, help='File name of the source CSV containing the activations')

    # Brain (target) specific
    parser.add_argument('--DATAROOT', default='/Users/gt/Documents/GitHub/drive_suppress_brains/data/',
                        type=str, help='Where to load the target dataset from')
    parser.add_argument('--target_dataset_fname', default='brain-lang-data_participant_20230728.csv', type=str,
                        help='Filename of target dataset. Will be loaded from DATAROOT.')
    parser.add_argument('--UID', default='848-853-865-875-876', type=str, help='Unique subject ID')
    parser.add_argument('--sess_num', default='1-2', type=str, help='Which session(s) to load data for')
    parser.add_argument('--FL', default='gs', type=str, help='First level (FL) identifier')
    parser.add_argument('--modeltype', default='d', type=str, help='Which GLMsingle model type to load')
    parser.add_argument('--preproc', default='swr', type=str, help='Input data to GLMsingle')
    parser.add_argument('--pcstop', default=5, type=int, help='Num of PCs removed in GLMsingle')
    parser.add_argument('--fracs', default=0.05, type=float, help='Ridge regression fraction')
    parser.add_argument('--func_thresh', default=90, type=int, help='Threshold for extracting functional ROIs')
    parser.add_argument('--norm', default='bySessVoxZ', type=str,
                        help='Which normalization to use (while extracting and packaging ROIs)')
    parser.add_argument('--regression_dict_type', default='rois', type=str,
                        help='Whether to load "rois" or "voxs" neural data dicts')
    parser.add_argument('--savestr_prefix', default='20221214a', type=str,
                        help='If the netw_dict was stored using a savestr_prefix, specify it here.')
    parser.add_argument('--stim_col', default='sentence', type=str, help='Which stimset column contains the stimuli')
    parser.add_argument('--specific_target', default=None, type=str2none,
                        help='Whether to run a specific target (str). If None, run all targets in the target dataset.')
    parser.add_argument('--stimsetid', default='beta-control-neural-T', type=str, help='The stimulus identifier.'
                           'For instance, the baseline set is denoted as "beta-neural-control-T" and the'
                           'model activations will be stored under this name.')

    # Mapping specific
    parser.add_argument('--langloc', default=False, type=str2bool, help='Whether to use the langloc masked activations')
    parser.add_argument('--mapping_class', default='ridgeCV', type=str, help='Which mapping class to use')
    parser.add_argument('--metric', default='pearsonr', type=str, help='Which metric to use')
    parser.add_argument('--preprocessor', default='None', type=str2none,
                        help='How to preprocess data prior to mapping. '
                             'Options are "None", "demean", "demean_std", "minmax", "pca10", "pca800"')
    parser.add_argument('--preprocess_X', default=False, type=str2bool,
                        help='Whether to preprocess X (regressors) prior to mapping. Must specify preprocessor if True.')
    parser.add_argument('--preprocess_y', default=False, type=str2bool,
                        help='Whether to preprocess y (targets) prior to mapping. Must specify preprocessor if True.')
    parser.add_argument('--run_cv', default=True, type=str2bool, help='Whether to run cross-validation')
    parser.add_argument('--k', default=5, type=int, help='How many folds to use for cross-validation if run_cv is True')
    parser.add_argument('--store_pred_per_fold', default=False, type=str2bool,
                        help='Whether to store predictions (y/y_pred) per fold')
    parser.add_argument('--store_full_mapping', default=False, type=str2bool,
                        help='Whether to store weights fitted on the full target dataset')
    parser.add_argument('--store_full_pred', default=True, type=str2bool,
                        help='Whether to store predictions on the full target dataset')
    parser.add_argument('--permute_X', default=None, type=str2none,
                        help='Whether to permute X (regressors) prior to mapping'
                             'Options are None, "shuffle_X_cols", "shuffle_X_rows", "shuffle_each_X_col"')

    # Misc
    parser.add_argument('--verbose', default=True, type=str2bool,
                        help='Whether to print output and not create a log file')

    ####### ARGUMENT LOGGING #######
    args = parser.parse_args(raw_args)
    print(json.dumps(vars(args), indent=2))
    args.source_model = args.source_CSV_fname
    # If layer is string, convert to int
    if isinstance(args.source_layer, str):
        args.source_layer = int(args.source_layer)

    sess_id = obtain_sess_id(UID=args.UID,
                             sess_num=args.sess_num,
                             d_UID_to_session=d_UID_to_session)
    DATAROOT = args.DATAROOT
    args_logger = ArgumentLogger(vars(args),
                                 script_name=script_name,
                                 add_args={'sess_id': sess_id,
                                           'script_name': script_name,
                                           'DATAROOT': DATAROOT,
                                           'RESULTROOT': RESULTROOT,
                                           'WEIGHTROOT': WEIGHTROOT,
                                           'LOGROOT': LOGROOT,
                                           'ACTVROOT': ACTVROOT,
                                           f'date_{script_name}': date},
                                 result_root=RESULTROOT,
                                 weight_root=WEIGHTROOT,
                                 log_root=LOGROOT,
                                 actv_root=ACTVROOT,
                                 )

    ####### LOAD NEURAL TARGET ########
    # Specify DATAROOT as where you saved the target dataset csv file (from https://github.com/gretatuckute/drive_suppress_brains)
    if args.target_dataset_fname.endswith('.csv'):
        df_data = pd.read_csv(f'{DATAROOT}/{args.target_dataset_fname}')
    elif args.target_dataset_fname.endswith('.pkl'):
        df_data = pd.read_pickle(f'{DATAROOT}/{args.target_dataset_fname}')
    else:
        raise ValueError(
            f'Specify loading of csv or pickle file. Currently, {args.target_dataset_fname} is not supported.')

    # Contrain to the specified UID. If joined by hyphens, average across subjects.
    if '-' in args.UID:
        # Split and convert to int
        UIDs = [int(UID) for UID in args.UID.split('-')]
        df_data_UID = df_data.query('target_UID in @UIDs').groupby(['item_id', 'roi']).mean().reset_index()
    else:
        UID = int(args.UID)
        df_data_UID = df_data.query('target_UID == @UID')

    # Generate a pivot table of item x ROI
    neural_data = df_data_UID.pivot(index='item_id', columns='roi',
                                    values='response_target')  # response_target is the neural data

    # The neural data is now indexed by item_id. We want to prefix it with the stimsetid to have a string identifier.
    neural_data.index = args.stimsetid + '.' + neural_data.index.astype(str)

    # Next, get the stimset
    # For each unique item_id, get the corresponding sentence. Assert that they're all the same for each unique item_id
    for item_id in df_data_UID['item_id'].unique():
        sentences = df_data.query('item_id == @item_id')['sentence'].unique()
        assert len(sentences) == 1, f'More than one sentence for item_id {item_id}'

    # Query the stimset such that we get the item_ids in df_data_UID
    stimset = df_data.query('item_id in @df_data_UID.item_id').groupby('item_id').first().reset_index()[
        ['item_id', 'sentence']]
    stimset.index = args.stimsetid + '.' + stimset['item_id'].astype(
        str)  # Prefix with stimsetid to have a string identifier

    ####### CREATE LOGGING FILE WITH CORRECT PARAMETERS (AFTER DATA LOADING) ########
    if not args.verbose:
        print(
            f'\nLogging output to file...\n {join(args_logger.LOGDIR, f"{script_name}_{args_logger.save_str}_{date}.log")}')
        sys.stdout = open(join(args_logger.LOGDIR, f'{script_name}_{args_logger.save_str}_{date}.log'), 'a+')

    print('\n' + ('*' * 40))
    print(json.dumps(vars(args), indent=2))
    print(('*' * 40) + '\n')
    args_logger.print_package_versions()

    ####### BRAIN ENCODER ########
    brain = BrainEncoder()
    brain.encode(stimset=stimset, neural_data=neural_data,
                 specific_target=args.specific_target)

    ####### ANN ENCODER ########
    ann = ANNEncoder(source_model=args.source_model,
                     sent_embed=args.sent_embed,
                     actv_cache_setting=None,
                     actv_cache_path=args_logger.ACTVDIR)

    ### Deal with langloc or not ###
    if args.langloc:
        args.source_layer = 0 # langloc is always at layer 0 as a mock, because we take units across several layers
        ann.encode_langloc_from_csv(stimset=stimset,
                            stim_col=args.stim_col,
                            CSVDIR=args.source_CSVDIR,
                            CSV_fname=args.source_CSV_fname,
                            cache_new_actv=False,
                            **{'stimsetid_suffix': f''})

    else: # use activations from specified csv
        ann.encode_from_csv(stimset=stimset,
                            stim_col=args.stim_col,
                            CSVDIR=args.source_CSVDIR,
                            CSV_fname=args.source_CSV_fname,
                            cache_new_actv=False,
                            **{'stimsetid_suffix': f''})
    sys.stdout.flush()

    ####### METRIC ########
    metric = Metric(metric=args.metric)

    ####### PREPROCESSOR ########
    preprocessor = Preprocessor(preprocess=args.preprocessor)

    ####### MAPPING ########
    mapping = Mapping(ANNEncoder=ann,
                      ann_layer=args.source_layer,
                      BrainEncoder=brain,
                      mapping_class=args.mapping_class,
                      metric=metric,
                      Preprocessor=preprocessor,
                      preprocess_X=args.preprocess_X,
                      preprocess_y=args.preprocess_y, )

    # We are just interested in crossvalidated model performance here.
    # df_full_scores, d_weights, d_full_pred = mapping.fit_full_mapping()

    # if args.store_full_mapping:
    #     args_logger.store(data=d_weights,
    #                       DIR='WEIGHTDIR',
    #                       prefix_str='mapping-full', )
    #
    # if args.store_full_pred:
    #     args_logger.store(data=d_full_pred,
    #                       DIR='RESULTDIR',
    #                       prefix_str='pred-full', )

    if args.run_cv:
        df_cv_scores, df_cv_scores_across_folds, d_CV_pred = mapping.CV_score(k=args.k,
                                                                              store_pred_per_fold=args.store_pred_per_fold,
                                                                              permute_X=args.permute_X,
                                                                              plot=args.specific_target)

        # df_cv_scores = pd.concat([df_cv_scores, df_full_scores], axis=1)  # add the full model scores to the cv scores
        df_cv_scores = args_logger.add_args_to_df(df_cv_scores)

        args_logger.store(data=df_cv_scores,
                          DIR='RESULTDIR',
                          prefix_str=f'CV-k-{args.k}')

        if args.store_pred_per_fold:  # store both the CV fold log as well as predictions concatenated across folds (y_CV_pred)
            df_cv_scores_across_folds = args_logger.add_args_to_df(df_cv_scores_across_folds)
            args_logger.store(data=df_cv_scores_across_folds,
                              DIR='RESULTDIR',
                              prefix_str=f'CV-k-{args.k}-folds')

            args_logger.store(data=d_CV_pred,
                              DIR='RESULTDIR',
                              prefix_str=f'pred-CV-k-{args.k}')

    sys.stdout.flush()


if __name__ == '__main__':
    main()
