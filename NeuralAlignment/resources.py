import numpy as np

####### UID related ########

d_UID_to_session = {'848': {'1': 'FED_20220420b_3T1',
                            '2': 'FED_20220427a_3T1',
                            '1-2': 'FED_20220420b_3T1-FED_20220427a_3T1',
                            '1-2-3': 'FED_20220420b_3T1-FED_20220427a-nan', },
                    '853': {'1': 'FED_20211008a_3T1',
                            '2': 'FED_20211013b_3T1',
                            '1-2': 'FED_20211008a_3T1-FED_20211013b_3T1',
                            '3': 'FED_20211020a_3T1',
                            '1-2-3': 'FED_20211008a_3T1-FED_20211013b_3T1-nan', },
                    '865': {'1': 'FED_20220414b_3T1',
                            '2': 'FED_20220415a_3T1',
                            '1-2': 'FED_20220414b_3T1-FED_20220415a_3T1',
                            '1-2-3': 'FED_20220414b_3T1-FED_20220415a_3T1-nan', },
                    '875': {'1': 'FED_20220408a_3T1',
                            '2': 'FED_20220411a_3T1',
                            '1-2': 'FED_20220408a_3T1-FED_20220411a_3T1',
                            '1-2-3': 'FED_20220408a_3T1-FED_20220411a_3T1-nan', },
                    '876': {'1': 'FED_20220413a_3T1',
                            '2': 'FED_20220420a_3T1',
                            '1-2': 'FED_20220413a_3T1-FED_20220420a_3T1',
                            '1-2-3': 'FED_20220413a_3T1-FED_20220420a_3T1-nan', },
                    '797': {'1': 'FED_20220810b_3T1',
                            '2': 'FED_20220811b_3T1',
                            '3': 'FED_20220815a_3T1',
                            '1-2': 'FED_20220810b_3T1-FED_20220811b_3T1',
                            '1-2-3': 'FED_20220810b_3T1-FED_20220811b_3T1-FED_20220815a_3T1', },
                    '841': {'1': 'FED_20220803a_3T1',
                            '2': 'FED_20220809a_3T1',
                            '3': 'FED_20220812a_3T1',
                            '1-2': 'FED_20220803a_3T1-FED_20220809a_3T1',
                            '1-2-3': 'FED_20220803a_3T1-FED_20220809a_3T1-FED_20220812a_3T1', },
                    '880': {'1': 'FED_20220802b_3T1',
                            '2': 'FED_20220804a_3T1',
                            '3': 'FED_20220812c_3T1',
                            '1-2': 'FED_20220802b_3T1-FED_20220804a_3T1',
                            '1-2-3': 'FED_20220802b_3T1-FED_20220804a_3T1-FED_20220812c_3T1', },
                    '946': {'1': 'FED_20221013b_3T1'},
                    '947': {'1': 'FED_20221014c_3T1'},
                    '866': {'1': 'FED_20221109a_3T1'},
                    '958': {'1': 'FED_20221109b_3T1'},
                    '1051': {'1': 'FED_20230518a_3T2'},
                    '837': {'1': 'FED_20221202a_3T1',
                            '2': 'FED_20221205b_3T1',
                            '3': 'FED_20221209a_3T1',
                            '1-2': 'FED_20221202a_3T1-FED_20221205b_3T1',
                            '1-2-3': 'FED_20221202a_3T1-FED_20221205b_3T1-FED_20221209a_3T1', },
                    '856': {'1': 'FED_20221129a_3T1',
                            '2': 'FED_20221201b_3T1',
                            '3': 'FED_20221205a_3T1',
                            '1-2': 'FED_20221129a_3T1-FED_20221201a_3T1',
                            '1-2-3': 'FED_20221129a_3T1-FED_20221201b_3T1-FED_20221205a_3T1', },
                    }

d_sess_type = {848: 'T',
               853: 'T',
               865: 'T',
               875: 'T',
               876: 'T',
               797: 'D',
               841: 'D',
               880: 'D',
               946: 'D',
               947: 'D',
               866: 'D',
               958: 'D',
               1051: 'D',
               837: 'S',
               856: 'S'}

blocked_UIDs = [946, 947, 866, 958, 1051]
search_UIDs = [797, 841, 880]
synth_UIDs = [837, 856]
UIDs_T = [848, 853, 865, 875, 876]
UIDs_D = [797, 841, 880, 837, 856]

####### PLOT related #######

d_roi_colors = {
    'lang_LH_netw': 'firebrick',
    'lang_LH_IFGorb': 'saddlebrown',
    'lang_LH_IFG': 'darkviolet',
    'lang_LH_MFG': 'gold',
    'lang_LH_AntTemp': 'plum',
    'lang_LH_PostTemp': 'orangered',
    'lang_LH_AngG': 'lightsalmon',
}

d_model_colors = {
    'pruned_seed0_embeddings': 'purple',
    'pruned_seed1_embeddings': 'purple',
    'pruned_seed2_embeddings': 'purple',
    'pruned_seed3_embeddings': 'purple',

    'random_seed0_embeddings': 'goldenrod',
    'random_seed1_embeddings': 'goldenrod',
    'random_seed2_embeddings': 'goldenrod',
    'random_seed3_embeddings': 'goldenrod',

    'dense_seed0_embeddings': 'darkseagreen',
    'dense_seed1_embeddings': 'darkseagreen',
    'dense_seed2_embeddings': 'darkseagreen',
    'dense_seed3_embeddings': 'darkseagreen',

    # 'GPT-100M-100Mtokens': 'gray',
    # 'GPT-100M-25percrandom-100Mtokens': 'pink',
    # 'GPT-100M-25perc-100Mtokens': 'purple',
    # 'GPT-100M-25perc-seed0-100Mtokens': 'SlateBlue',
    # 'GPT-100M-25perc-seed1-100Mtokens': 'MediumSlateBlue',
}

d_uid_colors = {
    '848': 'lightsteelblue',
    '853': 'goldenrod',
    '865': 'darkkhaki',
    '875': 'mediumaquamarine',
    '876': 'palevioletred', }

d_netw_colors = {
    'lang_LH_netw': 'firebrick',
    'lang_RH_netw': 'firebrick',
    'lang_LHRH_netw': 'firebrick',
    'lang_LH_IFGorb': 'firebrick',
    'lang_LH_IFG': 'firebrick',
    'lang_LH_MFG': 'firebrick',
    'lang_LH_AntTemp': 'firebrick',
    'lang_LH_PostTemp': 'firebrick',
    'lang_LH_AngG': 'firebrick',
    'lang_RH_IFGorb': 'firebrick',
    'lang_RH_IFG': 'firebrick',
    'lang_RH_MFG': 'firebrick',
    'lang_RH_AntTemp': 'firebrick',
    'lang_RH_PostTemp': 'firebrick',
    'lang_RH_AngG': 'firebrick',

    'md_LH_netw': 'mediumblue',
    'md_RH_netw': 'mediumblue',
    'md_LHRH_netw': 'mediumblue',
    'md_LH_PrecentralAprecG': 'mediumblue',
    'md_LH_PrecentralBIFGop': 'mediumblue',
    'md_LH_antParietal': 'mediumblue',
    'md_LH_insula': 'mediumblue',
    'md_LH_medialFrontal': 'mediumblue',
    'md_LH_midFrontal': 'mediumblue',
    'md_LH_midParietal': 'mediumblue',
    'md_LH_postParietal': 'mediumblue',
    'md_LH_supFrontal': 'mediumblue',
    'md_RH_PrecentralAprecG': 'mediumblue',
    'md_RH_PrecentralBIFGop': 'mediumblue',
    'md_RH_antParietal': 'mediumblue',
    'md_RH_insula': 'mediumblue',
    'md_RH_medialFrontal': 'mediumblue',
    'md_RH_midFrontal': 'mediumblue',
    'md_RH_midFrontalOrb': 'mediumblue',
    'md_RH_midParietal': 'mediumblue',
    'md_RH_postParietal': 'mediumblue',
    'md_RH_supFrontal': 'mediumblue',

    'dmn_LH_netw': 'forestgreen',
    'dmn_RH_netw': 'forestgreen',
    'dmn_LHRH_netw': 'forestgreen',
    'dmn_LH_AntTemp': 'forestgreen',
    'dmn_LH_FrontalMed': 'forestgreen',
    'dmn_LH_MidCing': 'forestgreen',
    'dmn_LH_PostCing': 'forestgreen',
    'dmn_LH_STGorInsula': 'forestgreen',
    'dmn_LH_TPJ': 'forestgreen',
    'dmn_RH_AntTemp': 'forestgreen',
    'dmn_RH_FrontalMed': 'forestgreen',
    'dmn_RH_MidCing': 'forestgreen',
    'dmn_RH_PostCing': 'forestgreen',
    'dmn_RH_STGorInsula': 'forestgreen',
    'dmn_RH_TPJ': 'forestgreen',

    # 'anatglasser_LH_V1': 'sienna',
    # 'anatglasser_RH_V1': 'sienna',
    # 'anatglasser_LHRH_V1': 'sienna',
    #
    # 'anatglasser_LH_AudPrimary': 'slategrey',
    # 'anatglasser_RH_AudPrimary': 'slategrey',
    # 'anatglasser_LHRH_AudPrimary': 'slategrey',
}

d_axes_legend = {'CV_score_mean': 'Cross-validated predictivity (mean ± fold SE)',
                 'CV_score_median': 'Cross-validated predictivity (median ± fold SE)',
                 'CV_alpha_mean': 'Cross-validated mean alpha',
                 'CV_alpha_median': 'Cross-validated median alpha',
                 'full_score': 'Held-out participant predictivity',
                 'cond': 'Condition',
                 'response_target': 'BOLD response (mean)',
                 'response': 'BOLD response',
                 'Actual (797-841-880-837-856-848-853-865-875-876)': 'BOLD response (mean)',
                 'std_over_items': 'item SD',
                 'sem_over_items': 'item SE',
                 'std_over_UIDs': 'participant SD',
                 'sem_over_UIDs': 'participant SE',
                 'std_within_UIDs': 'within-participant SD',
                 'sem_within_UIDs': 'within-participant SE',
                 'pred-full_from-848-853-865-875-876': 'Encoding model prediction',
                 # 'Predicted from 848-853-865-875-876'
                 'pred-CV-k-5_from-848-853-865-875-876': 'Predicted from 848-853-865-875-876 (CV)',
                 'pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean': 'Prediction from "GPT2-XL surprisal model"',
                 'pretransformer_pred-surprisal-5gram-surprisal-5gram_mean': 'Prediction from "5-gram surprisal model"',
                 'pretransformer_pred-surprisal-pcfg-surprisal-pcfg_mean': 'Prediction from "PCFG surprisal model"',
                 'pred-full_from-848-853-865-875-876_noise': 'Predicted from 848-853-865-875-876 NOISE',
                 'pretransformer_pred_noise': 'Pretransformer pred NOISE',
                 'nc': 'Noise ceiling (± split-half SE)',
                 'surprisal-gpt2-xl_mean': 'Surprisal',  # (GPT2-XL)
                 'surprisal-gpt2-xl_raw_mean': 'Log probability',
                 'surprisal-gpt2-xl_sum': 'Surprisal (sum)',
                 'surprisal-5gram_mean': 'Surprisal (5-gram)',
                 'surprisal-5gram_raw_mean': 'Log probability (5-gram)',
                 'surprisal-5gram_sum': 'Surprisal (5-gram, sum)',
                 'surprisal-pcfg_mean': 'Surprisal (PCFG)',
                 'surprisal-pcfg_raw_mean': 'Log probability (PCFG)',
                 'surprisal-pcfg_sum': 'Surprisal (PCFG, sum)',
                 'rating_arousal_mean': 'Arousal',
                 'rating_conversational_mean': 'Conversational frequency',
                 'rating_sense_mean': 'Plausibility',
                 'rating_gram_mean': 'Grammaticality',
                 'rating_frequency_mean': 'General frequency',
                 'rating_imageability_mean': 'Imageability',
                 'rating_memorability_mean': 'Memorability',
                 'rating_others_thoughts_mean': 'Mental states',
                 'rating_physical_mean': 'Physical objects',
                 'rating_places_mean': 'Places',
                 'rating_valence_mean': 'Valence',

                 'D_search': 'Search',
                 'D_synth': 'Modify',
                 'B_search': 'Search',
                 'B_synth': 'Modify',
                 'T': 'Baseline',

                 'T_1gram_overlap': '1-gram overlap',
                 'T_2gram_overlap': '2-gram overlap',
                 'T_3gram_overlap': '3-gram overlap',
                 'T_4gram_overlap': '4-gram overlap',
                 'T_5gram_overlap': '5-gram overlap',
                 'T_6gram_overlap': '6-gram overlap',
                 }

d_colors = {'B': 'royalblue',
            'D': 'tab:red',
            'T': 'grey',
            'D_search': 'tab:red',
            'B_search': 'royalblue',
            'D_synth': 'tab:red',
            'B_synth': 'royalblue',
            'lang': 'lightcoral',
            'md': 'royalblue',
            'dmn': 'mediumseagreen',
            'vis': 'orange',
            'aud': 'darkorchid'
            }

d_category_colors = {
    'pruned': 'purple',
    'random': 'goldenrod',
    'dense': 'darkseagreen',
}

d_colors_ngram = {0: 'gold',
                  1: 'goldenrod',
                  2: 'darkorange',
                  3: 'red',
                  4: 'mediumvioletred',
                  5: 'darkviolet',
                  6: 'indigo'}

# For cond_approach = T, circle. For cond_approach = D_search and B_search, square; for cond_approach = D_synth and B_synth, triangle
d_symbols = {'T': 'o',
             'D_search': 's',
             'D_synth': '^',
             'B_search': 's',
             'B_synth': '^', }

d_annotate = {True: '_annotated',
              False: ''}

d_include_T = {True: '_include-T', False: ''}

d_modeltypes = {'b': 'TYPEB_FITHRF',
                'c': 'TYPEC_FITHRF_GLMDENOISE',
                'd': 'TYPED_FITHRF_GLMDENOISE_RR'}

d_wlangloc = {True: '_wlangloc', False: ''}

d_source_layers = {'gpt2': np.arange(0, 13),
                   'gpt2-xl': np.arange(0, 49),
                   'bert-large-cased': np.arange(0, 25),

                   'pruned_seed0_embeddings': np.arange(0, 13),
                     'pruned_seed1_embeddings': np.arange(0, 13),
                        'pruned_seed2_embeddings': np.arange(0, 13),
                        'pruned_seed3_embeddings': np.arange(0, 13),
                        'random_seed0_embeddings': np.arange(0, 13),
                        'random_seed1_embeddings': np.arange(0, 13),
                        'random_seed2_embeddings': np.arange(0, 13),
                        'random_seed3_embeddings': np.arange(0, 13),
                        'dense_seed0_embeddings': np.arange(0, 13),
                        'dense_seed1_embeddings': np.arange(0, 13),
                        'dense_seed2_embeddings': np.arange(0, 13),
                        'dense_seed3_embeddings': np.arange(0, 13),


                   #  'GPT-100M-100Mtokens_last': np.arange(0, 13),
                   #  'GPT-100M-25percrandom-100Mtokens_last': np.arange(0, 13),
                   #  'GPT-100M-25perc-100Mtokens_last': np.arange(0, 13),
                   #  'GPT-100M-25perc-seed0-100Mtokens_last': np.arange(0, 13),
                   #  'GPT-100M-25perc-seed1-100Mtokens_last': np.arange(0, 13),
                   # 'resnet50': ['act1', 'layer1', 'layer2', 'layer3', 'layer4'],
                   }

######### PLOT related but for specific plots (e.g., axis limits etc) #########

## analyze_master_D_control.py: cond_barplot
d_ylim_cond_barplot = {'lang_LH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'md_LH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'md_RH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'dmn_LH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'dmn_RH_netw':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },

                       'lang_LH_IFG':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.52], },
                       'lang_LH_IFGorb':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_LH_MFG':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_LH_AntTemp':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_LH_PostTemp':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_IFG':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_IFGorb':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_MFG':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_AntTemp':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       'lang_RH_PostTemp':
                           {'None': [-0.2, 0.8],
                            'bySessVoxZ': [-0.5, 0.5], },
                       }


## analyze_master_D_control.py: item_scatter
def item_scatter_style(style_setting: str):
    """
	Available style settings for the item scatter: 'square', 'wide'

	"""

    if style_setting == 'square':
        plot_aspect_flag = 1
        add_identity_flag = True

        d_xlim = {'lang_LH_netw':
                      {'None':
                           {'797-841-880-837-856': None, },
                       'bySessVoxZ':
                           {'797-841-880-837-856': None,
                            '797-841-880': [-1.7, 1.7],
                            '837-856': [-2.4, 2.4],
                            '797': None,
                            '841': None,
                            '880': None,
                            '837': None,
                            '856': None,
                            }}}
        d_ylim = d_xlim

    elif style_setting == 'wide':
        plot_aspect_flag = 0.8
        add_identity_flag = True

        d_xlim = {'lang_LH_netw':
                      {'None':
                           {'797-841-880-837-856': None,
                            '797-841-880': [-0.75, 0.75],
                            '837-856': [-1, 1],  # synth has different preds
                            '797': [-0.75, 0.75],
                            '841': [-0.75, 0.75],
                            '880': [-0.75, 0.75],
                            '837': [-1, 1],
                            '856': [-1, 1], },
                       'bySessVoxZ':
                           {'797-841-880-837-856': None,
                            '797-841-880': [-0.75, 0.75],
                            '837-856': [-1, 1],  # synth has different preds
                            '797': [-0.75, 0.75],
                            '841': [-0.75, 0.75],
                            '880': [-0.75, 0.75],
                            '837': [-1, 1],
                            '856': [-1, 1],
                            }}}
        d_ylim = {'lang_LH_netw':
                      {'None':
                           {'797-841-880-837-856': None,
                            '797-841-880': [-2, 2],
                            '837-856': [-2.4, 2.4],
                            '797': None,
                            '841': None,
                            '880': None,
                            '837': None,
                            '856': None, },
                       'bySessVoxZ':
                           {'797-841-880-837-856': None,
                            '797-841-880': [-2, 2],
                            '837-856': [-2.4, 2.4],
                            '797': [-3, 3],
                            '841': [-3, 3],
                            '880': [-3, 3],
                            '837': [-3.5, 3.5],
                            '856': [-3.5, 3.5],
                            }}}
    else:
        raise ValueError(f'Invalid style_setting: {style_setting}')

    return plot_aspect_flag, add_identity_flag, d_xlim, d_ylim


######### ROI related #########

lst_all_normal_rois = ['aud_LH_TE11', 'aud_LH_TE12', 'aud_RH_TE11', 'aud_RH_TE12',
                       'cereb_LH_InfCereb', 'cereb_LH_MedCereb', 'cereb_LH_SupCereb',
                       'cereb_RH_InfCereb', 'cereb_RH_MedCereb', 'cereb_RH_SupCereb',
                       'dmn_LH_AntTemp', 'dmn_LH_FrontalMed', 'dmn_LH_MidCing',
                       'dmn_LH_PostCing', 'dmn_LH_STGorInsula', 'dmn_LH_TPJ', 'dmn_RH_AntTemp',
                       'dmn_RH_FrontalMed', 'dmn_RH_MidCing', 'dmn_RH_PostCing',
                       'dmn_RH_STGorInsula', 'dmn_RH_TPJ', 'lang_LH_AngG', 'lang_LH_AntTemp',
                       'lang_LH_IFG', 'lang_LH_IFGorb', 'lang_LH_MFG', 'lang_LH_PostTemp',
                       'lang_RH_AngG', 'lang_RH_AntTemp', 'lang_RH_IFG', 'lang_RH_IFGorb',
                       'lang_RH_MFG', 'lang_RH_PostTemp', 'md_LH_Precentral_A_pr',
                       'md_LH_Precentral_B_IF', 'md_LH_antParietal', 'md_LH_insula',
                       'md_LH_medialFrontal', 'md_LH_midFrontal', 'md_LH_midFrontalOrb',
                       'md_LH_midParietal', 'md_LH_postParietal', 'md_LH_supFrontal',
                       'md_RH_Precentral_A_pr', 'md_RH_Precentral_B_IF', 'md_RH_antParietal',
                       'md_RH_insula', 'md_RH_medialFrontal', 'md_RH_midFrontal',
                       'md_RH_midFrontalOrb', 'md_RH_midParietal', 'md_RH_postParietal',
                       'md_RH_supFrontal', 'vis_LH_Occipital_Inf', 'vis_LH_Occipital_Mid',
                       'vis_LH_Occipital_Sup', 'vis_RH_Occipital_Inf', 'vis_RH_Occipital_Mid',
                       'vis_RH_Occipital_Sup', 'lang_LHRH_netw', 'lang_LH_netw',
                       'lang_RH_netw', 'md_LHRH_netw', 'md_LH_netw', 'md_RH_netw',
                       'dmn_LHRH_netw', 'dmn_LH_netw', 'dmn_RH_netw', 'aud_LHRH_netw',
                       'aud_LH_netw', 'aud_RH_netw', 'vis_LHRH_netw', 'vis_LH_netw',
                       'vis_RH_netw',
                       'subcor_LHRH_netw',  # will be all the normal ROIs that start with subcor and LH or RH
                       'subcor_LH_netw', 'subcor_RH_netw', 'cereb_LHRH_netw', 'cereb_LH_netw',
                       'cereb_RH_netw'
                       ]

d_anatglasser_networks = {'lang':  # predefined ROIs from Lipkin et al. (2022)
                              {'anatglasser_LH_LangIFGorb': ['anatglasser_LH_47l', 'anatglasser_LH_45'],
                               'anatglasser_LH_LangIFG': ['anatglasser_LH_IFSp', 'anatglasser_LH_IFJa',
                                                          'anatglasser_LH_44'],
                               'anatglasser_LH_LangMFG': ['anatglasser_LH_FEF', 'anatglasser_LH_55b'],
                               'anatglasser_LH_LangAntTemp': ['anatglasser_LH_TA2', 'anatglasser_LH_STSva',
                                                              'anatglasser_LH_STSda',
                                                              'anatglasser_LH_STGa', 'anatglasser_LH_PI',
                                                              'anatglasser_LH_A5'],
                               'anatglasser_LH_LangPostTemp': ['anatglasser_LH_TPOJ2', 'anatglasser_LH_TPOJ1',
                                                               'anatglasser_LH_STV',
                                                               'anatglasser_LH_STSvp', 'anatglasser_LH_STSdp',
                                                               'anatglasser_LH_PSL',
                                                               'anatglasser_LH_PHT', 'anatglasser_LH_PGi'],
                               'anatglasser_LH_LangAngG': ['anatglasser_LH_TPOJ3', 'anatglasser_LH_PGs'],
                               'anatglasser_RH_LangIFGorb': ['anatglasser_RH_47l', 'anatglasser_RH_45'],
                               'anatglasser_RH_LangIFG': ['anatglasser_RH_IFSp', 'anatglasser_RH_IFJa',
                                                          'anatglasser_RH_44'],
                               'anatglasser_RH_LangMFG': ['anatglasser_RH_FEF', 'anatglasser_RH_55b'],
                               'anatglasser_RH_LangAntTemp': ['anatglasser_RH_TA2', 'anatglasser_RH_STSva',
                                                              'anatglasser_RH_STSda',
                                                              'anatglasser_RH_STGa', 'anatglasser_RH_PI',
                                                              'anatglasser_RH_A5'],
                               'anatglasser_RH_LangPostTemp': ['anatglasser_RH_TPOJ2', 'anatglasser_RH_TPOJ1',
                                                               'anatglasser_RH_STV',
                                                               'anatglasser_RH_STSvp', 'anatglasser_RH_STSdp',
                                                               'anatglasser_RH_PSL',
                                                               'anatglasser_RH_PHT', 'anatglasser_RH_PGi'],
                               'anatglasser_RH_LangAngG': ['anatglasser_RH_TPOJ3', 'anatglasser_RH_PGs'], },

                          'aud':
                              {'anatglasser_LH_AudPrimary': ['anatglasser_LH_A1', 'anatglasser_LH_LBelt', ],
                               'anatglasser_RH_AudPrimary': ['anatglasser_RH_A1', 'anatglasser_RH_LBelt', ],
                               'anatglasser_LHRH_AudPrimary': ['anatglasser_LH_A1', 'anatglasser_LH_LBelt',
                                                               'anatglasser_RH_A1', 'anatglasser_RH_LBelt', ], },
                          'vis':
                              {'anatglasser_LHRH_V1': ['anatglasser_LH_V1', 'anatglasser_RH_V1', ], }
                          # The ones per hemisphere are already separately defined in the anatglasser atlas
                          }

# Add in 'anatglasser_LH_LangNetw which consists of anatglasser_LH_LangIFGorb, anatglasser_LH_LangIFG, anatglasser_LH_LangMFG, anatglasser_LH_LangAntTemp, anatglasser_LH_LangPostTemp (just like the functinal ROIs)
d_anatglasser_networks['lang']['anatglasser_LH_LangNetw'] = d_anatglasser_networks['lang'][
                                                                'anatglasser_LH_LangIFGorb'] + \
                                                            d_anatglasser_networks['lang']['anatglasser_LH_LangIFG'] + \
                                                            d_anatglasser_networks['lang']['anatglasser_LH_LangMFG'] + \
                                                            d_anatglasser_networks['lang'][
                                                                'anatglasser_LH_LangAntTemp'] + \
                                                            d_anatglasser_networks['lang'][
                                                                'anatglasser_LH_LangPostTemp']
d_anatglasser_networks['lang']['anatglasser_RH_LangNetw'] = d_anatglasser_networks['lang'][
                                                                'anatglasser_RH_LangIFGorb'] + \
                                                            d_anatglasser_networks['lang']['anatglasser_RH_LangIFG'] + \
                                                            d_anatglasser_networks['lang']['anatglasser_RH_LangMFG'] + \
                                                            d_anatglasser_networks['lang'][
                                                                'anatglasser_RH_LangAntTemp'] + \
                                                            d_anatglasser_networks['lang'][
                                                                'anatglasser_RH_LangPostTemp']
d_anatglasser_networks['lang']['anatglasser_LHRH_LangNetw'] = d_anatglasser_networks['lang'][
                                                                  'anatglasser_LH_LangNetw'] + \
                                                              d_anatglasser_networks['lang']['anatglasser_RH_LangNetw']

# Get names of all the network ROIs in d_anatglasser_networks
anatglasser_networks_netw_names = [netw_roi for netw in d_anatglasser_networks.values() for netw_roi in netw.keys()]

lst_all_glasser_rois = ['anatglasser_LH_1', 'anatglasser_LH_10d', 'anatglasser_LH_10pp', 'anatglasser_LH_10r',
                        'anatglasser_LH_10v', 'anatglasser_LH_11l',
                        'anatglasser_LH_13l', 'anatglasser_LH_2', 'anatglasser_LH_23c', 'anatglasser_LH_23d',
                        'anatglasser_LH_24dd', 'anatglasser_LH_24dv',
                        'anatglasser_LH_25', 'anatglasser_LH_31a', 'anatglasser_LH_31pd', 'anatglasser_LH_31pv',
                        'anatglasser_LH_33pr', 'anatglasser_LH_3a',
                        'anatglasser_LH_3b', 'anatglasser_LH_4', 'anatglasser_LH_43', 'anatglasser_LH_44',
                        'anatglasser_LH_45', 'anatglasser_LH_46', 'anatglasser_LH_47l',
                        'anatglasser_LH_47m', 'anatglasser_LH_47s', 'anatglasser_LH_52', 'anatglasser_LH_55b',
                        'anatglasser_LH_5L', 'anatglasser_LH_5m', 'anatglasser_LH_5mv',
                        'anatglasser_LH_6a', 'anatglasser_LH_6d', 'anatglasser_LH_6ma', 'anatglasser_LH_6mp',
                        'anatglasser_LH_6r', 'anatglasser_LH_6v', 'anatglasser_LH_7PC',
                        'anatglasser_LH_7PL', 'anatglasser_LH_7Pm', 'anatglasser_LH_7m', 'anatglasser_LH_8Ad',
                        'anatglasser_LH_8Av', 'anatglasser_LH_8BL', 'anatglasser_LH_8BM',
                        'anatglasser_LH_8C', 'anatglasser_LH_9-46d', 'anatglasser_LH_9a', 'anatglasser_LH_9m',
                        'anatglasser_LH_9p', 'anatglasser_LH_A1', 'anatglasser_LH_A4',
                        'anatglasser_LH_A5', 'anatglasser_LH_AAIC', 'anatglasser_LH_AIP', 'anatglasser_LH_AVI',
                        'anatglasser_LH_AudPrimary', 'anatglasser_LH_DVT',
                        'anatglasser_LH_EC', 'anatglasser_LH_FEF', 'anatglasser_LH_FFC', 'anatglasser_LH_FOP1',
                        'anatglasser_LH_FOP2', 'anatglasser_LH_FOP3',
                        'anatglasser_LH_FOP4', 'anatglasser_LH_FOP5', 'anatglasser_LH_FST', 'anatglasser_LH_H',
                        'anatglasser_LH_IFJa', 'anatglasser_LH_IFJp',
                        'anatglasser_LH_IFSa', 'anatglasser_LH_IFSp', 'anatglasser_LH_IP0', 'anatglasser_LH_IP1',
                        'anatglasser_LH_IP2', 'anatglasser_LH_IPS1',
                        'anatglasser_LH_Ig', 'anatglasser_LH_LBelt', 'anatglasser_LH_LIPd', 'anatglasser_LH_LIPv',
                        'anatglasser_LH_LO1', 'anatglasser_LH_LO2',
                        'anatglasser_LH_LO3', 'anatglasser_LH_LangAnG', 'anatglasser_LH_LangAntTemp',
                        'anatglasser_LH_LangIFG', 'anatglasser_LH_LangIFGorb',
                        'anatglasser_LH_LangMFG', 'anatglasser_LH_LangPostTemp', 'anatglasser_LH_MBelt',
                        'anatglasser_LH_MI', 'anatglasser_LH_MIP', 'anatglasser_LH_MST',
                        'anatglasser_LH_MT', 'anatglasser_LH_OFC', 'anatglasser_LH_OP1', 'anatglasser_LH_OP2-3',
                        'anatglasser_LH_OP4', 'anatglasser_LH_PBelt',
                        'anatglasser_LH_PCV', 'anatglasser_LH_PEF', 'anatglasser_LH_PF', 'anatglasser_LH_PFcm',
                        'anatglasser_LH_PFm', 'anatglasser_LH_PFop',
                        'anatglasser_LH_PFt', 'anatglasser_LH_PGi', 'anatglasser_LH_PGp', 'anatglasser_LH_PGs',
                        'anatglasser_LH_PH', 'anatglasser_LH_PHA1',
                        'anatglasser_LH_PHA2', 'anatglasser_LH_PHA3', 'anatglasser_LH_PHT', 'anatglasser_LH_PI',
                        'anatglasser_LH_PIT', 'anatglasser_LH_POS1',
                        'anatglasser_LH_POS2', 'anatglasser_LH_PSL', 'anatglasser_LH_PeEc', 'anatglasser_LH_Pir',
                        'anatglasser_LH_PoI1', 'anatglasser_LH_PoI2',
                        'anatglasser_LH_PreS', 'anatglasser_LH_ProS', 'anatglasser_LH_RI', 'anatglasser_LH_RSC',
                        'anatglasser_LH_SCEF', 'anatglasser_LH_SFL',
                        'anatglasser_LH_STGa', 'anatglasser_LH_STSda', 'anatglasser_LH_STSdp', 'anatglasser_LH_STSva',
                        'anatglasser_LH_STSvp', 'anatglasser_LH_STV',
                        'anatglasser_LH_TA2', 'anatglasser_LH_TE1a', 'anatglasser_LH_TE1m', 'anatglasser_LH_TE1p',
                        'anatglasser_LH_TE2a', 'anatglasser_LH_TE2p',
                        'anatglasser_LH_TF', 'anatglasser_LH_TGd', 'anatglasser_LH_TGv', 'anatglasser_LH_TPOJ1',
                        'anatglasser_LH_TPOJ2', 'anatglasser_LH_TPOJ3',
                        'anatglasser_LH_V1', 'anatglasser_LH_V2', 'anatglasser_LH_V3', 'anatglasser_LH_V3A',
                        'anatglasser_LH_V3B', 'anatglasser_LH_V3CD',
                        'anatglasser_LH_V4', 'anatglasser_LH_V4t', 'anatglasser_LH_V6', 'anatglasser_LH_V6A',
                        'anatglasser_LH_V7', 'anatglasser_LH_V8', 'anatglasser_LH_VMV1',
                        'anatglasser_LH_VMV2', 'anatglasser_LH_VMV3', 'anatglasser_LH_VVC', 'anatglasser_LH_a10p',
                        'anatglasser_LH_a24', 'anatglasser_LH_a24pr',
                        'anatglasser_LH_a32pr', 'anatglasser_LH_a47r', 'anatglasser_LH_a9-46v', 'anatglasser_LH_d23ab',
                        'anatglasser_LH_d32', 'anatglasser_LH_i6-8',
                        'anatglasser_LH_p10p', 'anatglasser_LH_p24', 'anatglasser_LH_p24pr', 'anatglasser_LH_p32',
                        'anatglasser_LH_p32pr', 'anatglasser_LH_p47r',
                        'anatglasser_LH_p9-46v', 'anatglasser_LH_pOFC', 'anatglasser_LH_s32', 'anatglasser_LH_s6-8',
                        'anatglasser_LH_v23ab', 'anatglasser_RH_1',
                        'anatglasser_RH_10d', 'anatglasser_RH_10pp', 'anatglasser_RH_10r', 'anatglasser_RH_10v',
                        'anatglasser_RH_11l', 'anatglasser_RH_13l', 'anatglasser_RH_2',
                        'anatglasser_RH_23c', 'anatglasser_RH_23d', 'anatglasser_RH_24dd', 'anatglasser_RH_24dv',
                        'anatglasser_RH_25', 'anatglasser_RH_31a',
                        'anatglasser_RH_31pd', 'anatglasser_RH_31pv', 'anatglasser_RH_33pr', 'anatglasser_RH_3a',
                        'anatglasser_RH_3b', 'anatglasser_RH_4', 'anatglasser_RH_43',
                        'anatglasser_RH_44', 'anatglasser_RH_45', 'anatglasser_RH_46', 'anatglasser_RH_47l',
                        'anatglasser_RH_47m', 'anatglasser_RH_47s',
                        'anatglasser_RH_52', 'anatglasser_RH_55b', 'anatglasser_RH_5m', 'anatglasser_RH_5mv',
                        'anatglasser_RH_6a', 'anatglasser_RH_6d',
                        'anatglasser_RH_6ma', 'anatglasser_RH_6mp', 'anatglasser_RH_6r', 'anatglasser_RH_6v',
                        'anatglasser_RH_7PC', 'anatglasser_RH_7PL',
                        'anatglasser_RH_7Pm', 'anatglasser_RH_7m', 'anatglasser_RH_8Ad', 'anatglasser_RH_8Av',
                        'anatglasser_RH_8BL', 'anatglasser_RH_8BM',
                        'anatglasser_RH_8C', 'anatglasser_RH_9-46d', 'anatglasser_RH_9a', 'anatglasser_RH_9m',
                        'anatglasser_RH_9p', 'anatglasser_RH_A1',
                        'anatglasser_RH_A4', 'anatglasser_RH_A5', 'anatglasser_RH_AAIC', 'anatglasser_RH_AIP',
                        'anatglasser_RH_AVI', 'anatglasser_RH_AudPrimary',
                        'anatglasser_RH_DVT', 'anatglasser_RH_EC', 'anatglasser_RH_FEF', 'anatglasser_RH_FFC',
                        'anatglasser_RH_FOP1', 'anatglasser_RH_FOP2',
                        'anatglasser_RH_FOP3', 'anatglasser_RH_FOP4', 'anatglasser_RH_FOP5', 'anatglasser_RH_FST',
                        'anatglasser_RH_H', 'anatglasser_RH_IFJa',
                        'anatglasser_RH_IFJp', 'anatglasser_RH_IFSa', 'anatglasser_RH_IFSp', 'anatglasser_RH_IP0',
                        'anatglasser_RH_IP1', 'anatglasser_RH_IP2',
                        'anatglasser_RH_IPS1', 'anatglasser_RH_Ig', 'anatglasser_RH_LBelt', 'anatglasser_RH_LIPd',
                        'anatglasser_RH_LIPv', 'anatglasser_RH_LO1',
                        'anatglasser_RH_LO2', 'anatglasser_RH_LO3', 'anatglasser_RH_LangAnG',
                        'anatglasser_RH_LangAntTemp', 'anatglasser_RH_LangIFG',
                        'anatglasser_RH_LangIFGorb', 'anatglasser_RH_LangMFG', 'anatglasser_RH_LangPostTemp',
                        'anatglasser_RH_MBelt', 'anatglasser_RH_MI',
                        'anatglasser_RH_MIP', 'anatglasser_RH_MST', 'anatglasser_RH_MT', 'anatglasser_RH_OFC',
                        'anatglasser_RH_OP1', 'anatglasser_RH_OP2-3',
                        'anatglasser_RH_OP4', 'anatglasser_RH_PBelt', 'anatglasser_RH_PCV', 'anatglasser_RH_PEF',
                        'anatglasser_RH_PF', 'anatglasser_RH_PFcm',
                        'anatglasser_RH_PFm', 'anatglasser_RH_PFop', 'anatglasser_RH_PFt', 'anatglasser_RH_PGi',
                        'anatglasser_RH_PGp', 'anatglasser_RH_PGs',
                        'anatglasser_RH_PH', 'anatglasser_RH_PHA1', 'anatglasser_RH_PHA2', 'anatglasser_RH_PHA3',
                        'anatglasser_RH_PHT', 'anatglasser_RH_PI',
                        'anatglasser_RH_PIT', 'anatglasser_RH_POS1', 'anatglasser_RH_POS2', 'anatglasser_RH_PSL',
                        'anatglasser_RH_PeEc', 'anatglasser_RH_Pir',
                        'anatglasser_RH_PoI1', 'anatglasser_RH_PoI2', 'anatglasser_RH_PreS', 'anatglasser_RH_ProS',
                        'anatglasser_RH_RI', 'anatglasser_RH_RSC',
                        'anatglasser_RH_SCEF', 'anatglasser_RH_SFL', 'anatglasser_RH_STGa', 'anatglasser_RH_STSda',
                        'anatglasser_RH_STSdp', 'anatglasser_RH_STSva',
                        'anatglasser_RH_STSvp', 'anatglasser_RH_STV', 'anatglasser_RH_TA2', 'anatglasser_RH_TE1a',
                        'anatglasser_RH_TE1m', 'anatglasser_RH_TE1p',
                        'anatglasser_RH_TE2a', 'anatglasser_RH_TE2p', 'anatglasser_RH_TF', 'anatglasser_RH_TGd',
                        'anatglasser_RH_TGv', 'anatglasser_RH_TPOJ1',
                        'anatglasser_RH_TPOJ2', 'anatglasser_RH_TPOJ3', 'anatglasser_RH_V1', 'anatglasser_RH_V2',
                        'anatglasser_RH_V3', 'anatglasser_RH_V3A',
                        'anatglasser_RH_V3B', 'anatglasser_RH_V3CD', 'anatglasser_RH_V4', 'anatglasser_RH_V4t',
                        'anatglasser_RH_V6', 'anatglasser_RH_V6A',
                        'anatglasser_RH_V7', 'anatglasser_RH_V8', 'anatglasser_RH_VMV1', 'anatglasser_RH_VMV2',
                        'anatglasser_RH_VMV3', 'anatglasser_RH_VVC',
                        'anatglasser_RH_a10p', 'anatglasser_RH_a24', 'anatglasser_RH_a24pr', 'anatglasser_RH_a32pr',
                        'anatglasser_RH_a47r', 'anatglasser_RH_a9-46v',
                        'anatglasser_RH_d23ab', 'anatglasser_RH_d32', 'anatglasser_RH_i6-8', 'anatglasser_RH_p10p',
                        'anatglasser_RH_p24', 'anatglasser_RH_p24pr',
                        'anatglasser_RH_p32', 'anatglasser_RH_p32pr', 'anatglasser_RH_p47r', 'anatglasser_RH_p9-46v',
                        'anatglasser_RH_pOFC', 'anatglasser_RH_s32',
                        'anatglasser_RH_s6-8', 'anatglasser_RH_v23ab']
lst_all_rois = lst_all_normal_rois + lst_all_glasser_rois + anatglasser_networks_netw_names

rois_func_lang_md_dmn = [
    'lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp',
    'lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG', 'lang_RH_AntTemp', 'lang_RH_PostTemp',
    'md_LH_netw', 'md_LH_postParietal', 'md_LH_midParietal', 'md_LH_antParietal', 'md_LH_supFrontal',
    'md_LH_PrecentralAprecG', 'md_LH_PrecentralBIFGop', 'md_LH_midFrontal', 'md_LH_midFrontalOrb', 'md_LH_insula',
    'md_LH_medialFrontal',
    'md_RH_netw', 'md_RH_postParietal', 'md_RH_midParietal', 'md_RH_antParietal', 'md_RH_supFrontal',
    'md_RH_PrecentralAprecG', 'md_RH_PrecentralBIFGop', 'md_RH_midFrontal', 'md_RH_midFrontalOrb', 'md_RH_insula',
    'md_RH_medialFrontal',
    'dmn_LH_netw', 'dmn_LH_FrontalMed', 'dmn_LH_PostCing', 'dmn_LH_TPJ', 'dmn_LH_MidCing', 'dmn_LH_STGorInsula',
    'dmn_LH_AntTemp',
    'dmn_RH_netw', 'dmn_RH_FrontalMed', 'dmn_RH_PostCing', 'dmn_RH_TPJ', 'dmn_RH_MidCing', 'dmn_RH_STGorInsula',
    'dmn_RH_AntTemp', ]

d_roi_lists_names = {'lang_LH_netw': ['lang_LH_netw'],
                     'lang_RH_netw': ['lang_RH_netw'],
                     'lang_LH_IFGorb': ['lang_LH_IFGorb'],
                     'lang_RH_IFGorb': ['lang_RH_IFGorb'],
                     'lang_LH_IFG': ['lang_LH_IFG'],
                     'lang_RH_IFG': ['lang_RH_IFG'],
                     'lang_LH_MFG': ['lang_LH_MFG'],
                     'lang_RH_MFG': ['lang_RH_MFG'],
                     'lang_LH_AntTemp': ['lang_LH_AntTemp'],
                     'lang_RH_AntTemp': ['lang_RH_AntTemp'],
                     'lang_LH_PostTemp': ['lang_LH_PostTemp'],
                     'lang_RH_PostTemp': ['lang_RH_PostTemp'],

                     'md_LH_netw': ['md_LH_netw'],
                     'md_RH_netw': ['md_RH_netw'],
                     'dmn_LH_netw': ['dmn_LH_netw'],
                     'dmn_RH_netw': ['dmn_RH_netw'],
                     'lang_LH_ROIs': ['lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp',
                                      'lang_LH_PostTemp', ],
                     'lang_RH_ROIs': ['lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG', 'lang_RH_AntTemp',
                                      'lang_RH_PostTemp', ],
                     'lang_LHRH_ROIs': ['lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG',
                                        'lang_LH_AntTemp', 'lang_LH_PostTemp',
                                        'lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG',
                                        'lang_RH_AntTemp', 'lang_RH_PostTemp', ],
                     'audvis_anatglasser_ROIs': ['anatglasser_LH_AudPrimary', 'anatglasser_RH_AudPrimary',
                                                 'anatglasser_LH_V1', 'anatglasser_RH_V1', ],
                     'lang_anatglasser_ROIs': list(d_anatglasser_networks['lang'].keys()),
                     'anatglasser_LH_LangNetw': ['anatglasser_LH_LangNetw'],
                     'lang_LH_anatglasser_ROIs': [x for x in list(d_anatglasser_networks['lang'].keys()) if
                                                  x.startswith('anatglasser_LH_') and not x.endswith('AngG')],
                     'lang_RH_anatglasser_ROIs': [x for x in list(d_anatglasser_networks['lang'].keys()) if
                                                  x.startswith('anatglasser_RH_') and not x.endswith('AngG')],

                     'lang_LH_normal_anatglasser_ROIs': ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG',
                                                         'lang_LH_AntTemp', 'lang_LH_PostTemp', 'lang_LH_netw',
                                                         'anatglasser_LH_LangIFGorb', 'anatglasser_LH_LangIFG',
                                                         'anatglasser_LH_LangMFG', 'anatglasser_LH_LangAntTemp',
                                                         'anatglasser_LH_LangPostTemp', 'anatglasser_LH_LangNetw'],
                     'all_ROIs': lst_all_rois,
                     'rois_func_lang_md_dmn': rois_func_lang_md_dmn,
                     }


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
        rois_of_interest = None  # I.e. just use the string as the name, and pass a None such that we use all ROIs

    return rois_of_interest, rois_of_interest_name
