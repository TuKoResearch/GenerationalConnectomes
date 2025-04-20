import pandas as pd
import numpy as np
import os


def process_activations(fname_actv, fname_mask, sent_embed='last', output_dir=None):
    """
    Processes activations and applies a mask to filter units.

    Parameters:
    - fname_actv: str, path to the CSV file containing activations.
    - fname_mask: str, path to the NumPy file containing the mask.
    - sent_embed: str, keyword to filter activation layers (default: 'last').
    - output_dir: str, directory to save processed activations (default: same directory as input file).

    Returns:
    - selected_units: np.ndarray, filtered activations of shape [stim, selected units].
    """

    # Load activation data and mask
    df_csv_encoded = pd.read_csv(fname_actv)
    mask = np.load(fname_mask)

    # Find layer columns that match sent_embed
    layer_cols = [col for col in df_csv_encoded.columns if 'layer' in col and sent_embed in col]

    # Prepare activation DataFrame
    actv = pd.DataFrame()
    stimset_index = np.arange(1, df_csv_encoded.shape[0] + 1)

    for layer_col in layer_cols:
        # Convert string representations of arrays into actual NumPy arrays
        layer_matrix = df_csv_encoded[layer_col].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=",") if isinstance(x, str) else np.array([])
        ).to_numpy()

        # Stack into a proper 2D matrix
        actv_layer = np.vstack(layer_matrix)
        assert actv_layer.shape[0] == df_csv_encoded.shape[0]

        # Extract layer index
        layer_idx = int(layer_col.split('_')[0].replace('layer', ''))

        # Add as a multi-indexed DataFrame
        actv_layer_df = pd.DataFrame(actv_layer, index=stimset_index)
        actv_layer_df.columns = pd.MultiIndex.from_product([[layer_idx], actv_layer_df.columns])

        actv = pd.concat([actv, actv_layer_df], axis=1)

    # Drop layer 0 (embedding) if present
    if 0 in actv.columns.levels[0]:
        actv_no0 = actv.drop(0, axis=1, level=0)
    else:
        raise ValueError("No layer 0 found in the activations.")

    # Convert activations to NumPy array
    actv_flattened = actv_no0.to_numpy()  # Shape: [stim, layer * unit]

    # Flatten the mask
    mask_flattened = mask.flatten()  # Shape: [layer * unit]

    # Apply mask: Keep only units where mask == 1
    selected_units = actv_flattened[:, mask_flattened == 1]  # Shape: [stim, selected units]

    # Make into a pandas dataframe with multiindex (mock as 0)
    selected_units = pd.DataFrame(selected_units, index=stimset_index)
    selected_units.columns = pd.MultiIndex.from_product([[0], selected_units.columns])

    # Save selected activations as CSV
    # determine the name, use the actv base name with langloc_perc={perc}
    actv_base = os.path.basename(fname_actv).split('.')[0]
    # get the perc from the mask name
    perc = fname_mask.split('perc')[0].split('_')[-1]
    csv_output_path = os.path.join(output_dir, f'{actv_base}langlocperc={perc}.csv')
    print(f"Saving selected activations to {csv_output_path}")
    selected_units.to_csv(csv_output_path)

    return selected_units

# Key is the path to the activations, value is the path to the mask
d = {
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/pruned_seed0_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/pruned_seed0_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/pruned_seed1_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/pruned_seed1_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/pruned_seed2_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/pruned_seed2_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/pruned_seed3_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/pruned_seed3_1perc_localization.npy',

    '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/random_seed0_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/random_seed0_1perc_localization.npy',
    '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/random_seed1_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/random_seed1_1perc_localization.npy',
    '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/random_seed2_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/random_seed2_1perc_localization.npy',
    '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/random_seed3_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/random_seed3_1perc_localization.npy',

    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/dense_seed0_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/dense_seed0_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/dense_seed1_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/dense_seed1_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/dense_seed2_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/dense_seed2_1perc_localization.npy',
    # '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/dense_seed3_embeddings.csv': '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/langloc_masks/dense_seed3_1perc_localization.npy',

     }

outputdir = '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/post_langloc_mask'
os.makedirs(outputdir, exist_ok=True)

for fname_actv, fname_mask in d.items():
    process_activations(fname_actv, fname_mask, output_dir=outputdir)
