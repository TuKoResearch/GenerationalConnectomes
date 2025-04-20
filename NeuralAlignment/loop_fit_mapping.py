import os
import sys
import datetime
import fit_mapping

# Setup
log = True  # Set to False to run without logging
logs_dir = './logs/'
mapping_class = 'ridgeCV'
langloc = True
csv_files = [
    # 'pruned_seed0_embeddings',
    # 'pruned_seed1_embeddings',
    # 'pruned_seed2_embeddings',
    # 'pruned_seed3_embeddings',
    'random_seed0_embeddings',
    'random_seed1_embeddings',
    'random_seed2_embeddings',
    'random_seed3_embeddings',
    # 'dense_seed0_embeddings',
    # 'dense_seed1_embeddings',
    # 'dense_seed2_embeddings',
    # 'dense_seed3_embeddings',
]

if langloc:
    # suffix langlocperc=1
    csv_files = [f'{csv_file}langlocperc=1' for csv_file in csv_files]
    source_CSVDIR = '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/post_langloc_mask/'
    source_layers = [0]
else:
    source_CSVDIR = '/Users/gt/Documents/GitHub/ConnectomePruning/Embeddings/'
    source_layers = range(13)  # Layers 0 to 12

# Ensure the logs directory exists
if log and not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Generate analysis identifier
def generate_analysis_identifier(layer, csv_file):
    return f"layer_{layer}_{os.path.basename(csv_file)}"

# Main loop
for csv_file in csv_files:
    for layer in source_layers:
        print(f'\n\n')
        # Generate a unique identifier for this combination
        analysis_identifier = generate_analysis_identifier(layer, csv_file)
        datetime_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Set up logging
        log_file = None
        if log:
            log_file = os.path.join(logs_dir, f"{analysis_identifier}_{datetime_stamp}.txt")
            print(f"Logging to {log_file}")
            log_file_handle = open(log_file, 'w')
            original_stdout = sys.stdout  # Save the original stdout
            sys.stdout = log_file_handle  # Redirect stdout to the file

        # Run the analysis
        try:
            print(f"Running analysis for layer {layer} with CSV {csv_file}")
            # Construct the arguments for the main script
            args = [
                '--source_layer', str(layer),
                '--source_CSV_fname', csv_file,
                '--mapping_class', mapping_class,
                '--source_CSVDIR', source_CSVDIR,
                '--langloc', str(langloc),
            ]

            # Replace with the correct import and function call for your script
            fit_mapping.main(args)

            print(f"Completed analysis for layer {layer} with CSV {csv_file}")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            # Reset stdout if logging was enabled
            if log:
                sys.stdout = original_stdout
                log_file_handle.close()
