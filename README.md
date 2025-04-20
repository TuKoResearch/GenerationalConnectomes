# Model connectomes: A generational approach to data-efficient language models  
_Second Workshop on Representational Alignment at ICLR 2025_  

**Authors:** Klemen Kotar & Greta Tuckute

---

## Setup

### 1. Create Conda Environment

Make sure you have [Conda](https://docs.conda.io/) installed. Then:

```bash
# 1. Create a new env with Python 3.11
conda create -n genprune python=3.11 -y
conda activate genprune

# 2. Install PyTorch 2.6 (with CUDA support if needed)
# Replace cudatoolkit version with your local CUDA, e.g. 11.7
conda install -c pytorch pytorch==2.6.0 torchvision torchaudio cudatoolkit=11.7 -y

# 3. Install other dependencies (from setup.py)
pip install -e .
```  

> This will install `torch`, `scipy`, `tqdm`, `wandb`, `pandas`, `matplotlib`, `scikit-learn`, `transformers`, `tiktoken`, and `huggingface_hub`.

---

## LLM Training

Once your environment is ready, train the Generational Pruning GPT model from a pruned checkpoitn with:

```bash
# Single-GPU debug run
python train.py \
  --run_name my_experiment \
  --train_data_dir path/to/train/*.bin \
  --val_data_dir path/to/val/*.bin \
  --wandb            # (optional: log to Weights & Biases)

# Multi-GPU DDP run
torchrun --standalone --nproc_per_node=8 train.py \
  --run_name my_experiment \
  --train_data_dir path/to/train/*.bin \
  --val_data_dir path/to/val/*.bin \
  --per_device_batch_size 16 \
  --batch_size 512 \
  --wandb
```

**Key flags**:
- `--run_name`: name for output folder under `./out/` and (optionally) W&B run.  
- `--train_data_dir` / `--val_data_dir`: glob pattern for `.bin` tokenized data.  
- `--per_device_batch_size`: batch size per GPU.  
- `--batch_size`: total batch size (will be split across GPUs).  
- `--wandb`: enable logging to Weights & Biases.  
- `--push_to_hf`: after training, upload final model to Hugging Face Hub under repo name `--run_name`.

All other flags (learning rate, scheduler, pruning init, etc.) can be viewed with:

```bash
python train.py --help
```

In order to run the prunning training you can run:

# Single-GPU debug run
python train_itp.py \
  --run_name my_experiment \
  --train_data_dir path/to/train/*.bin \
  --val_data_dir path/to/val/*.bin \
  --wandb            # (optional: log to Weights & Biases)


This will save a checkpoint to `out/<my_experiment>` which you can use as your connectome for the inner loop trianing above.

---

## LLM Evaluations

### MMLU Benchmark

We provide an evaluation script for mmlu and hellaswag inside of `evals/`.
You can reproduce our evaluations by running the following evaluations using the model checkpoints from huggingface:

1. **Run mmlu**:
   ```bash
   python evals/mmlu.py \
     --model_name TuKoResearch/GenerationalPrunning100M \
     --tokenizer_name gpt2 \
     --device cuda:0
   ```

1=2. **Run mmlu**:
   ```bash
   python evals/hellaswag.py \
     --model_name TuKoResearch/GenerationalPrunning100M \
     --tokenizer_name gpt2 \
     --device cuda:0
   ```


---

## Behavioral alignment
We use the Futrell2018 reading time benchmark, which can be obtained from [brain-score language](https://github.com/brain-score/language) and can be loaded using an environment with `xarray` installed. 
The correlation with LLM surprisal can be run using: XX

---

## Neural alignment
We use the Tuckute2024 neural benchmark, which can be downloaded from the following [public repository](https://github.com/gretatuckute/drive_suppress_brains) or [brain-score language](https://github.com/brain-score/language). The cross-validation neural predictivity score can be run from [NeuralAlignment/fit_mapping.py](https://github.com/klemenkotar/ConnectomePruning/blob/main/NeuralAlignment/fit_mapping.py) and looped across layers and models using [NeuralAlignment/loop_fit_mapping.py](https://github.com/klemenkotar/ConnectomePruning/blob/main/NeuralAlignment/loop_fit_mapping.py).

In some of the analyses, we first localize the LLM language units, per the approach established in AlKhamissi et al., 2025 (_ACL_), from the [following repository](https://github.com/BKHMSI/llm-localization). We adapted this code (POINTER??) to output a binary mask which marks the LLM language units as 1. The [NeuralAlignment/apply_langloc_mask.py](https://github.com/klemenkotar/ConnectomePruning/blob/main/NeuralAlignment/apply_langloc_mask.py) script takes the the numpy binary mask for a given model, and saves the masked embedding values as a csv file, which can then serve as the input to [NeuralAlignment/fit_mapping.py](https://github.com/klemenkotar/ConnectomePruning/blob/main/NeuralAlignment/fit_mapping.py).

The regression outputs can be downloaded [here](https://huggingface.co/datasets/TuKoResearch/GenerationalPruningEmbeddings/resolve/main/SHARE.zip?download=true).


---

## Citation

If you use this code, please cite:

> Kotar, K., & Tuckute, G. (2025). Model connectomes: A generational approach to data-efficient language models. *Second Workshop on Representational Alignment at ICLR 2025*.


