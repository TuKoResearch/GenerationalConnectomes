# Model connectomes: A generational approach to data-efficient language models  
_Second Workshop on Representational Alignment at ICLR 2025_  

**By:** Klemen Kotar & Greta Tuckute

---

![Paper Figure](figs/generational_connectome_fig.png)

---

## Released models

We have released the following pretrained Generational Connectome GPT models on the Hugging Face Hub:

| Model | Description |
|-------|-------------|
| [TuKoResearch/ConnectomeGPT100M](https://huggingface.co/TuKoResearch/ConnectomeGPT100M/) | Generational Pruning GPT with learned connectome |
| [TuKoResearch/RandomConnectomeGPT100M](https://huggingface.co/TuKoResearch/RandomConnectomeGPT100M/) | Generational Pruning GPT with random connectome |
| [TuKoResearch/NoConnectomeGPT100M](https://huggingface.co/TuKoResearch/NoConnectomeGPT100M/) | Generational Pruning GPT without connectome |

You can evaluate any of these models on downstream NLP benchmarks by specifying the `--model_name` flag in the evaluation scripts.

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/TuKoResearch/GenerationalConnectomes.git
   cd GenerationalConnectomes
   ```

2. **Create & activate a Conda environment**  
   ```bash
   conda create -n GenerationalConnectomes python=3.11 -y
   conda activate GenerationalConnectomes
   ```

3. **Install PyTorch 2.6** (with the appropriate CUDA toolkit for your setup)
   
   Note that if you just want to evaluate the models (not train them), you can load them on CPU. In that case, omit the cuda command below.
   ```bash
   conda install -c pytorch pytorch==2.6.0 torchvision torchaudio cudatoolkit=11.7 -y
   ```

5. **Install the remaining dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Loading the model
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

# load tokenizer (note: you must use the GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# load model (note: you must trust remote code to load the model)
model = AutoModelForCausalLM.from_pretrained("TuKoResearch/ConnectomeGPT100M", trust_remote_code=True)
```

---

## NLP evaluations

We provide an evaluation script for mmlu and hellaswag inside of `evals/`.
You can reproduce our evaluations by running the following evaluations using the model checkpoints from huggingface:

1. **Run mmlu**:
   ```bash
   python evals/mmlu.py \
     --model_name TuKoResearch/ConnectomeGPT100M \
     --tokenizer_name gpt2 \
     --device cuda:0
   ```

2. **Run hellaswag**:
   ```bash
   python evals/hellaswag.py \
     --model_name TuKoResearch/ConnectomeGPT100M \
     --tokenizer_name gpt2 \
     --device cuda:0
   ```

---

## Behavioral alignment
We use the Futrell2018 reading time benchmark, which can be obtained from [brain-score language](https://github.com/brain-score/language) and can be loaded using an environment with `xarray` installed. The data can be downloaded [here](https://huggingface.co/datasets/TuKoResearch/GenerationalConnectomesData/resolve/main/assy_Futrell2018.nc?download=true).

Once downloaded place the Futrell2018 reading-time dataset (`assy_Futrell2018.nc`) in a directory called `data/`.

To run the surprisal evaluation script and compute the Pearson correlation between model surprisal and human reading times (for the final checkpoint), execute:

```bash
python surprisal_eval.py \
  --model_name TuKoResearch/ConnectomeGPT100M \
  --tokenizer_name gpt2 \
  --device cuda:0
```


---

## Neural alignment
We use the Tuckute2024 neural benchmark, which can be downloaded from the following [public repository](https://github.com/gretatuckute/drive_suppress_brains) or [brain-score language](https://github.com/brain-score/language). The cross-validation neural predictivity score can be run from [NeuralAlignment/fit_mapping.py](https://github.com/TuKoResearch/GenerationalConnectomes/blob/main/NeuralAlignment/fit_mapping.py) and looped across layers and models using [NeuralAlignment/loop_fit_mapping.py](https://github.com/TuKoResearch/GenerationalConnectomes/blob/main/NeuralAlignment/loop_fit_mapping.py).

In some of the analyses, we first localize the LLM language units, per the approach established in AlKhamissi et al., 2025 (_ACL_), from the [following repository](https://github.com/BKHMSI/llm-localization). We adapted this code to output a binary mask which marks the LLM language units as 1. The [NeuralAlignment/apply_langloc_mask.py](https://github.com/TuKoResearch/GenerationalConnectomes/blob/main/NeuralAlignment/apply_langloc_mask.py) script takes the the numpy binary mask for a given model, and saves the masked embedding values as a csv file, which can then serve as the input to [NeuralAlignment/fit_mapping.py](https://github.com/TuKoResearch/GenerationalConnectomes/blob/main/NeuralAlignment/fit_mapping.py).

The binary langloc masks, the model embeddings, and the regression outputs can be downloaded [here](https://huggingface.co/datasets/TuKoResearch/GenerationalConnectomesData/resolve/main/SHARE.zip?download=true).

---

## LLM training

Once your environment is ready, train the Generational Pruning GPT model from a pruned checkpoint with:

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

python train_itp.py \
  --run_name my_experiment \
  --train_data_dir path/to/train/*.bin \
  --val_data_dir path/to/val/*.bin \
  --wandb            # (optional: log to Weights & Biases)


This will save a checkpoint to `out/<my_experiment>` which you can use as your connectome for the inner loop trianing above.

---

## Citation

If you use this code, please cite:

> Kotar, K., & Tuckute, G. (2025). Model connectomes: A generational approach to data-efficient language models. *Second Workshop on Representational Alignment at ICLR 2025*.

