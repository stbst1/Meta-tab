# MetaTab

This is the guide released for the paper *"Evaluating the Robustness of Tabular Language Models via Metamorphic Testing."*  

![MetaTab Illustration](METATAB1.png)

---

## Requirements

- Python >= 3.10  
- Linux  

---

## Install

Clone this repository and run the following command in the root directory to install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation

Extract the dataset:

```bash
unzip assets/data.zip -d path/to/data
```

### 2. Model Setup

Set up language models such as TableGPT and TableLLM locally:

- **TableGPT**: https://huggingface.co/tablegpt/TableGPT2-7B  
- **TableLLM**: https://huggingface.co/RUCKBReasoning/TableLLM-7b  


### 3. Step-by-Step Example with TableGPT

#### Intermediate Program Generation (Original)

```bash
python run_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

#### Intermediate Program Generation (Perturbed)

- **PMR1: Shuffle**
```bash
python run_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation shuffle --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

- **PMR2: Column Shuffle**
```bash
python run_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation column_shuffle --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

- **PMR3: Transpose**
```bash
python run_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation transpose --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

- **PMR4: Reconstruction**
```bash
python run_reconstruction_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

---

### Data Modification Rules (DMR)

- **DMR1**
```bash
python run_tablegpt_agent_cut.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

- **DMR2**
```bash
python run_tablegpt_agent_c_cut.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

---

### Semantic Modification Rules (SMR)

- **SMR1**
```bash
python Symbolization_pure_numbers_to_workds.py
```

- **SMR2**
```bash
python Category_Anonymization.py
```

- **SMR3**
```bash
python filter_time_series_table.py
```

After preprocessing, run:

```bash
python run_tablegpt_agent.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5
```

---

## Evaluation

- **Error Rate**
```bash
python ./evaluate_agent_all_type.py
```

- **Recall, Precision, F1 Score**
```bash
python hhh_wtq.py
```

---

## Environment Requirements

- PyTorch  

---

## Results

We include the predictions of MetaTab in our dataset and its ablation results in the ```outputs/``` folder.  
