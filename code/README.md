# MDS5210-23fall

## File Structure

- project
    - code
        - configs.py #configuration of models
        - dataset.py #definition of datasets
        - evaluate.py #GPT4 evaluation API
        - gpt.py #definition of models
        - prepare_sft_dataset.py #script for downloading datasets
        - prompt_interactive.py #runable script for generating content from prompt interactively
        - README.md #this description markdown file
        - requirements.txt #package requirements
        - tokenizer.py #definition of tokenizer
        - train_sft.py #sft training run script
        - train_dpo.py #dpo training run script
        - trainers.py #file containing training details
        - utils.py #useful functions script
    - project.pdf #report file

## How to run

Make sure your python environment fulfill the requirements:

```python
pip install -r requirements.txt
```

Make sure you have downloaded the dataset:

```python
python prepare_sft_dataset.py
```

Before you train, you need to check some parameters whether be right or not in `configs.py` file and some local variables at `fit` function of `Trainer` class in `trainers.py` file. For instance, there is `wandb_on` inside `trainers.py`, and if you want to enable wandb, need to make sure you log in with your key.

If you want to train `SFT`, you just type:
```python
python train_sft.py
```

And if you want to train using `DPO`, type:

```python
python train_dpo.py
```

If you want to play with the fine-tuned GPT-2, you need to set the path to your checkpoint file of your model, and type:
```python
python prompt_interactive.py
```

## Reminders

If you are using wandb on Kaggle, you should use the code that will call the interactive chatbox:

```python
import wandb
from wandb.keras import WandbCallback
wandb.login()
```