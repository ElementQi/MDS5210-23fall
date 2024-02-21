import click
import torch
import copy
import os
from trainers import DPOTrainer
# from test_dpo import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic, GPT
from dataset import EYLSFTStaticDataset, RLHFDataset

# Avoid GPU version conflict (For Kaggle GPU only). Comment below two lines if you use local machine in order to speed up training.
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True


def train(pretrain, batch_size, exp_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg = get_configs("gpt2-medium/dropout")

    cfg = get_configs("gpt2-medium")
    cfg.max_steps = 200000 // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    assert pretrain == "huggingface"
    cfg.exp_name = exp_name

    # model_cache_path = f"./{cfg.model_name}"
    # if os.path.exists(model_cache_path):
    #     model = GPT.from_pretrained(model_cache_path)
    model = GPT.from_pretrained(cfg)
    ref_model = GPT.from_pretrained(cfg)

    train_ds = RLHFDataset(block_size=1024, split="train", max_examples=None, tokenizer_name="tiktoken/gpt2")
    test_ds = EYLSFTStaticDataset(block_size=1024, split="test", max_examples=None, tokenizer_name="tiktoken/gpt2")
    trainer = DPOTrainer(cfg, device, model, ref_model, train_ds, test_ds, beta=.1)
    trainer.fit()


@click.command()
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name)



if __name__ == "__main__":
    main()
