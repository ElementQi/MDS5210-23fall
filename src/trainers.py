import torch
import os
import json
import random
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from configs import TrainingConfig
from tqdm import tqdm


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')
    
class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # create a dataloader
        # get a batch of (data, label) by: x, y = self.train_dataloader
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        # TODO: complete the SFT training.

        # wandb section
        # import wandb
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="gpt2-12-18-5-16",
            
        #     # track hyperparameters and run metadata
        #     config=self.cfg.dict()
        # )

        # mount model to device, ready to train
        self.model.to(self.device)
        
        # we need to split train data into train and validation data. What need to mentioned is that we can't use test data since DATA SNOOPING.


        # print the number of parameters in the model
        # print(sum(p.numel() for p in self.model.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

        eval_interval = 10
        # choose eval_iters data from validation dataset
        eval_iters = 3

        print("start train")
        # for iter in tqdm(range(self.cfg.max_steps)):
        for iter in tqdm(range(20)):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == self.cfg.max_steps - 1:
                out = {}

                # set model to evaluation mode
                print("start validation")
                self.model.eval()
                for split in ['train', 'val']:
                    losses = torch.zeros(eval_iters)
                    for k in tqdm(range(eval_iters)):
                        X, Y = next(self.train_dataloader) if split == 'train' else next(self.test_dataloader)
                        X, Y = X.to(self.device), Y.to(self.device)
                        
                        logits_ = self.model(X)
                        B_, T_, C_ = logits_.shape
                        logits_ = logits_.view(B_*T_, C_)
                        targets_ = Y.view(B_*T_)
                        loss_ = nn.functional.cross_entropy(logits_, targets_)

                        losses[k] = loss_.item()
                    out[split] = losses.mean()

                # set model back to training mode
                self.model.train()

                train_loss = float(f"{out['train']:.4f}")
                test_loss = float(f"{out['val']:.4f}")

                print(f"step {iter}: train loss {train_loss}, val loss {test_loss}")

                # wandb.log({"training error": train_loss, "test error": test_loss})

                # save the model states in procedure
                # self.save_states(step=iter, is_last=False)

            # sample a batch of data
            xb, yb = next(self.train_dataloader)
            xb, yb = xb.to(self.device), yb.to(self.device)

            # do forward section
            logits = self.model(xb)

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = yb.view(B*T)

            loss = nn.functional.cross_entropy(logits, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # save the final model states
        # self.save_states(step=self.cfg.max_steps, is_last=True)

        # [optional] finish the wandb run, necessary in notebooks
        # wandb.finish()
