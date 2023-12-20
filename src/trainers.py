import torch
import os
import json
import random
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from configs import TrainingConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import generate_gpt2


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

        self.model.train()

        @torch.no_grad()
        def estimate_loss():
            out = {}

            # set model to evaluation mode
            print("start validation")
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                print(f"start {split} evaluation")
                for k in tqdm(range(eval_iters)):
                    X, Y = next(self.train_dataloader) if split == 'train' else next(self.test_dataloader)
                    X, Y = X.to(self.device), Y.to(self.device)
                    
                    logits = self.model(X)
                    B, T, C = logits.shape
                    logits = logits.view(B*T, C)
                    targets = Y.view(B*T)
                    loss = nn.functional.cross_entropy(logits, targets)

                    losses[k] = loss.item()
                out[split] = losses.mean()
            
            # set model back to training mode
            self.model.train()
            return out

        # load prompts from json
        # with open("prompts.json", 'r') as fd:
        #     import json
        #     prompts = json.load(fd)

        # init dialogue score judger
        reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

        rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
        rank_model.eval()

        # mount model to device, ready to train
        self.model.to(self.device)
        
        # we need to split train data into train and validation data. What need to mentioned is that we can't use test data since DATA SNOOPING.


        # print the number of parameters in the model
        # print(sum(p.numel() for p in self.model.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

        max_steps = 3000
        eval_iters = 50 # validation dataset batch size (200 required)
        eval_times = 10 # total number of validation times
        eval_interval = max_steps // eval_times
        prompts_num = 10

        mydict = self.cfg.dict()
        mydict["max_steps"] = max_steps
        mydict["eval_iters"] = eval_iters
        mydict["eval_times"] = eval_times
        mydict["additional"] = "with_gradient_accumulation"

        wandb_on = True
        
        # wandb section
        if wandb_on:
            import wandb
            wandb.init(
                # set the wandb project where this run will be logged
                project="gpt2-12-20-1k-GA",
                
                # track hyperparameters and run metadata
                config=mydict
            )

        prompts = [
            "As AI continues to evolve, what ethical guidelines should be implemented to ensure responsible use and development?",
            "What are the potential benefits and risks of colonizing Mars, and how should we prepare for them?",
            "Discuss the impact of single-use plastics on ocean ecosystems and potential solutions to mitigate this issue.",
            "How can we promote cultural understanding and appreciation in increasingly diverse societies?",
            "Evaluate the effects of universal basic income on a nation's economy and social welfare.",
            "What strategies can be employed to reduce the stigma surrounding mental health in the workplace?",
            "How might virtual reality technology revolutionize the educational system in the next decade?",
            "Discuss the role of AI in enhancing diagnostic accuracy and patient care in medicine.",
            "Examine the implications of cyber warfare on international relations and national security.",
            "Analyze the challenges and opportunities presented by the global shift towards renewable energy sources."
        ]

        
        print("start train")
        # for iter in tqdm(range(self.cfg.max_steps)):
        for iter in tqdm(range(max_steps)):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_steps - 1:
                
                scores = []
                # dialogue score section
                print("start calculate reward score")
                with torch.inference_mode():
                    for prompt in tqdm(prompts[:prompts_num]):
                        # response = generate_gpt2(self.model, f"Human: {prompt}\n\nAssistant: ", self.device)[len(f"Human: {prompt}\n\nAssistant: "):]

                        # only choose A1
                        response = generate_gpt2(self.model, f"Human: {prompt}\n\nAssistant: ", self.device).split()[3]
                        inputs = tokenizer(prompt, response, return_tensors='pt')
                        score = rank_model(**inputs).logits[0].item()
                        print(f"\n{score}\n")
                        scores.append(score)
                    final_score = float(f"{sum(scores) / len(scores):.4f}")
                    losses = estimate_loss()
                
                train_loss = float(f"{losses['train']:.4f}")
                test_loss = float(f"{losses['val']:.4f}")

                print(f"\n step {iter}: train loss {train_loss}, val loss {test_loss}, reward score {final_score}\n ")

                if wandb_on:
                    wandb.log({"training error": train_loss, "test error": test_loss, "reward score": final_score})

                # save the model states in procedure
                self.save_states(step=iter, is_last=False)

            # gradient accumulation section
            accumulation_steps = 2

            x, y = next(self.train_dataloader)
            for step in range(self.cfg.batch_size):
                xb = x[step:step+1, :].to(self.device)
                yb = y[step:step+1, :].to(self.device)

                # Forward pass
                logits = self.model(xb)

                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = yb.view(B*T)

                loss = nn.functional.cross_entropy(logits, targets)
                
                # accumulate
                loss = loss / accumulation_steps 
                loss.backward()

                # every accumulation_steps do gradient descent
                if (step + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)


            # no gradient accumulation
            # sample a batch of data
        
            # xb, yb = next(self.train_dataloader)
            # xb, yb = xb.to(self.device), yb.to(self.device)

            # # do forward section
            # logits = self.model(xb)

            # B, T, C = logits.shape
            # logits = logits.view(B*T, C)
            # targets = yb.view(B*T)

            # loss = nn.functional.cross_entropy(logits, targets)
            # self.optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # self.optimizer.step()

        # save the final model states
        self.save_states(step=self.cfg.max_steps, is_last=True)

        # [optional] finish the wandb run, necessary in notebooks
        if wandb_on:
            wandb.finish()
