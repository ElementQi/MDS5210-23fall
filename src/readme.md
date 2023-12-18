

what should we do now? 2023/12/18 1:17 midnight.

in configs.py, modify the block_size in get_configs <- not anymore... the batch size is inited as int 1.



### In train_sft.py

after line 36 `trainer.fit()`, we need to save the model.
    - but then, in `evaluate.py`, how do we load the fine-tuned model?
        - GPT.from_checkpoint(cfg, sft) ? is checkpoint meaning load the trained model? <- yes

```python
gpt_sft = torch.compile(GPT.from_checkpoint(
                cfg,
                sft))

#where gpt_sft is model
```


### In trainers.py


in `fit(self)` function, we should save the state, using the super()'s `save_states(self, step, is_last=False)` function.



in `SFTTrainer(Trainer)`
```python
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
```

`DataLoader` will stack automatically according to `batch_size`

So, we do not need to write a batch stacker.


How can we trigger the `forward` function of a `nn.Module`:
just call `model(x)`



The given code just splits the data into train and test data. What needs to be mentioned is that we can't use test data since DATA SNOOPING.
We need to get a new validation dataset generator iterator (which could be "nexted"). So that we can randomly get xb and yb from it. Otherwise, if we use list to store, the memory will explode.
But we firstly ignore **data snooping**, use test data.

### In evaluate.py

where is `prompts.csv`? should we collect?? but we have `responses.json` file. Therefore, we can grab the prompts from the `responses.json` file.

```json
{
    "vanilla": generate_gpt2(gpt_vanilla, f"Human: {prompt}\n\nAssistant: ", device)[
                len(f"Human: {prompt}\n\nAssistant: "):],
    "sft": generate_gpt2(gpt_sft, f"Human: {prompt}\n\nAssistant: ", device)[
            len(f"Human: {prompt}\n\nAssistant: "):],
    "ppo": generate_gpt2(gpt_ppo, f"Human: {prompt}\n\nAssistant: ", device)[
            len(f"Human: {prompt}\n\nAssistant: "):],
    "prompt": prompt
}
```

Moreover, I think this file is to created to chat with openai.
    - My thought is right. It is to let the chatgpt to be the judge, to say whether a model or the other model is better based on one specific prompt.

IT IS `GPT4 evaluation API` said by manual.








### RUNNING LOG

first time load
```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"
Downloading (…)"config.json";: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 718/718 [00:00<00:00, 735kB/s]
Downloading (…)"model.safetensors";: 100%|██████████████████████████████████████████████████████████████████████████████████| 1.52G/1.52G [01:06<00:00, 23.0MB/s]
Downloading (…)ration_config.json";: 100%|██████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 98.2kB/s]
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
406.286336 M parameters
```

second run

```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
wandb: Currently logged in as: 1033834827 (dorm_a603). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\wandb\run-20231218_053311-prf7v08x
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run light-durian-1
wandb:  View project at https://wandb.ai/dorm_a603/gpt2-12-18-5-16
wandb:  View run at https://wandb.ai/dorm_a603/gpt2-12-18-5-16/runs/prf7v08x
Traceback (most recent call last):
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 49, in <module>
    main()
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 45, in main
    train(pretrain, batch_size, exp_name)
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 36, in train
    trainer.fit()
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\trainers.py", line 110, in fit
    for iter in tqdm(range(self.cfg.max_iters)):
                           ^^^^^^^^^^^^^^^^^^
AttributeError: 'TrainingConfig' object has no attribute 'max_iters'
wandb:  View run light-durian-1 at: https://wandb.ai/dorm_a603/gpt2-12-18-5-16/runs/prf7v08x
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20231218_053311-prf7v08x\logs
```

so changed it into `max_steps`



third

```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
wandb: Currently logged in as: 1033834827 (dorm_a603). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\wandb\run-20231218_053518-u489m0n6
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run eager-haze-2
wandb:  View project at https://wandb.ai/dorm_a603/gpt2-12-18-5-16
wandb:  View run at https://wandb.ai/dorm_a603/gpt2-12-18-5-16/runs/u489m0n6
  0%|                                                                                                                                 | 0/200000 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 49, in <module>
    main()
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 45, in main
    train(pretrain, batch_size, exp_name)
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 36, in train
    trainer.fit()
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\trainers.py", line 123, in fit
    logits_ = self.model(X)
              ^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\gpt.py", line 232, in forward
    x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\gpt.py", line 194, in forward
    token_embeddings = self.token_embedding_layer(x)  # (B, T, d)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\sparse.py", line 162, in forward
    return F.embedding(
           ^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\functional.py", line 2233, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method 
wrapper_CUDA__index_select)
wandb:  View run eager-haze-2 at: https://wandb.ai/dorm_a603/gpt2-12-18-5-16/runs/u489m0n6
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20231218_053518-u489m0n6\logs

```

solution:

```
xb, yb = next(self.train_dataloader)
xb, yb = xb.to(self.device), yb.to(self.device)

X, Y = next(self.train_dataloader) if split == 'train' else next(self.test_dataloader)
X, Y = X.to(self.device), Y.to(self.device)

```

deploy on cityu dive:

```
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$ python train_sft.py 
2023-12-17 22:07:19.651178: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
start eval
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:58<00:00,  3.39it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:00<00:00,  3.33it/s]
Traceback (most recent call last):
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 49, in <module>
    main()
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 45, in main
    train(pretrain, batch_size, exp_name)
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 36, in train
    trainer.fit()
  File "/home/jovyan/Mengqi/1218/trainers.py", line 138, in fit
    train_loss = float(f"{losses['train']:.4f}")
                          ~~~~~~^^^^^^^^^
IndexError: too many indices for tensor of dimension 1

```


change:


```
train_loss = float(f"{losses['train']:.4f}")
test_loss = float(f"{losses['val']:.4f}")

# to

train_loss = float(f"{out['train']:.4f}")
test_loss = float(f"{out['val']:.4f}")
```

then

```
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$ python train_sft.py 
2023-12-17 22:12:48.464610: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
start eval
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:06<00:00,  3.13it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:05<00:00,  3.45it/s]
step 0: train loss 2.5694, val loss 2.6198
Traceback (most recent call last):
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 49, in <module>
    main()
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 45, in main
    train(pretrain, batch_size, exp_name)
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 36, in train
    trainer.fit()
  File "/home/jovyan/Mengqi/1218/trainers.py", line 155, in fit
    logits = self.model(xb)
             ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 232, in forward
    x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 203, in forward
    x = block(x, attention_mask)
        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 165, in forward
    x = self.mmsa(x, attention_mask)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 79, in forward
    attention = Q @ K.transpose(2, 3)
                ~~^~~~~~~~~~~~~~~~~~~
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 14.58 GiB of which 34.50 MiB is free. Process 4125680 has 14.54 GiB memory in use. Of the allocated memory 14.38 GiB is allocated by PyTorch, and 30.60 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

```

Since I have set too much batch_size (256)

<time: 2023/12/18 6:16>

Then I change it to 128

STILL
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 14.58 GiB of which 34.50 MiB is free. Process 4129129 has 14.54 GiB memory in use. Of the allocated memory 14.38 GiB is allocated by PyTorch, and 30.60 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

set it to 64, still.

then 32. still

Is somewhere the code wrong???
I think not. Since when i just use 1 in my own pc, it still crushed my gpu memory and main memory.

16 no.

then 8. no. then i set it to 1.

is the gpu locate issue? shuold i reboot? yes... 1 still dont work

in cityu:

```
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$ python train_sft.py 
2023-12-17 22:41:06.175262: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
  0%|                                                                                                                                                                                                                                     | 0/200 [00:00<?, ?it/s]start eval
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.47it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.49it/s]
step 0: train loss 2.4274, val loss 2.5685██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.49it/s]
 50%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                             | 100/200 [01:41<01:36,  1.03it/s]start eval
 20%|████████████████████████████████████████████▌                                                                                                                                                                                  | 1/5 [00:00<00:03,  1.24it/s]
 50%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                             | 100/200 [01:42<01:42,  1.02s/it]
Traceback (most recent call last):
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 49, in <module>
    main()
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 45, in main
    train(pretrain, batch_size, exp_name)
  File "/home/jovyan/Mengqi/1218/train_sft.py", line 36, in train
    trainer.fit()
  File "/home/jovyan/Mengqi/1218/trainers.py", line 127, in fit
    logits = self.model(xb)
             ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 232, in forward
    x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 203, in forward
    x = block(x, attention_mask)
        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 165, in forward
    x = self.mmsa(x, attention_mask)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/Mengqi/1218/gpt.py", line 79, in forward
    attention = Q @ K.transpose(2, 3)
                ~~^~~~~~~~~~~~~~~~~~~
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 14.58 GiB of which 52.50 MiB is free. Process 4157994 has 14.52 GiB memory in use. Of the allocated memory 14.36 GiB is allocated by PyTorch, and 31.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$
```


in my pc:
```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
start train
  0%|                                                                                                | 0/20 [00:00<?, ?it/s]start validation
100%|█████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:43<00:00,  8.71s/it] 
100%|█████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:39<00:00,  7.84s/it] 
step 0: train loss 2.4274, val loss 2.5685████████████████████████████████████████████████████| 5/5 [00:39<00:00,  7.98s/it] 
100%|███████████████████████████████████████████████████████████████████████████████████████| 20/20 [10:37<00:00, 31.85s/it] 
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project> 

```


not validation:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 14.58 GiB of which 52.50 MiB is free. Process 4157994 has 14.52 GiB memory in use. Of the allocated memory 14.36 GiB is allocated by PyTorch, and 31.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$ python train_sft.py 
2023-12-17 22:51:33.852872: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
  0%|                                                                                                                                                                                                                                     | 0/500 [00:00<?, ?it/s]tensor(1.8679, device='cuda:0', grad_fn=<NllLossBackward0>)
  2%|████▍                                                                                                                                                                                                                       | 10/500 [00:09<07:36,  1.07it/s]tensor(2.4127, device='cuda:0', grad_fn=<NllLossBackward0>)
  4%|████████▊                                                                                                                                                                                                                   | 20/500 [00:19<07:27,  1.07it/s]tensor(2.3688, device='cuda:0', grad_fn=<NllLossBackward0>)
  6%|█████████████▏                                                                                                                                                                                                              | 30/500 [00:28<07:20,  1.07it/s]tensor(2.3695, device='cuda:0', grad_fn=<NllLossBackward0>)
  8%|█████████████████▌                                                                                                                                                                                                          | 40/500 [00:37<07:12,  1.06it/s]tensor(2.0465, device='cuda:0', grad_fn=<NllLossBackward0>)
 10%|██████████████████████                                                                                                                                                                                                      | 50/500 [00:47<07:05,  1.06it/s]tensor(2.2343, device='cuda:0', grad_fn=<NllLossBackward0>)
 12%|██████████████████████████▍                                                                                                                                                                                                 | 60/500 [00:56<06:58,  1.05it/s]tensor(2.2206, device='cuda:0', grad_fn=<NllLossBackward0>)
 14%|██████████████████████████████▊                                                                                                                                                                                             | 70/500 [01:06<06:49,  1.05it/s]tensor(2.3710, device='cuda:0', grad_fn=<NllLossBackward0>)
 16%|███████████████████████████████████▏                                                                                                                                                                                        | 80/500 [01:15<06:42,  1.04it/s]tensor(2.3644, device='cuda:0', grad_fn=<NllLossBackward0>)
 18%|███████████████████████████████████████▌                                                                                                                                                                                    | 90/500 [01:25<06:33,  1.04it/s]tensor(2.0926, device='cuda:0', grad_fn=<NllLossBackward0>)
 20%|███████████████████████████████████████████▊                                                                                                                                                                               | 100/500 [01:35<06:26,  1.03it/s]tensor(2.2467, device='cuda:0', grad_fn=<NllLossBackward0>)
 22%|████████████████████████████████████████████████▏                                                                                                                                                                          | 110/500 [01:44<06:17,  1.03it/s]tensor(2.2301, device='cuda:0', grad_fn=<NllLossBackward0>)
 24%|████████████████████████████████████████████████████▌                                                                                                                                                                      | 120/500 [01:54<06:08,  1.03it/s]tensor(2.3035, device='cuda:0', grad_fn=<NllLossBackward0>)
 26%|████████████████████████████████████████████████████████▉                                                                                                                                                                  | 130/500 [02:04<05:59,  1.03it/s]tensor(1.9539, device='cuda:0', grad_fn=<NllLossBackward0>)
 28%|█████████████████████████████████████████████████████████████▎                                                                                                                                                             | 140/500 [02:13<05:51,  1.02it/s]tensor(2.3322, device='cuda:0', grad_fn=<NllLossBackward0>)
 30%|█████████████████████████████████████████████████████████████████▋                                                                                                                                                         | 150/500 [02:23<05:42,  1.02it/s]tensor(2.2120, device='cuda:0', grad_fn=<NllLossBackward0>)
 32%|██████████████████████████████████████████████████████████████████████                                                                                                                                                     | 160/500 [02:33<05:33,  1.02it/s]tensor(2.3949, device='cuda:0', grad_fn=<NllLossBackward0>)
 34%|██████████████████████████████████████████████████████████████████████████▍                                                                                                                                                | 170/500 [02:43<05:23,  1.02it/s]tensor(2.0672, device='cuda:0', grad_fn=<NllLossBackward0>)
 36%|██████████████████████████████████████████████████████████████████████████████▊                                                                                                                                            | 180/500 [02:53<05:14,  1.02it/s]tensor(2.2141, device='cuda:0', grad_fn=<NllLossBackward0>)
 38%|███████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                       | 190/500 [03:03<05:05,  1.02it/s]tensor(2.3212, device='cuda:0', grad_fn=<NllLossBackward0>)
 40%|███████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                   | 200/500 [03:12<04:54,  1.02it/s]tensor(2.0541, device='cuda:0', grad_fn=<NllLossBackward0>)
 42%|███████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                               | 210/500 [03:22<04:44,  1.02it/s]tensor(2.4786, device='cuda:0', grad_fn=<NllLossBackward0>)
 44%|████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                          | 220/500 [03:32<04:35,  1.02it/s]tensor(2.2161, device='cuda:0', grad_fn=<NllLossBackward0>)
 46%|████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                      | 230/500 [03:42<04:25,  1.02it/s]tensor(1.9866, device='cuda:0', grad_fn=<NllLossBackward0>)
 48%|█████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                  | 240/500 [03:52<04:15,  1.02it/s]tensor(2.1324, device='cuda:0', grad_fn=<NllLossBackward0>)
 50%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                             | 250/500 [04:01<04:06,  1.02it/s]tensor(2.3888, device='cuda:0', grad_fn=<NllLossBackward0>)
 52%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                         | 260/500 [04:11<03:55,  1.02it/s]tensor(2.5949, device='cuda:0', grad_fn=<NllLossBackward0>)
 54%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                    | 270/500 [04:21<03:45,  1.02it/s]tensor(2.0542, device='cuda:0', grad_fn=<NllLossBackward0>)
 56%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                | 280/500 [04:31<03:35,  1.02it/s]tensor(2.0523, device='cuda:0', grad_fn=<NllLossBackward0>)
 58%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                            | 290/500 [04:41<03:26,  1.02it/s]tensor(2.0085, device='cuda:0', grad_fn=<NllLossBackward0>)
 60%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                       | 300/500 [04:51<03:16,  1.02it/s]tensor(2.0795, device='cuda:0', grad_fn=<NllLossBackward0>)
 62%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                   | 310/500 [05:00<03:06,  1.02it/s]tensor(2.2237, device='cuda:0', grad_fn=<NllLossBackward0>)
 64%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                              | 320/500 [05:10<02:56,  1.02it/s]tensor(1.8911, device='cuda:0', grad_fn=<NllLossBackward0>)
 66%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                          | 330/500 [05:20<02:46,  1.02it/s]tensor(2.4335, device='cuda:0', grad_fn=<NllLossBackward0>)
 68%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                      | 340/500 [05:30<02:37,  1.02it/s]tensor(2.3083, device='cuda:0', grad_fn=<NllLossBackward0>)
 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                 | 350/500 [05:40<02:27,  1.02it/s]tensor(2.2183, device='cuda:0', grad_fn=<NllLossBackward0>)
 72%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                             | 360/500 [05:50<02:17,  1.02it/s]tensor(2.4153, device='cuda:0', grad_fn=<NllLossBackward0>)
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                         | 370/500 [05:59<02:07,  1.02it/s]tensor(2.2674, device='cuda:0', grad_fn=<NllLossBackward0>)
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                    | 380/500 [06:09<01:57,  1.02it/s]tensor(2.1896, device='cuda:0', grad_fn=<NllLossBackward0>)
 78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                | 390/500 [06:19<01:48,  1.02it/s]tensor(2.0686, device='cuda:0', grad_fn=<NllLossBackward0>)
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 400/500 [06:29<01:38,  1.02it/s]tensor(2.0794, device='cuda:0', grad_fn=<NllLossBackward0>)
 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 410/500 [06:39<01:28,  1.02it/s]tensor(2.3905, device='cuda:0', grad_fn=<NllLossBackward0>)
 84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                   | 420/500 [06:48<01:18,  1.02it/s]tensor(2.2079, device='cuda:0', grad_fn=<NllLossBackward0>)
 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                              | 430/500 [06:58<01:08,  1.02it/s]tensor(2.1548, device='cuda:0', grad_fn=<NllLossBackward0>)
 88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                          | 440/500 [07:08<00:58,  1.02it/s]tensor(2.1899, device='cuda:0', grad_fn=<NllLossBackward0>)
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                      | 450/500 [07:18<00:49,  1.02it/s]tensor(2.1329, device='cuda:0', grad_fn=<NllLossBackward0>)
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 460/500 [07:28<00:39,  1.02it/s]tensor(2.4076, device='cuda:0', grad_fn=<NllLossBackward0>)
 94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊             | 470/500 [07:38<00:29,  1.02it/s]tensor(2.1027, device='cuda:0', grad_fn=<NllLossBackward0>)
 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏        | 480/500 [07:47<00:19,  1.02it/s]tensor(1.9452, device='cuda:0', grad_fn=<NllLossBackward0>)
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌    | 490/500 [07:57<00:09,  1.02it/s]tensor(2.4978, device='cuda:0', grad_fn=<NllLossBackward0>)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:07<00:00,  1.03it/s]
jovyan@jupyter-chenxuliu8:~/Mengqi/1218$ 

```




we are using `EYLSFTStaticDataset` at all! But the guide book said that 

>  Anthropic’s "HH-RLHF" dataset for our project. In particular, we will use the
“helpful-base” division of the dataset. It consists of 43835 samples for training dataset and 2354
samples for test dataset.

but we got

```
Loading EYLSFTStaticDataset train split
Loaded 19934053 tokens from 84576 examples.
Loading EYLSFTStaticDataset test split
Loaded 844060 tokens from 3451 examples.
```


just use `dataset.RLHFDataset`


```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"       
Loading RLHF Dataset...
100%|███████████████████████████████████████████████████████████████████████████████████████| 21917/21917 [00:14<00:00, 1497.43it/s]
Loading RLHF Dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 1177/1177 [00:00<00:00, 1485.75it/s]
start train
  0%|                                                                                                        | 0/20 [00:00<?, ?it/s]start validation
  0%|                                                                                                         | 0/3 [00:00<?, ?it/s] 
  0%|                                                                                                        | 0/20 [00:00<?, ?it/s] 
Traceback (most recent call last):
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 52, in <module>
    main()
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1055, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\click\core.py", line 760, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 48, in main    train(pretrain, batch_size, exp_name)
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\train_sft.py", line 39, in train
    trainer.fit()
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\trainers.py", line 127, in fit 
    logits_ = self.model(X)
              ^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl      
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\gpt.py", line 232, in forward  
    x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl      
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\anaconda\anaconda\envs\pytorchForLLM\Lib\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src\gpt.py", line 191, in forward  
    B, T = x.size()
    ^^^^
ValueError: too many values to unpack (expected 2)

```


by using `dataset.SFTDataset`

```
PS H:\googleDriveSaver\classes\MDS 5210 - Machine Learning\final_project\gits\MDS5210-23fall\src> & H:/anaconda/anaconda/envs/pytorchForLLM/python.exe "h:/googleDriveSaver/classes/MDS 5210 - Machine Learning/final_project/gits/MDS5210-23fall/src/train_sft.py"       
Found cached dataset json (C:/Users/ElementQi/.cache/huggingface/datasets/Anthropic___json/Anthropic--hh-rlhf-bb7971723b14c46c/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51)
Loading SFT train split
Loaded 4294661 tokens from 21917 examples.
Found cached dataset json (C:/Users/ElementQi/.cache/huggingface/datasets/Anthropic___json/Anthropic--hh-rlhf-bb7971723b14c46c/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51)
Loading SFT test split
Loaded 235139 tokens from 1177 examples
```

ohh, that's just the half of 43835 and 2354




Moreover, we need to carefully choose our max_step, since the training process is quite slow.




## REQUIREMENTS


(A) Plot the curve of training error and test error. Since the dataset is relatively large, evaluating across the whole dataset is time consuming. To compensate the issue, you may estimate the loss by sampling a portion of data independently from the dataset. A recommended size for the estimation is **200**


