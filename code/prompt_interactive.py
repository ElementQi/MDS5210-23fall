from langchain.llms import OpenAI
import json
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from configs import get_configs
from tqdm import tqdm
import torch
import tiktoken
import json
from evaluate import generate_gpt2
    

def main():
    print("Run inference")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_configs("gpt2-medium")
    # sft = "./runs/sft_default_202312180912/sft_default_202312180912_final.pt"
    # sft = "./models/sft_tenprompts_202312220941_final.pt"
    sft = "./realmodel/sft_tenprompts_202312220941_final.pt"

    with torch.inference_mode():
        gpt_sft = torch.compile(GPT.from_checkpoint(cfg, sft))

        responses = []
        response=True
        prompt = None

        while prompt!='q' or prompt!='Q':

            prompt = str(input("Please enter the prompt(q or Q to quit): "))

            raw_list = generate_gpt2(gpt_sft, f"Human: {prompt}\n\nAssistant: ", device).split("\n")
            # response = generate_gpt2(gpt_sft, f"Human: {prompt}\n\nAssistant: ", device).split("\n")[2][len("Assistant: "): ]
            response = raw_list[2][len("Assistant: "): ]
            print(raw_list)

            print("sft:", response)

            responses.append({
                "sft": response,
                "prompt": prompt
            })

        with open("responses.json", "w") as fp:
            json.dump(responses, fp)

if __name__ == "__main__":
    main()