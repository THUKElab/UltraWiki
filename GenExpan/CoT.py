import re
import time
from utils import *

"""
CoTPromptGenerator is to generate the prompt that expands the entity with CoT

"""



_raw_cln_prompt = """Generate a class name that accurately represents the following entities. \
This class name should encompass all the given entities and reflect their shared characteristics. \
Examples:
[Tiger, Lion, Cheetah] -> Big Cats
[Shakespeare, Tolstoy, Hemingway] -> Famous Authors
[Mercury, Venus, Mars] -> Planets in the Solar System
[{}] ->"""



_raw_ent_prompt = """They are all Animals: \
Lion, Elephant, Giraffe, and Zebra.
They are all Authors: \
Ernest Hemingway, F. Scott Fitzgerald, Mark Twain, and John Steinbeck.
They are all {cln}: \
{ents_text} and"""


class CoTPromptGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        generated_clns_save_path,
        max_cln_tokens=20,
        max_attrs_tokens=30,
        save_step=10
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cln_tokens = max_cln_tokens
        self.max_attrs_tokens = max_attrs_tokens
        self.device = next(model.parameters()).device
        self.generated_clns_save_path = generated_clns_save_path
        self.cln = None
        self.attrs_text = None
        self.generated_clns = []
        if os.path.exists(self.generated_clns_save_path):
            self.generated_clns = read_json(self.generated_clns_save_path)
        self.add_count = 0
        self.save_step = save_step

    def save_generated_clns(self):
        cur_generated_clns = []
        if os.path.exists(self.generated_clns_save_path):
            cur_generated_clns = read_json(self.generated_clns_save_path)
        for cln in cur_generated_clns:
            if cln not in self.generated_clns:
                self.generated_clns.append(cln)
        write_json(self.generated_clns, self.generated_clns_save_path)

    @torch.inference_mode()
    def inference(self, prompt, max_new_tokens) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        batch_output = self.model.generate(**inputs, 
            max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        output_sent = self.tokenizer.batch_decode(
            batch_output, skip_special_tokens=True
        )[0]
        return output_sent[len(prompt) :]

    def parse_to_cln(self, sent: str) -> str:
        sent = sent.strip()
        if "\n" in sent:
            sent = sent[: sent.index("\n")]
        return sent.strip()

    def generate_cln(self, seeds: list[str]) -> str:
        seeds_text = ", ".join(seeds)
        cln_prompt = _raw_cln_prompt.format(seeds_text)
        sent = self.inference(prompt=cln_prompt, max_new_tokens=self.max_cln_tokens)
        return self.parse_to_cln(sent=sent)

    def search_list(self, pos_seeds: list[str], neg_seeds: list[str]) -> int:
        for i in range(len(self.generated_clns)):
            _pos_seeds = self.generated_clns[i]["pos_seeds"]
            _neg_seeds = self.generated_clns[i]["neg_seeds"]
            if _pos_seeds == pos_seeds and _neg_seeds == neg_seeds:
                return i
        return -1

    def init_seeds(self, pos_seeds: list[str], neg_seeds: list[str]):
        index = self.search_list(pos_seeds, neg_seeds)
        if index != -1:
            self.cln = self.generated_clns[index]["cln"]
        else:
            self.cln=self.generate_cln(pos_seeds)
            t = {
                "pos_seeds": pos_seeds,
                "neg_seeds": neg_seeds,
                "cln": self.cln,
            }
            self.generated_clns.append(t)
            self.add_count += 1
            if self.add_count % self.save_step == 0:
                self.save_generated_clns()

    def get_CoT_gen_ents_prompt(self, ents_text: str) -> str:
        ent_prompt = _raw_ent_prompt.format(
            cln=self.cln,
            ents_text=ents_text,
        )
        return ent_prompt


if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ckpt_dir = "./train_output/"
    all_ckpts = [
        dir_name
        for dir_name in os.listdir(ckpt_dir)
        if dir_name.startswith("checkpoint")
    ]
    all_ckpts = [os.path.join(ckpt_dir, dir_name) for dir_name in all_ckpts]
    ckpt = all_ckpts[0]

    model_name = "llama-7b"
    model_path = os.path.expanduser(f"~/local_transformers/{model_name}")
    # model_path = ckpt

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path).half()
    # model.cuda(0)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    generator = CoTPromptGenerator(
        model=model, tokenizer=tokenizer,
        generated_clns_save_path="./data/generated_clns.json",
        save_step=1,
    )
    

    while True:
        pos_seeds_text = input("input pos_seeds: \n")
        neg_seeds_text = input("input neg_seeds: \n")
        pos_seeds = pos_seeds_text.split(",")
        neg_seeds = neg_seeds_text.split(",")
        if len(pos_seeds + neg_seeds) <= 2:
            break

        generator.init_seeds(pos_seeds=pos_seeds,neg_seeds=neg_seeds)
        prompt=generator.get_CoT_gen_ents_prompt("LQS, LQQ, BBQ")
        print(f"prompt: ")
        print(prompt)
        print()
