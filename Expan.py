import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
from CoT import CoTPromptGenerator
from beam_search import beam_search
# np.random.seed(10)


class Expan:
    _prompt_gen_ent = '''iron, copper, aluminum and zinc.
math, physics, chemistry and biology.
{} and'''


    _prompt_match_ent1 = '{} is similar to'
    _prompt_match_ent2 = '{}'

    def __init__(self, args, model_name="llama-7b"):
        self.args = args
        if args.model_path is None:
            model_path = model_name
        else:
            model_path = args.model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path).half()
        self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ents, _ = load_entities(args.entities)
        if args.ent2etext is not None:
            self.ent2etext = read_json(args.ent2etext)
        else:
            self.ent2etext = None
        self.eos_token = '.\n'
        self.eos_token_id = self.tokenizer.encode(
            '.\n', add_special_tokens=False)[-1]
        encoded_ents = convert_to_tokens_ids(
            self.ents, self.tokenizer, self.eos_token)
        self.trie = EntityTrie(encoded_ents)
        self.num_ents_per_epoch = args.num_ents_per_epoch
        self.early_stopping_cnt = args.early_stopping_cnt
        
        if args.CoT:
            self.cot_prompt_generator = CoTPromptGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                generated_clns_save_path=args.generated_clns,
            )


    def cal_match_rate_sum(self, candidate_ents, seeds):
        prompts = []
        masks = []
        n, m = len(candidate_ents), len(seeds)
        for i, ent in enumerate(candidate_ents):
            for seed in seeds:
                prompt1 = Expan._prompt_match_ent1.format(ent)
                prompt2 = Expan._prompt_match_ent2.format(seed)
                prompt1 = self.tokenizer.encode(
                    prompt1, add_special_tokens=True)
                prompt2 = self.tokenizer.encode(
                    prompt2, add_special_tokens=False)
                prompt = prompt1+prompt2
                mask = [0]*len(prompt1)+[1]*len(prompt2)
                prompts.append(prompt)
                masks.append(mask)
        prompts = ids_lists_to_tensor(prompts, pad_token_id=0)  # (n*m,L+1)
        masks = ids_lists_to_tensor(masks, pad_token_id=0)  # (n*m,L+1)
        match_rate = cal_probs(prompts, masks, self.model)  # (n*m)
        match_rate = match_rate.reshape(n, m).sum(dim=-1).cpu().numpy()
        return match_rate

    def topp_select(self, arr, p=0.9, descending=True):
        normalized_arr = arr / np.sum(arr)
        step = -1 if descending else 1
        pre_indices = np.argsort(normalized_arr)[::step]
        sorted_arr = normalized_arr[pre_indices]
        cumsum_arr = np.cumsum(sorted_arr)
        sorted_top_indices = np.where(cumsum_arr <= p)[0]
        if len(sorted_top_indices) == 0:
            sorted_top_indices = [0]
        return pre_indices[sorted_top_indices]

    def generate_ents(self, pos_seeds, neg_seeds, target_size=200):
        trie = self.trie
        sample_weight = []

        L_cur = []
        early_stopping_cnt = self.early_stopping_cnt
        epoch_idx = 1
        while len(L_cur) < target_size:
            pre_L_len = len(L_cur)

            if epoch_idx == 1:
                _ents = np.random.choice(pos_seeds, 3, replace=False)
            else:
                _ents1 = np.random.choice(pos_seeds, 2, replace=False)
                p = np.array(sample_weight, dtype=np.float64)
                p = p/p.sum()
                _ents2 = np.random.choice(L_cur, 1, replace=False, p=p)
                _ents = np.concatenate([_ents1, _ents2])
            
            if self.ent2etext is not None:
                ents_texts=[]
                for ent in _ents:
                    etext=self.ent2etext.get(ent,"")
                    if len(etext)>0:
                        ents_texts.append(etext+"\n")
                aug_text="".join(ents_texts)
            else:
                aug_text=""
            ents_text = ", ".join(_ents)
            if self.args.CoT:
                prompt = self.cot_prompt_generator.get_CoT_gen_ents_prompt(ents_text)
            else:
                prompt = Expan._prompt_gen_ent.format(ents_text)
            prompt = aug_text+prompt

            prompt_ids = self.tokenizer.encode(prompt)
            res = beam_search(
                input_ids=prompt_ids,
                num_beams=self.num_ents_per_epoch,
                model=self.model,
                max_new_tokens=20,
                score_fn=lambda prod_probs, length: prod_probs.pow(
                    (((5 + length) / (5 + 1)) ** 0.6).reciprocal()
                ),
                eos_token_id=self.eos_token_id,
                entity_trie=trie,
                return_prod_probs=False,
                return_num_new_tokens=False,
                only_return_generated_tokens=True,
            )

            output_ids = res.output_ids
            tokens_list = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True)
            expanded = []
            for tokens in tokens_list:
                ent = clear_eos_and_leading_space(tokens, self.eos_token)
                if ent is not None:
                    expanded.append(ent)
            candidate_ents = expanded

            pos_match_rates = self.cal_match_rate_sum(
                candidate_ents=candidate_ents, seeds=pos_seeds)
            # neg_match_rates = self.cal_match_rate_sum(
            #     candidate_ents=candidate_ents, seeds=neg_seeds)

            indices = self.topp_select(
                pos_match_rates, p=self.args.topp_prob, descending=True)

            for index in indices:
                ent = expanded[index]
                if ent in L_cur:
                    index = L_cur.index(ent)
                    sample_weight[index] += 1
                elif (ent not in pos_seeds) and (ent not in neg_seeds):
                    L_cur.append(ent)
                    sample_weight.append(1)

            epoch_idx += 1

            if len(L_cur) == pre_L_len:
                early_stopping_cnt -= 1
                if early_stopping_cnt == 0:
                    break
            else:
                early_stopping_cnt = self.early_stopping_cnt

        L_cur = L_cur[:target_size] if len(L_cur) > target_size else L_cur
        return L_cur

    def rank(self, ents, pos_seeds, neg_seeds):
        pos_match_rates = self.cal_match_rate_sum(
            candidate_ents=ents, seeds=pos_seeds)
        indices = pos_match_rates.argsort()[::-1]
        if not self.args.no_neg_rank:
            ents = [ents[i] for i in indices]
            seg_length = self.args.seg_length
            neg_match_rates = self.cal_match_rate_sum(
                candidate_ents=ents, seeds=neg_seeds)
            neg_match_rates = torch.from_numpy(neg_match_rates)
            indices = seg_sort(neg_match_rates, seg_length=seg_length, descending=False)
        return [ents[i] for i in indices]

    def expand(self, pos_seeds, neg_seeds, target_size):
        if self.args.CoT:
            self.cot_prompt_generator.init_seeds(pos_seeds, neg_seeds)

        ents = self.generate_ents(pos_seeds, neg_seeds, target_size)
        ents = self.rank(ents, pos_seeds, neg_seeds)
        return ents
