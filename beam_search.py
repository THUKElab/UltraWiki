import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from collections import namedtuple
from EntityTrie import EntityTrie



BeamSearchResult = namedtuple('BeamSearchResult', [
                              'output_ids', 'prod_probs', 'num_new_tokens'])


# set the probability of the tokens that are not in the next_tokens of prefix to zero
# this function is used in prefix-constrained beam search
def reset_invalid_probs(
    probs: torch.Tensor,
    prefix: torch.Tensor,
    entity_trie: EntityTrie,
    return_num_next_tokens: bool = False,
) -> tuple:
    next_tokens = entity_trie.get_next_tokens(prefix)
    vocab_size = probs.shape[-1]
    prob_mask = torch.ones(vocab_size, dtype=torch.bool, device=probs.device)
    prob_mask[next_tokens] = False
    probs[prob_mask] = 0.0
    res = [probs / probs.sum()]
    if return_num_next_tokens:
        res.append(len(next_tokens))
    return tuple(res)


# if entity_trie is not None, it will execute a constrained beam search
@torch.inference_mode()
def beam_search(
    input_ids: list[int],
    num_beams: int,
    model: nn.Module,
    eos_token_id: int,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] =
        lambda prod_probs, length: prod_probs.pow(
            (((5 + length) / 6) ** 0.6).reciprocal()
    ),
    entity_trie: EntityTrie = None,
    max_new_tokens: int = 5,
    return_prod_probs: bool = False,
    return_num_new_tokens: bool = False,
    only_return_generated_tokens: bool = False,
    num_beams_return: int = 0,
    *args,
    **kwargs,
) -> BeamSearchResult:
    model.eval()
    device = next(model.parameters()).device
    num_prompt_tokens = len(input_ids)

    beams = torch.tensor(input_ids, device=device).unsqueeze(0)
    num_new_tokens = torch.zeros(num_beams, dtype=torch.int, device=device)
    # prod_probs[i] denotes the product of probabilities of the ith beam.
    prod_probs = torch.ones(num_beams, dtype=torch.float, device=device)
    # if eos_unreached[beam_id] == False,
    #   the beam should be ignored when calculating scores and updating length
    eos_unreached = torch.ones(num_beams, dtype=torch.bool, device=device)

    for i in range(max_new_tokens):
        logits = model.forward(beams).logits[:, -1, :].to(torch.float32)
        vocab_size = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)  # (k,V):cuda

        # each beam expands to form many new beams
        # each new beam formed can be obtained from
        #   the id of its original beam that forms it and the id of the new token
        # So we define new_beams_id and new_tokens_id, their shapes are both (num_beams)
        # new_beams_id[i] denotes which previous beam expand to obtain current ith beam
        # new_tokens_id[i] denotes the last token of current ith beam

        if i == 0:  # only one beam needs to expand in round one
            if entity_trie is not None:
                probs[0], num_next_tokens = reset_invalid_probs(
                    probs=probs[0],
                    prefix=beams[0, num_prompt_tokens:],
                    entity_trie=entity_trie,
                    return_num_next_tokens=True,
                )
                if num_next_tokens < num_beams:
                    raise RuntimeError(
                        f"The first generation is not enough to match {num_beams} of tokens"
                    )

            probs = probs.flatten()  # (1*V):cuda
            num_new_tokens += 1
            _, indices = probs.topk(num_beams)
            new_beams_id = torch.zeros(
                num_beams, dtype=torch.long, device=device
            )  # (k):cuda
            new_tokens_id = indices  # (V):cuda
        else:
            if entity_trie is not None:
                for beam_id in torch.arange(num_beams)[eos_unreached.cpu()]:
                    probs[beam_id] = reset_invalid_probs(
                        probs=probs[beam_id],
                        prefix=beams[beam_id, num_prompt_tokens:],
                        entity_trie=entity_trie,
                    )[0]

            # formed_prod_probs denotes the cumulative probability of the formed beams in expansion
            # formed_eos_unreached is used to filter those formed beams whose prod_probs need to be calculated
            probs = probs.flatten()  # (k*V):cuda
            num_new_tokens[eos_unreached] += 1
            formed_prod_probs = prod_probs.repeat_interleave(
                vocab_size)  # (k*V):cuda
            formed_eos_unreached = eos_unreached.repeat_interleave(
                vocab_size
            )  # (k*V):cuda
            formed_prod_probs[formed_eos_unreached] *= probs[formed_eos_unreached]

            # calculate the score for each formed beams in expansion
            scores = score_fn(
                formed_prod_probs, num_new_tokens.repeat_interleave(vocab_size)
            )  # (k*V):cuda

            # construct candidate beams table
            #   (including those original beams that have reached eos and those formed beams expanded by the original beams that haven't reached eos)
            # for those original beams that have reached eos, only one item needed to be added in candidate beams table
            # beams_id_unreach_eos is the id of the original beams that haven't reached eos
            beams_id_unreach_eos = torch.arange(num_beams, device=device)[
                eos_unreached
            ]  # (k'):cuda
            # num_unreach_eos is the number of original beams that heven't reached eos, denoted as k'
            num_unreach_eos = beams_id_unreach_eos.shape[0]
            # candidate_beams_id_unreach_eos is the beam_id of the formed beams
            #   expanded by those original beams that have not yet reached eos
            candidate_beams_id_unreach_eos = beams_id_unreach_eos.repeat_interleave(
                vocab_size
            )  # (k'*V):cuda
            # candidate_tokens_id_unreach_eos is the token_id of the formed beams
            #   expanded by those original beams that have not yet reached eos
            candidate_tokens_id_unreach_eos = torch.arange(
                vocab_size, device=device
            ).repeat(
                num_unreach_eos
            )  # (k'*V):cuda
            candidate_beams_unreach_eos = torch.hstack(
                [
                    candidate_beams_id_unreach_eos.reshape(-1, 1),
                    candidate_tokens_id_unreach_eos.reshape(-1, 1),
                ]
            )
            candidate_scores_unreach_eos = scores[formed_eos_unreached]
            # now, candidate_beams are only those formed beams expanded by the original beams that haven't reached eos
            # the original beams that have reached eos should also be added in it

            # candidate_beams_reach_eos are the original beams that have reached eos
            candidate_beams_reach_eos, candidate_scores_reach_eos = [], []
            for beam_id in torch.arange(num_beams)[(~eos_unreached).cpu()]:
                candidate_beams_reach_eos.append([beam_id, 0])
                candidate_scores_reach_eos.append(scores[beam_id * vocab_size])
            candidate_beams_reach_eos = torch.tensor(
                candidate_beams_reach_eos, dtype=torch.long, device=device
            )
            candidate_scores_reach_eos = torch.tensor(
                candidate_scores_reach_eos, dtype=torch.float, device=device
            )
            candidate_beams = torch.cat(
                [candidate_beams_unreach_eos, candidate_beams_reach_eos], dim=0
            )
            candidate_scores = torch.cat(
                [candidate_scores_unreach_eos, candidate_scores_reach_eos], dim=0
            )

            # select k best beams
            _, indices = candidate_scores.topk(num_beams)
            candidate_beams = candidate_beams[indices]
            new_beams_id = candidate_beams[:, 0]
            new_tokens_id = candidate_beams[:, 1]

        # update beams
        beams = torch.hstack([beams[new_beams_id], new_tokens_id[:, None]])
        num_new_tokens = num_new_tokens[new_beams_id]
        # update prod_probs (cumulative multiplication)
        new_tokens_probs = probs[new_beams_id * vocab_size + new_tokens_id]
        new_tokens_probs[~eos_unreached[new_beams_id]] = 1.0
        prod_probs = prod_probs[new_beams_id] * new_tokens_probs
        # once the new token of a beam is eos, set the corresponding eos_unreached state to False
        eos_unreached = eos_unreached[new_beams_id] & (
            new_tokens_id != eos_token_id)

        # if all beams reach eos, it is impossible to expand
        if not eos_unreached.any():
            break

    # fill the invalid tokens with eos
    length = num_prompt_tokens + num_new_tokens
    mask = (
        torch.arange(
            beams.shape[-1], device=beams.device)[None, :] >= length[:, None]
    )
    beams[mask] = eos_token_id

    output_ids = beams[:,
                       num_prompt_tokens:] if only_return_generated_tokens else beams
    if num_beams_return != 0:
        output_ids = output_ids[:num_beams_return]
    if not return_prod_probs:
        prod_probs = None
    if not return_num_new_tokens:
        num_new_tokens = None
    return BeamSearchResult(output_ids, prod_probs, num_new_tokens)

