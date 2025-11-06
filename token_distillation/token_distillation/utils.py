import random
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

SPIECE_WHITESPACE = "▁"
GPT_BPE_WHITESPACE = "Ġ"


def seed_everything(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_new_phrase_tokenized_ids(new_phrase, tokenizer: PreTrainedTokenizer, tokenizer_path="") -> torch.Tensor:
    """
    It is sometimes difficult to reliably tokenize a span of text that doesn't start with a whitespace.
    `add_prefix_space=False` is supposed to handle this, but it didn't work as expected in my tests.
    We use the following strategy to reliably split a span of text (e.g. `new_phrase`) into tokens and correctly handle the prefix whitespace:
    (1) We add the BOS token to the beginning of the text.
    (2) We tokenize the text with the tokenizer.
    (3) We remove the BOS token id from the output.
    """
    whitespace_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(f"{tokenizer.bos_token} ", add_special_tokens=False))[1]
    # print(f"Whitespace token for tokenizer at '{tokenizer_path}': '{whitespace_token}'")
    new_phrase_starts_with_whitespace = new_phrase.startswith(" ") or new_phrase.startswith(whitespace_token)

    new_phrase_w_bos = f"{tokenizer.bos_token}{new_phrase}"
    new_phrase_tokenized_ids = tokenizer.encode(new_phrase_w_bos, return_tensors="pt", add_special_tokens=False)[0][1:]

    # sanity checks
    # print(
    #     f"New phrase: '{new_phrase}' | Tokenized IDs: {new_phrase_tokenized_ids.tolist()} | Tokens: {tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids.tolist())}"
    # )
    assert new_phrase_tokenized_ids[0] != tokenizer.bos_token_id
    if new_phrase_starts_with_whitespace:
        assert tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)
    else:
        assert not tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)

    return new_phrase_tokenized_ids


def generate_samples_with_patterns(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokens_to_pattern: dict[str, Iterable[int]],
    num_samples_per_pattern: int,
    seed=42,
    max_length=50,
):
    seed_everything(seed)

    max_length = max_length + 1  # +1 for the BOS token

    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is None:
        # raise ValueError("Tokenizer does not have a BOS token. Please use a tokenizer that supports BOS tokens.")
        print(
            "[generate_samples_with_patterns] Warning: Tokenizer does not have a BOS token. Proceeding generation without it."
        )
    token_to_samples = {}

    for token, pattern in tqdm(tokens_to_pattern.items(), desc="Generating samples", total=len(tokens_to_pattern)):
        input = torch.tensor(list(pattern), device=model.device).unsqueeze(0)
        if tokenizer.bos_token_id is not None:
            input = torch.tensor([tokenizer.bos_token_id] + list(pattern), device=model.device).unsqueeze(0)

        token_to_samples[token] = model.generate(
            input,
            do_sample=True,
            max_new_tokens=max_length,
            # pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            num_return_sequences=num_samples_per_pattern,
            # doesn't work unfortunately
            # constraints=[PhrasalConstraint(token_ids=[ele.item() for ele in pattern])],
            # custom_generate="transformers-community/constrained-beam-search",
            # trust_remote_code=True,
            # num_beams=8,
        ).to("cpu")
        # print up to 10 sample sequences for this token
        max_to_print = min(10, token_to_samples[token].size(0))

        # remove/truncate at the first EOS token in each generated sequence
        if tokenizer.eos_token_id is not None:
            filtered_seqs = []
            for seq in token_to_samples[token]:
                seq_tensor = seq
                if tokenizer.eos_token_id is not None:
                    eos_idx = (seq_tensor == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_idx.numel() > 0:
                        seq_tensor = seq_tensor[: eos_idx[0]]
                filtered_seqs.append(seq_tensor)
            token_to_samples[token] = filtered_seqs
        print(f"\nToken '{token}' (pattern: {tokenizer.decode(pattern, skip_special_tokens=True)}) - sample sequences:")
        for i in range(max_to_print):
            sample_sequence = token_to_samples[token][i]
            print(f"[{i + 1}/{max_to_print}] Generated text: {tokenizer.decode(sample_sequence, skip_special_tokens=False)}")
        print("-----------\n\n")
    return token_to_samples
