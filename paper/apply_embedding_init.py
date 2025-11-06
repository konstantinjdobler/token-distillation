import copy
import json
import os
import time
from functools import partial
from typing import Literal

import clm
import torch
from ahocorasick import collect_snippets_with_patterns_from_dataset, map_int_seq_to_str
from datasets import Dataset, load_dataset
from fire import Fire
from token_distillation_utils import (
    GPT_BPE_WHITESPACE,
    SPIECE_WHITESPACE,
    generate_samples_with_patterns,
    get_new_phrase_tokenized_ids,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel

import token_distillation

DATASET_ROOT = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")) + "/tokenized_datasets"


def tokenize_dataset(tokenizer, dataset_path, tokenizer_repr, batch_size=16_000, name=None):
    """
    Tokenize and cache a dataset for efficient reuse.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        dataset_path (str): Path or name of the dataset to load.
        tokenizer_repr (str): String representation of the tokenizer for cache naming.
        batch_size (int): Batch size for tokenization. Defaults to 16,000.
        name (str, optional): Dataset subset name. Defaults to None.

    Returns:
        None. Saves tokenized dataset to disk.
    """
    if os.path.exists(f"{DATASET_ROOT}/tokenized_dataset_{dataset_path}_{tokenizer_repr}/"):
        return
    if dataset_path == "uonlp/CulturaX":
        dataset = load_dataset("uonlp/CulturaX", name or "de", split="train", streaming=True)
    elif dataset_path == "ncbi/pubmed":
        dataset = load_dataset(dataset_path, split="train", streaming=True, revision="refs/pr/17", trust_remote_code=True)
        dataset = dataset.map(
            lambda x: {"text": x["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]},
            batched=False,
            remove_columns=dataset.column_names,
        )
    else:
        dataset = load_dataset(dataset_path, name=name, split="train", streaming=True)

    tokenized_dataset = dataset.map(
        lambda x: {"tokens": tokenizer(x["text"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]},
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    # in the paper, we only use up to 768,000 samples
    MAX_INDS = 1_000_000
    ds = Dataset.from_generator(
        partial(gen_from_iterable_dataset, tokenized_dataset.take(MAX_INDS)),
        features=tokenized_dataset.features,
    )

    ds.save_to_disk(f"{DATASET_ROOT}/tokenized_dataset_{dataset_path}_{tokenizer_repr}/")


def main(
    model_path: str = "mistralai/Mistral-7B-v0.1",
    new_tokens_source: str = None,
    build_target_method: Literal["tokenizer", "ngrams", "ner", "wordlist"] = "tokenizer",
    out_path: str = "./new_model",
    init_method: str = "token_distillation",
    dataset_path: str = "ncbi/pubmed",
    dataset_name: str | None = None,
    loss_method: list[str] = ["MSE-on-hiddens"],
    token_distillation_lr: float = 1e-4,
    token_distillation_epochs: int = 1,
    token_distillation_batch_size: int = 16,
    snippet_len: int = 50,
    snippet_num: int = 25,
    stopping_condition: str = "num_docs:760_000",
    set_output_to_zero_if_untied: bool = True,
    seed=42,
    target_layer=-1,
    use_generated_snippets: bool = False,
    filter_by_dataset_occurrence: bool = True,
    mixed_precision: bool = True,
):
    """
    Main function for initializing embeddings for new tokens in a pretrained language model.

    This function implements various initialization methods including token distillation,
    subtoken mean, random initialization, and ZeTT.

    Args:
        model_path (str): Path or name of the pretrained model. Defaults to "mistralai/Mistral-7B-v0.1".
        new_tokens_source (str, optional): Path to file containing new tokens. Defaults to None.
        build_target_method (Literal): Method for building target vocabulary. Defaults to "tokenizer".
        out_path (str): Output path for the extended model. Defaults to "./new_model".
        init_method (str): Initialization method to use. Defaults to "token_distillation".
        dataset_path (str): Path to dataset for snippet collection. Defaults to "ncbi/pubmed".
        dataset_name (str, optional): Dataset subset name. Defaults to None.
        loss_method (list[str]): Loss functions to use for training. Defaults to ["MSE-on-hiddens"].
        token_distillation_lr (float): Learning rate for token distillation. Defaults to 1e-4.
        token_distillation_epochs (int): Number of training epochs. Defaults to 1.
        token_distillation_batch_size (int): Batch size for training. Defaults to 16.
        snippet_len (int): Length of snippets for training. Defaults to 50.
        snippet_num (int): Number of snippets per token. Defaults to 25.
        stopping_condition (str): Condition for stopping snippet collection. Defaults to "num_docs:760_000".
        set_output_to_zero_if_untied (bool): Whether to zero output embeddings if not tied. Defaults to True.
        seed (int): Random seed for reproducibility. Defaults to 42.
        target_layer (int): Target layer for hidden state distillation. Defaults to -1.
        use_generated_snippets (bool): Whether to use generated snippets. Defaults to False.
        filter_by_dataset_occurrence (bool): Whether to filter tokens by dataset occurrence. Defaults to True.
        mixed_precision (bool): Whether to use mixed precision training. Defaults to True.

    Returns:
        None. Saves the extended model to the specified output path.
    """  
    if init_method == "zett-post":
        # guard against running whole script on unsupported models for config
        from zett.transfer import valid_model_for_zett_transfer

        if not valid_model_for_zett_transfer(model_path):
            print(f"Invalid model for zett transfer: {model_path}")
            return
    print(f"loss_method: {loss_method}")
    # we need to use these kwargs - especially `add_prefix_space=False` to get the correct tokenization of single tokens from target tokenizer
    source_tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_prefix_space=False)
    SOURCE_TOKENIZER_WHITESPACE = (
        SPIECE_WHITESPACE if source_tokenizer.get_vocab().get(SPIECE_WHITESPACE) is not None else GPT_BPE_WHITESPACE
    )
    tokenizer_repr = model_path.split("/")[-1]
    print(
        f"Detected whitespace token: {SOURCE_TOKENIZER_WHITESPACE} for source tokenizer {tokenizer_repr} from model {model_path}."
    )

    if build_target_method == "ner":
        # originally we used NER but in the paper we use a simpler method to construct new tokens
        COUNT_CUTOFF = 5
        with open(new_tokens_source, "r") as f:
            ners = [json.loads(line) for line in f]

        # apply filters against noise
        ners = [ner for ner in ners if ner["count"] >= COUNT_CUTOFF]
        ners = [ner for ner in ners if not any(c.isdigit() for c in ner["entity"])]  # skip all numbers
        ners = [ner for ner in ners if not any(c in "[]{}()<>.,;:!?@#$%^&*+_=|\\/" for c in ner["entity"])]

        ners = list(sorted(ners, key=lambda x: x["count"], reverse=True))
        target_vocab = [" " + ner["entity"] if not ner["entity"].startswith(" ") else ner["entity"] for ner in ners]
        target_tokenizer = copy.deepcopy(source_tokenizer)
        tgt_vocab = target_tokenizer.get_vocab()
        print(f"potential tokens: {len(target_vocab)}")
        target_vocab = [ne for ne in target_vocab if ne.replace(" ", SOURCE_TOKENIZER_WHITESPACE) not in tgt_vocab]
        print(f"filtered tokens: {len(target_vocab)}")
        target_tokenizer.add_tokens(target_vocab)
    elif build_target_method == "wordlist":
        words = []
        with open(new_tokens_source, "r") as f:
            words_by_category = json.load(f)
        for category, words_list in words_by_category.items():
            words.extend(words_list)
        target_tokenizer = copy.deepcopy(source_tokenizer)
        tgt_vocab = target_tokenizer.get_vocab()
        print(f"potential tokens: {len(words)}", words)
        words = [w for w in words if w.replace(" ", SOURCE_TOKENIZER_WHITESPACE) not in tgt_vocab]
        print(f"filtered tokens: {len(words)}", words)
        target_tokenizer.add_tokens(words)
    else:
        raise ValueError("Invalid build_target_method")

    model: PreTrainedModel = (AutoModelForMaskedLM if "bert" in model_path.lower() else AutoModelForCausalLM).from_pretrained(
        model_path, device_map="cuda:0", attn_implementation="sdpa"
    )
    model.eval()
    print(model)

    new_vocab = {}
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = sorted(target_tokenizer.get_vocab().items())
    todo_tokens = []
    for token, token_id in tqdm(target_vocab):
        # do some normalization around whitespace representation
        token = token.replace(GPT_BPE_WHITESPACE, SOURCE_TOKENIZER_WHITESPACE).replace(
            SPIECE_WHITESPACE, SOURCE_TOKENIZER_WHITESPACE
        )
        if source_vocab.get(token.replace(" ", SOURCE_TOKENIZER_WHITESPACE)) is not None:
            new_vocab[token_id] = model.get_input_embeddings().weight[
                source_vocab[token.replace(" ", SOURCE_TOKENIZER_WHITESPACE)]
            ]
        elif token in target_tokenizer.all_special_tokens:
            continue
        elif token.startswith("<0x") and token.endswith(">"):
            # skip special byte-level tokens in llama-style tokenizer
            continue
        else:
            todo_tokens.append((token, token_id))
    print(f"Found {len(new_vocab)} tokens in source vocab")
    print(f"Found {len(todo_tokens)} tokens in target vocab that need processing")

    if use_generated_snippets and not filter_by_dataset_occurrence:
        new_phrases_ids = []
        new_phrases_tokens = []

        for token, token_id in todo_tokens:
            new_phrases_ids.append(torch.tensor(get_new_phrase_tokenized_ids(token, source_tokenizer, model_path)))
            new_phrases_tokens.append(token)
        stopping_cond = "none"  # used for cache path later
        SNIPPET_LEN = snippet_len
        NUM_SNIPPETS_PER_TOKEN = snippet_num
        dataset_path = "wordlist" + new_tokens_source.replace(".json", "").split("/")[-1]
    else:
        source_tokenizer_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        patterns_ids = [get_new_phrase_tokenized_ids(t[0], source_tokenizer, model_path) for t in todo_tokens]
        stopping_cond = stopping_condition
        tokenize_dataset(
            source_tokenizer_fast,
            dataset_path=dataset_path,
            tokenizer_repr=tokenizer_repr,
            batch_size=16_000,
            name=dataset_name,
        )
        dataset = Dataset.load_from_disk(f"{DATASET_ROOT}/tokenized_dataset_{dataset_path}_{tokenizer_repr}/")
        collected_snippets = collect_snippets_with_patterns_from_dataset(
            patterns_ids, source_tokenizer_fast, dataset, stopping_condition=stopping_cond
        )
        print("Collected snippets for all tokens")

        SNIPPET_LEN = snippet_len
        NUM_SNIPPETS_PER_TOKEN = snippet_num
        MAX_PHRASES = 1_000_000
        new_phrases_ids = []
        new_phrases_tokens = []
        new_phrases_snippets_ids = []
        new_phrases_token_ids_in_target_vocab = []
        todo_tokens_not_enough_snippets = []
        """
        Filter out tokens that do not have enough snippets of the desired length.
        """
        for token, token_id in tqdm(
            sorted(
                todo_tokens,
                key=lambda x: len(
                    collected_snippets.get(
                        map_int_seq_to_str(get_new_phrase_tokenized_ids(x[0], source_tokenizer, model_path)), []
                    )
                ),
            )
        ):
            # do `list(...)` s.t. we don't have a tensor that might get modified in-place by `hash_int_seq`
            og_tokens_for_token = list(get_new_phrase_tokenized_ids(token, source_tokenizer, model_path))
            snippets = collected_snippets.get(map_int_seq_to_str(og_tokens_for_token), [])
            if len(snippets) < NUM_SNIPPETS_PER_TOKEN:
                todo_tokens_not_enough_snippets.append((token, token_id))
                continue

            # truncate snippets to window around target token
            truncated_snippets = []
            for i, snippet in enumerate(snippets):
                snippet_tokens = snippet[0]
                pattern_start_idx_in_snippet = snippet[1]
                pattern_len = len(og_tokens_for_token)
                if pattern_len > SNIPPET_LEN:
                    snippets[i] = []  # skip this snippet
                    continue
                num_non_pattern_tokens_in_snippet = SNIPPET_LEN - pattern_len

                max_possible_buffer_before = pattern_start_idx_in_snippet
                max_possible_buffer_after = len(snippet_tokens) - pattern_start_idx_in_snippet - pattern_len

                buffer_before = max(0, num_non_pattern_tokens_in_snippet // 2)
                buffer_after = num_non_pattern_tokens_in_snippet - buffer_before
                if buffer_after > max_possible_buffer_after:
                    buffer_before = num_non_pattern_tokens_in_snippet - max_possible_buffer_after
                    buffer_after = max_possible_buffer_after
                if buffer_before > max_possible_buffer_before:
                    buffer_after = num_non_pattern_tokens_in_snippet - max_possible_buffer_before
                    buffer_before = max_possible_buffer_before

                truncated_snippets.append(
                    snippet_tokens[
                        pattern_start_idx_in_snippet - buffer_before : pattern_start_idx_in_snippet + pattern_len + buffer_after
                    ]
                )

            snippets = truncated_snippets

            BOS = source_tokenizer.bos_token_id if source_tokenizer.bos_token_id is not None else source_tokenizer.cls_token_id

            snippets = [[BOS] + s[:SNIPPET_LEN] for s in snippets if len(s) >= SNIPPET_LEN][:NUM_SNIPPETS_PER_TOKEN]
            snippets = [{"input_ids": torch.tensor(s).unsqueeze(0)} for s in snippets]
            if len(snippets) < NUM_SNIPPETS_PER_TOKEN:
                todo_tokens_not_enough_snippets.append((token, token_id))
                continue
            if MAX_PHRASES != -1 and len(new_phrases_ids) >= MAX_PHRASES:
                todo_tokens_not_enough_snippets.append((token, token_id))
                continue
            new_phrases_ids.append(torch.tensor(og_tokens_for_token))
            new_phrases_tokens.append(token)
            new_phrases_snippets_ids.append([s["input_ids"].squeeze(0) for s in snippets])
            new_phrases_token_ids_in_target_vocab.append(token_id)
        print(f"Enough snippets for {len(new_phrases_ids)} new tokens")
        print(f"Not enough snippets for {len(todo_tokens_not_enough_snippets)} new tokens")

    if use_generated_snippets:
        # do here s.t. we use the exact same tokens as when we would retrieve from a corpus when `filter_by_dataset_occurrence=True`
        # when generating, we can of course also generate for all tokens (`filter_by_dataset_occurrence=False`) but this is for fair comparison

        generated_snippet_cache_path = (
            f"{DATASET_ROOT}/generated_snippets_{dataset_path}_{stopping_cond}_{tokenizer_repr}_{NUM_SNIPPETS_PER_TOKEN}.pkl"
        )
        t0 = time.perf_counter()
        if os.path.exists(generated_snippet_cache_path):
            with open(generated_snippet_cache_path, "r") as f:
                tokens_to_new_snippets = json.load(f)
            new_phrases_snippets_ids = [torch.tensor(tokens_to_new_snippets[t], device="cpu") for t in new_phrases_tokens]
            print(f"Loaded generated snippets from {generated_snippet_cache_path} in {time.perf_counter() - t0:.2f}s")
        else:
            tokens_to_pattern = {t: ids for t, ids in zip(new_phrases_tokens, new_phrases_ids)}
            tokens_to_new_snippets = generate_samples_with_patterns(
                model,
                source_tokenizer,
                tokens_to_pattern,
                num_samples_per_pattern=NUM_SNIPPETS_PER_TOKEN,
                seed=seed,
                max_length=SNIPPET_LEN,
            )
            new_phrases_snippets_ids = [tokens_to_new_snippets[t] for t in new_phrases_tokens]

            os.makedirs(os.path.dirname(generated_snippet_cache_path), exist_ok=True)
            # Save the generated snippets to a cache file for reuse
            with open(generated_snippet_cache_path, "w") as f:
                # Convert tensors to lists for JSON serialization
                serializable_snippets = {t: [s.tolist() for s in snips] for t, snips in tokens_to_new_snippets.items()}
                json.dump(serializable_snippets, f, ensure_ascii=False, indent=4)
            print(f"Saved generated snippets to {generated_snippet_cache_path} in {time.perf_counter() - t0:.2f}s")

    if init_method == "zett-post":
        from zett.transfer import zett_transfer

        # ZeTT expects GPT_BPE_WHITESPACE
        input_embs, output_embs = zett_transfer(
            [w.replace(" ", GPT_BPE_WHITESPACE) for w in new_phrases_tokens],
            model,
        )
        if input_embs is None or output_embs is None:
            print("invalid model for zett transfer")
            return

        new_tokens_to_emb = {}
        for token, new_embedding in zip(new_phrases_tokens, input_embs):
            new_tokens_to_emb[token] = new_embedding

        new_tokens_to_outputembs = {}
        for token, new_embedding in zip(new_phrases_tokens, output_embs):
            new_tokens_to_outputembs[token] = new_embedding

        if model.config.tie_word_embeddings:
            new_tokens_to_outputembs = None

        extend_pretrained_with_tokens_and_embeddings(
            out_path,
            model,
            new_tokens_to_emb,
            source_tokenizer,
            new_tokens_to_outputembs=new_tokens_to_outputembs,
            set_output_to_zero_if_untied=False,
        )
        return

    if init_method == "subtoken-mean":
        new_tokens = {}
        new_tokens_out = {}
        for token in new_phrases_tokens:
            embs = model.get_input_embeddings()
            token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(embs.weight.device)

            new_tokens[token] = torch.mean(embs(token_ids), dim=0)

            out_embs = model.get_output_embeddings()
            token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(out_embs.weight.device)

            new_tokens_out[token] = torch.mean(out_embs.weight.data[token_ids], dim=0)
        if set_output_to_zero_if_untied or model.config.tie_word_embeddings:
            new_tokens_out = None
        extend_pretrained_with_tokens_and_embeddings(
            out_path,
            model,
            new_tokens,
            source_tokenizer,
            set_output_to_zero_if_untied=set_output_to_zero_if_untied,
            new_tokens_to_outputembs=new_tokens_out,
        )
        return

    if init_method == "random":
        """Handle tokens that do not have enough snippets."""
        new_tokens = {}
        gen = torch.Generator(device="cuda:0").manual_seed(seed)
        embs = model.get_input_embeddings()
        embs_mean = torch.mean(embs.weight, dim=0)
        embs_std = torch.std(embs.weight, dim=0)
        for token in new_phrases_tokens:
            token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(embs.weight.device)
            new_tokens[token] = torch.normal(embs_mean, embs_std, generator=gen).to(embs.weight.device).to(embs.weight.dtype)
        extend_pretrained_with_tokens_and_embeddings(
            out_path,
            model,
            new_tokens,
            source_tokenizer,
            set_output_to_zero_if_untied=set_output_to_zero_if_untied,
            new_tokens_to_outputembs=new_tokens if not model.config.tie_word_embeddings else None,
        )
        return

    if (
        init_method == "fvt-clm"
        or init_method == "fvt-clm-no-og"
        or init_method == "fvt-clm-pp"
        or init_method == "fvt-clm-no-og-pp"
    ):
        """Handle tokens that do not have enough snippets."""
        print("FVT init with further CLM training")
        new_tokens = {}
        new_tokens_out = {}
        for token in new_phrases_tokens:
            embs = model.get_input_embeddings()
            token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(embs.weight.device)
            new_tokens[token] = torch.mean(embs(token_ids), dim=0)

            out_embs = model.get_output_embeddings()
            token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(out_embs.weight.device)
            new_tokens_out[token] = torch.mean(out_embs.weight.data[token_ids], dim=0)
        if init_method == "fvt-clm" or init_method == "fvt-clm-no-og" or model.config.tie_word_embeddings:
            new_tokens_out = None
        extend_pretrained_with_tokens_and_embeddings(
            out_path,
            model,
            new_tokens,
            source_tokenizer,
            set_output_to_zero_if_untied=init_method in ["fvt-clm", "fvt-clm-no-og"],
            new_tokens_to_outputembs=new_tokens_out,
        )
        del model
        model = AutoModelForCausalLM.from_pretrained(out_path, device_map="cuda:0", attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(out_path)
        model = clm.train_embeddings(
            model,
            new_phrases_snippets_ids,
            {phrase_ids: i + len(source_tokenizer) for i, phrase_ids in enumerate(new_phrases_ids)},
            tokenizer,
            epochs=token_distillation_epochs,
            batch_size=token_distillation_batch_size,
            learning_rate=token_distillation_lr,
            preserve_og_embs={"fvt-clm": True, "fvt-clm-no-og": False, "fvt-clm-pp": True, "fvt-clm-no-og-pp": False}[
                init_method
            ],
            seed=seed,
            mixed_precision=mixed_precision,
        )

        if not model.config.tie_word_embeddings and set_output_to_zero_if_untied:
            # when not tieing keys, set to 0 as we have not learned embs for it.
            print("Not tying keys, setting output embeddings to 0.")
            model.get_output_embeddings().weight.data[len(source_tokenizer) :] = torch.zeros_like(
                model.get_output_embeddings().weight.data[len(source_tokenizer) :]
            )
        model.save_pretrained(out_path, safe_serialization=False)
        return
    if (
        init_method == "token_distillation"
        or init_method == "token_distillation-randinit"
        or init_method == "token_distillation-zett"
    ):
        """Handle tokens that do not have enough snippets."""
        print("Using token_distillation!")

        if init_method != "token_distillation-zett":
            new_tokens = {}
            if init_method == "token_distillation-randinit":
                gen = torch.Generator(device="cuda:0").manual_seed(seed)
                embs = model.get_input_embeddings()
                embs_mean = torch.mean(embs.weight, dim=0)
                embs_std = torch.std(embs.weight, dim=0)
                print(f"Embedding mean shape: {embs_mean.shape}, std shape: {embs_std.shape}")
            for token in new_phrases_tokens:
                embs = model.get_input_embeddings()
                if init_method == "token_distillation":
                    token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer, model_path).to(embs.weight.device)
                    new_tokens[token] = torch.mean(embs(token_ids), dim=0)
                elif init_method == "token_distillation-randinit":
                    new_tokens[token] = torch.normal(embs_mean, embs_std, generator=gen).to(embs.weight.device)

            t0 = time.perf_counter()
            extend_pretrained_with_tokens_and_embeddings(
                out_path, model, new_tokens, source_tokenizer, set_output_to_zero_if_untied=True
            )
            print(f"Saved model to {out_path} in {time.perf_counter() - t0:.2f}s")
        del model

        t0 = time.perf_counter()
        model_load_path = out_path
        if init_method == "token_distillation-zett":
            # load ZeTT-inited model as starting point
            model_load_path = out_path.split("-MEDICAL-")[0] + "-MEDICAL-zett-precursor"
        model = AutoModelForCausalLM.from_pretrained(model_load_path, device_map="cuda:0", attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        print(f"Loaded model in {time.perf_counter() - t0:.2f}s")
        model = token_distillation.train_embeddings(
            model,
            new_phrases_snippets_ids,
            {phrase_ids: i + len(source_tokenizer) for i, phrase_ids in enumerate(new_phrases_ids)},
            tokenizer,
            epochs=token_distillation_epochs,
            batch_size=token_distillation_batch_size,
            loss_methods=loss_method,
            learning_rate=token_distillation_lr,
            seed=seed,
            target_layer=target_layer,
            mixed_precision=mixed_precision,
        )

        if not model.config.tie_word_embeddings and set_output_to_zero_if_untied:
            # when not tieing keys, set to 0 as we have not learned embs for it.
            print("Not tying keys, setting output embeddings to 0.")
            model.get_output_embeddings().weight.data[len(source_tokenizer) :] = torch.zeros_like(
                model.get_output_embeddings().weight.data[len(source_tokenizer) :]
            )

        model.save_pretrained(out_path, safe_serialization=False)
        tokenizer.save_pretrained(out_path)
        return
    raise NotImplementedError(f"init_method {init_method} not implemented.")


def extend_pretrained_with_tokens_and_embeddings(
    out_path: str,
    model: PreTrainedModel,
    new_tokens_to_embs: dict,
    source_tokenizer,
    set_output_to_zero_if_untied=False,
    new_tokens_to_outputembs: dict = None,
):
    """
    Extend a pretrained model's vocabulary with new tokens and their embeddings.

    Args:
        out_path (str): Path to save the extended model.
        model (PreTrainedModel): The model to extend.
        new_tokens_to_embs (dict): Mapping of new tokens to their input embeddings.
        source_tokenizer: The original tokenizer to extend.
        set_output_to_zero_if_untied (bool): Whether to zero output embeddings if not tied. Defaults to False.
        new_tokens_to_outputembs (dict, optional): Mapping of new tokens to their output embeddings. Defaults to None.

    Returns:
        None. Saves the extended model and tokenizer to disk.
    """
    if model.config.tie_word_embeddings:
        assert new_tokens_to_outputembs is None, "Cannot have new output embs when tying keys"

    target_tokenizer = copy.deepcopy(source_tokenizer)
    target_tokenizer.add_tokens(list(new_tokens_to_embs.keys()))

    model.resize_token_embeddings(len(target_tokenizer))
    new_embedding_weights = model.get_input_embeddings().weight.data
    for token, embedding in new_tokens_to_embs.items():
        token_id = target_tokenizer.convert_tokens_to_ids(token)
        new_embedding_weights[token_id] = embedding
    model.get_input_embeddings().weight.data = new_embedding_weights

    if new_tokens_to_outputembs is not None:
        new_output_embs = model.get_output_embeddings().weight.data
        for token, embedding in new_tokens_to_outputembs.items():
            token_id = target_tokenizer.convert_tokens_to_ids(token)
            new_output_embs[token_id] = embedding
        model.get_output_embeddings().weight.data = new_output_embs

    if not model.config.tie_word_embeddings:
        if set_output_to_zero_if_untied:
            # when not tying keys, set to 0 as we have not learned embs for it.
            print("Not tying keys, setting output embeddings to 0")
            model.get_output_embeddings().weight.data[len(source_tokenizer) :] = torch.zeros_like(
                model.get_output_embeddings().weight.data[len(source_tokenizer) :]
            )
        else:
            print("Not tying keys, leaving output embeddings as they are")
    else:
        model.tie_weights()
        assert torch.equal(model.get_input_embeddings().weight.data, model.get_output_embeddings().weight.data)

    model.save_pretrained(out_path, safe_serialization=False)
    target_tokenizer.save_pretrained(out_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    Fire(main)
