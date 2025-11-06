from __future__ import annotations

import copy
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, List, Literal, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .ahocorasick import collect_snippets_with_patterns_from_dataset, map_int_seq_to_str
from .train_loop import train_embeddings
from .utils import generate_samples_with_patterns, get_new_phrase_tokenized_ids

GPT_BPE_WHITESPACE = "Ġ"
SPIECE_WHITESPACE = "▁"
DATASET_ROOT = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")) + "/tokenized_datasets"


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------
@dataclass
class GeneratedDataSource:
    seed: int = 42


@dataclass
class HFDataSource:
    """Configuration for collecting snippets from an HF dataset."""

    dataset_path: str
    name: str | None = None
    split: str = "train"
    tokenization_batch_size: int = 16_000
    max_docs: int | None = 1_000_000  # None = all available from stream
    revision: str | None = None
    trust_remote_code: bool = False
    map_to_text_fn: Callable | None = None  # optional: customize mapping to a "text" field
    # Example for `map_to_text_fn` for `ncbi/pubmed`:
    # lambda x: {"text": x["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]}


class OutputEmbeddingInit(str, Enum):
    ZERO = "zero"
    SUBTOKEN_MEAN = "subtoken_mean"
    TRAIN_WITH_CE = "train_with_ce"


@dataclass
class DistillationConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    loss_methods: list[str] = ("MSE-on-hiddens",)
    seed: int = 42
    target_layer: int = -1
    mixed_precision: bool = True


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def detect_whitespace_token(source_tokenizer) -> str:
    """Detect which whitespace surrogate token the tokenizer uses."""
    return SPIECE_WHITESPACE if source_tokenizer.get_vocab().get(SPIECE_WHITESPACE) is not None else GPT_BPE_WHITESPACE


def extend_pretrained_with_tokens_and_embeddings(
    out_path: str,
    model: PreTrainedModel,
    new_tokens_to_input_embs: dict[str, torch.Tensor],
    source_tokenizer,
    new_tokens_to_output_embs: dict[str, torch.Tensor] | None = None,
    save: bool = True,
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """Extend a pretrained model's vocabulary and set new token weights.

    Accepts separate input/output maps and cleanly handles tied vs. untied heads.
    """
    if model.config.tie_word_embeddings:
        assert new_tokens_to_output_embs is None, "Cannot set output embeddings explicitly when tying weights."

    target_tokenizer = copy.deepcopy(source_tokenizer)
    target_tokenizer.add_tokens(list(new_tokens_to_input_embs.keys()))

    model.resize_token_embeddings(len(target_tokenizer))

    # Input embeddings
    in_w = model.get_input_embeddings().weight.data
    for token, emb in new_tokens_to_input_embs.items():
        token_id = target_tokenizer.convert_tokens_to_ids(token)
        in_w[token_id] = emb
    model.get_input_embeddings().weight.data = in_w

    # Output embeddings (if untied and provided)
    if new_tokens_to_output_embs is not None:
        out_w = model.get_output_embeddings().weight.data
        for token, emb in new_tokens_to_output_embs.items():
            token_id = target_tokenizer.convert_tokens_to_ids(token)
            out_w[token_id] = emb
        model.get_output_embeddings().weight.data = out_w

    if model.config.tie_word_embeddings:
        model.tie_weights()
        assert torch.equal(
            model.get_input_embeddings().weight.data,
            model.get_output_embeddings().weight.data,
        )
    if save:
        os.makedirs(out_path, exist_ok=True)
        model.save_pretrained(out_path, safe_serialization=False)
        target_tokenizer.save_pretrained(out_path)
    return model, target_tokenizer


# -----------------------------------------------------------------------------
# Snippet builders
# -----------------------------------------------------------------------------


def _cache_dir(dataset_path: str, name: str | None, tokenizer_repr: str, max_docs: int | None) -> str:
    name_part = name or "default"
    max_part = str(max_docs) if max_docs is not None else "all"
    return f"{DATASET_ROOT}/tokenized_{dataset_path}_{name_part}_{tokenizer_repr}_{max_part}/"


def _tokenize_dataset_if_needed(
    tokenizer_fast,
    dataset_cfg: HFDataSource,
    tokenizer_repr: str,
) -> str:
    cache_dir = _cache_dir(dataset_cfg.dataset_path, dataset_cfg.name, tokenizer_repr, dataset_cfg.max_docs)
    if os.path.exists(cache_dir):
        print(f"[tokdist] Using cached tokenized dataset at {cache_dir}")
        return cache_dir
    print(
        f"[tokdist] Tokenizing dataset {dataset_cfg.max_docs} samples from {dataset_cfg.dataset_path} (this may take a while)...\n  Caching to {cache_dir}"
    )

    ds = load_dataset(
        dataset_cfg.dataset_path,
        name=dataset_cfg.name,
        split=dataset_cfg.split,
        streaming=True,
        revision=dataset_cfg.revision,
        trust_remote_code=dataset_cfg.trust_remote_code,
    )

    # Optional mapping to a flat `text` column
    if dataset_cfg.map_to_text_fn is not None:
        ds = ds.map(dataset_cfg.map_to_text_fn, batched=False, remove_columns=ds.column_names)

    # Tokenize stream → small on-disk Dataset for reuse
    tok_stream = ds.map(
        lambda x: {"tokens": tokenizer_fast(x["text"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]},
        batched=True,
        batch_size=dataset_cfg.tokenization_batch_size,
        remove_columns=ds.column_names,
    )

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    take_n = dataset_cfg.max_docs if dataset_cfg.max_docs is not None else None
    if take_n is not None:
        tok_stream = tok_stream.take(take_n)

    ds_tok = Dataset.from_generator(
        partial(gen_from_iterable_dataset, tok_stream),
        features=tok_stream.features,
    )

    os.makedirs(cache_dir, exist_ok=True)
    ds_tok.save_to_disk(cache_dir)
    return cache_dir


def build_snippets_for_tokens_from_hf(
    model_path: str,
    source_tokenizer,
    todo_tokens: List[Tuple[str, int]],
    snippet_len: int,
    snippets_per_token: int,
    dataset_cfg: HFDataSource,
    tokenizer_repr: str,
):
    """Collect snippets via Aho-Corasick for each new token string.

    Returns: (new_phrases_ids, new_phrases_snippets_ids, skipped)
    """
    source_tokenizer_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    patterns_ids = [get_new_phrase_tokenized_ids(t[0], source_tokenizer, model_path) for t in todo_tokens]

    cache_dir = _tokenize_dataset_if_needed(source_tokenizer_fast, dataset_cfg, tokenizer_repr)
    dataset = Dataset.load_from_disk(cache_dir)

    collected = collect_snippets_with_patterns_from_dataset(
        patterns_ids,
        source_tokenizer_fast,
        dataset,
        max_docs=dataset_cfg.max_docs,
        offset_before=snippet_len * 2,
        offset_after=snippet_len * 2,
        max_necessary_samples=snippets_per_token * 2,  # collect extra to allow filtering
    )

    new_phrases_ids: dict[torch.Tensor] = {}
    new_phrases_snippets_ids: dict[list[torch.Tensor]] = {}
    skipped: list[Tuple[str, int, int]] = []
    token_id_to_token = {tid: t for t, tid in todo_tokens}

    bos = source_tokenizer.bos_token_id or source_tokenizer.cls_token_id
    bos_prefix = [bos] if bos is not None else []

    for token, token_id in tqdm(todo_tokens):
        pattern_ids = list(get_new_phrase_tokenized_ids(token, source_tokenizer, model_path))
        snippets = collected.get(map_int_seq_to_str(pattern_ids), [])
        available_raw = len(snippets)
        if available_raw < snippets_per_token:
            skipped.append((token, token_id, available_raw))
            continue

        # truncate to a window around the pattern
        truncated: list[list[int]] = []
        for snippet_tokens, pattern_start_idx in snippets:
            pat_len = len(pattern_ids)
            if pat_len > snippet_len:
                continue
            free = snippet_len - pat_len
            max_before = pattern_start_idx
            max_after = len(snippet_tokens) - pattern_start_idx - pat_len
            before = min(max_before, free // 2)
            after = min(max_after, free - before)
            start = pattern_start_idx - before
            end = pattern_start_idx + pat_len + after
            truncated.append(snippet_tokens[start:end])

        # shuffle to avoid bias if original dataset is sorted
        random.shuffle(truncated)
        seqs = [bos_prefix + s[:snippet_len] for s in truncated if len(s) >= snippet_len][:snippets_per_token]
        seqs = [torch.tensor(s) for s in seqs]
        available_final = len(seqs)
        if available_final < snippets_per_token:
            skipped.append((token, token_id, available_final))
            continue

        new_phrases_ids[token_id] = torch.tensor(pattern_ids)
        new_phrases_snippets_ids[token_id] = seqs

    # print ten example snippets for the first few tokens
    for token_id in list(new_phrases_snippets_ids.keys())[:5]:
        token = token_id_to_token[token_id]
        print(f"[tokdist] Example snippets for token {token!r}:")
        for snippet_ids in new_phrases_snippets_ids[token_id][:10]:
            decoded = source_tokenizer.decode(snippet_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print(f"           - {decoded!r}")

    # convert dicts to ordered lists
    new_phrases_ids_list: list[torch.Tensor] = []
    new_phrases_snippets_ids_list: list[list[torch.Tensor]] = []
    for token_id in new_phrases_ids.keys():
        new_phrases_ids_list.append(new_phrases_ids[token_id])
        new_phrases_snippets_ids_list.append(new_phrases_snippets_ids[token_id])

    return new_phrases_ids_list, new_phrases_snippets_ids_list, skipped


def build_snippets_for_tokens_generated(
    model,
    model_path: str,
    source_tokenizer,
    todo_tokens: List[Tuple[str, int]],
    snippet_len: int,
    snippets_per_token: int,
    seed: int,
    device: str = "cuda:0",
):
    """Generate snippets for each token using your constrained generation helper.

    Returns: (new_phrases_ids, new_phrases_snippets_ids)
    """
    prev_device = model.device
    model.to(device=device)
    new_phrases_ids: list[torch.Tensor] = []
    new_phrases_tokens: list[str] = []

    for token, _ in todo_tokens:
        new_phrases_ids.append(torch.tensor(get_new_phrase_tokenized_ids(token, source_tokenizer, model_path)))
        new_phrases_tokens.append(token)

    tokens_to_pattern = {t: ids for t, ids in zip(new_phrases_tokens, new_phrases_ids)}
    tokens_to_new_snippets = generate_samples_with_patterns(
        model,
        source_tokenizer,
        tokens_to_pattern,
        num_samples_per_pattern=snippets_per_token,
        seed=seed,
        max_length=snippet_len,
    )

    new_phrases_snippets_ids = [tokens_to_new_snippets[t] for t in new_phrases_tokens]
    model.to(device=prev_device)
    return new_phrases_ids, new_phrases_snippets_ids


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------


class TokenDistillation:
    """High-level orchestrator for adding and distilling new tokens."""

    def __init__(
        self,
        model_path: str,
        device: str | None = "cuda:0",
        attn_impl: str | None = "sdpa",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.attn_impl = attn_impl

        # load model + source tokenizer
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation=self.attn_impl,
        )
        self.model_has_tied_embeddings = self.model.config.tie_word_embeddings
        self.model.eval()
        self.source_tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_prefix_space=False)
        self._src_ws = detect_whitespace_token(self.source_tokenizer)
        self._tokenizer_repr = model_path.split("/")[-1]

    # ----------------------------- Public API ----------------------------- #
    def run(
        self,
        new_tokens: List[str] | "AutoTokenizer",
        out_path: str = None,
        data: HFDataSource | GeneratedDataSource = GeneratedDataSource(),
        snippet_len: int = 50,
        snippets_per_token: int = 100,
        output_emb_policy: OutputEmbeddingInit = OutputEmbeddingInit.TRAIN_WITH_CE,
        training: DistillationConfig = DistillationConfig(),
        pre_init_strategy: Literal["fvt", "adapti-vocab"] = "fvt",
        save: bool = False,
    ) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """Add tokens, build/generate snippets, distill, and save."""
        if out_path is None:
            if save:
                raise ValueError("out_path must be specified if save=True")
            out_path = tempfile.mkdtemp(prefix="tokdist_")
            print(f"[tokdist] Using temporary output path for intermediate files: {out_path}")

        todo_tokens = self._build_target_tokenizer(new_tokens)

        # 1) Build snippets for new tokens
        skipped: list[Tuple[str, int, int]] = []
        if isinstance(data, HFDataSource):
            new_phrases_ids, new_phrases_snippets_ids, skipped = build_snippets_for_tokens_from_hf(
                self.model_path,
                self.source_tokenizer,
                todo_tokens,
                snippet_len,
                snippets_per_token,
                data,
                tokenizer_repr=self._tokenizer_repr,
            )
        else:
            new_phrases_ids, new_phrases_snippets_ids = build_snippets_for_tokens_generated(
                self.model,
                self.model_path,
                self.source_tokenizer,
                todo_tokens,
                snippet_len,
                snippets_per_token,
                seed=data.seed,
                device=self.device,
            )

        if skipped:
            print(f"[tokdist] Skipped {len(skipped)} tokens due to insufficient snippets:")
            for token, _, available in skipped:
                print(f"           - {token!r}: collected {available}/{snippets_per_token} usable snippets")
            print(
                "[tokdist] Recommendations:"
                " (1) Increase `HFDataSource.max_docs`"
                " (2) Switch to a corpus with more in-domain coverage"
                " (3) Fall back to generated snippets"
                " (4) Relax skipping safeguards if few samples are acceptable"
            )
            skipped_ids = {token_id for _, token_id, _ in skipped}
            todo_tokens = [item for item in todo_tokens if item[1] not in skipped_ids]
            if not todo_tokens:
                raise RuntimeError("No tokens left to distill after filtering skipped tokens.")

        # 2) Pre-init embeddings for new tokens via subtoken mean
        input_preinit = self._compute_subtoken_means(todo_tokens, input_or_output="input", method=pre_init_strategy)
        output_preinit = None
        if not self.model_has_tied_embeddings:
            output_preinit = self._compute_subtoken_means(todo_tokens, input_or_output="output", method=pre_init_strategy)
        if self.model_has_tied_embeddings and output_emb_policy == OutputEmbeddingInit.TRAIN_WITH_CE:
            print(
                "[tokdist] Cannot use 'OutputEmbeddingInit.TRAIN_WITH_CE' output embedding policy to learn output embeddings separately when model has tied embeddings. Proceeding without it."
            )

        # 3) Save a temporary extended model so training can address new token ids
        t0 = time.perf_counter()
        extend_pretrained_with_tokens_and_embeddings(
            out_path,
            self.model,
            input_preinit,
            self.source_tokenizer,
            new_tokens_to_output_embs=output_preinit,
            save=True,
        )
        print(f"[tokdist] Saved temp-extended model to {out_path} in {time.perf_counter() - t0:.2f}s")

        # 4) Reload model/tokenizer from the extended path for training
        model = AutoModelForCausalLM.from_pretrained(
            out_path, device_map=self.device, attn_implementation=self.attn_impl, dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(out_path)

        # 5) Train
        phrase_to_new_id = {phrase_ids: i + len(self.source_tokenizer) for i, phrase_ids in enumerate(new_phrases_ids)}
        model = train_embeddings(
            model,
            new_phrases_snippets_ids,
            phrase_to_new_id,
            tokenizer,
            epochs=training.epochs,
            batch_size=training.batch_size,
            loss_methods=list(training.loss_methods),
            learning_rate=training.learning_rate,
            seed=training.seed,
            target_layer=training.target_layer,
            mixed_precision=training.mixed_precision,
            learn_output_with_ce=(output_emb_policy == OutputEmbeddingInit.TRAIN_WITH_CE)
            and not self.model_has_tied_embeddings,
        )

        # 6) Post-process output embeddings if untied
        if self.model_has_tied_embeddings:
            print("[tokdist] Output head is tied; no post-processing needed.")
        else:
            if output_emb_policy == OutputEmbeddingInit.ZERO:
                start = len(self.source_tokenizer)
                model.get_output_embeddings().weight.data[start:] = torch.zeros_like(
                    model.get_output_embeddings().weight.data[start:]
                )
                print("[tokdist] New output embeddings set to zero.")
            elif output_emb_policy == OutputEmbeddingInit.SUBTOKEN_MEAN:
                out_map = self._compute_subtoken_means(todo_tokens)
                out_w = model.get_output_embeddings().weight.data
                for tok, emb in out_map.items():
                    tok_id = tokenizer.convert_tokens_to_ids(tok)
                    out_w[tok_id] = emb
                model.get_output_embeddings().weight.data = out_w
                print("[tokdist] New output embeddings set to subtoken means.")
            elif output_emb_policy == OutputEmbeddingInit.TRAIN_WITH_CE:
                print("[tokdist] New output embeddings were trained separately with CE loss.")

        # # 7) Save final or remove temp
        if save:
            os.makedirs(out_path, exist_ok=True)
            model.save_pretrained(out_path, safe_serialization=False)
            tokenizer.save_pretrained(out_path)
            print(f"[tokdist] Final model saved to {out_path}")
        else:
            print(f"[tokdist] Removing temporary files from {out_path}")
            shutil.rmtree(out_path)
        return model, tokenizer

    # ----------------------------- Internals ----------------------------- #
    def _build_target_tokenizer(self, new_tokens: List[str] | "AutoTokenizer"):
        target_tokenizer = copy.deepcopy(self.source_tokenizer)
        src_vocab = self.source_tokenizer.get_vocab()

        if isinstance(new_tokens, list):
            candidate_tokens = [t for t in new_tokens if t]
        else:
            candidate_tokens = list(new_tokens.get_vocab().keys())

        filtered: list[str] = []
        seen: set[str] = set()
        for token in candidate_tokens:
            if token in seen:
                continue
            seen.add(token)
            token_norm = token.replace(GPT_BPE_WHITESPACE, self._src_ws).replace(SPIECE_WHITESPACE, self._src_ws)
            if src_vocab.get(token_norm.replace(" ", self._src_ws)) is not None:
                continue
            if token in target_tokenizer.all_special_tokens:
                continue
            if token.startswith("<0x") and token.endswith(">"):
                continue
            filtered.append(token)

        if filtered:
            target_tokenizer.add_tokens(filtered)
        todo_tokens = [(token, target_tokenizer.convert_tokens_to_ids(token)) for token in filtered]
        print(f"[tokdist] {len(todo_tokens)} tokens need initialization")
        return todo_tokens

    @torch.no_grad()
    def _compute_subtoken_means(
        self, todo_tokens: List[tuple[str, int]], input_or_output: str = "input", method: Literal["fvt", "adapti-vocab"] = "fvt"
    ) -> dict[str, torch.Tensor]:
        """
        Compute a pooled embedding for each new token based on its subtokens' embeddings. We implement two methods:
        - 'fvt': Mean of subtoken embeddings (From "Fast Vocabulary Transfer for Language Model Compression", https://arxiv.org/html/2402.09977)
        - 'adapti-vocab': Exponentially weighted mean of subtoken embeddings based on their position (from "AdaptiVocab", https://arxiv.org/abs/2503.19693)
        """
        if input_or_output == "input":
            embs = self.model.get_input_embeddings().weight
        elif input_or_output == "output":
            embs = self.model.get_output_embeddings().weight
        else:
            raise ValueError("input_or_output must be 'input' or 'output'")
        out: dict[str, torch.Tensor] = {}
        for token, _ in todo_tokens:
            token_ids = get_new_phrase_tokenized_ids(token, self.source_tokenizer, self.model_path).to(embs.device)
            if method == "adapti-vocab":
                k = len(token_ids)
                if input_or_output == "input":
                    weights = torch.exp(2 * torch.arange(1, k + 1, device=embs.device))
                else:  # output
                    weights = torch.exp(-2 * torch.arange(1, k + 1, device=embs.device))
                weights = weights / weights.sum()
                weighted_embs = embs[token_ids] * weights.unsqueeze(1)
                out[token] = torch.sum(weighted_embs, dim=0)
            else:  # method == "fvt"
                out[token] = torch.mean(embs[token_ids], dim=0)
        return out
