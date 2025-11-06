import itertools
import time
from typing import Dict, Iterable, List, Tuple

import ahocorasick_rs
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# Fast UTF-8 surrogate-safe mapping helpers
UTF8_SURROGATE_CODEPOINT_MIN = 0xD800
UTF8_SURROGATE_CODEPOINT_MAX = 0xDFFF
NUM_UTF8_SURROGATE_CODEPOINTS = UTF8_SURROGATE_CODEPOINT_MAX - UTF8_SURROGATE_CODEPOINT_MIN + 1
NUM_UTF8_CHARS = 1_114_112 - NUM_UTF8_SURROGATE_CODEPOINTS  # max unique codepoints we can use safely


def my_chr(x: int) -> str:
    x = int(x)
    if x >= UTF8_SURROGATE_CODEPOINT_MIN:
        x += NUM_UTF8_SURROGATE_CODEPOINTS
    return chr(x)


def my_ord(x: str) -> int:
    v = ord(x)
    if v >= UTF8_SURROGATE_CODEPOINT_MIN:
        v -= NUM_UTF8_SURROGATE_CODEPOINTS
    return v


def map_int_seq_to_str(sequence: Iterable[int]) -> str:
    return "".join(my_chr(x) for x in sequence)


def unmap_int_seq_from_str(string: str) -> List[int]:
    return [my_ord(c) for c in string]


def collect_snippets_with_patterns_from_dataset(
    patterns_ids: List[List[int]],
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset | Iterable[List[int]],
    *,
    max_docs: int,
    offset_before: int = 50,
    offset_after: int = 150,
    batch_start: int = 500,
    batch_max: int = 256_000,
    max_necessary_samples: int | None = None,
    verbose: bool = True,
    log_topk_patterns: int = 10,
    print_snippets_at_end: bool = False,
) -> Dict[str, List[Tuple[List[int], int]]]:
    """
    Encodes token IDs to a surrogate-safe UTF-8 string and runs Aho-Corasick.
    Processes up to `max_docs` documents (stopping earlier if either the dataset ends or
    all patterns have reached `max_necessary_samples`).

    Returns:
        dict[str, list[(snippet_tokens, start_index_in_snippet)]],
        keyed by map_int_seq_to_str(pattern).
    """
    if len(tokenizer) >= NUM_UTF8_CHARS:
        raise ValueError(f"Tokenizer length {len(tokenizer)} exceeds fast-mapper capacity ({NUM_UTF8_CHARS}). ")
    if max_docs <= 0:
        raise ValueError("max_docs must be a positive integer")

    # Pre-encode patterns
    stringified_patterns = [map_int_seq_to_str(p) for p in patterns_ids]
    collected: Dict[str, List[Tuple[List[int], int]]] = {p: [] for p in stringified_patterns}

    # Build an iterator of tokenized documents (each: List[int])
    if isinstance(dataset, Dataset):
        tokens_iter = iter(dataset["tokens"])
        ds_kind = "datasets.Dataset"
    elif isinstance(dataset, Iterable):
        tokens_iter = iter(dataset)
        ds_kind = "Iterable[List[int]]"
    else:
        raise TypeError("dataset must be a datasets.Dataset or an Iterable[List[int]] of tokenized docs")

    if verbose:
        print("[aho] ===== Aho-Corasick Sample Retrieval from Dataset =====")
        print(f"[aho] patterns: {len(stringified_patterns)}")
        print(f"[aho] dataset type: {ds_kind}")
        print(f"[aho] max_docs: {max_docs}")
        print(f"[aho] offsets: before={offset_before}, after={offset_after}")
        print(f"[aho] batch: start={batch_start}, max={batch_max}")
        cap_txt = str(max_necessary_samples) if max_necessary_samples is not None else "no cap"
        print(f"[aho] per-pattern cap: {cap_txt}")
        print("[aho] ================================================")

    total_docs = 0
    total_batches = 0
    proc_bs = batch_start
    t_total = time.perf_counter()

    # helper for compact pattern stats
    def _print_pattern_stats(prefix: str = "[aho]"):
        if not verbose:
            return
        # Show top-K by collected count
        items = sorted(((p, len(v)) for p, v in collected.items()), key=lambda x: x[1], reverse=True)[:log_topk_patterns]
        if not items:
            print(f"{prefix} no patterns tracked")
            return
        pretty = []
        for p, n in items:
            name = tokenizer.decode(unmap_int_seq_from_str(p))
            pretty.append(f"{name}: {n}")
        print(f"{prefix} top-{len(pretty)} pattern counts: " + ", ".join(pretty))

    while total_docs < max_docs and stringified_patterns:
        remaining = max_docs - total_docs
        this_bs = min(proc_bs, remaining)

        t_batch_start = time.perf_counter()
        batch = list(itertools.islice(tokens_iter, this_bs))
        t_batch_elapsed = time.perf_counter() - t_batch_start
        if verbose:
            print(f"[aho] fetched batch#{total_batches + 1} of size {len(batch)} in {t_batch_elapsed:.3f}s")
        if not batch:
            if verbose:
                print("[aho] dataset exhausted.")
            break

        # (Re)build automaton with the *current* hot patterns
        t_init = time.perf_counter()
        ac = ahocorasick_rs.AhoCorasick(stringified_patterns, implementation=ahocorasick_rs.Implementation.DFA)
        t_init_elapsed = time.perf_counter() - t_init

        if verbose:
            print(
                f"[aho] batch#{total_batches + 1} init: patterns={len(stringified_patterns)} "
                f"build={t_init_elapsed:.3f}s docs={len(batch)}"
            )

        # Scan the batch
        t_scan_start = time.perf_counter()
        batch_tokens = 0
        pre_total_snippets = sum(len(v) for v in collected.values())

        for doc in tqdm(batch, leave=False, disable=not verbose):
            batch_tokens += len(doc)
            sdoc = map_int_seq_to_str(doc)
            for pattern_idx, start, end in ac.find_matches_as_indexes(sdoc, overlapping=True):
                patt = stringified_patterns[pattern_idx]
                safe_start = max(0, start - offset_before)
                safe_end = min(len(doc), end + offset_after)
                snippet = doc[safe_start:safe_end]
                start_in_snippet = start - safe_start
                collected[patt].append((snippet, start_in_snippet))

        t_scan_elapsed = time.perf_counter() - t_scan_start
        post_total_snippets = sum(len(v) for v in collected.values())
        new_snippets = post_total_snippets - pre_total_snippets

        docs_per_s = (len(batch) / t_scan_elapsed) if t_scan_elapsed > 0 else float("inf")
        toks_per_s = (batch_tokens / t_scan_elapsed) if t_scan_elapsed > 0 else float("inf")

        if verbose:
            print(
                f"[aho] batch#{total_batches + 1} scan: {t_scan_elapsed:.3f}s "
                f"docs={len(batch)} ({docs_per_s:,.1f}/s) "
                f"tokens={batch_tokens:,} ({toks_per_s:,.0f}/s) "
                f"new_snippets={new_snippets}"
            )
            _print_pattern_stats()

        total_docs += len(batch)
        total_batches += 1

        # === Prune patterns that have reached the cap ===
        if max_necessary_samples is not None:
            done = [p for p in stringified_patterns if len(collected[p]) >= max_necessary_samples]
            if done:
                # Keep them in `collected`, but remove from further matching
                before = len(stringified_patterns)
                stringified_patterns = [p for p in stringified_patterns if len(collected[p]) < max_necessary_samples]
                after = len(stringified_patterns)
                if verbose:
                    print(f"[aho] pruning: finished_patterns={len(done)} (patterns {before} -> {after})")
            # Early stop if everything is satisfied
            if not stringified_patterns:
                if verbose:
                    print("[aho] all patterns reached the cap; stopping early.")
                break

        # Exponential backoff on batch size for better throughput
        if proc_bs < batch_max:
            proc_bs = min(proc_bs * 2, batch_max)
            if verbose:
                print(f"[aho] next batch size: {proc_bs}")

    t_total_elapsed = time.perf_counter() - t_total
    if verbose:
        print(f"[aho] processed {total_docs}/{max_docs} docs in {t_total_elapsed:.2f}s over {total_batches} batch(es).")
        total_snips = sum(len(v) for v in collected.values())
        print(f"[aho] total collected snippets: {total_snips}")
        if max_necessary_samples is not None:
            satisfied = sum(1 for v in collected.values() if len(v) >= max_necessary_samples)
            print(f"[aho] patterns satisfied (>= {max_necessary_samples}): {satisfied}")
        print("[aho] ===== Aho-Corasick Sample Retrieval from Dataset FINISHED =====")

    # Optional: print all collected snippets per pattern (off by default)
    if print_snippets_at_end:
        for patt, snippets in collected.items():
            patt_txt = tokenizer.decode(unmap_int_seq_from_str(patt))
            print(f"Pattern '{patt_txt}' - collected {len(snippets)} snippets")
            for snippet, start in snippets:
                print(f"  Snippet (start at {start}): {tokenizer.decode(snippet, skip_special_tokens=False)}")

    return collected
