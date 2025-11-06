import time
from typing import Iterable

import ahocorasick_rs
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

"""
Who decided to put these at a random place inside the range (0,2**16) instead of the end? This was a "fun" bug to track down.
"""
UTF8_SURROGATE_CODEPOINT_MIN = 0xD800  # 55296
UTF8_SURROGATE_CODEPOINT_MAX = 0xDFFF  # 57343
NUM_UTF8_SURROGATE_CODEPOINTS = UTF8_SURROGATE_CODEPOINT_MAX - UTF8_SURROGATE_CODEPOINT_MIN + 1


def my_chr(x):
    """
    Like `chr`, but avoids mapping to integers that are reserved for UTF-16 surrogate codepoints.
    Surrogate codepoints make trouble when encoding to utf-8 for the Rust-based Aho-Corasick implementation.
    """
    x = int(x)  # for safety if we get a tensor
    if x >= UTF8_SURROGATE_CODEPOINT_MIN:
        x += NUM_UTF8_SURROGATE_CODEPOINTS
    return chr(x)


def my_ord(x):
    """
    Like `ord`, but complementary to `my_chr` for `chr`.
    """
    x = ord(x)
    if x >= UTF8_SURROGATE_CODEPOINT_MIN:
        x -= NUM_UTF8_SURROGATE_CODEPOINTS
    return x


def map_int_seq_to_str(sequence):
    """
    Maps a sequence of integers to a string by converting each integer to a character.

    Args:
        sequence (iterable of int): The sequence of integers to be converted.

    Returns:
        str: A string representation of the sequence where each integer is represented by a single character.
    """
    return "".join(my_chr(x) for x in sequence)


def unmap_int_seq_from_str(string):
    """
    Reverses the mapping from `map_int_seq_to_str`, converting a string back to a sequence of integers.

    Args:
        string (str): The string to be converted back to integers.

    Returns:
        list[int]: A list of integers corresponding to the characters in the string.
    """
    return [my_ord(x) for x in string]


def collect_snippets_with_patterns_from_dataset(
    patterns_ids, tokenizer: PreTrainedTokenizer, dataset: Dataset | Iterable[int], stopping_condition="num_docs:16000"
) -> dict[str, list[tuple[list[int], int]]]:
    """
    Cast the problem of detecting occurrences of a sequence of tokens in a larger document as a string matching problem.
    We use an optimized Rust-based Aho-Corasick implementation with iterative pruning of the search patterns.

    The main performance bottlenecks are loading the data into memory and allocating memory for the results. The actual Aho-Corasick search is very fast, especially with our iterative pruning of the search patterns.

    TODO -- for really large-scale document corpus -- implement more memory efficient results storage / pre-allocate memory once (not possible with python lists in a sensible way).

    Returns: A dictionary with `map_int_seq_to_str(pattern)` as keys and the found snippets as values, along with the start position of the pattern in the snippet.
    """
    print("Starting Aho-Corasick snippet for patterns from `pattern_ids` in `dataset`...")

    map_int_seq_fn = map_int_seq_to_str
    unmap_int_seq_fn = unmap_int_seq_from_str
    FAST_MAP = True

    # chr cannot encode integers larger than this to single chars
    # additionally, we need to avoid the UTF-8 surrogate codepoints
    NUM_UTF8_CHARS = 1_114_112 - NUM_UTF8_SURROGATE_CODEPOINTS
    print("tokenizer length:", len(tokenizer))
    if len(tokenizer) >= NUM_UTF8_CHARS:
        print(
            f"Tokenizer length {len(tokenizer)} is larger than the number of UTF-8 characters {NUM_UTF8_CHARS}. Using slow list[int]=>str mapping function."
        )

        def my_hash_fn(x, return_boundaries=False):
            stringified = [str(int(i)) for i in x]
            stringified_to_regular_pos_map = {}
            stringified_final = ""
            for i, s in enumerate(stringified):
                stringified_to_regular_pos_map[len(stringified_final)] = i
                stringified_final += f"|{s}|"
            if return_boundaries:
                return stringified_final, stringified_to_regular_pos_map
            return stringified_final

        map_int_seq_fn = my_hash_fn

        unmap_int_seq_fn = lambda x: [int(i) for i in x.strip("|").split("||")]
        FAST_MAP = False

    print("Using map function FAST_HASH:", FAST_MAP)
    stringified_patterns = [map_int_seq_fn(pattern) for pattern in patterns_ids]
    print(f"Number of patterns: {len(stringified_patterns)}")
    print(stringified_patterns[:20], [len(p) for p in stringified_patterns[:20]])

    OFFSET_BEFORE = 50
    OFFSET_AFTER = 150
    DONE = False

    min_num_of_collected_snippets = 150
    collected_snippets = {pattern: [] for pattern in stringified_patterns}

    def collection_ahocorasick_f(tokenized_docs: list[list[int]]):
        # rebuild every new iter since we can prune patterns that have enough snippets
        t_before_init = time.perf_counter()
        ac = ahocorasick_rs.AhoCorasick(stringified_patterns, implementation=ahocorasick_rs.Implementation.DFA)
        t_after_init = time.perf_counter()
        print(f"Init time for Aho-Corasick: {t_after_init - t_before_init}")

        t0 = time.perf_counter()
        for doc in tqdm(tokenized_docs, leave=False):
            if FAST_MAP:
                stringified_doc = map_int_seq_fn(doc)
            else:
                stringified_doc, boundaries = map_int_seq_fn(doc, return_boundaries=True)
            matches = ac.find_matches_as_indexes(stringified_doc, overlapping=True)

            for match in matches:
                pattern_id, start, end = match
                pattern_mapped = stringified_patterns[pattern_id]
                if FAST_MAP:
                    safe_start_of_snippet = max(0, start - OFFSET_BEFORE)
                    safe_end_of_snippet = min(len(doc), end + OFFSET_AFTER)
                    snippet = doc[safe_start_of_snippet:safe_end_of_snippet]

                    start_pos_of_pattern_in_snippet = start - safe_start_of_snippet
                else:
                    start_pos_of_pattern_in_unmapped = boundaries[start]
                    # at least 0, and as many as we have before the pattern in the doc up to OFFSET_BEFORE many
                    offset_before_in_doc = max(0, start_pos_of_pattern_in_unmapped - OFFSET_BEFORE)

                    # OFFSET_BEFORE if we have at least OFFSET_BEFORE tokens before the pattern; else start_pos_of_pattern_in_unhashed since the snippet starts at the beginning of the doc
                    start_pos_of_pattern_in_snippet = min(OFFSET_BEFORE, start_pos_of_pattern_in_unmapped)
                    end_pos_of_pattern_in_unmapped = start_pos_of_pattern_in_unmapped + len(unmap_int_seq_fn(pattern_mapped))

                    # at most the end of the doc, and as many as we have after the pattern in the doc up to OFFSET_AFTER many
                    offset_after_in_doc = min(len(doc), end_pos_of_pattern_in_unmapped + OFFSET_AFTER)
                    snippet = doc[offset_before_in_doc:offset_after_in_doc]

                collected_snippets[pattern_mapped].append((snippet, start_pos_of_pattern_in_snippet))

        print(f"Collection for {len(tokenized_docs)} done in {time.perf_counter() - t0:.4f} seconds")

    proc_bs = 500
    print("Starting collection")
    iter_counter = 0

    if stopping_condition.startswith("num_docs:"):
        MAX_NUM_DOCS = int(stopping_condition.split(":")[1])
    else:
        raise NotImplementedError(f"Stopping condition {stopping_condition} not implemented")
    MAX_BATCH_SIZE = 256_000

    total_docs_processed = 0
    total_t0 = time.perf_counter()
    while not DONE:
        print(f"\n\n--------------\nStarting iteration {iter_counter} with {proc_bs} documents\n\n----------")
        data_fetch_t0 = time.perf_counter()
        print("Fetching data...")
        if isinstance(dataset, Dataset):
            batch = dataset.take(proc_bs)["tokens"]
            dataset = dataset.skip(proc_bs)
        else:
            batch = dataset[total_docs_processed : total_docs_processed + proc_bs]
        data_fetch_t1 = time.perf_counter()
        print(f"Data fetch time: {data_fetch_t1 - data_fetch_t0:.4f} seconds")

        collection_ahocorasick_f(batch)
        total_docs_processed += proc_bs
        iter_counter += 1
        print(f"----Summary for processed {total_docs_processed} documents")
        collected_docs_lens = {k: len(v) for k, v in collected_snippets.items()}
        least_num_snippets = min(collected_docs_lens.values())
        if least_num_snippets >= min_num_of_collected_snippets:
            print("All tokens have enough snippets")
            DONE = True
            break
        print(
            f"Num tokens with enough snippets: {len([k for k, v in collected_docs_lens.items() if v >= min_num_of_collected_snippets])}"
        )

        # Prune patterns that have enough snippets
        stringified_patterns = [k for k, v in collected_docs_lens.items() if v < min_num_of_collected_snippets]
        print(
            f"Hot patterns: {len(stringified_patterns)} | new num snippets: {sum([v for k, v in collected_docs_lens.items() if v < min_num_of_collected_snippets])}"
        )

        dict_num_snippets_to_count = {}
        for k, v in collected_docs_lens.items():
            dict_num_snippets_to_count[v] = dict_num_snippets_to_count.get(v, 0) + 1
        print("Collected counts", sorted(dict_num_snippets_to_count.items())[:30])
        print(
            f"Num tokens with least snippets ({least_num_snippets}): {len([k for k, v in collected_docs_lens.items() if v == least_num_snippets])}"
        )
        print(f"Num tokens with fewer than 50 snippets: {len([k for k, v in collected_docs_lens.items() if v < 50])}")
        print(
            "Example tokens with least snippets:",
            [
                tokenizer.convert_ids_to_tokens(unmap_int_seq_fn(k))
                for k, v in collected_docs_lens.items()
                if v == least_num_snippets
            ][:10],
        )
        print(f"Total time for {iter_counter} iterations: {time.perf_counter() - total_t0:.4f} seconds")
        if DONE or total_docs_processed >= MAX_NUM_DOCS:
            break
        if proc_bs < MAX_BATCH_SIZE:
            proc_bs *= 2
    if not FAST_MAP:
        collected_snippets = {map_int_seq_to_str(unmap_int_seq_fn(k)): v for k, v in collected_snippets.items()}
    return collected_snippets
