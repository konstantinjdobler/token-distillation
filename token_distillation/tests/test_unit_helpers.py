import copy

import pytest
import torch

from token_distillation.tokdist import GPT_BPE_WHITESPACE, TokenDistillation
from token_distillation.train_loop import collate_fn, transform_input_token_format


class FakeTokenizer:
    def __init__(self, vocab, special_tokens=None):
        self._vocab = dict(vocab)
        self.all_special_tokens = list(special_tokens or [])

    def get_vocab(self):
        return dict(self._vocab)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def __deepcopy__(self, memo):
        return FakeTokenizer(copy.deepcopy(self._vocab, memo), list(self.all_special_tokens))


def test_transform_input_token_format_merges_only_assigned_phrase():
    tokenized_texts = [
        [[10, 11, 12, 13], [99, 10, 11, 77]],
        [[20, 21, 22]],
    ]
    new_phrase_to_new_id = {
        (10, 11): 1000,
        (20, 21): 2000,
    }
    assigned_new_phrases = [
        (10, 11),
        torch.tensor([20, 21]),
    ]

    merged = transform_input_token_format(
        tokenized_texts=tokenized_texts,
        new_phrase_to_new_id=new_phrase_to_new_id,
        pad_token_id=0,
        assigned_new_phrases=assigned_new_phrases,
    )

    assert merged == [
        {
            "merged_seq": [1000, 12, 13, 0],
            "original_seq": [10, 11, 12, 13],
            "unmerged_to_merged_mask": [0, 1, 1, 1],
        },
        {
            "merged_seq": [99, 1000, 77, 0],
            "original_seq": [99, 10, 11, 77],
            "unmerged_to_merged_mask": [1, 0, 1, 1],
        },
        {
            "merged_seq": [2000, 22, 0],
            "original_seq": [20, 21, 22],
            "unmerged_to_merged_mask": [0, 1, 1],
        },
    ]


def test_collate_fn_pads_to_max_length():
    batch = [
        {
            "merged_seq": [1000, 12, 13],
            "original_seq": [10, 11, 12],
            "unmerged_to_merged_mask": [0, 1, 1],
        },
        {
            "merged_seq": [2000, 22],
            "original_seq": [20, 21],
            "unmerged_to_merged_mask": [0, 1],
        },
    ]

    collated = collate_fn(batch, pad_id=99)

    assert torch.equal(collated["merged_seq"], torch.tensor([[1000, 12, 13], [2000, 22, 99]]))
    assert torch.equal(collated["original_seq"], torch.tensor([[10, 11, 12], [20, 21, 99]]))
    assert torch.equal(collated["unmerged_to_merged_mask"], torch.tensor([[0, 1, 1], [0, 1, 0]]))


def test_build_target_tokenizer_filters_existing_special_duplicate_and_byte_tokens(capsys):
    tokdist = TokenDistillation.__new__(TokenDistillation)
    tokdist.source_tokenizer = FakeTokenizer(
        {
            f"{GPT_BPE_WHITESPACE}existing": 0,
            "<special>": 1,
            "plain": 2,
        },
        special_tokens=["<special>"],
    )
    tokdist._src_ws = GPT_BPE_WHITESPACE

    todo_tokens = tokdist._build_target_tokenizer(
        [
            " existing",
            "fresh",
            "fresh",
            "<special>",
            "<0xAB>",
            "plain",
            "",
            " second",
        ]
    )

    assert todo_tokens == [("fresh", 3), (" second", 4)]
    assert "2 tokens need initialization" in capsys.readouterr().out


def test_build_target_tokenizer_accepts_tokenizer_like_source():
    tokdist = TokenDistillation.__new__(TokenDistillation)
    tokdist.source_tokenizer = FakeTokenizer({"base": 0})
    tokdist._src_ws = GPT_BPE_WHITESPACE

    target_tokenizer = FakeTokenizer({"base": 0, "new": 1, "other": 2})
    todo_tokens = tokdist._build_target_tokenizer(target_tokenizer)

    assert todo_tokens == [("new", 1), ("other", 2)]


def test_collate_fn_requires_pad_id():
    with pytest.raises(ValueError, match="pad_id must be provided"):
        collate_fn([], pad_id=None)
