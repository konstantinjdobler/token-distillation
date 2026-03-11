# Token Distillation: Attention-aware Input Embeddings for New Tokens

Token Distillation quickly learns input embeddings for newly added tokens by distilling hidden states from the original tokenization into a single embedding.

This package provides the reusable Python implementation from our ICLR 2026 paper
"Token Distillation: Attention-aware Input Embeddings for New Tokens"
([arXiv:2505.20133](https://arxiv.org/abs/2505.20133)).

## Quickstart

```python
from token_distillation import GeneratedDataSource, TokenDistillation

tokdist = TokenDistillation(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    attn_impl="sdpa",
    device="cuda:0",
)

model, tokenizer = tokdist.run(
    new_tokens=[" Krankenwagen", " Schmetterling"],
    data=GeneratedDataSource(seed=42),
    out_path="outputs/llama3.1-german",
    save=True # `save=True` saves the model & tokenizer in `out_path`
)
```

This will:

1. Add the requested tokens (if missing).
2. Generate snippets containing those tokens.
3. Distill embeddings for the new tokens.
4. Save updated model/tokenizer artifacts.

## Use a Corpus Instead of Generated Snippets

```python
from token_distillation import HFDataSource, TokenDistillation

tokdist = TokenDistillation(model_path="meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")

model, tokenizer = tokdist.run(
    new_tokens=[" Krankenwagen", " Schmetterling"],
    data=HFDataSource(
        dataset_path="HuggingFaceFW/fineweb-2",
        name="deu_Latn",
        split="train",
        max_docs=1_000_000,
    ),
    snippet_len=50,
    snippets_per_token=100,
)
```

## Training Configuration

```python
from token_distillation import DistillationConfig, HFDataSource, TokenDistillation

tokdist = TokenDistillation(model_path="meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")

model, tokenizer = tokdist.run(
    new_tokens=[" Krankenwagen", " Schmetterling"],
    data=HFDataSource(dataset_path="HuggingFaceFW/fineweb-2", name="deu_Latn", split="train"),
    training=DistillationConfig(
        epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        loss_methods=["MSE-on-hiddens"],
        # Example combined objective:
        # loss_methods=["MSE-on-hiddens", "CE-auto-weighted"],
        seed=1234,
        target_layer=-1,
        mixed_precision=True,
    ),
)
```

## Notes

- For untied heads, output embedding behavior is controlled via setting `output_emb_policy`. We recommend combining Token Distillation for input embeddings with other methods to obtain string output embeddings.
- We find that earlier target layers (at around the one third mark) can yield even better results in many cases than distilling from the last layer's hidden states.

## Citation

```bibtex
@inproceedings{dobler2026token,
    title={Token Distillation: Attention-Aware Input Embeddings for New Tokens},
    author={Konstantin Dobler and Desmond Elliott and Gerard de Melo},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=n20ml5nGEo}
}
```
