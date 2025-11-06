from typing import List

from fire import Fire
from transformers import AutoTokenizer

from token_distillation import DistillationConfig, GeneratedDataSource, HFDataSource, OutputEmbeddingInit, TokenDistillation


def main(
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    out_path: str = "./outputs/llama3.1-german",
    # new tokens
    new_tokens_source: str = "list",  # "list" or "tokenizer"
    tokens: str = " Krankenwagen, Schmetterling",  # comma-separated when new_tokens_source="list"
    target_tokenizer_path: str | None = None,
    # data
    data_source: str = "hf",  # "hf" or "gen"
    dataset_path: str = "HuggingFaceFW/fineweb-2",
    dataset_name: str | None = "deu_Latn",
    dataset_split: str = "train",
    tokenization_batch_size: int = 16_000,
    max_docs: int | None = 1_000_000,
    revision: str | None = None,
    trust_remote_code: bool = False,
    # snippets
    snippet_len: int = 50,
    snippets_per_token: int = 100,
    # training
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    loss_methods: str = "MSE-on-hiddens",  # comma-separated string if multiple losses
    target_layer: int = -1,
    seed: int = 42,
    mixed_precision: bool = True,
    # output emb policy
    output_emb_policy: str = "train_with_ce",  # zero | subtoken_mean | train_with_ce
):
    tokdist = TokenDistillation(model_path=model_path)

    # new token source
    if new_tokens_source == "list":
        new_tokens: List[str] = [t for t in [s for s in tokens.split(",")] if t]
    elif new_tokens_source == "tokenizer":
        if not target_tokenizer_path:
            raise ValueError("target_tokenizer_path is required when new_tokens_source='tokenizer'")
        new_tokens = AutoTokenizer.from_pretrained(target_tokenizer_path)
    else:
        raise ValueError("new_tokens_source must be 'list' or 'tokenizer'")

    # data source
    if data_source == "hf":
        data = HFDataSource(
            dataset_path=dataset_path,
            name=dataset_name,
            split=dataset_split,
            tokenization_batch_size=tokenization_batch_size,
            max_docs=max_docs,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    elif data_source == "gen":
        data = GeneratedDataSource(seed=seed)
    else:
        raise ValueError("data_source must be 'hf' or 'gen'")

    tokdist.run(
        new_tokens=new_tokens,
        data=data,
        out_path=out_path,
        snippet_len=snippet_len,
        snippets_per_token=snippets_per_token,
        output_emb_policy=OutputEmbeddingInit(output_emb_policy),
        training=DistillationConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_methods=[m.strip() for m in loss_methods.split(",") if m.strip()],
            seed=seed,
            target_layer=target_layer,
            mixed_precision=mixed_precision,
        ),
        save=True,
        pre_init_strategy="fvt",
    )


if __name__ == "__main__":
    Fire(main)
