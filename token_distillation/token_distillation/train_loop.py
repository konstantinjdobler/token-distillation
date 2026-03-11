import time
from typing import Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

from .utils import seed_everything

TokenLike = Sequence[int] | torch.Tensor
PhraseKey = tuple[int, ...]
GroupedTokenizedTexts = Sequence[Sequence[TokenLike]]


class TextDataset(Dataset):
    """Dataset wrapper for token distillation training data."""

    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return {k: v for k, v in self.tokenized_texts[idx].items()}


def _to_int_list(values: TokenLike) -> list[int]:
    """Normalize token containers (tensor/list/tuple) to a Python list of ints."""
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return list(values)

def transform_input_token_format(
    tokenized_texts: GroupedTokenizedTexts,
    new_phrase_to_new_id: dict[PhraseKey, int],
    pad_token_id: int,
    assigned_new_phrases: Sequence[TokenLike],
):
    """
    Transform tokenized texts by merging only each group's assigned phrase into a single token.

    Args:
        tokenized_texts: List of tokenized text batches.
        new_phrase_to_new_id: Mapping from phrase token sequences to new merged token IDs.
            Keys must be tuples to ensure deterministic lookup.
        pad_token_id: Token ID to use for padding.
        assigned_new_phrases: List aligned with the outer dimension of tokenized_texts.
            Only the assigned phrase for each group will be merged (all other phrases are ignored).

    Returns:
        list: Transformed texts with merged sequences and alignment masks.
    """
    merged_texts = []
    if len(assigned_new_phrases) != len(tokenized_texts):
        raise ValueError("assigned_new_phrases must match the outer length of tokenized_texts")

    for texts, assigned_phrase in tqdm(zip(tokenized_texts, assigned_new_phrases), total=len(tokenized_texts)):
        assigned_phrase_key = tuple(_to_int_list(assigned_phrase))
        if assigned_phrase_key not in new_phrase_to_new_id:
            raise KeyError(f"Assigned phrase not found in new_phrase_to_new_id: {assigned_phrase_key}")
        assigned_new_id = new_phrase_to_new_id[assigned_phrase_key]
        assigned_len = len(assigned_phrase_key)

        for text in tqdm(texts, leave=False):
            text = _to_int_list(text)
            i = 0
            current_text = []
            unmerged_to_merged_mask = [None] * len(text)
            old_len = len(text)
            while i < len(text):
                if i + assigned_len <= len(text) and tuple(text[i : i + assigned_len]) == assigned_phrase_key:
                    # merged assigned phrase starting at position i
                    current_text.append(assigned_new_id)
                    unmerged_to_merged_mask[i : i + assigned_len] = [0] * assigned_len
                    unmerged_to_merged_mask[i + assigned_len - 1] = 1
                    i += assigned_len
                else:
                    current_text.append(text[i])
                    unmerged_to_merged_mask[i] = 1
                    i += 1

            if any(mask_value is None for mask_value in unmerged_to_merged_mask):
                raise RuntimeError("Some tokens were not assigned a mask.")
            if sum(unmerged_to_merged_mask) != len(current_text):
                raise RuntimeError("The mask is not the same length as the text.")
            current_text += [pad_token_id] * (old_len - len(current_text))
            merged_texts.append(
                {"merged_seq": current_text, "original_seq": text, "unmerged_to_merged_mask": unmerged_to_merged_mask}
            )

    return merged_texts


def _next_token_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Compute next-token CE for autoregressive logits/targets with padding ignored."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )


def collate_fn(batch, pad_id=None):
    """Collate function that pads variable-length examples in a batch.

    Args:
        batch: list of samples, each sample is a dict with keys: merged_seq, original_seq, unmerged_to_merged_mask
        pad_id: pad token id (int). If None, caller must replace pad tokens later.

    Returns:
        dict of tensors: merged_seq, original_seq, unmerged_to_merged_mask
    """
    if pad_id is None:
        raise ValueError("pad_id must be provided to collate_fn")

    max_len = max(len(sample["merged_seq"]) for sample in batch)
    merged_seqs = []
    original_seqs = []
    masks = []
    for sample in batch:
        m = list(sample["merged_seq"])  # already padded to original length in transform_input_token_format
        o = list(sample["original_seq"])
        mask = list(sample["unmerged_to_merged_mask"])
        pad_len = max_len - len(m)
        if pad_len > 0:
            m = m + [pad_id] * pad_len
            o = o + [pad_id] * pad_len
            mask = mask + [0] * pad_len
        merged_seqs.append(m)
        original_seqs.append(o)
        masks.append(mask)

    return {
        "merged_seq": torch.tensor(merged_seqs, dtype=torch.long),
        "original_seq": torch.tensor(original_seqs, dtype=torch.long),
        "unmerged_to_merged_mask": torch.tensor(masks, dtype=torch.long),
    }


def train_embeddings(
    model,
    tokenized_texts: GroupedTokenizedTexts,
    new_phrase_to_new_id: dict[PhraseKey, int],
    assigned_new_phrases: Sequence[TokenLike],
    tokenizer,
    epochs=1,
    batch_size=1,
    learning_rate=1e-4,
    loss_methods=None,
    preserve_original_embeddings=True,
    seed=42,
    original_token_ids=None,
    target_layer=-1,
    mixed_precision=True,
    learn_output_with_ce=True,
):
    """
    Train embeddings for new tokens using token distillation.

    Implements token distillation to learn embeddings for new (merged) tokens by
    distilling information from the model's hidden states and/or logits computed
    on the original (unmerged) multi-token sequences.

    Args:
        model: The model to train (PreTrainedModel).
        tokenized_texts: List of batches of tokenized text sequences, grouped by new phrase
            (one group per new phrase). Each group is assigned to that phrase and only that
            phrase will be merged in its samples.
        new_phrase_to_new_id: Mapping from phrase token sequences to new merged token IDs.
        assigned_new_phrases: List of phrase token sequences aligned with tokenized_texts outer dimension.
        tokenizer: Tokenizer used for pad/eos ids and vocab size information.
        epochs (int): Number of training epochs. Default: 1.
        batch_size (int): Training batch size. Default: 1.
        learning_rate (float): Learning rate. Default: 1e-4.
        loss_methods (list[str] | None): Loss functions to use. If None defaults to ["MSE-on-hiddens"].
            Valid options: "MSE-on-hiddens", "MSE-on-logits", "KL-on-logits", "CE", "CE-auto-weighted".
        preserve_original_embeddings (bool): If True, prevent updates to original token embeddings. Default: True.
        seed (int): Random seed. Default: 42.
        original_token_ids (list[int] | None): IDs of original tokens to preserve/monitor. If None,
            defaults to range(len(tokenizer) - len(new_phrase_to_new_id)).
        target_layer (int): Layer index to use for hidden-state distillation. Default: -1 (last layer).
        mixed_precision (bool): Whether to use mixed precision. Default: True.
        learn_output_with_ce (bool): Whether to update output embeddings via a CE loss (separate step). Default: True.

    Returns:
        The model with trained embeddings.
    """
    t0 = time.perf_counter()

    seed_everything(seed)
    if loss_methods is None:
        loss_methods = ["MSE-on-hiddens"]
    valid_loss_methods = ["MSE-on-hiddens", "MSE-on-logits", "KL-on-logits", "CE", "CE-auto-weighted"]
    if "CE-auto-weighted" in loss_methods:
        if not (
            "MSE-on-hiddens" in loss_methods or "MSE-on-logits" in loss_methods or "KL-on-logits" in loss_methods
        ):
            raise ValueError("CE-auto-weighted requires one of MSE-on-hiddens, MSE-on-logits, or KL-on-logits")
    if not all(method in valid_loss_methods for method in loss_methods):
        raise ValueError(f"Invalid loss methods: {loss_methods}")
    if tokenizer.pad_token_id is None:
        if tokenizer.get_vocab().get("<|finetune_right_pad_id|>") is not None:
            tokenizer.pad_token_id = tokenizer.get_vocab().get("<|finetune_right_pad_id|>")
        elif tokenizer.get_vocab().get("<|padding|>") is not None:
            tokenizer.pad_token_id = tokenizer.get_vocab().get("<|padding|>")
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Setting pad token id to eos token id: {tokenizer.eos_token_id} | {tokenizer.eos_token}")

        if tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer must have a pad token")
    if not tokenized_texts or not isinstance(tokenized_texts[0], (list, tuple)):
        raise ValueError("tokenized_texts must be grouped by new phrase (list of lists).")
    if len(tokenized_texts) != len(new_phrase_to_new_id):
        raise ValueError("tokenized_texts outer length must match new_phrase_to_new_id.")
    if len(assigned_new_phrases) != len(tokenized_texts):
        raise ValueError("assigned_new_phrases outer length must match tokenized_texts.")

    dataset = TextDataset(
        transform_input_token_format(
            tokenized_texts,
            new_phrase_to_new_id,
            tokenizer.pad_token_id,
            assigned_new_phrases=assigned_new_phrases,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_token_id),
    )

    # don't train the model
    for p in model.parameters():
        p.requires_grad = False
    # only train the embeddings
    model.get_output_embeddings().weight.requires_grad = True
    model.get_input_embeddings().weight.requires_grad = True

    original_input_embs = model.get_input_embeddings().weight.clone().detach()
    original_output_embs = model.get_output_embeddings().weight.clone().detach()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad()

    # paper used linear warmup + decay but constant works fine and might even converge faster
    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        # num_warmup_steps=int(0.5 * len(dataloader)),  # warmup for half the steps of the first epoch, heuristic used in the paper.
        # num_training_steps=epochs * len(dataloader),
    )
    model.train()

    if original_token_ids is None:
        original_token_ids = list(range(len(tokenizer) - len(new_phrase_to_new_id)))

    print(model)
    device = model.device
    print(f"Training startup time: {time.perf_counter() - t0}")
    for epoch in range(epochs):
        epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch: {epoch}")
        running_window_losses = []
        for step_idx, batch in epoch_bar:
            merged_seq = batch["merged_seq"].to(device, non_blocking=True)
            unmerged_to_merged_mask = batch["unmerged_to_merged_mask"].to(device, non_blocking=True)
            unmerged_seq = batch["original_seq"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                # student fwd
                merged_out = model(merged_seq, output_hidden_states=True)

                if all(method in ["CE", "CE-auto-weighted"] for method in loss_methods):
                    pass  # Skip teacher forward pass if only using CE-based loss
                else:
                    # teacher fwd - only needed for non-CE losses
                    with torch.no_grad():
                        unmerged_out = model(unmerged_seq, output_hidden_states=True)

            loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if "MSE-on-hiddens" in loss_methods:
                token_distillation_hiddens = merged_out["hidden_states"][target_layer].float()
                og_hiddens = unmerged_out["hidden_states"][target_layer].float()

                # take only the tokens that were not merged + the last token of the merged phrase
                og_hiddens = og_hiddens[unmerged_to_merged_mask == 1]

                # remove padding
                token_distillation_hiddens = token_distillation_hiddens[merged_seq != tokenizer.pad_token_id]

                token_distillation_hiddens = token_distillation_hiddens.view(-1, token_distillation_hiddens.size(-1))
                og_hiddens = og_hiddens.view(-1, og_hiddens.size(-1))
                loss = loss + torch.nn.functional.mse_loss(token_distillation_hiddens, og_hiddens)
            if "MSE-on-logits" in loss_methods or "KL-on-logits" in loss_methods:
                token_distillation_logits = merged_out["logits"].float()
                og_logits = unmerged_out["logits"].float()

                # take only the tokens that were not merged + the last token of the merged phrase; remove padding
                token_distillation_logits = token_distillation_logits[merged_seq != tokenizer.pad_token_id]
                og_logits = og_logits[unmerged_to_merged_mask == 1]

                if len(token_distillation_logits.shape) != 2:
                    raise RuntimeError("token_distillation_logits should be 2D")
                if len(og_logits.shape) != 2:
                    raise RuntimeError("og_logits should be 2D")

                # only calculate loss on logits for original vocabulary
                token_distillation_logits = token_distillation_logits[:, original_token_ids]
                og_logits = og_logits[:, original_token_ids]

                if "MSE-on-logits" in loss_methods:
                    loss = loss + torch.nn.functional.mse_loss(token_distillation_logits, og_logits)
                if "KL-on-logits" in loss_methods:
                    loss = loss + torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(token_distillation_logits),
                        torch.nn.functional.log_softmax(og_logits),
                        reduction="batchmean",
                        log_target=True,
                    )
            if "CE" in loss_methods or "CE-auto-weighted" in loss_methods:
                token_distillation_logits = merged_out["logits"].float()
                targets = merged_seq[:, 1:]
                logits = token_distillation_logits[:, :-1]
                ce_loss = _next_token_ce_loss(logits, targets, tokenizer.pad_token_id)
                if "CE" in loss_methods:
                    loss = loss + ce_loss
                if "CE-auto-weighted" in loss_methods:
                    # this is \alpha from the paper, `.item()` is like `stop_gradient`
                    scaling_factor = loss.item() / ce_loss.item()
                    loss = loss + ce_loss * scaling_factor

            if learn_output_with_ce:
                token_distillation_logits = merged_out["logits"].float()
                targets = merged_seq[:, 1:]
                logits = token_distillation_logits[:, :-1]
                ce_loss = _next_token_ce_loss(logits, targets, tokenizer.pad_token_id)
                torch.autograd.backward(
                    [ce_loss],
                    inputs=[model.get_output_embeddings().weight],
                    retain_graph=True,
                )

            # only accumulate gradients for input embeddings
            # prevents issues if coupling CE loss with `learn_output_with_ce`
            loss.backward(inputs=[model.get_input_embeddings().weight])

            if preserve_original_embeddings:
                # gradient surgery, zero out the gradients of the original tokens
                # this is very beneficial and prevents degradation of og embs
                model.get_input_embeddings().weight.grad[original_token_ids] = 0
                if model.get_output_embeddings().weight.grad is not None:
                    model.get_output_embeddings().weight.grad[original_token_ids] = 0

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_window_losses.append(loss.item())
            avg_loss = sum(running_window_losses) / len(running_window_losses)
            epoch_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}, Running Loss: {avg_loss}")

            if len(running_window_losses) == int(len(dataloader) * 0.1):
                running_window_losses = []
                print(f"Epoch: {epoch}, Step: {step_idx}, Loss: {loss.item()}, Running Loss: {avg_loss}")

    if preserve_original_embeddings:
        if not torch.equal(
            model.get_input_embeddings().weight.data[original_token_ids], original_input_embs[original_token_ids]
        ):
            raise RuntimeError("The original input embeddings have changed.")
        if not torch.equal(
            model.get_output_embeddings().weight.data[original_token_ids], original_output_embs[original_token_ids]
        ):
            raise RuntimeError("The original output embeddings have changed.")
    else:
        if torch.equal(model.get_input_embeddings().weight.data[original_token_ids], original_input_embs[original_token_ids]):
            raise RuntimeError(
                "The original input embeddings have not changed. This is unexpected if `preserve_original_embeddings` is False."
            )

    t_end = time.perf_counter()
    print(f"Total end-to-end time: {t_end - t0}")

    return model
