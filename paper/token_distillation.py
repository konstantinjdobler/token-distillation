import time

import torch
from token_distillation_utils import seed_everything
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler


class TextDataset(Dataset):
    """Dataset wrapper for token distillation training data."""

    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.tokenized_texts[idx].items()}


def transform_input_token_format(
    tokenized_texts: list[list[list[int]]], new_phrase_to_new_id: dict[list[int], int], pad_token_id: int
):
    """
    Transform tokenized texts by merging sequences that match new phrases into single tokens.

    Args:
        tokenized_texts: List of tokenized text batches.
        new_phrase_to_new_id: Mapping from phrase token sequences to new merged token IDs.
        pad_token_id: Token ID to use for padding.

    Returns:
        list: Transformed texts with merged sequences and alignment masks.
    """
    merged_texts = []
    maximum_new_phrase_len = max(len(phrase) for phrase in new_phrase_to_new_id.keys())
    new_phrase_per_len_to_new_id = [
        {tuple(phrase.tolist()): new_id for phrase, new_id in new_phrase_to_new_id.items() if len(phrase) == i}
        for i in range(maximum_new_phrase_len + 1)
    ]
    for texts in tqdm(tokenized_texts):
        for text in tqdm(texts, leave=False):
            # print(text)
            text = text.tolist()
            i = 0
            current_text = []
            unmerged_to_merged_mask = [None] * len(text)
            old_len = len(text)
            while i < len(text):
                for new_phrase_len in range(maximum_new_phrase_len, 0, -1):
                    potential_new_phrase = tuple(text[i : i + new_phrase_len])
                    if potential_new_phrase in new_phrase_per_len_to_new_id[new_phrase_len]:
                        # merged phrase starting at position i found (of length new_phrase_len)
                        current_text.append(new_phrase_per_len_to_new_id[new_phrase_len][potential_new_phrase])
                        unmerged_to_merged_mask[i : i + new_phrase_len] = [0] * new_phrase_len
                        unmerged_to_merged_mask[i + new_phrase_len - 1] = 1  # last token of the phrase
                        i += new_phrase_len
                        break
                else:
                    # no merged phrase starting at position i found
                    current_text.append(text[i])
                    unmerged_to_merged_mask[i] = 1
                    i += 1

            assert all(i is not None for i in unmerged_to_merged_mask), "Some tokens were not assigned a mask"
            assert sum(unmerged_to_merged_mask) == len(current_text), "The mask is not the same length as the text"
            current_text += [pad_token_id] * (old_len - len(current_text))
            merged_texts.append(
                {"merged_seq": current_text, "original_seq": text, "unmerged_to_merged_mask": unmerged_to_merged_mask}
            )

    return merged_texts


def train_embeddings(
    model,
    tokenized_texts,
    new_phrase_to_new_id,
    tokenizer,
    epochs=3,
    batch_size=1,
    learning_rate=5e-5,
    loss_methods=None,
    preserve_og_embs=True,
    seed=42,
    original_token_ids=None,
    target_layer=-1,
    mixed_precision=False,
):
    """
    Train embeddings for new tokens using token distillation.

    This function implements the token distillation method to learn embeddings for new tokens
    by distilling knowledge from the teacher model's hidden states when processing the original
    multi-token sequences.

    Args:
        model: The model to train.
        tokenized_texts: Training data as tokenized text sequences.
        new_phrase_to_new_id: Mapping from phrase sequences to new token IDs.
        tokenizer: Tokenizer for the model.
        epochs (int): Number of training epochs. Defaults to 3.
        batch_size (int): Training batch size. Defaults to 1.
        learning_rate (float): Learning rate. Defaults to 5e-5.
        loss_methods (list[str], optional): Loss functions to use. Defaults to ["MSE-on-hiddens"].
        preserve_og_embs (bool): Whether to preserve original embeddings. Defaults to True.
        seed (int): Random seed. Defaults to 42.
        original_token_ids (list, optional): IDs of original tokens to preserve. Defaults to None.
        target_layer (int): Target layer for hidden state distillation. Defaults to -1.
        mixed_precision (bool): Whether to use mixed precision training. Defaults to False.

    Returns:
        PreTrainedModel: The trained model.
    """
    t0 = time.perf_counter()

    seed_everything(seed)
    if loss_methods is None:
        loss_methods = ["MSE-on-hiddens"]
    VALID_LOSS_METHODS = ["MSE-on-hiddens", "MSE-on-logits", "KL-on-logits", "CE", "CE-auto-weighted"]
    if "CE-auto-weighted" in loss_methods:
        assert "MSE-on-hiddens" in loss_methods or "MSE-on-logits" in loss_methods or "KL-on-logits" in loss_methods, (
            "CE-auto-weighted requires one of MSE-on-hiddens, MSE-on-logits, or KL-on-logits"
        )
    assert all(i in VALID_LOSS_METHODS for i in loss_methods), f"Invalid loss methods: {loss_methods}"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.get_vocab().get("<|finetune_right_pad_id|>")
            or tokenizer.get_vocab().get("<|padding|>")
            or tokenizer.eos_token_id
        )
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad token"
    dataset = TextDataset(transform_input_token_format(tokenized_texts, new_phrase_to_new_id, tokenizer.pad_token_id))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

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

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.5 * len(dataloader)),  # warmup for half the steps of the first epoch
        num_training_steps=epochs * len(dataloader),
    )
    model.train()

    if original_token_ids is None:
        original_token_ids = list(range(len(tokenizer) - len(new_phrase_to_new_id)))

    print(model)
    device = model.device
    print(f"Training startup time: {time.perf_counter() - t0}")
    first_batch_done = False
    t0_first_batch = time.perf_counter()
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

                assert len(token_distillation_logits.shape) == 2, "token_distillation_logits should be 2D"
                assert len(og_logits.shape) == 2, "og_logits should be 2D"

                # only caluclate loss on logits for og vocabulary
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
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.pad_token_id
                )
                if "CE" in loss_methods:
                    loss = loss + ce_loss
                if "CE-auto-weighted" in loss_methods:
                    # this is \alpha from the paper, `.item()` is like `stop_gradient`
                    scaling_factor = loss.item() / ce_loss.item()
                    loss = loss + ce_loss * scaling_factor

            loss.backward()

            if preserve_og_embs:
                # gradient surgery, zero out the gradients of the original tokens
                # this is very beneficial and prevents degradation of og embs
                model.get_input_embeddings().weight.grad[original_token_ids] = 0
                if model.get_output_embeddings().weight.grad is not None:
                    model.get_output_embeddings().weight.grad[original_token_ids] = 0

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_window_losses.append(loss.item())
            if len(running_window_losses) == 100 or len(running_window_losses) == 0:
                avg_loss = sum(running_window_losses) / len(running_window_losses)
                running_window_losses = []
                epoch_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}, Running Loss: {avg_loss}")
                print(f"Epoch: {epoch}, Step: {step_idx}, Loss: {loss.item()}, Running Loss: {avg_loss}")
            if not first_batch_done:
                print(f"First batch time: {time.perf_counter() - t0_first_batch}")
            first_batch_done = True

    if preserve_og_embs:
        assert torch.equal(
            model.get_input_embeddings().weight.data[original_token_ids], original_input_embs[original_token_ids]
        ), "The original input embeddings have changed."
        assert torch.equal(
            model.get_output_embeddings().weight.data[original_token_ids], original_output_embs[original_token_ids]
        ), "The original ouptut embeddings have changed."
    else:
        assert not torch.equal(
            model.get_input_embeddings().weight.data[original_token_ids], original_input_embs[original_token_ids]
        ), "The original input embeddings have not changed! This is unexpected if `preserve_og_embs` is False."

    t_end = time.perf_counter()
    print(f"Total end-to-end time: {t_end - t0}")

    return model
