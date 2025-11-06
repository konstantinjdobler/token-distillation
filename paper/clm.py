
import torch
from token_distillation_utils import seed_everything
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler


class TextDataset(Dataset):
    """Dataset wrapper for tokenized text sequences."""

    def __init__(self, tokenized_texts):
        assert (
            isinstance(tokenized_texts, list)
            and all(isinstance(i, list) for i in tokenized_texts)
            and all(isinstance(j, int) for i in tokenized_texts for j in i)
        ), "tokenized_texts must be a list of list of int"
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_texts[idx], dtype=torch.long)


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
        list: Transformed tokenized texts with phrases merged into single tokens.
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
            old_len = len(text)
            while i < len(text):
                for new_phrase_len in range(maximum_new_phrase_len, 0, -1):
                    potential_new_phrase = tuple(text[i : i + new_phrase_len])
                    if potential_new_phrase in new_phrase_per_len_to_new_id[new_phrase_len]:
                        current_text.append(new_phrase_per_len_to_new_id[new_phrase_len][potential_new_phrase])
                        i += new_phrase_len
                        break
                else:
                    current_text.append(text[i])
                    i += 1

            current_text += [pad_token_id] * (old_len - len(current_text))
            merged_texts.append(current_text)

    return merged_texts


def train_embeddings(
    model,
    tokenized_texts,
    new_phrase_to_new_id,
    tokenizer,
    epochs=3,
    batch_size=1,
    learning_rate=5e-5,
    preserve_og_embs=True,
    seed=42,
    mixed_precision=False,
):
    """
    Train embeddings for new tokens using causal language modeling.

    Args:
        model: The model to train.
        tokenized_texts: Training data as tokenized text sequences.
        new_phrase_to_new_id: Mapping from phrase sequences to new token IDs.
        tokenizer: Tokenizer for the model.
        epochs (int): Number of training epochs. Defaults to 3.
        batch_size (int): Training batch size. Defaults to 1.
        learning_rate (float): Learning rate. Defaults to 5e-5.
        preserve_og_embs (bool): Whether to preserve original embeddings. Defaults to True.
        seed (int): Random seed. Defaults to 42.
        mixed_precision (bool): Whether to use mixed precision training. Defaults to False.

    Returns:
        PreTrainedModel: The trained model.
    """
    seed_everything(seed)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            or tokenizer.get_vocab().get("<|finetune_right_pad_id|>")
            or tokenizer.get_vocab().get("<|padding|>")
        )
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad token"
    dataset = TextDataset(transform_input_token_format(tokenized_texts, new_phrase_to_new_id, tokenizer.pad_token_id))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

    for p in model.parameters():
        p.requires_grad = False

    model.get_output_embeddings().weight.requires_grad = True
    model.get_input_embeddings().weight.requires_grad = True

    original_input_embs = model.get_input_embeddings().weight.clone().detach()
    original_output_embs = model.get_output_embeddings().weight.clone().detach()
    first_added_token_idx = len(tokenizer) - len(new_phrase_to_new_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad()

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.5 * len(dataloader)),
        num_training_steps=epochs * len(dataloader),
    )
    model.train()

    print(model)
    device = model.device
    for epoch in range(epochs):
        epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch: {epoch}")
        running_window_losses = []
        for step_idx, batch in epoch_bar:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                out = model(batch)
            logits = out["logits"]

            targets = batch[:, 1:]
            logits = logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.pad_token_id
            )

            loss.backward()
            if preserve_og_embs:
                # gradient surgery, zero out the gradients of the original tokens
                # this is very beneficial and prevents degradation of og embs
                model.get_input_embeddings().weight.grad[:first_added_token_idx] = 0
                if model.get_output_embeddings().weight.grad is not None:
                    model.get_output_embeddings().weight.grad[:first_added_token_idx] = 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_window_losses.append(loss.item())
            if len(running_window_losses) == 100 or len(running_window_losses) == 0:
                avg_loss = sum(running_window_losses) / len(running_window_losses)
                running_window_losses = []
                epoch_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}, Running Loss: {avg_loss}")
                print(f"Epoch: {epoch}, Step: {step_idx}, Loss: {loss.item()}, Running Loss: {avg_loss}")

    if preserve_og_embs:
        assert torch.equal(
            model.get_input_embeddings().weight.data[:first_added_token_idx], original_input_embs[:first_added_token_idx]
        ), f"The first {first_added_token_idx} embeddings have changed."
        assert torch.equal(
            model.get_output_embeddings().weight.data[:first_added_token_idx], original_output_embs[:first_added_token_idx]
        ), f"The first {first_added_token_idx} embeddings have changed."
    else:
        assert not torch.equal(
            model.get_input_embeddings().weight.data[:first_added_token_idx], original_input_embs[:first_added_token_idx]
        ), f"The first {first_added_token_idx} embeddings have not changed. This is unexpected if `preserve_og_embs` is False."

    return model
