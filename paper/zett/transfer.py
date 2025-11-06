# based on https://github.com/bminixhofer/zett/blob/016775824664d3479dfd6215d78fd73d51dce833/README.md?plain=1#L93C1-L119C4

import torch
from transformers import AutoModel, AutoTokenizer

from .utils import get_surface_form_matrix


def valid_model_for_zett_transfer(base_model_name):
    if "mistralai/mistral-7b-v0.1" in base_model_name.lower():
        return "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
    elif "meta-llama/meta-llama-3-8b" in base_model_name.lower():
        return "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
    else:
        return False


def zett_transfer(words, base_model):
    base_model_name = base_model.config.name_or_path

    HP_PATH = valid_model_for_zett_transfer(base_model_name)
    if not HP_PATH:
        raise ValueError(
            f"Model {base_model_name} is not supported for transfer to Zett. Supported models are Mistral-7B-v0.1 and Meta-Llama-3-8B(-Instruct)"
        )

    hypernet = AutoModel.from_pretrained(HP_PATH, trust_remote_code=True, device_map="cuda:0")

    source_embeddings = torch.concatenate(
        [
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ],
        axis=1,
    )

    hn_tokenizer = AutoTokenizer.from_pretrained(HP_PATH)

    target_surface_forms = get_surface_form_matrix(
        words,  # ["Ġhello", "Ġworld"],  # byte representation of the tokens to predict
        maxlen=hypernet.config.hn_surface_maxlen,
        tokenizer_to_use=hn_tokenizer,
    )[0]

    # the last output is the predicted bias in case the model uses a bias (e.g. XLM-R)
    predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
        torch.from_numpy(target_surface_forms).to("cuda:0"), source_embeddings=source_embeddings
    )
    print("output shapes:", predicted_input_embeddings.shape, predicted_output_embeddings.shape)
    print("input shapes:", source_embeddings.shape, target_surface_forms.shape)
    return predicted_input_embeddings, predicted_output_embeddings
