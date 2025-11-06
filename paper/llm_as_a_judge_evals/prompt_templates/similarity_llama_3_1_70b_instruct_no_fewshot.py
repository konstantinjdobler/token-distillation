def basic_prompt(line, token):
    """
    Annotator prompt adapted from into Llama-3 style:
    https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_farm/chatml_b1_chat_v0_without_inputs.txt

    Llama-3 prompt style is taken from:
    https://github.com/meta-llama/llama-recipes
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3
    https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L222
    """
    out = f"""<|start_header_id|>system<|end_header_id|>
    
You are a pattern-following assistant that can only answer with "Yes" or "No". Your goal is to determine whether a predicted definition conveys a similar enough meaning to the ground truth definition provided for a given word.<|eot_id|><|start_header_id|>user<|end_header_id|>

Remember to answer with one word either "Yes" or "No".

### Instruction:
Determine if the predicted definition conveys a similar meaning to the ground truth definition. The word is "{token.lstrip()}".

### Ground truth definition:
{line["base_definition"].strip()}

### Predicted definition:
{line["model_definition"].strip()}

### Does the predicted definition convey a similar meaning to the ground truth definition (Yes or No)?:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

    return out


def basic_parser(input_str):
    if "Yes" not in input_str and "No" not in input_str:
        return None
    elif "Yes" in input_str and "No" not in input_str:
        return 1
    elif "No" in input_str and "Yes" not in input_str:
        return 0
    elif input_str.find("Yes") < input_str.find("No"):
        return 1
    elif input_str.find("No") < input_str.find("Yes"):
        return 0
    else:
        return None
