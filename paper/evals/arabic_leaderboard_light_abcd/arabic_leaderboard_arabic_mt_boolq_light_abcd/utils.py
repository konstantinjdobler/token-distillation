import datasets
import numpy as np

# fmt: off
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# fmt: on

def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        question = doc["question"]
        passage = doc["passage"]
        instruction = "بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا"
        query = f"""{instruction}
        المقطع :
        {passage}
        السؤال:
        {question}
        A) نعم
        B) لا
        الإجابة:
        """
        letter_choices = ["A", "B"]

        return {
            "query": query,
            # "choices": ["نعم", "لا"],
            "choices": letter_choices,
            "gold": 0 if doc["answer"] else 1,
        }

    return dataset.map(_process_doc)
