import datasets
import numpy as np

# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on


# fmt: off
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# fmt: on



def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        question = doc["question"]
        answer = doc["answer"]
        
        choices = ["صح", "خطأ"]
        choices_formatted = [
            f" {LETTER_INDICES[i]}) {choice}\n" for i, choice in enumerate(choices)
        ]
        query = f"السؤال: {question}\n"
        query += "\n".join(choices_formatted)
        query += "\nالإجابة:"

        return {
            "query": query,
            "choices": LETTER_INDICES[:2],
            "gold": ["صح", "خطأ"].index(answer),
        }

    return dataset.map(_process_doc)
