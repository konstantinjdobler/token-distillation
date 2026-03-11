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
        question = doc["query"]
        answer_index = int(doc["label"])
        # Dynamically determining the choices by excluding '__few_shots', 'query' and 'label'
        choices_keys = [
            key for key in doc.keys() if key not in ["query", "label", "__few_shots"]
        ]
        choices = [doc[key] for key in choices_keys]

        instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
        query = f"{instruction}السؤال: {question}\n"
        letter_choices = []
        for index, choice in enumerate(choices):
            query += f"{LETTER_INDICES[index]}) {choice}\n"
            letter_choices.append(LETTER_INDICES[index])
        query += "الإجابة:"

        return {"query": query, "choices": letter_choices, "gold": answer_index}

    return dataset.map(_process_doc)
