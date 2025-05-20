# %% [markdown]
# ### Code to be followed

# %%
""" Random seed """
from transformers import set_seed
set_seed(11207330)

# %%
""" Load dataset"""

from pathlib import Path
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

def load_data(dataset_name_or_path: str, 
              prompt_template_path: str = None):
    """ Init: multiple choices answer """
    all_choices = ["A", "B", "C", "D"]
    
    """ Init: prompt template"""
    if prompt_template_path is not None:
        try:
            with Path(prompt_template_path).open("r", encoding="utf-8") as file:
                prompt_template = file.read()
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load the prompt template: {e}") from e
    else:
        prompt_template = "{question}"

    
    """ preprocess function: use the prompt template to format the question """
    def _preprocess(example):
        example["question"] = (
            f"{example['instruction']}\n"
            + f"{example['question']}\n"
            + "".join(example[f"option{i + 1}"] for i in range(len(all_choices)))
        )
        example["answer"] = example["answer"].replace("A", "0").replace("B", "1").replace("C", "2").replace("D", "3") # anwer to index: 這邊是用 0,1,2,3來表示答案
        example["question"] = prompt_template.format(question=example["question"])
        return example
    
    dataset = load_dataset(
                    "json",
                    data_files=dataset_name_or_path,
                    split="train",
                )
    
    return_dataset = Dataset.from_list([
                _preprocess(example)
                for example in tqdm(dataset, desc="Processing dataset", unit="example")
            ])
    return return_dataset

# load_data(dataset_name_or_path="TOCFL-MultiBench/TOCFL-MultiBench.json", prompt_template_path="prompt/base.txt")

# %%
""" metrics"""
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(
    all_choices: list,
    all_answers: list,
    all_response: list,
    all_index2ans: list = None,
    allow_random: bool = True,
) -> dict:
    """calculate_metrics"""
    if all_index2ans is None:
        all_index2ans = [None] * len(all_response)

    predictions = [
        parse_multi_choice_response(response, all_choices, index2ans, allow_random)
        for response, index2ans in zip(all_response, all_index2ans)
    ]

    accuracy = accuracy_score(all_answers, predictions)
    f1 = f1_score(all_answers, predictions, average="weighted", zero_division=1)
    precision = precision_score(all_answers, predictions, average="weighted", zero_division=1)
    recall = recall_score(all_answers, predictions, average="weighted", zero_division=1)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def parse_multi_choice_response(
    response: str,
    all_choices: list = ["A", "B", "C", "D"],
    index2ans: dict = None,
    allow_random: bool = True,
) -> str:
    """parse_multi_choice_response"""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)

    if index2ans is not None and len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        if allow_random:
            pred_index = random.choice(all_choices)
        else:
            pred_index = ""

    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index

# %% [markdown]
# ### Test Area

# %%
# data = load_data(dataset_name_or_path="TOCFL-MultiBench/TOCFL-MultiBench.json", prompt_template_path="prompt/base.txt")

# """ Usage: 問題底家 """
# print(data)
# print(data['question'])
# print(data['answer']) # answer 被弄成數值，以搭配calculate_metrics。

# %%
# 1. 定義可選項目（模型可能輸出的 label）
# all_choices   = ["A", "B", "C", "D"]

# 2. 定義「正確答案」列表（ground truth）
#    比方說我們有四題，正確答案分別是 A, B, C, A
# all_answers   = ["A", "B", "C", "A"]

# 3. 定義模型回傳的「原始字串」列表
#    這裡假設模型回了跟正確一樣的 four responses
# all_response  = ["A", "B", "C", "A"]

# 4. （選擇性）定義 index2ans 映射
#    當模型回的是文字（例如 "dog"、"cat"）而非 A/B/C/D 時，用這個 dict 幫它對回標籤。
#    key 是選項字母，value 是對應的文字答案。
#    如果你的模型只回 A/B/C/D，就可以不傳這個參數（預設會是全 None）。
# all_index2ans = None

# 呼叫 calculate_metrics
# metrics = calculate_metrics(
#     all_choices=all_choices,
#     all_answers=all_answers,
#     all_response=all_response,
#     all_index2ans=all_index2ans,
#     allow_random=True,         # 若 parse 不出答案，是否隨機選一個
# )

# print(metrics)


