# Load model directly
# from transformers import AutoProcessor, AutoModel

import gc
import torch
from datasets import Dataset, load_dataset
import time
from tqdm.auto import tqdm

def clear_resources(name: str) -> None:
    # if hasattr(self, name):
    #     delattr(self, name)
    torch.cuda.empty_cache()
    gc.collect()

def load_data(dataset_name_or_path: str, 
              prompt_template_path: str = None) -> Dataset:
    """ Init """
    all_choices = ["A", "B", "C", "D"]
    
    """ preprocess function"""
    def _preprocess(example, prompt_template):
        example["question"] = (
            f"{example['instruction']}\n"
            + f"{example['question']}\n"
            + "".join(example[f"option{i + 1}"] for i in range(len(all_choices)))
        )
        example["answer"] = example["answer"].replace("A", "0").replace("B", "1").replace("C", "2").replace("D", "3")
        return example
    
    prompt_template = open(prompt_template_path, "r").read()
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

def load_model(model_name_or_path: str):
    pass
    
    # """ Load model (sample code) """
    # from transformers import AutoProcessor, AutoModelForCausalLM
    # processor = AutoProcessor.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # return processor, model

def evaluate():
    ###############
    start_time = time.time()
    """ write your evaluation code here"""
    end_time = time.time()
    ###############
    pass

def main():
    """ Parameter """
    dataset_name_or_path = "TOCFL-MultiBench/TOCFL-MultiBench.json"
    prompt_template_path = None
    model_name_or_path = "???"
    tensor_type = "???" # "bf16", "auto"
    
    """ Load dataset """
    dataset = load_data(dataset_name_or_path, prompt_template_path)
    print(f"Dataset loaded: {dataset}")
    
    ###############
    """ Load model """
    """ Evaluation """
    ###############
    
if __name__ == "__main__":
    main()
