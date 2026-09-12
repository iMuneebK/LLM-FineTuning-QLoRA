from datasets import load_dataset
import pandas as pd
import json

def load_and_prepare_data(dataset_path):
    """
    Loads dataset and formats it for instruction tuning.
    Expects a JSONL file with 'instruction', 'input', and 'output' fields.
    """
    try:
        # Assuming local JSONL file for this example
        dataset = load_dataset('json', data_files={'train': dataset_path}, split='train')
    except Exception as e:
        print(f"Failed to load local dataset: {e}")
        print("Falling back to a sample huggingface dataset...")
        dataset = load_dataset('databricks/databricks-dolly-15k', split='train[:1000]')

    def format_instruction(sample):
        """Formats the input into a standard prompt template"""
        instruction = f"### Instruction:\n{sample.get('instruction', sample.get('question', ''))}\n"
        context = f"### Input:\n{sample.get('input', sample.get('context', ''))}\n" if sample.get('input', sample.get('context', '')) else ""
        response = f"### Response:\n{sample.get('output', sample.get('response', ''))}"
        
        full_prompt = instruction + context + response
        return {"text": full_prompt}

    print("Formatting dataset...")
    formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    print(f"Prepared {len(formatted_dataset)} examples.")
    
    return formatted_dataset

if __name__ == "__main__":
    # Test execution
    load_and_prepare_data("sample_data.jsonl")
