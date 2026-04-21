#TruthDrift: An ML model designed to detect hallucinations within AI generated text. 

from datasets import load_dataset
feverFile = load_dataset("json", data_files="train.jsonl", trust_remote_code=True)
