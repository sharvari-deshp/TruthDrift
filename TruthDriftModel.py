#TruthDrift: An ML model designed to detect hallucinations within AI generated text. 
import json

feverFile = []
with open("FEVER Dataset JSON Files/train (1).jsonl", "r") as parse:
    for line in parse: 
        text = json.loads(line) #converts to dict type

        if (text['label'] != "NOT ENOUGH INFO"):
            feverFile.append(text)
            
        
#test if dataset is imported
print(feverFile[0])
print(len(feverFile))
