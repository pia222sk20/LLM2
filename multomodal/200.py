from datasets import load_dataset

dataset = load_dataset("food101", split="train")
label_names = dataset.features["label"].names

for cls in ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare']:
    print(cls, "→", label_names.index(cls))
