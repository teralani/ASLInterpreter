
import json
import random

INPUT = "data/combined_dict.json"
OUTPUT = "data/combined_dict.json"

TRAIN_RATIO = 40154 / 83399
VAL_RATIO = 10304 / 83399

SEED = 42

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

blank_class = next(
    (item for item in data if item.get("gloss") == "blank"),
    None
)

if blank_class is None:
    raise ValueError()

instances = blank_class["instances"]

random.seed(SEED)
random.shuffle(instances)

total = len(instances)

train_n = round(total * TRAIN_RATIO)
val_n = round(total * VAL_RATIO)
test_n = total - train_n - val_n

print(f"Blank instances: {total}")
print(f"Train: {train_n}")
print(f"Val:   {val_n}")
print(f"Test:  {test_n}")

for i, instance in enumerate(instances):
    if i < train_n:
        instance["split"] = "train"
    elif i < train_n + val_n:
        instance["split"] = "val"
    else:
        instance["split"] = "test"

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved to: {OUTPUT}")
