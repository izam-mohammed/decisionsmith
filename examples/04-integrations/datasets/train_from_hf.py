"""Train and evaluate on a Hugging Face dataset by name: `model.train("hf:<dataset>")`, `evaluate("hf:<dataset>:test")`.

Here the dataset is a local folder of Parquet files (with a `ClassLabel` column, as many hub datasets have);
a hub id like `hf:my-org/tickets` works the same way. Label columns hold option names (or ClassLabel ids).
"""

import os

from datasets import ClassLabel, Dataset, Features, Value

import decisionsmith as ds

TEAMS = ["billing", "technical", "sales"]
EXAMPLES = {
    "billing": ["I was charged twice", "My invoice is wrong", "Please refund my payment", "The card charge failed"],
    "technical": ["The app crashes on login", "Sync is broken", "Uploads time out", "The page shows an error"],
    "sales": ["How much is the plan?", "Can I get a quote?", "Do you offer discounts?", "I want to upgrade"],
}
rows = [(f"{text} ({i})", TEAMS.index(team)) for i in range(8) for team, xs in EXAMPLES.items() for text in xs]
os.makedirs("tickets", exist_ok=True)
features = Features({"text": Value("string"), "label": ClassLabel(names=TEAMS)})
for split, part in (("train", rows[:72]), ("test", rows[72:])):
    Dataset.from_dict({"text": [r[0] for r in part], "label": [r[1] for r in part]}, features=features).to_parquet(
        "tickets/%s.parquet" % split
    )

model = ds.model(TEAMS)
model.train("hf:tickets", epochs=1)
print(model.evaluate("hf:tickets:test"))
