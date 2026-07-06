# split_train_val.py
import json, random, shutil
from pathlib import Path

SRC = Path("data/singlecell_previews")
DST = Path("data/singlecell_previews_split")
VAL_FRACTION = 0.05  
SEED = 42

def main():
    rows = [json.loads(l) for l in (SRC / "metadata.jsonl").open()]
    random.Random(SEED).shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    for split_name, split_rows in [("train", train_rows), ("validation", val_rows)]:
        split_dir = DST / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "metadata.jsonl").open("w") as f:
            for r in split_rows:
                shutil.copy(SRC / r["file_name"], split_dir / r["file_name"])
                f.write(json.dumps(r) + "\n")

    print(f"train={len(train_rows)}  validation={len(val_rows)}")

if __name__ == "__main__":
    main()