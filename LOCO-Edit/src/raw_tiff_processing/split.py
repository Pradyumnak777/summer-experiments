import json
import random
import shutil

from .config import (
    CAPTION_A,
    CAPTION_B,
    CAPTION_COMPOSITE,
    OUT_DIR_A,
    OUT_DIR_B,
    OUT_DIR_MASK,
    OUT_DIR_PREVIEW,
    SEED,
    SPLIT_SUFFIX,
    VAL_FRACTION,
)

#(source dir, filename suffix, caption) - the mask is label data so it gets no caption manifest
SPLIT_TARGETS = [
    (OUT_DIR_A, "_chA.png", CAPTION_A),
    (OUT_DIR_B, "_chB.png", CAPTION_B),
    (OUT_DIR_MASK, "_mask.png", None),
    (OUT_DIR_PREVIEW, "_preview.png", CAPTION_COMPOSITE),
]


def read_base_names():
    """base names in extraction order, read off the chA manifest so the shuffle is reproducible"""
    suffix = "_chA.png"
    rows = [json.loads(line) for line in (OUT_DIR_A / "metadata.jsonl").open()]
    return [row["file_name"][: -len(suffix)] for row in rows]


def split_base_names(base_names):
    #one partition drawn once and reused for every channel, so chA/chB/mask/preview stay aligned
    shuffled = list(base_names)
    random.Random(SEED).shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    return shuffled[n_val:], shuffled[:n_val]


def write_split(src_dir, suffix, caption, train_names, val_names):
    if not src_dir.exists():
        print(f"SKIP  {src_dir}  (not extracted)")
        return

    dst_root = src_dir.parent / f"{src_dir.name}{SPLIT_SUFFIX}"
    for split_name, names in [("train", train_names), ("validation", val_names)]:
        split_dir = dst_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for base in names:
            file_name = f"{base}{suffix}"
            src_path = src_dir / file_name
            #masks are optional per cell, so a missing file here is expected rather than an error
            if not src_path.exists():
                continue
            shutil.copy(src_path, split_dir / file_name)
            rows.append({"file_name": file_name, "text": caption})

        if caption is not None:
            with (split_dir / "metadata.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        print(f"{dst_root.name}/{split_name}: {len(rows)} files")


def main():
    base_names = read_base_names()
    train_names, val_names = split_base_names(base_names)
    print(f"train={len(train_names)}  validation={len(val_names)}  seed={SEED}")

    for src_dir, suffix, caption in SPLIT_TARGETS:
        write_split(src_dir, suffix, caption, train_names, val_names)


if __name__ == "__main__":
    main()
