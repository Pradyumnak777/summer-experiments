# make_controlnet_pairs.py
import json
from pathlib import Path

CH_A = Path("data/microscopy_lora_chA").resolve()   # condition (Sox2/halo549)
CH_B = Path("data/microscopy_lora_chB").resolve()   # target    (MCP/snap646)
OUT  = Path("data/controlnet_pairs")
OUT.mkdir(parents=True, exist_ok=True)

CAPTION = "an image of cells in fluroscent microscopy"

rows = 0
with (OUT / "metadata.jsonl").open("w") as f:
    for fa in sorted(CH_A.glob("*_chA.png")):
        base = fa.name[:-len("_chA.png")]
        fb = CH_B / f"{base}_chB.png"
        if not fb.exists():
            continue
        # symlink the target (chB) into the dataset dir so imagefolder can decode it
        link = OUT / f"{base}_chB.png"
        if not link.exists():
            link.symlink_to(fb)
        f.write(json.dumps({
            "file_name": f"{base}_chB.png",   # relative -> becomes the decoded "image" column
            "conditioning_image": str(fa),    # absolute -> we cast this to an image in the train script
            "text": CAPTION,
        }) + "\n")
        rows += 1

print(f"wrote {rows} pairs -> {OUT}/metadata.jsonl")