"""Download WikiText-2 and save each split as a text file in data/raw/wikitext-2/.

The files are already committed, so this is only needed to re-create them.
We use the tokenized version (wikitext-2-v1), where rare words are already
replaced by <unk>.
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "wikitext-2"

# Hugging Face split name -> file name from the original WikiText release
FILE_NAMES = {
    "train": "wiki.train.tokens",
    "validation": "wiki.valid.tokens",
    "test": "wiki.test.tokens",
}


def main():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, file_name in FILE_NAMES.items():
        rows = dataset[split]["text"]
        path = OUT_DIR / file_name

        # Blank rows come through as "" with no newline, so strip the newline
        # and add it back to keep exactly one row per line.
        # newline="\n" stops Windows from writing \r\n.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(row.rstrip("\n") + "\n")

        print(f"{split}: {len(rows):,} lines -> {path}")


if __name__ == "__main__":
    main()
