# Data

## WikiText-2 (`raw/wikitext-2/`)

| File | Split | Lines | Articles |
|---|---|---|---|
| `wiki.train.tokens` | train | 36,718 | 600 |
| `wiki.valid.tokens` | validation | 3,760 | 60 |
| `wiki.test.tokens` | test | 4,358 | 60 |

WikiText-2 is a set of Good and Featured English Wikipedia articles made for language modeling. We use the tokenized version (`wikitext-2-v1`):

- Words and punctuation are already separated by spaces.
- Rare words are replaced with `<unk>`; every word left in the train split appears at least 3 times.
- Article titles look like ` = Title = ` and section headings like ` = = Heading = = `.
- Numbers are split with `@-@`, `@,@` and `@.@`, e.g. `1 @,@ 000` for "1,000".

Full statistics are in [`results/data_stats/stats.md`](../results/data_stats/stats.md).

**Source:** Hugging Face [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-v1`, downloaded with `scripts/download_wikitext2.py`. Each Hugging Face row is written as one line; the text is otherwise unchanged. Run the script to re-create the files.

**License:** [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) (also GFDL), as stated on the dataset card. You can share and adapt the data if you credit the source; a modified version you share must use the same license.

**Citation:** Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. *Pointer Sentinel Mixture Models*. arXiv:1609.07843.

## Generated files (`processed/`)

The vocabulary and encoded splits are built from the raw text by our scripts. They are not committed.
