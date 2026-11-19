# Rare-token analysis (test split)

Test targets grouped by how often the word occurs in train. WikiText-2 already replaced the words
seen fewer than 3 times with `<unk>`, so the rarest group starts at 3. Transformer rows average two seeds.

| Group | 1-10 | 11-100 | 101-1k | >1k | `<unk>` |
|---|---:|---:|---:|---:|---:|
| Test tokens | 10,331 (4.2%) | 36,810 (15.0%) | 57,713 (23.5%) | 125,497 (51.1%) | 15,218 (6.2%) |

## Cross-entropy (nats per token)

| Model | 1-10 | 11-100 | 101-1k | >1k | `<unk>` |
|---|---:|---:|---:|---:|---:|
| unigram (k=0.0001) | 12.800 | 10.903 | 8.801 | 4.595 | 3.644 |
| bigram (k=0.003) | 12.395 | 10.219 | 7.677 | 3.997 | 3.902 |
| trigram (k=0.001) | 11.331 | 10.529 | 8.895 | 6.484 | 6.565 |
| interpolated trigram | 12.146 | 9.797 | 7.187 | 3.357 | 3.399 |
| Transformer k=6 | 11.600 | 8.631 | 6.199 | 2.781 | 3.459 |
| Transformer k=32 | 11.160 | 8.141 | 5.868 | 2.666 | 3.464 |
| Transformer k=64 | 11.055 | 8.039 | 5.823 | 2.663 | 3.349 |
| Transformer k=128 | 11.060 | 8.028 | 5.816 | 2.700 | 3.415 |

## Top-1 accuracy

| Model | 1-10 | 11-100 | 101-1k | >1k | `<unk>` |
|---|---:|---:|---:|---:|---:|
| unigram (k=0.0001) | 0.00% | 0.00% | 0.00% | 11.16% | 0.00% |
| bigram (k=0.003) | 0.34% | 1.28% | 3.15% | 31.49% | 23.20% |
| trigram (k=0.001) | 0.89% | 2.31% | 5.30% | 29.00% | 21.42% |
| interpolated trigram | 1.02% | 2.46% | 4.86% | 33.04% | 23.76% |
| Transformer k=6 | 0.61% | 2.85% | 8.55% | 39.88% | 25.29% |
| Transformer k=32 | 0.69% | 3.43% | 10.28% | 42.64% | 24.42% |
| Transformer k=64 | 0.57% | 3.48% | 10.47% | 42.72% | 26.42% |
| Transformer k=128 | 0.58% | 3.47% | 10.45% | 41.96% | 24.71% |
