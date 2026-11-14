# Validation context truncation: ablation/k128_seed0

| Run | CE (nats) | CE (bits) | Perplexity | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|---|
| trained k=128, scored k=6 | 4.9972 | 7.2094 | 148.00 | 23.35% | 43.92% | 51.82% |
| trained k=128, scored k=32 | 4.7251 | 6.8169 | 112.74 | 25.58% | 46.41% | 54.46% |
| trained k=128, scored k=64 | 4.6903 | 6.7666 | 108.88 | 25.72% | 46.69% | 54.76% |
| trained k=128, scored k=128 | 4.6802 | 6.7521 | 107.79 | 25.63% | 46.64% | 54.81% |
