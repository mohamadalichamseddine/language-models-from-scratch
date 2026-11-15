# Validation context truncation: ablation/k128_seed1

| Run | CE (nats) | CE (bits) | Perplexity | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|---|
| trained k=128, scored k=6 | 4.9983 | 7.2110 | 148.16 | 23.28% | 43.94% | 51.88% |
| trained k=128, scored k=32 | 4.7263 | 6.8185 | 112.87 | 25.39% | 46.33% | 54.50% |
| trained k=128, scored k=64 | 4.6896 | 6.7656 | 108.80 | 25.68% | 46.60% | 54.84% |
| trained k=128, scored k=128 | 4.6762 | 6.7464 | 107.37 | 25.66% | 46.63% | 54.79% |
