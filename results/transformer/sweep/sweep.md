# Transformer sweep at context 128

Learning rate x dropout; every other setting is the default of scripts/train_transformer.py.
Validation perplexity comes from sliding-window scoring. The best run is in bold.

| Learning rate | Dropout | Val perplexity | Top-1 | Best step | Steps trained | Minutes |
|---|---|---|---|---|---|---|
| 0.0003 | 0.1 | 114.38 | 25.22% | 4000 | 4000 | 11.2 |
| 0.0003 | 0.2 | 122.44 | 24.33% | 4000 | 4000 | 11.3 |
| 0.0003 | 0.3 | 136.15 | 23.26% | 3900 | 4000 | 11.3 |
| 0.0006 | 0.1 | **107.24** | 25.48% | 3300 | 3800 | 10.7 |
| 0.0006 | 0.2 | 107.70 | 25.36% | 4000 | 4000 | 11.2 |
| 0.0006 | 0.3 | 117.91 | 24.25% | 4000 | 4000 | 11.2 |
| 0.001 | 0.1 | 113.25 | 25.41% | 2400 | 2900 | 8.2 |
| 0.001 | 0.2 | 107.64 | 25.70% | 4000 | 4000 | 11.3 |
| 0.001 | 0.3 | 114.06 | 24.68% | 4000 | 4000 | 11.3 |

Trained on: Tesla T4.
