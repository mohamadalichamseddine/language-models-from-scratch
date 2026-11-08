# Add-k tuning on the validation split

Validation perplexity for each k. The best k for each model is in bold.

| k | Unigram | Bigram | Trigram |
|---|---|---|---|
| 1 | 967.60 | 2019.37 | 12071.26 |
| 0.3 | 965.69 | 1240.03 | 8349.95 |
| 0.1 | 965.27 | 862.12 | 5998.27 |
| 0.03 | 965.14 | 639.78 | 4367.97 |
| 0.01 | 965.10 | 539.57 | 3481.83 |
| 0.003 | 965.09 | **504.41** | 2965.44 |
| 0.001 | 965.09 | 525.52 | **2824.38** |
| 0.0003 | 965.09 | 601.24 | 3028.63 |
| 0.0001 | **965.09** | 719.09 | 3625.45 |

| Model | Best k | Validation perplexity | Tokens with zero probability at k = 0 |
|---|---|---|---|
| Unigram | 0.0001 | 965.09 | 0.00% |
| Bigram | 0.003 | 504.41 | 21.35% |
| Trigram | 0.001 | 2824.38 | 38.47% |
