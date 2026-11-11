# Validation set comparison (only available scored models)

| Run | CE (nats) | CE (bits) | Perplexity | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|---|
| unigram (k=0.0001) | 6.8722 | 9.9145 | 965.09 | 5.81% | 22.11% | 31.72% |
| bigram (k=0.003) | 6.2234 | 8.9785 | 504.41 | 18.40% | 37.87% | 45.97% |
| trigram (k=0.001) | 7.9460 | 11.4637 | 2824.38 | 17.59% | 35.28% | 43.70% |
| interpolated trigram | 5.6627 | 8.1695 | 287.91 | 19.81% | 39.64% | 46.85% |
| Transformer | 4.6750 | 6.7446 | 107.24 | 25.48% | 46.54% | 54.69% |
