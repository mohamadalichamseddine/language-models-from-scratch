# Where a longer context helps: Transformer k=128 vs k=6 (test split)

Gain = loss with k=6 minus loss with k=128 (nats per token, each averaged over two seeds);
positive means the longer context helps. Over all test tokens the gain is 0.2472 nats.
"7-128 back only" are the copy-like cases: the target word appeared in the previous 128 tokens
but not in the previous 6, so only the k=128 model could see it.

## By where the target word last appeared

| Group | Tokens | Share | Loss k=6 | Loss k=128 | Gain | k=128 better | Share of total gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| within previous 6 | 22,821 | 9.3% | 2.214 | 2.150 | +0.064 | 51.7% | 2.4% |
| 7-128 back only | 108,514 | 44.2% | 3.601 | 3.274 | +0.327 | 60.5% | 58.4% |
| not in previous 128 | 114,234 | 46.5% | 6.615 | 6.407 | +0.208 | 56.3% | 39.2% |

## By word class

| Group | Tokens | Share | Loss k=6 | Loss k=128 | Gain | k=128 better | Share of total gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| function word | 85,801 | 34.9% | 3.233 | 3.179 | +0.055 | 53.0% | 7.7% |
| content word | 97,832 | 39.8% | 7.583 | 7.093 | +0.490 | 62.7% | 79.0% |
| `<unk>` | 15,218 | 6.2% | 3.459 | 3.415 | +0.044 | 52.5% | 1.1% |
| punctuation, numbers, `<eos>` | 46,718 | 19.0% | 2.676 | 2.518 | +0.158 | 57.5% | 12.2% |

## By word class and where the word last appeared

| Group | Tokens | Share | Loss k=6 | Loss k=128 | Gain | k=128 better | Share of total gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| function word / within previous 6 | 6,852 | 2.8% | 2.130 | 2.100 | +0.030 | 52.3% | 0.3% |
| function word / 7-128 back only | 51,174 | 20.8% | 2.695 | 2.615 | +0.080 | 54.4% | 6.7% |
| function word / not in previous 128 | 27,775 | 11.3% | 4.498 | 4.483 | +0.015 | 50.8% | 0.7% |
| content word / within previous 6 | 1,007 | 0.4% | 6.775 | 5.904 | +0.871 | 72.3% | 1.4% |
| content word / 7-128 back only | 21,412 | 8.7% | 7.321 | 6.145 | +1.176 | 77.1% | 41.5% |
| content word / not in previous 128 | 75,413 | 30.7% | 7.669 | 7.379 | +0.290 | 58.5% | 36.1% |
| `<unk>` / within previous 6 | 6,051 | 2.5% | 2.986 | 2.923 | +0.063 | 54.3% | 0.6% |
| `<unk>` / 7-128 back only | 9,072 | 3.7% | 3.771 | 3.734 | +0.037 | 51.5% | 0.5% |
| `<unk>` / not in previous 128 | 95 | 0.0% | 3.839 | 4.295 | -0.456 | 26.3% | -0.1% |
| punctuation, numbers, `<eos>` / within previous 6 | 8,911 | 3.6% | 1.240 | 1.239 | +0.001 | 47.2% | 0.0% |
| punctuation, numbers, `<eos>` / 7-128 back only | 26,856 | 10.9% | 2.304 | 2.086 | +0.218 | 61.9% | 9.7% |
| punctuation, numbers, `<eos>` / not in previous 128 | 10,951 | 4.5% | 4.755 | 4.617 | +0.138 | 55.0% | 2.5% |
