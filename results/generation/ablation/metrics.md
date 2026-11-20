# Full generation analysis

Validation prompts: 30 per run, 10 tokens each; 100 generated tokens per continuation. Prompt-selection seed: 42.

Scores are macro-averages over generated tokens only. Context is the maximum number of preceding tokens; training seeds are separate rows. N-grams use their full frozen distribution; Transformers use temperature 1 and top-k 50.

Distinct-n is unique n-grams / all n-grams. Repetition is duplicate 4-grams / all 4-grams. Samples shorter than n receive zero. These are diversity measures, not measures of fluency or correctness.

| Model | Context | Training seed | Samples | Distinct-1 | Distinct-2 | Repetition-4 |
|---|---:|---:|---:|---:|---:|---:|
| unigram | 0 | — | 30 | 0.7573 | 0.9943 | 0.0000 |
| bigram | 1 | — | 30 | 0.9330 | 0.9889 | 0.0027 |
| interpolated trigram | 2 | — | 30 | 0.7653 | 0.9875 | 0.0003 |
| trigram | 2 | — | 30 | 0.9983 | 1.0000 | 0.0000 |
| Transformer | 6 | 0 | 30 | 0.6430 | 0.9327 | 0.0062 |
| Transformer | 6 | 1 | 30 | 0.6407 | 0.9360 | 0.0065 |
| Transformer | 32 | 0 | 30 | 0.6397 | 0.9300 | 0.0031 |
| Transformer | 32 | 1 | 30 | 0.6500 | 0.9458 | 0.0017 |
| Transformer | 64 | 0 | 30 | 0.5923 | 0.9088 | 0.0141 |
| Transformer | 64 | 1 | 30 | 0.6343 | 0.9327 | 0.0017 |
| Transformer | 128 | 0 | 30 | 0.6017 | 0.9034 | 0.0052 |
| Transformer | 128 | 1 | 30 | 0.6007 | 0.9047 | 0.0041 |
