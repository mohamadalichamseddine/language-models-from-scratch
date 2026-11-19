# T23: where a longer context helps

The [long-context table](../results/long_context/long_context.md) compares the Transformers trained
with 128 and with 6 tokens of context on every test token (two seeds each). A token's gain is its loss
with k=6 minus its loss with k=128. Over the whole test split the gain is 0.247 nats per token
(perplexity 130.9 → 102.2). The table and the examples come from:

```bash
python scripts/long_context.py
```

It needs the T21 test predictions: unzip `results/t21_results.zip` from the repo root, or run
`scripts/final_evaluation.py`.

## Findings

**Most of the gain comes from words the model can copy.** 44% of test tokens are copy-like: the
target word appeared 7-128 tokens back, which only the k=128 model can see. They gain 0.33 nats
each and give 58% of the total gain. Content words in this group gain the most of all, 1.18 nats
each (k=128 is better on 77% of them), so 8.7% of the tokens give 42% of the total gain. The
examples are names and terms repeated from earlier in the article, like "University" in a list of
universities, or "typhoon" in an article about a typhoon.

**Content words need the context; function words barely do.** Content words gain 0.49 nats on
average and give 79% of the total gain. Function words gain 0.055 nats (8% of the total), so
grammar is handled by the nearest 6 tokens almost as well as by 128.

**A longer context helps even when there is nothing to copy.** Content words that did not appear
in the previous 128 tokens still gain 0.29 nats and give 36% of the total gain: the longer window
tells the model what the article is about, so topic words become more likely.

**Some gains are structural.** Punctuation, numbers and `<eos>` gain 0.22 nats when they last
appeared 7-128 tokens back. One printed example is a closing quotation mark whose opening mark is
more than 6 tokens back, so the k=6 model cannot know a quotation is open.

**`<unk>` gains almost nothing (0.04 nats).** It stands for many different rare words, so seeing an
earlier `<unk>` says little about the next one.

**Little is gained beyond about 64 tokens.** In T19, the k=128 models lose only about 1 perplexity
point on test when their history is cut to 64 tokens (103.3 vs 102.2), 4.5 points at 32 tokens and
37 points at 6 tokens. The model trained at k=64 is even slightly better than the one trained at
k=128 (100.2 vs 102.2, see [the final table](../results/final/final.md)).

## Comparison with Khandelwal et al. (2018)

Khandelwal, He, Qi and Jurafsky, *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use
Context* (ACL 2018), shuffled, replaced or dropped parts of the context of an LSTM language model at
test time and measured how much worse it got. Their main findings, and how ours compare:

- **The nearby context matters most.** They found that the LSTM uses about 200 tokens of context
  but treats the nearest ~50 sharply and the distant history only as a fuzzy topic. Our Transformer
  shows the same shape over a shorter range: most of the benefit comes from the nearest 32-64
  tokens, and going from 64 to 128 adds about 1%.
- **Content words need far more context than function words.** They found this for the LSTM, and
  we find it for the Transformer: content words gain about 9 times more from the longer context
  than function words.
- **Copying from the distant context.** They found that the LSTM benefits from a neural cache
  (Grave et al., 2017) to copy words from the distant context. Our Transformer needs no cache:
  self-attention can look up an earlier token directly, and copy-like content words are where it
  gains most.
- **The distant context as a topic.** Their "fuzzy far away" matches our gain on content words
  that did not appear in the previous 128 tokens: the context helps without anything to copy.

The comparison has limits. They studied LSTMs with contexts of hundreds of tokens and perturbed
the context of one trained model at test time. We compare separately trained Transformers with at
most 128 tokens; only T19 perturbs one model's context at test time, and only by cutting it.

## Reference

Urvashi Khandelwal, He He, Peng Qi and Dan Jurafsky. 2018. Sharp Nearby, Fuzzy Far Away: How Neural
Language Models Use Context. In *Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (ACL)*, pages 284–294. https://aclanthology.org/P18-1027/
