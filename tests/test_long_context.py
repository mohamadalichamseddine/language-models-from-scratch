import numpy as np

from lm.data import Vocab
from lm.long_context import CONTEXT_GROUPS, distance_to_previous, context_groups, word_classes


def test_distance_counts_back_to_the_last_occurrence():
    assert distance_to_previous([5, 7, 5, 5]).tolist() == [np.inf, np.inf, 2, 1]


def test_context_groups_split_at_6_and_128_tokens_back():
    filler = list(range(10, 200))  # tokens that never repeat
    ids = [1] + filler[:5] + [1]  # 1 again 6 tokens back
    ids += filler[5:11] + [1]  # 7 tokens back
    ids += filler[11:139] + [1]  # 129 tokens back
    groups = context_groups(ids)
    assert groups[6] == CONTEXT_GROUPS[0]
    assert groups[13] == CONTEXT_GROUPS[1]
    assert groups[142] == CONTEXT_GROUPS[2]
    assert groups[0] == CONTEXT_GROUPS[2]


def test_word_classes():
    vocab = Vocab(["<eos>", "<unk>", "The", "cat", "'s", "@-@", "1999", "H2O"])
    classes = word_classes(np.arange(len(vocab)), vocab).tolist()
    assert classes == ["punctuation, numbers, <eos>", "<unk>", "function word", "content word",
                       "function word", "punctuation, numbers, <eos>", "punctuation, numbers, <eos>",
                       "content word"]
