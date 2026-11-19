"""Rule baseline: edit-distance lexicon matcher (technique area B).

A message is OFFENSIVE if any of its tokens matches an entry of the hand-written lexicon
(resources/offensive_lexicon_tr.txt).  Matching uses a *weighted* minimum edit distance
(dynamic programming, Jurafsky & Martin ch. 2): insertion / deletion / substitution cost 1,
but substituting a masking symbol ('*', '.', '_') or a leetspeak digit that maps to the
expected letter costs 0, so "s*ktir", "4mk" and "ger1zekalı" match at distance 0.

  - entries shorter than 8 letters: only distance 0 - a one-letter typo of a short word is
    usually another real word (yavşak/yasak, kaltak/kalmak, yarrak/yaprak, götünü/gözünü)
  - entries of 8+ letters: distance <= 1 ("şerefsz", "gerizekali")
  - Turkish letters are folded to ASCII except ı (adı "name" vs. adi "vile")
  - "stem*" entries are compared against token prefixes of length len(stem) - 1 .. len(stem) + 1

The matched words are also returned, so the application can mask them in the chat.

Run:  python -m src.lexicon      (tunes nothing; evaluates on val, then on test)
"""
from pathlib import Path

import numpy as np

from src.normalize import LEET, normalize, tokenize

ROOT = Path(__file__).resolve().parents[1]
LEXICON_FILE = ROOT / "resources" / "offensive_lexicon_tr.txt"
WILDCARDS = set("*._")
# fold ç ş ğ ö ü â î û to ASCII but keep ı: folding ı -> i makes "adı" (name) = "adi" (vile)
# and "çıktığı" one edit away from "siktiği"
_LEX_FOLD = str.maketrans("çğöşüâîû", "cgosuaiu")
MAX_WILDCARDS = 2


def lex_fold(s: str) -> str:
    return s.translate(_LEX_FOLD)


LEET_FOLDED = {k: lex_fold(v) for k, v in LEET.items()}


def load_lexicon(path=LEXICON_FILE):
    exact, stems = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (stems if line.endswith("*") else exact).append(lex_fold(line.rstrip("*")))
    return exact, stems


def _sub_cost(a: str, b: str, free_wildcards: bool = True) -> int:
    """Cost of turning token char a into lexicon char b."""
    if a == b or LEET_FOLDED.get(a) == b or (free_wildcards and a in WILDCARDS):
        return 0
    return 1


def weighted_edit_distance(token: str, entry: str) -> int:
    # masking symbols are free only inside a word and at most MAX_WILDCARDS times ("s*kt*r"),
    # so "**dram**" or "*allah" do not match anything
    token = token.strip("".join(WILDCARDS))
    free = sum(c in WILDCARDS for c in token) <= MAX_WILDCARDS
    n, m = len(token), len(entry)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = min(d[j] + 1,                     # deletion
                      d[j - 1] + 1,                 # insertion
                      prev + _sub_cost(token[i - 1], entry[j - 1], free))  # substitution
            prev, d[j] = d[j], cur
    return d[m]


class LexiconMatcher:
    def __init__(self, path=LEXICON_FILE):
        self.exact, self.stems = load_lexicon(path)
        self.exact_set = set(self.exact)

    @staticmethod
    def _max_dist(entry):
        return 0 if len(entry) < 8 else 1

    def match_token(self, tok: str):
        """Return the lexicon entry a (folded) token matches, or None."""
        if tok in self.exact_set:
            return tok
        if not any(c.isalpha() for c in tok):
            return None
        for e in self.exact:
            if abs(len(tok) - len(e)) <= self._max_dist(e) and \
                    weighted_edit_distance(tok, e) <= self._max_dist(e):
                return e
        for s in self.stems:
            k = self._max_dist(s)
            for L in range(len(s) - k, len(s) + k + 1):
                if L <= len(tok) and weighted_edit_distance(tok[:L], s) <= k:
                    return s + "*"
        return None

    def find(self, text: str):
        """Matched (original token, lexicon entry) pairs in a raw message."""
        hits = []
        for tok in tokenize(normalize(text)):
            e = self.match_token(lex_fold(tok))
            if e:
                hits.append((tok, e))
        return hits

    def predict_proba(self, texts):
        return np.array([1.0 if self.find(t) else 0.0 for t in texts])


if __name__ == "__main__":
    from src.data import load
    from src.evaluate import metrics, report

    m = LexiconMatcher()
    print(f"lexicon: {len(m.exact)} exact entries, {len(m.stems)} stems")
    for t in ["s*ktir git", "4mk ya", "ŞEREFSİZLER", "orspu çocuğu", "götürdüm eve", "sıkıldım ya",
              "malzeme aldım", "g e r i z e k a l ı"]:
        print(f"  {t!r:28} -> {m.find(t)}")
    val = load("val")
    print("val:", metrics(val.label, (m.predict_proba(val.text) >= 0.5).astype(int)))
    report("lexicon_rules", m.predict_proba, load("test"), 0.5, [LEXICON_FILE],
           extra={"family": "classical (rules)", "precision": "n/a"})
