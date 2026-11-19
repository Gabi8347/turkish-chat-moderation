"""Regex normalization of noisy Turkish chat text (technique area B).

Every model in the ladder sees the output of `normalize()`, so the comparison is fair.
Rules (applied in this order):
  1. placeholders     : @USER -> @user, URL/http... -> url
  2. spaced letters   : "İ S T E M İ Y O R U M" / "s.i.k" -> "İSTEMİYORUM" / "sik"
  3. leetspeak        : one digit/symbol between two letters -> letter ("gelm3ye" -> "gelmeye",
                        "S1F1R" -> "SIFIR"). Skipped for hashtags, e-mails, links,
                        mixed-case IDs and tokens with >3 digits (uuids), where digits are
                        real ("#GTHB5BİN", "wig1K7qbXM").
                        Word-initial digits are left alone: in this corpus they are almost
                        always numerals with a suffix ("5dk", "3gün"); "4mk"-style spellings
                        are caught later by the edit-distance lexicon matcher.
  4. hashtags         : '#tag' -> 'tag'
  5. Turkish lowercase: "I" -> "ı", "İ" -> "i" before str.lower()
  6. repeated letters : 3+ identical letters -> 1  ("çoooook" -> "çok", "ammmk" -> "amk")
  7. whitespace       : collapse
`fold()` additionally maps Turkish letters to ASCII (ş->s, ı->i ...) for the lexicon matcher,
so "şerefsiz", "serefsiz" and "SEREFSİZ" all meet in the same form.

Run:  python -m src.normalize    (prints before/after examples)
"""
import re

_LETTER = "a-zA-ZçğıöşüÇĞİÖŞÜâîû"
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "€": "e"}

_RE_USER = re.compile(r"@USER\b")
_RE_URL = re.compile(r"\bURL\b|https?://\S+")
_RE_HASH = re.compile(rf"#([{_LETTER}0-9_]+)")
# >=3 single letters separated by a space . _ or -: "s.i.k", "İ S T E M"
_RE_SPACED = re.compile(rf"(?<![{_LETTER}])((?:[{_LETTER}][ ._\-]){{2,}}[{_LETTER}])(?![{_LETTER}])")
# one leet symbol with a letter on both sides
_RE_LEET = re.compile(rf"(?<=[{_LETTER}])([013457$€])(?=[{_LETTER}])")
_RE_MIXED_CASE = re.compile(r"\S[a-zçğıöşü]\S*[A-ZÇĞİÖŞÜ]|\S[A-ZÇĞİÖŞÜ]\S*[a-zçğıöşü]\S*[A-ZÇĞİÖŞÜ]")
_RE_REPEAT = re.compile(rf"([{_LETTER}])\1{{2,}}")
_RE_WS = re.compile(r"\s+")
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
# tokens: @user, words (may contain '*' masking, e.g. "s*k"), emojis / other symbols one by one
_RE_TOKEN = re.compile(rf"@user|[{_LETTER}0-9*]+|[^\s{_LETTER}0-9]")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _deleet(token: str) -> str:
    if (token[0] == "#" or any(c in token for c in "@./:") or _RE_MIXED_CASE.match(token)
            or sum(c.isdigit() for c in token) > 3):  # ids, uuids, phone numbers
        return token
    return _RE_LEET.sub(lambda m: LEET[m.group(1)], token)


def normalize(text: str) -> str:
    text = _RE_USER.sub(" @user ", text)
    text = _RE_URL.sub("url", text)
    text = _RE_SPACED.sub(lambda m: re.sub(r"[ ._\-]", "", m.group(1)), text)
    text = " ".join(_deleet(t) for t in text.split())
    text = _RE_HASH.sub(r"\1", text)
    text = tr_lower(text)
    text = _RE_REPEAT.sub(r"\1", text)
    return _RE_WS.sub(" ", text).strip()


def fold(text: str) -> str:
    """ASCII-fold Turkish letters (use after normalize)."""
    return text.translate(_FOLD)


def tokenize(text: str) -> list[str]:
    """Simple regex tokenizer used by tf-idf, the lexicon matcher and word2vec."""
    return _RE_TOKEN.findall(text)


if __name__ == "__main__":
    examples = [
        "@USER Bu ne Saçmalık yaaaaa 😂 http://t.co/xyz",
        "İ S T E M İ Y O R U M artık #HerŞeyBitti",
        "s.i.k.t.i.r git b*k herif",
        "gelm3ye korkuyo musun 4mk S1F1R",
        "ISPARTA'DA ŞEREFSİZLER VAR",
        "#GTHB5BİN 5dk sonra 3gün @USER@USER",
    ]
    for e in examples:
        n = normalize(e)
        print(f"{e!r}\n  -> {n!r}\n  fold -> {fold(n)!r}\n  tokens -> {tokenize(n)}\n")
