"""A short reply to a search, written only from what we know about the works found.

Two writers, given the same facts (facts()):
- template_reply: fixed sentences filled with the facts, so it can't invent anything.
- LLMReplier: a small local language model (Qwen2.5-0.5B-Instruct) writes a friendlier reply. It can mix facts up,
  so check_reply flags what the reply says that the facts don't, and reply() then shows the template instead.
  check_reply is a tripwire, not a proof: notebooks/09_search.ipynb also reads the replies by eye.

Run it: python -m museum.chat "a stormy sea" [--llm]   (without a query: several questions in a row)
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
import torch

from museum.data import IMAGE_DIR
from museum.guide import DISPLAY_NAMES
from museum.search import Collection
from museum.stories import load_stories

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GENRES = {"still_life": "still life", "genre_painting": "everyday scene", "sketch_and_study": "sketch",
          "landscape": "landscape"}
STYLE_WORDS = {"Baroque": r"baroque", "Cubism": r"cubis[mt]", "Impressionism": r"impressionis[mt]",
               "Pop_Art": r"pop[ -]?art", "Ukiyo_e": r"ukiyo[ -]?e\b"}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
# Painters named in stories.md, so a reply can't start a sentence with one of them unnoticed.
NAMES_IN_STORIES = frozenset({"caravaggio", "rubens", "rembrandt", "vermeer", "velazquez", "picasso", "braque", "gris",
                              "leger", "monet", "renoir", "pissarro", "degas", "morisot", "gogh", "warhol",
                              "lichtenstein", "oldenburg", "hokusai", "hiroshige"})
SYSTEM_PROMPT = ("You are a museum guide talking to a visitor. Answer in 2 short sentences of plain text, not a "
                 "list. Use only the facts you are given: never add a name, title, date or place that is not in "
                 "them, and never guess an unknown artist. The works are the closest ones in the collection, not "
                 "exact matches, so say so.")
# One worked example before the real question: a small model copies an example much better than it follows rules.
EXAMPLE_QUERY = "a sailing boat"
EXAMPLE_WORKS = pd.DataFrame({"label": ["Impressionism", "Impressionism", "Cubism"],
                              "artist": ["Unknown Artist", "Unknown Artist", "georges-braque"],
                              "genre": ["Unknown Genre", "landscape", "Unknown Genre"]})
EXAMPLE_REPLY = ("These are the closest works I found, not exact matches: two Impressionism paintings by unknown "
                 "artists and a Cubism work by Georges Braque. Impressionism comes from France, from the 1860s to "
                 "the 1880s.")


def style_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def artist_name(slug) -> str | None:
    """'peter-paul-rubens' -> 'Peter Paul Rubens'; None when metadata.csv doesn't know the artist."""
    if not isinstance(slug, str) or slug == "Unknown Artist":
        return None
    return " ".join(w if w in {"van", "de", "der"} else w.capitalize() for w in slug.split("-"))


def describe(work) -> str:
    """One work as a line of facts: its style, and its artist and genre when known."""
    artist = artist_name(work.artist)
    parts = [style_name(work.label), f"by {artist}" if artist else "artist unknown"]
    if work.genre in GENRES:
        parts.append(GENRES[work.genre])
    return ", ".join(parts)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def style_counts(results: pd.DataFrame) -> str:
    """'5 Impressionism and 1 Baroque', the most frequent style first."""
    return _and([f"{n} {style_name(label)}" for label, n in results["label"].value_counts().items()])


def main_style(results: pd.DataFrame) -> str:
    return results["label"].value_counts().index[0]


def facts(query: str, results: pd.DataFrame, stories: dict | None = None) -> str:
    """Everything a reply may say: the works found (best first), the style counts, and the period of the most
    frequent style (only that one, so a reply can't give one style's dates to another)."""
    stories = stories or load_stories()
    lines = [f'The visitor asked: "{query}"', f"The {len(results)} closest works, best first:"]
    lines += [f"{i}. {describe(work)}" for i, work in enumerate(results.itertuples(), 1)]
    lines.append(f"Styles: {style_counts(results)}")
    lines.append(f"{style_name(main_style(results))} period: {stories[main_style(results)]['Period']}")
    return "\n".join(lines)


def template_reply(query: str, results: pd.DataFrame, stories: dict | None = None) -> str:
    stories = stories or load_stories()
    best = results.iloc[0]
    artist = artist_name(best["artist"])
    text = [f'Here are the {len(results)} works that look closest to "{query}": {style_counts(results)}.',
            f"The closest one is by {artist or 'an unknown artist'} ({style_name(best['label'])})."]
    others = results.iloc[1:].drop_duplicates("artist")
    named = [f"{artist_name(w.artist)} ({style_name(w.label)})" for w in others.itertuples()
             if artist_name(w.artist) and w.artist != best["artist"]]
    if named:
        text.append(f"Among the others: {_and(named)}.")
    text.append(f"{style_name(main_style(results))}: {stories[main_style(results)]['Period']}")
    return " ".join(text)


# ---------------------------------------------------------------- checking a reply against the facts

def _plain(text: str) -> str:
    """Lower case without accents, so 'Léger' matches 'Leger'."""
    text = text.replace("\u2019", "'")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _words(text: str) -> list[str]:
    """Words as written, 'Ukiyo-e' and "Hokusai's" included (digits split words)."""
    return re.findall(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*", text)


def _sentences(text: str) -> list[str]:
    """Sentences and lines, without list markers ('1. ', '- ', '**') in front."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s for s in (re.sub(r"^[\s\d.*#•-]+", "", part) for part in parts) if s]


@lru_cache(maxsize=1)
def _artist_words() -> frozenset[str]:
    """Every artist name we know of: the collection's (metadata.csv) and the ones the stories mention."""
    slugs = pd.read_csv(IMAGE_DIR / "metadata.csv", usecols=["artist"])["artist"].unique()
    return frozenset(w for slug in slugs for w in slug.split("-") if w not in {"van", "de", "der", "unknown"}) \
        | NAMES_IN_STORIES


def check_reply(reply: str, query: str, results: pd.DataFrame, stories: dict | None = None) -> list[str]:
    """What the reply says that the facts don't. Empty list = nothing found (which doesn't prove it's right).

    - a word with a capital letter that isn't in the facts: a name, title or place made up (the first word of a
      sentence only counts when it is an artist name we know, since every sentence starts with a capital)
    - an artist named right after a style that isn't theirs in the results ("a Pop Art work by Juan Gris")
    - a style that none of the works found has
    - a number that isn't in the facts (a date), or a wrong count of works in a style ("two Baroque paintings")
    - "unknown artist" while an artist the results do know isn't named
    - and two ways of not answering: a list instead of sentences, or a refusal
    """
    known = _plain(facts(query, results, stories))
    known_words = {w.removesuffix("'s") for w in _words(known)}
    any_style = "|".join(STYLE_WORDS.values())
    problems = []
    if re.search(r"^\s*(\d+\.|[-*•])\s", reply, flags=re.M) or "**" in reply:
        problems.append("written as a list, not as sentences")
    if re.search(r"can(?:'|no)t (?:assist|help)", _plain(reply)):
        problems.append("refuses to answer")
    for sentence in _sentences(reply):
        plain_sentence = _plain(sentence)
        for i, word in enumerate(_words(sentence)):
            plain = _plain(word).removesuffix("'s")
            if word[0].isupper() and plain not in known_words and plain != "i" \
                    and not re.fullmatch(any_style, plain) and (i > 0 or plain in _artist_words()):
                problems.append(f'"{word}" is not in the facts')
        styles_here = [(m.start(), label) for label, pattern in STYLE_WORDS.items()
                       for m in re.finditer(pattern, plain_sentence)]
        for work in results.drop_duplicates("artist").itertuples():
            artist = artist_name(work.artist)
            named = artist and re.search(rf"\b{re.escape(_plain(artist.split()[-1]))}\b", plain_sentence)
            if not named or not styles_here:
                continue
            before = [label for start, label in sorted(styles_here) if start < named.start()]
            label = before[-1] if before else min(styles_here)[1]     # the style word just before the name
            if label != work.label:
                problems.append(f"{artist} is put with {style_name(label)}, "
                                f"but the work found is {style_name(work.label)}")
    found = set(results["label"])
    for label, pattern in STYLE_WORDS.items():
        if label not in found and re.search(pattern, _plain(reply)):
            problems.append(f"{style_name(label)} is not among the works found")
    numbers = set(re.findall(r"\d+", known))
    problems += [f"the number {n} is not in the facts" for n in re.findall(r"\d+", reply) if n not in numbers]
    counts = results["label"].value_counts()
    count_words = "|".join([*NUMBER_WORDS, r"\d+"])
    for said, pattern in re.findall(rf"\b({count_words}) (?:[a-z]+ )?({any_style})", _plain(reply)):
        n = int(said) if said.isdigit() else NUMBER_WORDS[said]
        label = next(label for label, style in STYLE_WORDS.items() if re.fullmatch(style, pattern))
        if label in counts and n != counts[label]:
            problems.append(f"says {n} {style_name(label)}, but {counts[label]} were found")
    if "unknown" in _plain(reply):
        missing = [artist_name(a) for a in results["artist"].unique()
                   if artist_name(a) and not re.search(rf"\b{_plain(artist_name(a).split()[-1])}\b", _plain(reply))]
        if missing:
            problems.append(f"says the artist is unknown, but doesn't name {_and(missing)}")
    return list(dict.fromkeys(problems))


# ---------------------------------------------------------------- the reply shown to the visitor

def prompt_messages(query: str, results: pd.DataFrame, stories: dict | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": facts(EXAMPLE_QUERY, EXAMPLE_WORKS, stories)},
            {"role": "assistant", "content": EXAMPLE_REPLY},
            {"role": "user", "content": facts(query, results, stories)}]


def _full_sentences(text: str) -> str:
    """Drop a last sentence cut off by the token limit (when at least one full sentence is left)."""
    cut = max(text.rfind(c) for c in ".!?")
    return text[:cut + 1] if cut > 0 else text


class LLMReplier:
    """Qwen2.5-0.5B-Instruct, run locally. Greedy decoding: the same question always gets the same reply."""

    def __init__(self, model_id: str = LLM_ID, max_new_tokens: int = 80, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # optional, see requirements-search.txt

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval().to(device)
        self.max_new_tokens, self.device = max_new_tokens, device

    @torch.inference_mode()
    def __call__(self, query: str, results: pd.DataFrame) -> str:
        inputs = self.tokenizer.apply_chat_template(prompt_messages(query, results), add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _full_sentences(text.strip())


@dataclass
class Reply:
    text: str
    writer: str                          # "template" or "llm"
    rejected: str | None = None          # the language model's draft, when check_reply found a problem in it
    problems: list[str] = field(default_factory=list)


def reply(query: str, results: pd.DataFrame, llm=None) -> Reply:
    """The language model's reply if there is one and check_reply finds nothing wrong, else the template."""
    if llm is None:
        return Reply(template_reply(query, results), "template")
    draft = llm(query, results)
    problems = check_reply(draft, query, results) if draft.strip() else ["empty reply"]
    if problems:
        return Reply(template_reply(query, results), "template", draft, problems)
    return Reply(draft, "llm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="what to look for; leave out to ask several questions in a row")
    parser.add_argument("-k", type=int, default=6, help="how many works to show")
    parser.add_argument("--llm", action="store_true", help=f"write the reply with {LLM_ID} (checked)")
    args = parser.parse_args()
    collection = Collection.load()
    llm = LLMReplier() if args.llm else None
    queries = [args.query] if args.query else iter(lambda: input("\nWhat are you looking for? ").strip(), "")
    for query in queries:
        results = collection.search(query, args.k)
        for i, work in enumerate(results.itertuples(), 1):
            print(f"{i}. {describe(work):45s} similarity {work.score:.2f}  {work.path}")
        answer = reply(query, results, llm)
        print(f"\n{answer.text}")
        if answer.rejected:
            print(f"\n(the language model's reply was replaced: {'; '.join(answer.problems)})")
