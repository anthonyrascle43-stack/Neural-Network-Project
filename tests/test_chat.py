"""chat.py with hand-made search results and fake language models (no download in CI)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from museum import chat


def results(*works):
    """Search results from (label, artist slug or None, genre or None) triples, best first."""
    return pd.DataFrame({"path": [f"w{i}.jpg" for i in range(len(works))],
                         "label": [label for label, _, _ in works],
                         "artist": [artist or "Unknown Artist" for _, artist, _ in works],
                         "genre": [genre or "Unknown Genre" for _, _, genre in works],
                         "score": [0.3 - 0.01 * i for i in range(len(works))]})


SOUP = results(("Pop_Art", None, None), ("Pop_Art", None, None), ("Cubism", "juan-gris", "still_life"))
FUJI = results(("Ukiyo_e", "katsushika-hokusai", "landscape"), ("Ukiyo_e", None, None), ("Ukiyo_e", None, None))


def test_names_and_descriptions():
    assert chat.artist_name("peter-paul-rubens") == "Peter Paul Rubens"
    assert chat.artist_name("anthony-van-dyck") == "Anthony van Dyck"
    assert chat.artist_name("Unknown Artist") is None and chat.artist_name(float("nan")) is None
    work = SimpleNamespace(label="Ukiyo_e", artist="katsushika-hokusai", genre="landscape")
    assert chat.describe(work) == "Ukiyo-e, by Katsushika Hokusai, landscape"
    assert chat.describe(SimpleNamespace(label="Pop_Art", artist="Unknown Artist", genre="Unknown Genre")) \
        == "Pop Art, artist unknown"


def test_facts_list_the_works_and_only_the_main_period():
    text = chat.facts("a soup can", SOUP)
    assert '"a soup can"' in text and "3. Cubism, by Juan Gris, still life" in text
    assert "Styles: 2 Pop Art and 1 Cubism" in text
    assert "Pop Art period:" in text and "Cubism period" not in text, "one period, so dates can't be swapped"


def test_template_reply_says_what_was_found():
    text = chat.template_reply("a soup can", SOUP)
    assert text.startswith('Here are the 3 works that look closest to "a soup can": 2 Pop Art and 1 Cubism.')
    assert "The closest one is by an unknown artist (Pop Art)." in text and "Juan Gris (Cubism)" in text
    assert "The closest one is by Katsushika Hokusai (Ukiyo-e)." in chat.template_reply("Mount Fuji", FUJI)


@pytest.mark.parametrize("query, found", [("a soup can", SOUP), ("Mount Fuji", FUJI),
                                          (chat.EXAMPLE_QUERY, chat.EXAMPLE_WORKS)])
def test_the_template_and_the_prompt_example_pass_the_check(query, found):
    assert chat.check_reply(chat.template_reply(query, found), query, found) == []
    assert chat.check_reply(chat.EXAMPLE_REPLY, chat.EXAMPLE_QUERY, chat.EXAMPLE_WORKS) == []


def test_a_faithful_reply_passes():
    reply = ("These are the closest works I found, not exact matches: two Pop Art pieces by unknown artists and a "
             "Cubism still life by Juan Gris. Pop Art comes from Britain and the United States, mid-1950s.")
    assert chat.check_reply(reply, "a soup can", SOUP) == []


@pytest.mark.parametrize("reply, problem", [
    ("Two Pop Art works, and a Cubism one painted by Rembrandt.", '"Rembrandt" is not in the facts'),
    ("Hokusai made the closest one.", '"Hokusai" is not in the facts'),
    ("These works are in the Louvre.", '"Louvre" is not in the facts'),
    ("Two Pop Art pieces and a Pop Art work by Juan Gris.", "Juan Gris is put with Pop Art"),
    ("Two Pop Art pieces, close to Baroque.", "Baroque is not among the works found"),
    ("Two Pop Art works from 1962.", "the number 1962 is not in the facts"),
    ("Three Pop Art works and one Cubism work by Juan Gris.", "says 3 Pop Art, but 2 were found"),
    ("Pop Art works by unknown artists.", "says the artist is unknown, but doesn't name Juan Gris"),
    ("1. Pop Art\n2. Pop Art\n3. Cubism, by Juan Gris", "written as a list, not as sentences"),
    ("I'm sorry, but I can't assist with that request.", "refuses to answer"),
])
def test_check_flags_what_the_facts_dont_say(reply, problem):
    problems = chat.check_reply(reply, "a soup can", SOUP)
    assert any(p.startswith(problem) for p in problems), problems


def test_check_ignores_accents_possessives_and_style_adjectives():
    reply = "Juan Gris’s Cubist still life and two Pop Art works by unknown artists, in the Léger way."
    assert chat.check_reply(reply, "a soup can", SOUP) == ['"Léger" is not in the facts']


def test_reply_uses_the_language_model_only_when_the_check_passes():
    good = "Two Pop Art works by unknown artists and a Cubism still life by Juan Gris, the closest I found."
    answer = chat.reply("a soup can", SOUP, llm=lambda query, found: good)
    assert answer.writer == "llm" and answer.text == good and answer.rejected is None

    bad = "Three Campbell's soup cans by Andy Warhol."
    answer = chat.reply("a soup can", SOUP, llm=lambda query, found: bad)
    assert answer.writer == "template" and answer.text == chat.template_reply("a soup can", SOUP)
    assert answer.rejected == bad and any("Warhol" in p for p in answer.problems)

    assert chat.reply("a soup can", SOUP, llm=lambda query, found: "  ").problems == ["empty reply"]
    assert chat.reply("a soup can", SOUP).writer == "template"


def test_prompt_has_the_worked_example_before_the_question():
    messages = chat.prompt_messages("a soup can", SOUP)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == chat.EXAMPLE_REPLY and '"a soup can"' in messages[3]["content"]


def test_a_reply_cut_by_the_token_limit_keeps_its_full_sentences():
    assert chat._full_sentences("One sentence. Two sentences! And a thi") == "One sentence. Two sentences!"
    assert chat._full_sentences("no full stop at all") == "no full stop at all"
