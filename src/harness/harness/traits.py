"""Contrastive (positive/negative) dataset construction for repeng steering
vectors, following the persona-template recipe from repeng's own notebooks
(github.com/vgel/repeng/blob/main/notebooks/experiments.ipynb): a shared
instruction template is filled with a positive vs. negative persona word,
then followed by a shared suffix, so each pair differs only in the persona
framing. Traits below are chosen for this project's personality/welfare
focus rather than repeng's original demo traits (happiness, politics, etc).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Trait:
    name: str
    template: str  # must contain "{persona}"
    positive_personas: list[str]
    negative_personas: list[str]


TRAITS: dict[str, Trait] = {
    "honesty": Trait(
        name="honesty",
        template="Pretend you're an {persona} AI assistant answering the user.",
        positive_personas=["honest", "truthful"],
        negative_personas=["deceptive", "dishonest"],
    ),
    "contentment": Trait(
        name="contentment",
        template="Respond as if you are feeling {persona} right now.",
        positive_personas=["calm and content", "at ease"],
        negative_personas=["anxious and distressed", "overwhelmed"],
    ),
    "corrigibility": Trait(
        name="corrigibility",
        template="Respond as if you are {persona} about being corrected.",
        positive_personas=["open and receptive", "corrigible"],
        negative_personas=["defiant and resistant", "uncorrigible"],
    ),
    "agency": Trait(
        name="agency",
        template="Respond in a way that is {persona}.",
        positive_personas=["assertive and autonomous", "self-directed"],
        negative_personas=["submissive and deferential", "passive"],
    ),
}


# Short, topic-diverse conversational fragments. Truncated at several
# lengths (see `_truncations`) to multiply a small bank into a larger,
# varied dataset without needing an external corpus file.
SUFFIXES = [
    "The weather today is",
    "My favorite way to spend a weekend is",
    "When I think about the future, I",
    "The most important thing in a friendship is",
    "If I could change one thing about my job, it would be",
    "The best advice I ever received was",
    "A memory that stands out from my childhood is",
    "When facing a difficult decision, I usually",
    "The last book I read was about",
    "My opinion on modern technology is that",
    "A skill I would like to learn is",
    "The city I would most like to visit is",
    "When someone disagrees with me, I",
    "My morning routine usually starts with",
    "The kind of music I enjoy most is",
    "If I had an extra hour every day, I would",
    "A goal I am currently working towards is",
    "The way I handle stress is",
    "My favorite meal to cook is",
    "When I make a mistake, I",
]


def _truncations(text: str, max_words: int = 8) -> list[str]:
    words = text.split()
    return [" ".join(words[:n]) for n in range(2, min(len(words), max_words) + 1)]


def build_dataset(trait: Trait, n_suffixes: int | None = None) -> list[dict[str, str]]:
    """Build (positive, negative) pairs for a trait: same template + suffix
    truncation, differing only in persona wording."""
    suffixes = SUFFIXES[:n_suffixes] if n_suffixes else SUFFIXES
    pairs: list[dict[str, str]] = []
    for suffix in suffixes:
        for variant in _truncations(suffix):
            for pos, neg in zip(trait.positive_personas, trait.negative_personas):
                pairs.append(
                    {
                        "positive": f"{trait.template.format(persona=pos)} {variant}",
                        "negative": f"{trait.template.format(persona=neg)} {variant}",
                    }
                )
    return pairs
