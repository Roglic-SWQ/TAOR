"""Rendering choices and duplicate detection for grounded image alternatives."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata

IMAGE_PRESENTATION_POLICY = "grounded_image_diversity_v1"
MAX_PRESENTATION_REWRITES = 2


def image_rendering_plan(index: int) -> dict[str, str]:
    """Vary framing/layout, never the target's factual attributes or geometry."""
    choices = (
        ("medium framing", "centered subject, balanced composition"),
        ("close framing of the same supported subject", "centered subject, subject fills the frame"),
        ("wide framing with neutral surrounding space", "off-center subject, ample negative space"),
        ("medium framing", "subject on the left, neutral space on the right"),
        ("close framing of the same supported subject", "subject on the right, restrained negative space"),
        ("wide framing with neutral surrounding space", "centered subject, balanced composition"),
        ("medium framing", "subject on the right, neutral space on the left"),
        ("close framing of the same supported subject", "subject on the left, restrained negative space"),
        ("wide framing with neutral surrounding space", "subject on the left, neutral space on the right"),
    )
    framing, placement = choices[index % len(choices)]
    return {"framing": framing, "subject_placement": placement,
            "evidence_policy": "Preserve supported identity and attributes. Keep unknown surroundings neutral; "
                               "never invent unseen sides, facial features, objects or settings."}


def _words(prompt: str) -> list[str]:
    # A changed candidate number must not turn an identical scene into a new
    # alternative. Changed numeric facts are not presentation variety either.
    return [word for word in re.findall(r"\w+", unicodedata.normalize("NFKC", prompt).casefold())
            if not word.isdigit()]


def _rendering_tags(prompt: str) -> set[str]:
    text = " ".join(_words(prompt))
    patterns = {
        "close": r"\b(close up|closeup|close framing|tight(?:er)? (?:crop|framing)|fills? the frame)\b",
        "wide": r"\b(wide(?:r)? (?:view|shot|framing|composition)|distant view)\b",
        "medium": r"\b(medium (?:shot|framing|composition)|mid shot)\b",
        "off_center": r"\b(off cent(?:er|re)|rule of thirds|subject on the (?:left|right))\b",
        "left": r"\b(subject on the left|left aligned|left side of (?:the )?frame)\b",
        "right": r"\b(subject on the right|right aligned|right side of (?:the )?frame)\b",
        "center": r"\b(centered|centred|central subject|symmetrical)\b",
        "negative_space": r"\b(negative space|empty space)\b",
        "photo": r"\b(photograph|photographic|photorealistic)\b",
        "illustration": r"\b(illustration|illustrated|drawing|sketch)\b",
    }
    return {tag for tag, pattern in patterns.items() if re.search(pattern, text)}


def image_presentation_signature(prompt: str) -> list[str]:
    """Audit actual wording, rather than treating the requested layout as proof."""
    return sorted(_rendering_tags(prompt))


def duplicate_image_prompt(prompt: str, previous: list[str]) -> tuple[int, str] | None:
    """Catch copies, reordered words and very close copies without rejecting a changed framing."""
    words = _words(prompt)
    counts = Counter(words)
    for index, earlier in enumerate(previous):
        other = _words(earlier)
        if words == other or (counts == Counter(other)
                              and _rendering_tags(prompt) == _rendering_tags(earlier)):
            return index, "same_words"
        # The same factual basis naturally produces substantial overlap. A
        # changed framing/layout is useful variety, even with shared wording.
        if _rendering_tags(prompt) != _rendering_tags(earlier):
            continue
        common = len(set(words) & set(other))
        union = len(set(words) | set(other))
        if (union and common / union >= 0.9
                and SequenceMatcher(None, words, other, autojunk=False).ratio() >= 0.9):
            return index, "near_copy"
    return None
