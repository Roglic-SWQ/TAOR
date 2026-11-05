"""Rendering choices and duplicate detection for grounded image alternatives."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata
import json

IMAGE_PRESENTATION_POLICY = "grounded_image_focus_v4"
MAX_PRESENTATION_REWRITES = 2
IMAGE_REQUESTS_PER_CANDIDATE = 6
MIN_IMAGE_CANDIDATE_REQUESTS = 3


class ImageFocusValidationError(ValueError):
    """A proposed focus or its realized prompt failed the diversity contract."""


def _presentation_text(text):
    """Normalize common rendering paraphrases, without claiming semantic equivalence."""
    text = unicodedata.normalize('NFKC', text).casefold()
    aliases = {
        'monochrome': r'black[ -]and[ -]white|gr[ae]yscale|monochromatic',
        'photo': r'photographic|photography|photograph|photorealistic',
        'illustration': r'illustrative|illustrated|drawing|sketch',
        'close': r'close[ -]up|closeup|tight crop|close framing',
        'wide': r'wide framing|wide shot|wide view|distant view',
        'side': r'side[ -]lit|side[ -]lighting|lateral',
        'backlit': r'back[ -]lit|back[ -]lighting|rim[ -]lit|rim[ -]lighting',
        'soft': r'diffuse|diffused|gentle',
        'glamorous': r'glamour|glamorous|stylish|elegant|fashionable',
        'centered': r'centrally (?:positioned|placed)|centred|central(?= subject)',
    }
    for term, pattern in aliases.items():
        text = re.sub(r'\b(?:' + pattern + r')\b', term, text)
    # Saying "not a close-up" does not specify a new shot. Ignore common
    # negated rendering labels instead of counting their keywords as novelty.
    return re.sub(r'\b(?:not|no|without)\s+(?:(?:a|an|the)\s+)?'
                  r'(?:close|wide|medium|centered|photo|illustration|side|backlit|soft)\b', '', text)


def _focus_terms(text):
    text = _presentation_text(text)
    filler = set(('a an the of and or in on with for to from by as same new different alternative '
                  'focus emphasis view image depiction subject rendering presentation framing '
                  'composition style lighting choice neutral unknown unspecified appropriate '
                  'supported evidence preserve consistent portrait').split())
    return set(_words(text)) - filler


def validate_image_focus(plan, facts, previous):
    """Check referenced facts and concrete changes, never a self-reported difference alone."""
    fields = {'visual_focus', 'supporting_fact_ids', 'rendering', 'difference_from_previous'}
    axes = {'framing', 'viewpoint', 'subject_placement', 'style', 'lighting'}
    if not isinstance(plan, dict) or set(plan) != fields:
        raise ImageFocusValidationError('Return exactly visual_focus, supporting_fact_ids, rendering, difference_from_previous')
    if (not isinstance(plan['visual_focus'], str) or not plan['visual_focus'].strip()
            or len(plan['visual_focus']) > 240):
        raise ImageFocusValidationError('visual_focus must be concise nonempty text, at most 240 characters')
    rendering = plan['rendering']
    if (not isinstance(rendering, dict) or set(rendering) != axes
            or any(not isinstance(value, str) or not value.strip() or len(value) > 160 for value in rendering.values())):
        raise ImageFocusValidationError('rendering requires five concise string fields: ' + ', '.join(sorted(axes)))
    refs = plan['supporting_fact_ids']
    allowed = {fact['fact_id'] for fact in facts if fact.get('fact_id')}
    if (not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs)
            or len(refs) != len(set(refs)) or not set(refs) <= allowed):
        raise ImageFocusValidationError('supporting_fact_ids must be distinct IDs from the shared source_facts')
    if not isinstance(plan['difference_from_previous'], str) or len(plan['difference_from_previous']) > 300:
        raise ImageFocusValidationError('difference_from_previous must be concise text, at most 300 characters')
    current = _focus_terms(plan['visual_focus'])
    changes = []
    for prior in previous:
        old = prior.get('focus_plan', prior)
        old_focus = _focus_terms(old.get('visual_focus', ''))
        focus_change = len(current - old_focus) >= 2
        altered = {axis: sorted(_focus_terms(rendering[axis]) - _focus_terms(old.get('rendering', {}).get(axis, '')))
                   for axis in axes}
        altered = {axis: terms for axis, terms in altered.items() if terms}
        if not focus_change and not altered:
            raise ImageFocusValidationError('The plan repeats an accepted visual emphasis and rendering. '
                'Choose a different supported visual emphasis OR a meaningful rendering change. '
                'A new visual emphasis may keep all five rendering values unchanged; '
                'a rendering change may keep the same supported focus. '
                'Synonyms or difference_from_previous alone do not establish a change.')
        changes.append({'focus_changed': focus_change, 'changed_axes': altered,
                        'new_focus_terms': sorted(current - old_focus)})
    return {'changes_from_previous': changes, 'fact_ids_checked': sorted(set(refs))}


def validate_focus_realization(text, plan, audit, previous_texts):
    """Require the actual description/prompt to express at least one planned difference."""
    if duplicate_image_prompt(text, previous_texts) is not None:
        raise ImageFocusValidationError('Actual output repeats an accepted prompt; realize the selected new focus instead of paraphrasing it')
    terms = _focus_terms(text)
    for index, change in enumerate(audit['changes_from_previous']):
        prior_terms = _focus_terms(previous_texts[index]) if index < len(previous_texts) else set()
        newly_expressed = terms - prior_terms
        rendering_realized = any(set(values) & newly_expressed for values in change['changed_axes'].values())
        # A feature can become the new emphasis even if an earlier prompt
        # already mentioned it in passing. Exact and near copies are checked
        # above; do not require the feature itself to be new factual content.
        focus_realized = change['focus_changed'] and len(set(change['new_focus_terms']) & terms) >= 2
        if not rendering_realized and not focus_realized:
            raise ImageFocusValidationError('Actual output omits the planned difference. Express the chosen '
                'new focus or rendering in the image text: ' + json.dumps(change, ensure_ascii=False))


def _words(prompt: str) -> list[str]:
    # A changed candidate number must not turn an identical scene into a new
    # alternative. Changed numeric facts are not presentation variety either.
    return [word for word in re.findall(r"\w+", unicodedata.normalize("NFKC", prompt).casefold())
            if not word.isdigit()]


def _rendering_tags(prompt: str) -> set[str]:
    text = " ".join(_words(_presentation_text(prompt)))
    patterns = {
        "close": r"\b(close|tight(?:er)? (?:crop|framing)|fills? the frame)\b",
        "wide": r"\b(wide|wider (?:view|shot|framing|composition)|distant view)\b",
        "medium": r"\b(medium (?:shot|framing|composition)|mid shot)\b",
        "off_center": r"\b(off cent(?:er|re)|rule of thirds|subject on the (?:left|right))\b",
        "left": r"\b(subject on the left|left aligned|left side of (?:the )?frame)\b",
        "right": r"\b(subject on the right|right aligned|right side of (?:the )?frame)\b",
        "center": r"\b(centered|centred|central subject|symmetrical)\b",
        "negative_space": r"\b(negative space|empty space)\b",
        "photo": r"\bphoto\b",
        "illustration": r"\b(illustration|illustrated|drawing|sketch)\b",
    }
    return {tag for tag, pattern in patterns.items() if re.search(pattern, text)}


def image_presentation_signature(prompt: str) -> list[str]:
    """Audit actual wording, rather than treating the requested layout as proof."""
    return sorted(_rendering_tags(prompt))


def duplicate_image_prompt(prompt: str, previous: list[str]) -> tuple[int, str] | None:
    """Catch copies, reordered words and very close copies without rejecting a changed framing."""
    words = _words(_presentation_text(prompt))
    counts = Counter(words)
    for index, earlier in enumerate(previous):
        other = _words(_presentation_text(earlier))
        if words == other or (counts == Counter(other)
                              and _rendering_tags(prompt) == _rendering_tags(earlier)):
            return index, "same_words"
        # The same factual basis naturally produces substantial overlap. A
        # changed framing/layout is useful variety, even with shared wording.
        if _rendering_tags(prompt) != _rendering_tags(earlier):
            continue
        common = len(set(words) & set(other))
        union = len(set(words) | set(other))
        smaller, larger = min(len(set(words)), len(set(other))), max(len(set(words)), len(set(other)))
        if (smaller >= 8 and common / smaller >= 0.92 and common / larger >= 0.5):
            return index, 'shortened_copy'
        if (union and common / union >= 0.9
                and SequenceMatcher(None, words, other, autojunk=False).ratio() >= 0.9):
            return index, "near_copy"
    return None
