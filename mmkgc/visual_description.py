"""Policy for describing entity-relevant appearance without graph coverage."""
from copy import deepcopy

TASK = 'entity_visual_description_v1'
FEATURE_RULES = (
    'Describe visual semantics relevant to the supplied entity, not every property in every image. '
    'First locate the depicted subject that could represent that entity. Retain a few complementary '
    'visible structural, morphological, component or material properties that help characterize it. '
    'Keep what the pixels show; abstract away incidental scene details, not the visual evidence itself. '
    'Do not substitute a dictionary definition or knowledge of the label for observed features. '
    'For a physical tool, its elongated body and enlarged end may matter; the wall behind it does not. '
    'For a bird, beak structure or coherent plumage pattern may matter; its perch and blurred background do not. '
    'A profession, religious role or abstract concept is not characterized by a pictured person’s '
    'age, hair, beard, shirt, headwear or pose. Do not infer that role from appearance. '
    'A bystander, worn garment, snow or backdrop must not become a component of the target object. '
    'Select color only when it meaningfully characterizes the actual target, not incidental clothing. '
    'Distinguish a represented subject’s structure from the paint and weathering of its depiction. '
    'Clear relevant morphology can be useful even when exact identity remains unverified. '
    'Do not resolve an unclear property or source identity by guessing. Keep image-specific attribution, '
    'meaningful qualifications and conflicting views separate. Merge exact duplicate observations with their sources. '
    'There is no per-image quota and no requirement to use every image. When no entity-relevant visual '
    'feature is supported, return no features rather than a description of unrelated visible content. '
)


def enabled(knowledge):
    return (knowledge.get('generation_task') == TASK
            or knowledge.get('stage1_v3', {}).get('generation_task') == TASK)


def selection_view(knowledge):
    """Keep the full graph in audit artifacts, outside this task's model input."""
    view = deepcopy(knowledge)
    view['relation_facts'] = [f for f in view['relation_facts']
        if f.get('generation_eligible', True) and f.get('visual_observations')
        and not any(source != 'image' for source in f.get('source_modalities', []))]
    view.pop('relevance_selection', None)
    # Admission diagnostics may discuss graph assertions. Feature qualifications
    # stay attached to each observation; full diagnostics stay in the original.
    view['uncertainties'] = []
    view['stage1_v3']['uncertain_aspects'] = []
    return view
