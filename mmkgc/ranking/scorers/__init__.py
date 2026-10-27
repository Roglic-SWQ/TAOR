"""Text scorer adapters for Stage 4 model-fusion ranking."""

__all__ = [
    "BLIPITMRankingScorer",
    "BERTScoreAuxScorer",
    "CLIPRankingScorer",
    "FixedTextScorer",
    "ImageJudgeResult",
    "PairJudgeResult",
    "GPT4OMiniJudgeParseError",
    "GPT4OMiniRankingJudge",
    "SentenceBERTScorer",
    "TextSemanticScorer",
    "TextJudgeResult",
    "build_image_judge_prompt",
    "build_pair_judge_prompt",
    "build_text_graph_extraction_prompt",
    "build_text_judge_prompt",
    "build_visual_graph_extraction_prompt",
    "parse_image_judge_response",
    "parse_pair_judge_response",
    "parse_text_graph_response",
    "parse_text_judge_response",
    "parse_visual_graph_response",
]


_EXPORT_MODULES = {
    "BLIPITMRankingScorer": "mmkgc.ranking.scorers.blip_itm",
    "BERTScoreAuxScorer": "mmkgc.ranking.scorers.bertscore",
    "CLIPRankingScorer": "mmkgc.ranking.scorers.clip",
    "FixedTextScorer": "mmkgc.ranking.scorers.base",
    "ImageJudgeResult": "mmkgc.ranking.scorers.gpt4omini_judge",
    "PairJudgeResult": "mmkgc.ranking.scorers.gpt4omini_judge",
    "GPT4OMiniJudgeParseError": "mmkgc.ranking.scorers.gpt4omini_judge",
    "GPT4OMiniRankingJudge": "mmkgc.ranking.scorers.gpt4omini_judge",
    "SentenceBERTScorer": "mmkgc.ranking.scorers.sentence_bert",
    "TextSemanticScorer": "mmkgc.ranking.scorers.base",
    "TextJudgeResult": "mmkgc.ranking.scorers.gpt4omini_judge",
    "build_image_judge_prompt": "mmkgc.ranking.scorers.gpt4omini_judge",
    "build_pair_judge_prompt": "mmkgc.ranking.scorers.gpt4omini_judge",
    "build_text_graph_extraction_prompt": "mmkgc.ranking.scorers.gpt4omini_judge",
    "build_text_judge_prompt": "mmkgc.ranking.scorers.gpt4omini_judge",
    "build_visual_graph_extraction_prompt": "mmkgc.ranking.scorers.gpt4omini_judge",
    "parse_image_judge_response": "mmkgc.ranking.scorers.gpt4omini_judge",
    "parse_pair_judge_response": "mmkgc.ranking.scorers.gpt4omini_judge",
    "parse_text_graph_response": "mmkgc.ranking.scorers.gpt4omini_judge",
    "parse_text_judge_response": "mmkgc.ranking.scorers.gpt4omini_judge",
    "parse_visual_graph_response": "mmkgc.ranking.scorers.gpt4omini_judge",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
