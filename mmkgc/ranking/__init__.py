"""Stage 4 model-fusion ranking exports."""

__all__ = ["ModelFusionRankingEngine"]


def __getattr__(name: str):
    if name == "ModelFusionRankingEngine":
        from mmkgc.ranking.engine import ModelFusionRankingEngine

        return ModelFusionRankingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
