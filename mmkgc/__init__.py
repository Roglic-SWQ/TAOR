"""MMKGC: relation-guided missing modality completion for MKG-W."""

from mmkgc.defaults import build_default_mainline_pipeline, build_default_mainline_runtime
from mmkgc.pipeline import MMKGCPipeline

__version__ = "0.1.0"

__all__ = [
    "MMKGCPipeline",
    "build_default_mainline_pipeline",
    "build_default_mainline_runtime",
]
