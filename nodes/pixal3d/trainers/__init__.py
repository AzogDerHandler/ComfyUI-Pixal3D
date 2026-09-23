"""Slimmed trainer package -- only inference-path symbols.

The training-time classes (BasicTrainer, FlowMatchingTrainer, ..VaeTrainer,
text/image-conditioned mixins, etc.) were removed for the ComfyUI-Pixal3D
wrapper since none of them are touched during pipeline.run(). Only the
DinoV3Proj{,MultiView}FeatureExtractor are loaded, lazily via __getattr__ below.
"""
import importlib

__attributes = {
    "DinoV3ProjFeatureExtractor": "flow_matching.mixins.image_conditioned_proj",
    "DinoV3ProjMultiViewFeatureExtractor": "flow_matching.mixins.image_conditioned_proj",
}

__all__ = list(__attributes.keys())


def __getattr__(name):
    if name in __attributes:
        module = importlib.import_module(f".{__attributes[name]}", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__} has no attribute {name}")
