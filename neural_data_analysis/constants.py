"""
brain_area_dict
brain_area_abbreviations_lower
brain_area_abbreviations_upper
embedder_configs
gpt_configs
"""

from typing import TypedDict, Optional


class EmbedderConfig(TypedDict):
    embedding_name: str
    batch_size: int
    model: Optional[str]
    processor: Optional[str]
    embedding_description: Optional[str]
    detection_threshold: Optional[float]


brain_area_dict = {
    "all": [
        # AMYGDALA
        "amy",
        "amygdala",
        "Amygdala",
        "AMY",
        # HIPPOCAMPUS
        "hpc",
        "hippocampus",
        "Hippocampus",
        "HPC",
        # PARAHIPPOCAMPAL CORTEX
        "parahippocampal_gyrus",
        "parahippocampal_cortex",
        "PHC",
        "phc",
        # ORBITOFRONTAL CORTEX
        "ofc",
        "orbitofrontal cortex",
        "OFC",
        # ANTERIOR CINGULATE CORTEX
        "acc",
        "ACC",
        "anterior cingulate cortex",
        # DORSAL ANTERIOR CINGULATE CORTEX
        "dorsal_anterior_cingulate_cortex",
        "dACC",
        # SUPPLEMENTARY MOTOR AREA
        "sma",
        "SMA",
        "supplementary motor area",
        # PRE-SUPPLEMENTARY MOTOR AREA
        "presma",
        "preSMA",
        "pre-supplementary motor area",
        "pre_supplementary_motor_area",
        # VENTROMEDIAL PREFRONTAL CORTEX
        "vmpfc",
        "vmPFC",
        "ventromedial prefrontal cortex",
        "ventral_medial_prefrontal_cortex",
    ],
    "mtl": [
        "amy",
        "AMY",
        "amygdala",
        "Amygdala",
        "hpc",
        "HPC",
        "hippocampus",
        "Hippocampus",
    ],
    "amygdala": ["amy", "AMY", "amygdala", "Amygdala"],
    "amy": ["amy", "AMY", "amygdala", "Amygdala"],
    "hippocampus": ["hpc", "HPC", "hippocampus", "Hippocampus"],
    "hpc": ["hpc", "HPC", "hippocampus", "Hippocampus"],
    "orbitofrontal cortex": ["orbitofrontal cortex", "ofc", "OFC"],
    "ofc": ["orbitofrontal cortex", "ofc", "OFC"],
    "anterior cingulate cortex": ["anterior cingulate cortex", "acc", "ACC"],
    "acc": ["anterior cingulate cortex", "acc", "ACC"],
    "ACC": ["anterior cingulate cortex", "acc", "ACC"],
    "supplementary motor area": ["supplementary motor area", "SMA", "sma"],
    "sma": ["supplementary motor area", "SMA", "sma"],
    "presma": ["pre-supplementary motor area", "preSMA", "presma"],
    "vmpfc": ["ventromedial prefrontal cortex", "vmPFC", "vmpfc"],
}
brain_area_abbreviations_upper = {
    "anterior cingulate cortex": "ACC",
    "dorsal_anterior_cingulate_cortex": "dACC",
    "ACC": "ACC",
    "all": "ALL",
    "Amygdala": "AMY",
    "amygdala": "AMY",
    "Hippocampus": "HPC",
    "hippocampus": "HPC",
    "orbitofrontal cortex": "OFC",
    "parahippocampal_cortex": "PHC",
    "parahippocampal_gyrus": "PHC",
    "pre_supplementary_motor_area": "preSMA",
    "preSMA": "preSMA",
    "supplementary motor area": "SMA",
    "ventral_medial_prefrontal_cortex": "vmPFC",
    "vmPFC": "vmPFC",
    "RSPE": "RSPE",
}
brain_area_abbreviations_lower = {
    "anterior cingulate cortex": "acc",
    "dorsal_anterior_cingulate_cortex": "dacc",
    "ACC": "acc",
    "all": "all",
    "Amygdala": "amy",
    "amygdala": "amy",
    "Hippocampus": "hpc",
    "hippocampus": "hpc",
    "orbitofrontal cortex": "ofc",
    "parahippocampal_cortex": "phc",
    "parahippocampal_gyrus": "phc",
    "preSMA": "presma",
    "pre_supplementary_motor_area": "presma",
    "supplementary motor area": "sma",
    "ventral_medial_prefrontal_cortex": "vmpfc",
    "vmPFC": "vmpfc",
    "RSPE": "rspe",
}

# noinspection Annotator
embedder_configs: dict[str, EmbedderConfig] = dict(
    VGG16Embedder=dict(embedding_name="vgg16", batch_size=64),
    ResNet50Embedder=dict(embedding_name="resnet", batch_size=64),
    CLIPEmbedder=dict(
        embedding_name="clip",
        batch_size=512,
        model="openai/clip-vit-base-patch32",
        processor="openai/clip-vit-base-patch32",
    ),
    DETREmbedder=dict(
        embedding_name="detr",
        embedding_description="DEtection TRansformer for object detection",
        model="facebook/detr-resnet-50",
        processor="facebook/detr-resnet-50",
        detection_threshold=0.5,
        batch_size=16,
    ),
    BLIPEmbedder=dict(
        embedding_name="blip",
        embedding_description="BLIP: Learning Better Language Models by Encoding Images in Text Sequences",
        batch_size=64,
        model="Salesforce/blip-image-captioning-base",
        processor="Salesforce/blip-image-captioning-base",
    ),
    BLIP2Embedder=dict(
        embedding_name="blip2",
        embedding_description="BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and "
        "Large Language Models",
        batch_size=1,
        model="Salesforce/blip2-opt-2.7b",
        processor="Salesforce/blip2-opt-2.7b",
        make_specific_binary_vector=False,
    ),
    DINOEmbedder=dict(
        embedding_name="dino",
        embedding_description="DINO: Emerging Properties in Self-Supervised Vision Transformers",
        batch_size=1,
        model="facebook/dinov2-base",
        processor="facebook/dinov2-base",
    ),
    ViT_B_16Embedder=dict(
        embedding_name="vit",
        embedding_description="Vision Transformer Base 16",
        batch_size=64,
    ),
    SGPTEmbedder=dict(embedding_name="sgpt"),
)

gpt_configs = {
    "gpt-4": {
        "gpt_engine": "gpt-4",
        "openai_api_key": None,
        "pregenerate_prompts": False,
    }
}
