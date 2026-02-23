"""medetect – utilities for ONNX model post-processing."""

import onnx

# Training-data license string embedded in the ONNX model.
# Update this value if you switch to a different dataset.
TRAINING_DATA_LICENSE = (
    "DOTA (Dataset for Object deTection in Aerial images) – "
    "Academic use only. "
    "https://captain-whu.github.io/DOTA/index.html"
)

METADATA_KEY_TRAINING_DATA_LICENSE = "training_data_license"


def add_training_data_license(
    src_path: str,
    dst_path: str | None = None,
    license_text: str = TRAINING_DATA_LICENSE,
) -> None:
    """Add training-data license metadata to an ONNX model file.

    Args:
        src_path: Path to the source ONNX model.
        dst_path: Path to write the modified model.
                  Defaults to *src_path* (in-place update).
        license_text: License text to embed.
                      Defaults to :data:`TRAINING_DATA_LICENSE`.
    """
    model = onnx.load(src_path)

    # Remove any existing entry with the same key to avoid duplicates.
    for p in list(model.metadata_props):
        if p.key == METADATA_KEY_TRAINING_DATA_LICENSE:
            model.metadata_props.remove(p)

    entry = model.metadata_props.add()
    entry.key = METADATA_KEY_TRAINING_DATA_LICENSE
    entry.value = license_text

    onnx.save(model, dst_path or src_path)
