from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.ranking.image_input import ImageInput, ImageInputError


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(color: tuple[int, int, int] = (0, 255, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes())


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jpeg_bytes())


def test_image_input_from_bytes_opens_pil_and_encodes_data_url() -> None:
    image_bytes = _png_bytes()

    image_input = ImageInput.from_bytes(
        entity_key="Q1",
        candidate_id="image_001",
        image_bytes=image_bytes,
    )

    assert image_input.media_type == "image/png"
    assert image_input.image_bytes == image_bytes
    assert image_input.image_path is None

    with image_input.open_pil() as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)

    prefix, encoded = image_input.as_data_url().split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == image_bytes


def test_image_input_from_bytes_accepts_jpeg_bytes() -> None:
    image_input = ImageInput.from_bytes(
        entity_key="Q1",
        candidate_id="image_002",
        image_bytes=_jpeg_bytes(),
    )

    assert image_input.media_type == "image/jpeg"
    with image_input.open_pil() as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)


def test_image_input_from_bytes_rejects_non_image_bytes() -> None:
    with pytest.raises(ImageInputError, match="PNG or JPEG"):
        ImageInput.from_bytes(
            entity_key="Q1",
            candidate_id="image_001",
            image_bytes=b"not an image",
        )


def test_image_input_constructor_requires_one_image_source() -> None:
    with pytest.raises(ImageInputError, match="exactly one"):
        ImageInput(
            entity_key="Q1",
            candidate_id="image_001",
            media_type="image/png",
        )


def test_image_input_constructor_rejects_both_image_sources(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    with pytest.raises(ImageInputError, match="exactly one"):
        ImageInput(
            entity_key="Q1",
            candidate_id="image_001",
            media_type="image/png",
            image_bytes=_png_bytes(),
            image_path=image_path,
        )


def test_image_input_constructor_rejects_non_image_path_bytes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(ImageInputError, match="PNG or JPEG"):
        ImageInput(
            entity_key="Q1",
            candidate_id="image_001",
            media_type="image/png",
            image_path=image_path,
        )


def test_image_input_constructor_rejects_path_media_type_mismatch(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.jpg"
    _write_jpeg(image_path)

    with pytest.raises(ImageInputError, match="media type"):
        ImageInput(
            entity_key="Q1",
            candidate_id="image_001",
            media_type="image/png",
            image_path=image_path,
        )


def test_image_input_constructor_rejects_bytes_media_type_mismatch() -> None:
    with pytest.raises(ImageInputError, match="media type"):
        ImageInput(
            entity_key="Q1",
            candidate_id="image_001",
            media_type="image/png",
            image_bytes=_jpeg_bytes(),
        )


def test_image_input_from_entity_path_opens_pil_and_encodes_data_url(
    tmp_path: Path,
) -> None:
    entity_dir = tmp_path / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.png"
    _write_png(image_path)

    image_input = ImageInput.from_entity_path(
        entity_key="Q1",
        candidate_id="image_001",
        entity_dir=entity_dir,
        relative_path="generated_images/image_001.png",
    )

    assert image_input.media_type == "image/png"
    assert image_input.image_bytes is None
    assert image_input.image_path == image_path.resolve()
    with image_input.open_pil() as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)
    assert image_input.as_data_url().startswith("data:image/png;base64,")


def test_image_input_from_entity_path_rejects_non_image_bytes(
    tmp_path: Path,
) -> None:
    entity_dir = tmp_path / "Q1"
    bad_path = entity_dir / "generated_images" / "image_001.png"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not an image")

    with pytest.raises(ImageInputError, match="PNG or JPEG"):
        ImageInput.from_entity_path(
            entity_key="Q1",
            candidate_id="image_001",
            entity_dir=entity_dir,
            relative_path="generated_images/image_001.png",
        )


def test_image_input_from_entity_path_derives_jpeg_media_type(
    tmp_path: Path,
) -> None:
    entity_dir = tmp_path / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.jpg"
    _write_jpeg(image_path)

    image_input = ImageInput.from_entity_path(
        entity_key="Q1",
        candidate_id="image_001",
        entity_dir=entity_dir,
        relative_path="generated_images/image_001.jpg",
    )

    assert image_input.media_type == "image/jpeg"
    assert image_input.as_data_url().startswith("data:image/jpeg;base64,")


def test_image_input_from_entity_path_rejects_mismatched_media_type(
    tmp_path: Path,
) -> None:
    entity_dir = tmp_path / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.jpg"
    _write_jpeg(image_path)

    with pytest.raises(ImageInputError, match="media type"):
        ImageInput.from_entity_path(
            entity_key="Q1",
            candidate_id="image_001",
            entity_dir=entity_dir,
            relative_path="generated_images/image_001.jpg",
            media_type="image/png",
        )


def test_image_input_path_data_url_uses_verified_file_bytes_and_media_type(
    tmp_path: Path,
) -> None:
    entity_dir = tmp_path / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.jpg"
    _write_jpeg(image_path)

    image_input = ImageInput.from_entity_path(
        entity_key="Q1",
        candidate_id="image_001",
        entity_dir=entity_dir,
        relative_path="generated_images/image_001.jpg",
        media_type="image/jpeg",
    )

    prefix, encoded = image_input.as_data_url().split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == image_path.read_bytes()


def test_image_input_rejects_path_outside_entity_dir(tmp_path: Path) -> None:
    entity_dir = tmp_path / "Q1"
    entity_dir.mkdir()
    outside = tmp_path / "outside.png"
    _write_png(outside)

    with pytest.raises(ImageInputError, match="inside entity directory"):
        ImageInput.from_entity_path(
            entity_key="Q1",
            candidate_id="image_001",
            entity_dir=entity_dir,
            relative_path="../outside.png",
        )


def test_image_input_rejects_absolute_entity_path(tmp_path: Path) -> None:
    entity_dir = tmp_path / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.png"
    _write_png(image_path)

    with pytest.raises(ImageInputError, match="relative"):
        ImageInput.from_entity_path(
            entity_key="Q1",
            candidate_id="image_001",
            entity_dir=entity_dir,
            relative_path=image_path.resolve(),
        )


def test_image_input_rejects_symlink_escape(tmp_path: Path) -> None:
    entity_dir = tmp_path / "Q1"
    entity_dir.mkdir()
    outside = tmp_path / "outside.png"
    _write_png(outside)
    link = entity_dir / "escape.png"
    link.symlink_to(outside)

    with pytest.raises(ImageInputError, match="inside entity directory"):
        ImageInput.from_entity_path(
            entity_key="Q1",
            candidate_id="image_001",
            entity_dir=entity_dir,
            relative_path="escape.png",
        )
