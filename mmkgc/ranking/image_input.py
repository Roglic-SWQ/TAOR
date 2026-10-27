from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal


ImageMediaType = Literal["image/png", "image/jpeg"]


class ImageInputError(ValueError):
    """Raised when an image input cannot be decoded or addressed safely."""


@dataclass(frozen=True)
class ImageInput:
    entity_key: str
    candidate_id: str
    media_type: ImageMediaType
    image_bytes: bytes | None = None
    image_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entity_key, str) or not self.entity_key.strip():
            raise ImageInputError("entity_key must be a non-empty string")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ImageInputError("candidate_id must be a non-empty string")
        if self.media_type not in ("image/png", "image/jpeg"):
            raise ImageInputError(f"unsupported image media type: {self.media_type}")

        has_bytes = self.image_bytes is not None
        has_path = self.image_path is not None
        if has_bytes == has_path:
            raise ImageInputError(
                "exactly one of image_bytes or image_path must be provided"
            )

        if self.image_bytes is not None:
            copied_bytes = bytes(self.image_bytes)
            actual_media_type = _detect_image_media_type(copied_bytes)
            if actual_media_type != self.media_type:
                raise ImageInputError(
                    "image media type does not match decoded bytes: "
                    f"{self.media_type} != {actual_media_type}"
                )
            object.__setattr__(self, "image_bytes", copied_bytes)

        if self.image_path is not None:
            path = Path(self.image_path)
            if not path.is_file():
                raise ImageInputError(f"image file does not exist: {path}")
            actual_media_type = _detect_image_media_type(path.read_bytes())
            if actual_media_type != self.media_type:
                raise ImageInputError(
                    "image media type does not match decoded file bytes: "
                    f"{self.media_type} != {actual_media_type}"
                )
            object.__setattr__(self, "image_path", path)

    @classmethod
    def from_bytes(
        cls,
        *,
        entity_key: str,
        candidate_id: str,
        image_bytes: bytes,
    ) -> "ImageInput":
        media_type = _detect_image_media_type(image_bytes)
        return cls(
            entity_key=entity_key,
            candidate_id=candidate_id,
            media_type=media_type,
            image_bytes=bytes(image_bytes),
        )

    @classmethod
    def from_entity_path(
        cls,
        *,
        entity_key: str,
        candidate_id: str,
        entity_dir: Path,
        relative_path: str | Path,
        media_type: ImageMediaType | None = None,
    ) -> "ImageInput":
        return cls.from_allowed_path(
            entity_key=entity_key,
            candidate_id=candidate_id,
            allowed_root=entity_dir,
            relative_path=relative_path,
            media_type=media_type,
        )

    @classmethod
    def from_allowed_path(
        cls,
        *,
        entity_key: str,
        candidate_id: str,
        allowed_root: Path,
        relative_path: str | Path,
        media_type: ImageMediaType | None = None,
    ) -> "ImageInput":
        if media_type is not None and media_type not in ("image/png", "image/jpeg"):
            raise ImageInputError(f"unsupported image media type: {media_type}")

        relative = Path(relative_path)
        if relative.is_absolute():
            raise ImageInputError("image path must be a relative path")
        if any(part == ".." for part in relative.parts):
            raise ImageInputError("image path must stay inside entity directory")

        root = Path(allowed_root).resolve()
        candidate = root / relative
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ImageInputError("image path must stay inside entity directory") from exc

        if not resolved.is_file():
            raise ImageInputError(f"image file does not exist: {resolved}")

        actual_media_type = _detect_image_media_type(resolved.read_bytes())
        if media_type is not None and media_type != actual_media_type:
            raise ImageInputError(
                "image media type does not match decoded file bytes: "
                f"{media_type} != {actual_media_type}"
            )

        return cls(
            entity_key=entity_key,
            candidate_id=candidate_id,
            media_type=actual_media_type,
            image_path=resolved,
        )

    def open_pil(self):
        try:
            from PIL import Image
        except Exception as exc:
            raise ImageInputError("Pillow is required to open image inputs") from exc

        try:
            if self.image_bytes is not None:
                with Image.open(BytesIO(self.image_bytes)) as image:
                    return image.convert("RGB")
            if self.image_path is not None:
                with Image.open(self.image_path) as image:
                    return image.convert("RGB")
        except Exception as exc:
            raise ImageInputError("failed to decode image input as PNG or JPEG") from exc

        raise ImageInputError("image input must contain bytes or a path")

    def as_data_url(self) -> str:
        if self.image_bytes is not None:
            payload = self.image_bytes
        elif self.image_path is not None:
            payload = self.image_path.read_bytes()
        else:
            raise ImageInputError("image input must contain bytes or a path")

        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


def _detect_image_media_type(image_bytes: bytes) -> ImageMediaType:
    try:
        from PIL import Image
    except Exception as exc:
        raise ImageInputError("Pillow is required to decode image inputs") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format
    except Exception as exc:
        raise ImageInputError("image bytes must decode as PNG or JPEG") from exc

    if image_format == "PNG":
        return "image/png"
    if image_format == "JPEG":
        return "image/jpeg"
    raise ImageInputError("image bytes must decode as PNG or JPEG")
