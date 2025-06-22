import os
from typing import Optional, Any
import logging

import cv2

logger = logging.getLogger(__name__)
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions

# Enum for delegate selection (CPU/GPU)
Delegate = _BaseOptions.Delegate

# Global cache to reuse MediaPipe detector instances across wrappers
_DETECTOR_CACHE: dict[tuple[str, str, str], object] = {}  # key: (wrapper, model_path, delegate) -> detector

# Assume models are downloaded to a specific path
MODELS_PATH = os.path.join(os.path.dirname(__file__), "mediapipe_models")
FACE_LANDMARKER_MODEL = os.path.join(MODELS_PATH, "face_landmarker.task")
HAND_LANDMARKER_MODEL = os.path.join(MODELS_PATH, "hand_landmarker.task")
POSE_LANDMARKER_MODEL = os.path.join(MODELS_PATH, "pose_landmarker_full.task")


class _OptionBuilderMixin:
    """Mixin that builds task options with overridable DEFAULT_PARAMS and OPTIONS_CLS."""

    OPTIONS_CLS: type | None = None  # to be set by subclass
    DEFAULT_PARAMS: dict = {}

    @classmethod
    def build_options(cls, base_options: BaseOptions, running_mode: vision.RunningMode, **overrides):
        params = {**cls.DEFAULT_PARAMS, **overrides}
        if cls.OPTIONS_CLS is None:
            raise NotImplementedError("Subclasses must define OPTIONS_CLS")
        return cls.OPTIONS_CLS(base_options=base_options, running_mode=running_mode, **params)


class BaseLandmarker(_OptionBuilderMixin):
    OPTIONS_CLS = None  # subclasses define
    DEFAULT_PARAMS = {}

    def __init__(
        self,
        model_path: str,
        running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
        delegate: str = "cpu",
        **kwargs,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file not found at {model_path}")

        # Select CPU/GPU delegate
        delegate_enum = Delegate.GPU if delegate.lower() == "gpu" else Delegate.CPU
        base_options = BaseOptions(model_asset_path=model_path, delegate=delegate_enum)

        self.options = self.build_options(base_options, running_mode, **kwargs)
        self.detector = self._get_detector(model_path, delegate_enum)

    def _create_options(self, base_options: BaseOptions, running_mode: vision.RunningMode, **kwargs):
        raise NotImplementedError

    def _create_detector(self, options):
        raise NotImplementedError

    def _get_detector(self, model_path, delegate_enum):
        cache_key = (self.__class__.__name__, model_path, delegate_enum)
        if cache_key in _DETECTOR_CACHE:
            return _DETECTOR_CACHE[cache_key]
        detector = self._create_detector(self.options)
        _DETECTOR_CACHE[cache_key] = detector
        return detector

    def detect(self, image: np.ndarray) -> Any:
        """Run landmark detection and return MediaPipe result.
        All errors are caught and logged as warnings to avoid crashing pipelines.
        """
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return self.detector.detect(mp_image)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("%s.detect failed: %s", self.__class__.__name__, e)
            return None

    def close(self):
        """Close the underlying detector and remove from cache if no longer referenced."""
        # Find cache key(s) pointing to this detector
        keys_to_remove = [k for k, v in _DETECTOR_CACHE.items() if v is self.detector]
        for k in keys_to_remove:
            _DETECTOR_CACHE.pop(k, None)
        self.detector.close()


class FaceLandmarkerWrapper(BaseLandmarker):
    OPTIONS_CLS = vision.FaceLandmarkerOptions
    DEFAULT_PARAMS = {
        "output_face_blendshapes": False,
        "output_facial_transformation_matrixes": False,
        "num_faces": 1,
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    def _create_detector(self, options):
        return vision.FaceLandmarker.create_from_options(options)


class HandLandmarkerWrapper(BaseLandmarker):
    OPTIONS_CLS = vision.HandLandmarkerOptions
    DEFAULT_PARAMS = {
        "num_hands": 2,
        "min_hand_detection_confidence": 0.5,
        "min_hand_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    def _create_detector(self, options):
        return vision.HandLandmarker.create_from_options(options)


class PoseLandmarkerWrapper(BaseLandmarker):
    OPTIONS_CLS = vision.PoseLandmarkerOptions
    DEFAULT_PARAMS = {
        "output_segmentation_masks": False,
        "num_poses": 1,
        "min_pose_detection_confidence": 0.5,
        "min_pose_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    def _create_detector(self, options):
        logger.debug("PoseLandmarkerWrapper: Creating detector with options: %s", options)
        detector = vision.PoseLandmarker.create_from_options(options)
        logger.debug("PoseLandmarkerWrapper: Detector created successfully.")
        return detector
