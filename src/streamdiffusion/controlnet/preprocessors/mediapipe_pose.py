import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from typing import Union, Optional, List, Tuple, Dict
import logging
import time
logger = logging.getLogger(__name__)

# Heavy constant maps moved to separate module for memory efficiency
from .constants import (
    MEDIAPIPE_TO_OPENPOSE_MAP,
    OPENPOSE_LIMB_SEQUENCE,
    OPENPOSE_COLORS,
    OPENPOSE_FACE_CONNECTIONS,
    FACE_COLORS,
    MEDIAPIPE_TO_OPENPOSE_FACE_MAP,
)
from .base import BasePreprocessor

try:
    import mediapipe as mp

    from .mediapipe_landmarkers import (
        FaceLandmarkerWrapper,
        HandLandmarkerWrapper,
        PoseLandmarkerWrapper,
        FACE_LANDMARKER_MODEL,
        HAND_LANDMARKER_MODEL,
        POSE_LANDMARKER_MODEL,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class MediaPipePosePreprocessor(BasePreprocessor):
    """
    MediaPipe-based pose preprocessor for ControlNet that outputs OpenPose-style annotations.

    This preprocessor uses the latest MediaPipe Solutions API to perform modular detection of
    pose, face, and hand landmarks. It converts the detected keypoints into an OpenPose-compatible
    format for use with ControlNet.

    Features:
    - Modular detection: Enable or disable pose, face, and hand detection independently.
    - OpenPose compatibility: Converts MediaPipe landmarks to a 25-keypoint OpenPose skeleton.
    - Temporal smoothing: Reduces jitter in video streams for more stable animations.
    """

    def __init__(
        self,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        enable_pose: bool = True,
        enable_face: bool = True,
        enable_hands: bool = True,
        line_thickness: int = 2,
        circle_radius: int = 4,
        confidence_threshold: float = 0.3,
        enable_smoothing: bool = True,
        smoothing_factor: float = 0.7,
        pose_options: Optional[Dict] = None,
        face_options: Optional[Dict] = None,
        hand_options: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initializes the MediaPipePosePreprocessor.

        Args:
            detect_resolution: The resolution for landmark detection.
            image_resolution: The output image resolution.
            enable_pose: Whether to enable pose detection.
            enable_face: Whether to enable face landmark detection.
            enable_hands: Whether to enable hand landmark detection.
            line_thickness: The thickness of the drawn skeleton lines.
            circle_radius: The radius of the drawn keypoint circles.
            confidence_threshold: The minimum confidence score for a keypoint to be rendered.
            enable_smoothing: Whether to apply temporal smoothing to the keypoints.
            smoothing_factor: The strength of the temporal smoothing (0-1).
            pose_options: Custom options for the PoseLandmarker.
            face_options: Custom options for the FaceLandmarker.
            hand_options: Custom options for the HandLandmarker.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for MediaPipe pose preprocessing. "
                "Install it with: pip install mediapipe"
            )

        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            enable_pose=enable_pose,
            enable_face=enable_face,
            enable_hands=enable_hands,
            line_thickness=line_thickness,
            circle_radius=circle_radius,
            confidence_threshold=confidence_threshold,
            enable_smoothing=enable_smoothing,
            smoothing_factor=smoothing_factor,
            pose_options=pose_options,
            face_options=face_options,
            hand_options=hand_options,
            **kwargs,
        )

        self.enable_pose = enable_pose
        self.enable_face = enable_face
        self.enable_hands = enable_hands

        self.pose_detector = None
        self.face_detector = None
        self.hand_detector = None

        logger.debug("Initializing Pose Landmarker...")
        if self.enable_pose:
            self.pose_detector = PoseLandmarkerWrapper(
                model_path=POSE_LANDMARKER_MODEL, **(pose_options or {})
            )
            logger.debug("Pose Landmarker initialized. Type: %s", type(self.pose_detector))
        else:
            logger.debug("Pose Landmarker disabled.")

        logger.debug("Initializing Face Landmarker...")
        if self.enable_face:
            self.face_detector = FaceLandmarkerWrapper(
                model_path=FACE_LANDMARKER_MODEL, **(face_options or {})
            )
            logger.debug("Face Landmarker initialized.")
        else:
            logger.debug("Face Landmarker disabled.")

        logger.debug("Initializing Hand Landmarker...")
        if self.enable_hands:
            self.hand_detector = HandLandmarkerWrapper(
                model_path=HAND_LANDMARKER_MODEL, **(hand_options or {})
            )

        # Buffer storing previous smoothed keypoints per unique pose id
        self._smoothing_buffers: Dict[str, List[List[float]]] = {}

        # Copy ctor args to explicit attributes so helpers avoid hidden `self.params`
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        self._face_idx = np.fromiter(
            [MEDIAPIPE_TO_OPENPOSE_FACE_MAP[i] for i in range(70)], dtype=np.int32
        )
    
    def __call__(self, input_image: Union[Image.Image, np.ndarray], **kwargs) -> Image.Image:
        """
        Process an input image to detect and draw pose, face, and hand landmarks.

        Args:
            input_image: The input image in PIL or NumPy format.
            **kwargs: Additional keyword arguments.

        Returns:
            A PIL Image with the detected landmarks drawn.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed")

        # Convert incoming image to BGR once and keep that space for the whole pipeline.
        if isinstance(input_image, Image.Image):
            # PIL images are RGB; convert directly to BGR numpy array
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        elif isinstance(input_image, np.ndarray):
            if input_image.shape[2] == 4:
                # RGBA → BGR
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)
            # else assume already BGR (OpenCV default)

        detect_resolution = self.detect_resolution
        image_resolution = self.image_resolution

        image_resized = cv2.resize(input_image, (detect_resolution, detect_resolution))

        canvas = np.zeros_like(image_resized)

        if self.enable_pose and self.pose_detector:
            pose_results = self.pose_detector.detect(image_resized)
            if pose_results and pose_results.pose_landmarks:
                for landmarks in pose_results.pose_landmarks:
                    openpose_keypoints = self._mediapipe_to_openpose(
                        landmarks, detect_resolution, detect_resolution
                    )
                    if self.enable_smoothing:
                        openpose_keypoints = self._apply_smoothing(openpose_keypoints)
                    canvas = self._draw_openpose_skeleton(canvas, openpose_keypoints)

        if self.enable_face and self.face_detector:
            face_results = self.face_detector.detect(image_resized)
            if face_results and face_results.face_landmarks:
                for landmarks in face_results.face_landmarks:
                    canvas = self._draw_face_keypoints(canvas, landmarks)

        if self.enable_hands and self.hand_detector:
            hand_results = self.hand_detector.detect(image_resized)
            if hand_results and hand_results.hand_landmarks:
                for i, landmarks in enumerate(hand_results.hand_landmarks):
                    is_left = hand_results.handedness[i][0].category_name == 'Left'
                    canvas = self._draw_hand_keypoints(canvas, landmarks, is_left)

        if image_resolution != detect_resolution:
            canvas = cv2.resize(canvas, (image_resolution, image_resolution), interpolation=cv2.INTER_AREA)

        # Convert BGR canvas back to RGB for PIL output
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(canvas_rgb)
    
    def _apply_smoothing(self, keypoints: List[List[float]], pose_id: str | None = None) -> List[List[float]]:
        """
        Apply TouchDesigner-inspired temporal smoothing
        
        Args:
            keypoints: Current frame keypoints
            pose_id: Unique identifier for this pose
            
        Returns:
            Smoothed keypoints
        """
        # Fast-exit if smoothing disabled or missing keypoints
        if not self.enable_smoothing or not keypoints:
            return keypoints

        # Derive a stable pose_id if none supplied – use a simple hash of the first visible
        # landmark positions so multi-person frames don’t overwrite each other.
        if pose_id is None:
            try:
                first_kp = next(pt for pt in keypoints if pt[2] > 0.1)
                pose_id = f"{hash((round(first_kp[0],2), round(first_kp[1],2)))}"
            except StopIteration:
                pose_id = "default"

        # Initialise history buffer lazily
        if pose_id not in self._smoothing_buffers:
            self._smoothing_buffers[pose_id] = keypoints.copy()
            return keypoints

        smoothing_factor = self.smoothing_factor
            
        # Apply exponential smoothing (simplified 1-euro filter style)
        smoothed = []
        previous = self._smoothing_buffers[pose_id]
        
        for i, (current_point, prev_point) in enumerate(zip(keypoints, previous)):
            if current_point[2] > 0.1:  # Only smooth if confidence is good
                smoothed_x = prev_point[0] * smoothing_factor + current_point[0] * (1 - smoothing_factor)
                smoothed_y = prev_point[1] * smoothing_factor + current_point[1] * (1 - smoothing_factor)
                smoothed_conf = current_point[2]  # Keep current confidence
                smoothed.append([smoothed_x, smoothed_y, smoothed_conf])
            else:
                smoothed.append(current_point)
        
        # Update buffer
        self._smoothing_buffers[pose_id] = smoothed
        return smoothed
    
    def _mediapipe_to_openpose(
        self, mediapipe_landmarks: List, image_width: int, image_height: int
    ) -> List[List[float]]:
        """
        Convert MediaPipe landmarks to OpenPose format.

        Args:
            mediapipe_landmarks: A list of MediaPipe pose landmarks.
            image_width: The width of the image.
            image_height: The height of the image.

        Returns:
            A list of OpenPose keypoints in [x, y, confidence] format.
        """
        if not mediapipe_landmarks:
            return []
        
        # Initialize OpenPose keypoints array (25 points x 3 values)
        openpose_keypoints = [[0.0, 0.0, 0.0] for _ in range(25)]
        
        # Convert MediaPipe landmarks to pixel coordinates
        mp_points = []
        for landmark in mediapipe_landmarks:
            x = landmark.x * image_width
            y = landmark.y * image_height
            confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            mp_points.append([x, y, confidence])
        
        # Map MediaPipe points to OpenPose format
        for openpose_idx, mediapipe_idx in MEDIAPIPE_TO_OPENPOSE_MAP.items():
            if mediapipe_idx is not None and mediapipe_idx < len(mp_points):
                openpose_keypoints[openpose_idx] = mp_points[mediapipe_idx]
        
        # Calculate derived points
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Neck (1): midpoint between shoulders
        if (len(mp_points) > 12 and mp_points[11][2] > confidence_threshold and 
            mp_points[12][2] > confidence_threshold):
            neck_x = (mp_points[11][0] + mp_points[12][0]) / 2
            neck_y = (mp_points[11][1] + mp_points[12][1]) / 2
            neck_conf = min(mp_points[11][2], mp_points[12][2])
            openpose_keypoints[1] = [neck_x, neck_y, neck_conf]
        
        # MidHip (8): midpoint between hips
        if (len(mp_points) > 24 and mp_points[23][2] > confidence_threshold and 
            mp_points[24][2] > confidence_threshold):
            midhip_x = (mp_points[23][0] + mp_points[24][0]) / 2
            midhip_y = (mp_points[23][1] + mp_points[24][1]) / 2
            midhip_conf = min(mp_points[23][2], mp_points[24][2])
            openpose_keypoints[8] = [midhip_x, midhip_y, midhip_conf]
        
        return openpose_keypoints
    
    def _draw_openpose_skeleton(
        self, image: np.ndarray, keypoints: List[List[float]]
    ) -> np.ndarray:
        """
        Draw an OpenPose-style skeleton on an image.

        Args:
            image: The input image as a NumPy array.
            keypoints: A list of OpenPose keypoints.

        Returns:
            The image with the skeleton drawn on it.
        """
        if not keypoints or len(keypoints) != 25:
            return image
        
        h, w = image.shape[:2]
        line_thickness = self.params.get('line_thickness', 2)
        circle_radius = self.params.get('circle_radius', 4)
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Draw limbs
        for i, (start_idx, end_idx) in enumerate(OPENPOSE_LIMB_SEQUENCE):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > confidence_threshold and keypoints[end_idx][2] > confidence_threshold):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                # Use standard OpenPose colors
                color = OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]
                
                cv2.line(image, start_point, end_point, color, line_thickness)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint[2] > confidence_threshold:
                center = (int(keypoint[0]), int(keypoint[1]))
                color = OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]
                cv2.circle(image, center, circle_radius, color, -1)
        
        return image
    
    def _draw_hand_keypoints(self, image: np.ndarray, hand_landmarks: List, is_left_hand: bool = True) -> np.ndarray:
        """
        Draw hand keypoints in OpenPose style - FIXED coordinate mapping
        
        Args:
            image: Input image
            hand_landmarks: MediaPipe hand landmarks
            is_left_hand: Whether this is the left hand
            
        Returns:
            Image with hand keypoints drawn
        """
        if not hand_landmarks:
            return image
        
        h, w = image.shape[:2]
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Standard hand connections (21 landmarks per hand)
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17),
        ]
        
        # Convert to pixel coordinates - FIXED
        hand_points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            hand_points.append((x, y))
        
        # Standard hand colors
        hand_color = [255, 128, 0] if is_left_hand else [0, 255, 255]  # Orange for left, cyan for right
        
        # Draw connections
        for start_idx, end_idx in hand_connections:
            if start_idx < len(hand_points) and end_idx < len(hand_points):
                cv2.line(image, hand_points[start_idx], hand_points[end_idx], hand_color, 2)
        
        # Draw keypoints
        for point in hand_points:
            cv2.circle(image, point, 3, hand_color, -1)
        
        return image
    
    def _draw_face_keypoints(self, image: np.ndarray, face_landmarks: List) -> np.ndarray:
        """
        Draw face landmarks in OpenPose style.

        Args:
            image: Input image canvas.
            face_landmarks: MediaPipe face landmarks (468 points).

        Returns:
            Image with face skeleton drawn.
        """
        if not face_landmarks:
            return image

        h, w = image.shape[:2]
        line_thickness = self.params.get('line_thickness', 2)
        confidence_threshold = self.params.get('confidence_threshold', 0.3)

        # Vectorised conversion of 468 face landmarks (x,y) and mapping to 70-point OpenPose order
        pts = np.stack([(lm.x * w, lm.y * h) for lm in face_landmarks], axis=0).astype(np.float32)

        # Map to 70-point OpenPose order in a single take
        openpose_pts = pts[self._face_idx]  # (70,2)

        # Draw connections
                # Draw connections using the mapped 70-point array
        for i, (start_idx, end_idx) in enumerate(OPENPOSE_FACE_CONNECTIONS):
            if start_idx < openpose_pts.shape[0] and end_idx < openpose_pts.shape[0]:
                start_point = tuple(openpose_pts[start_idx].astype(int))
                end_point = tuple(openpose_pts[end_idx].astype(int))
                color = FACE_COLORS[i % len(FACE_COLORS)]
                cv2.line(image, start_point, end_point, color, line_thickness)

        return image
    
    # DEPRECATED - old method, keep for reference
    def process(self, image: Union[Image.Image, np.ndarray]):
        """
        Apply MediaPipe pose detection and create OpenPose-style annotation

        Args:
            image: Input image

        Returns:
            PIL Image with OpenPose-style pose skeleton on black background
        """
        return self(image)
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply MediaPipe pose detection and create OpenPose-style annotation
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with OpenPose-style pose skeleton on black background
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize for detection
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        # Convert to RGB numpy array for MediaPipe
        rgb_image = np.asarray(image_resized)  # Already RGB, avoid extra conversion
        
        pose_results = None
        hand_results = None
        face_results = None

        if self.enable_pose and self.pose_detector:
            pose_results = self.pose_detector.detect(rgb_image)
        
        if self.enable_hands and self.hand_detector:
            hand_results = self.hand_detector.detect(rgb_image)

        if self.enable_face and self.face_detector:
            face_results = self.face_detector.detect(rgb_image)
        
        # Create black background for pose annotation
        pose_image = np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)
        
        # Draw pose skeleton if detected
        if pose_results and pose_results.pose_landmarks:
            # Convert MediaPipe to OpenPose format
            openpose_keypoints = self._mediapipe_to_openpose(
                pose_results.pose_landmarks[0], # Assuming single person detection for this path
                detect_resolution, 
                detect_resolution
            )
            
            # Apply TouchDesigner-style smoothing
            openpose_keypoints = self._apply_smoothing(openpose_keypoints, "main_pose")
            
            # Draw OpenPose-style skeleton
            pose_image = self._draw_openpose_skeleton(pose_image, openpose_keypoints)
        
        # Draw hands if enabled
        draw_hands = self.params.get('draw_hands', True)
        if draw_hands and self.enable_hands and hand_results and hand_results.hand_landmarks:
            for i, landmarks_list in enumerate(hand_results.hand_landmarks):
                if hand_results.handedness and i < len(hand_results.handedness):
                    is_left = hand_results.handedness[i][0].category_name == 'Left'
                    pose_image = self._draw_hand_keypoints(
                        pose_image, landmarks_list, is_left_hand=is_left
                    )

        
        # Draw face if enabled
        draw_face = self.params.get('draw_face', True)
        if draw_face and self.enable_face and face_results and face_results.face_landmarks:
            for landmarks_list in face_results.face_landmarks:
                pose_image = self._draw_face_keypoints(
                    pose_image, landmarks_list
                )
        
        # Convert back to PIL
        pose_pil = Image.fromarray(pose_image[:, :, ::-1])  # BGR -> RGB with channel flip
        
        # Resize to target resolution
        image_resolution = self.params.get('image_resolution', 512)
        if pose_pil.size != (image_resolution, image_resolution):
            pose_pil = pose_pil.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return pose_pil
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid unnecessary CPU transfers
        
        Args:
            image_tensor: Input image tensor on GPU
            
        Returns:
            Processed pose tensor on GPU
        """
        # For MediaPipe, we need to go through CPU anyway, so use standard process
        pil_image = self.tensor_to_pil(image_tensor)
        processed_pil = self.process(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    def reset_smoothing_buffers(self):
        """Reset smoothing buffers (useful for new sequences)"""
        logger.info("MediaPipePosePreprocessor.reset_smoothing_buffers: Clearing smoothing buffers")
        self._smoothing_buffers.clear()
    
    def __del__(self):
        """Cleanup MediaPipe detector"""
        if hasattr(self, '_detector') and self._detector is not None:
            self._detector.close() 