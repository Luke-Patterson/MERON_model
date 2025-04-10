import cv2
import dlib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict

class FacialLandmarksProcessor:
    """Process images to extract facial landmark features for malnutrition detection."""
    
    def __init__(self, landmark_model_path: Optional[str] = None):
        """
        Initialize the facial landmarks processor.
        
        Args:
            landmark_model_path: Path to dlib's facial landmark model file.
                               If None, will look in default data directory.
        """
        # Setup logging
        self.logger = logging.getLogger("MERON_landmarks")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize landmark detector
        if landmark_model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            landmark_model_path = str(base_dir / 'data' / 'shape_predictor_68_face_landmarks.dat')
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_model_path)
        
        # Define facial region indices
        # Based on dlib's 68-point facial landmark scheme
        self.FACIAL_REGIONS = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_mouth': list(range(48, 60)),
            'inner_mouth': list(range(60, 68))
        }
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Array of shape (68, 2) containing (x,y) coordinates for each landmark,
            or None if no face detected
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            if len(faces) == 0:
                self.logger.warning("No face detected in image")
                return None
            
            # Get landmarks for first face
            shape = self.predictor(gray, faces[0])
            landmarks = np.array([[shape.part(i).x, shape.part(i).y] 
                                for i in range(68)])
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error detecting landmarks: {str(e)}")
            return None
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be invariant to scale and translation.
        
        Args:
            landmarks: Array of shape (68, 2) containing landmark coordinates
            
        Returns:
            Normalized landmarks array of same shape
        """
        # Center landmarks around origin
        centered = landmarks - landmarks.mean(axis=0)
        
        # Scale to unit size
        scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
        normalized = centered / scale
        
        return normalized
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> Dict:
        """
        Extract geometric features from facial landmarks.
        
        Args:
            landmarks: Array of shape (68, 2) containing facial landmarks
            
        Returns:
            Dictionary of geometric features
        """
        try:
            features = {}
            
            # Basic measurements
            features['cheek_symmetry'] = self._calculate_cheek_symmetry(landmarks)
            features['temple_width'] = self._calculate_temple_width(landmarks)
            features['face_ratio'] = self._calculate_face_ratio(landmarks)
            features['eye_symmetry'] = self._calculate_eye_symmetry(landmarks)
            features['mouth_ratio'] = self._calculate_mouth_ratio(landmarks)
            
            # Area measurements
            features['face_area'] = self._calculate_face_area(landmarks)
            features['left_cheek_area'] = self._calculate_cheek_area(landmarks, 'left')
            features['right_cheek_area'] = self._calculate_cheek_area(landmarks, 'right')
            features['left_eye_area'] = self._calculate_eye_area(landmarks, 'left')
            features['right_eye_area'] = self._calculate_eye_area(landmarks, 'right')
            
            # Angular measurements
            features['jaw_angle'] = self._calculate_jaw_angle(landmarks)
            features['forehead_angle'] = self._calculate_forehead_angle(landmarks)
            features['left_jaw_angle'] = self._calculate_jaw_angle(landmarks, side='left')
            features['right_jaw_angle'] = self._calculate_jaw_angle(landmarks, side='right')
            
            # Advanced symmetry measures
            features['facial_symmetry'] = self._calculate_facial_symmetry(landmarks)
            features['eye_mouth_symmetry'] = self._calculate_eye_mouth_symmetry(landmarks)
            
            # Curvature measurements
            features['jaw_curvature'] = self._calculate_curvature(landmarks[self.FACIAL_REGIONS['jaw']])
            features['left_eye_curvature'] = self._calculate_curvature(landmarks[self.FACIAL_REGIONS['left_eye']])
            features['right_eye_curvature'] = self._calculate_curvature(landmarks[self.FACIAL_REGIONS['right_eye']])
            features['mouth_curvature'] = self._calculate_curvature(landmarks[self.FACIAL_REGIONS['outer_mouth']])
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting geometric features: {str(e)}")
            raise
    
    def process_image(self, image: np.ndarray) -> Optional[Dict]:
        """
        Process a single image to extract all facial landmark features.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary containing:
                'landmarks': Raw landmark coordinates
                'normalized_landmarks': Scale and translation invariant landmarks
                'geometric_features': Computed geometric features
            Returns None if face detection fails
        """
        # Detect landmarks
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return None
            
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks(landmarks)
        
        # Extract geometric features
        geometric_features = self.extract_geometric_features(normalized_landmarks)
        
        return {
            'landmarks': landmarks,
            'normalized_landmarks': normalized_landmarks,
            'geometric_features': geometric_features
        }
    
    def create_feature_vector(self, processed_data: Dict) -> np.ndarray:
        """
        Convert processed landmark data into a feature vector for modeling.
        
        Args:
            processed_data: Output from process_image()
            
        Returns:
            1D numpy array containing all features concatenated
        """
        features = []
        
        # Add normalized landmark coordinates
        features.extend(processed_data['normalized_landmarks'].flatten())
        
        # Add geometric features in consistent order
        geometric = processed_data['geometric_features']
        features.extend([
            geometric['cheek_symmetry'],
            geometric['temple_width'],
            geometric['face_ratio'],
            geometric['eye_symmetry'],
            geometric['mouth_ratio']
        ])
        
        return np.array(features)

    def _calculate_cheek_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate symmetry between left and right cheeks."""
        left_cheek = landmarks[self.FACIAL_REGIONS['jaw'][3]]
        right_cheek = landmarks[self.FACIAL_REGIONS['jaw'][-4]]
        nose_tip = landmarks[self.FACIAL_REGIONS['nose_tip'][-1]]
        return np.linalg.norm(left_cheek - nose_tip) / np.linalg.norm(right_cheek - nose_tip)

    def _calculate_temple_width(self, landmarks: np.ndarray) -> float:
        """Calculate width between temples."""
        left_temple = landmarks[self.FACIAL_REGIONS['left_eyebrow'][0]]
        right_temple = landmarks[self.FACIAL_REGIONS['right_eyebrow'][-1]]
        return np.linalg.norm(left_temple - right_temple)

    def _calculate_face_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate ratio of face height to width."""
        jaw_width = np.linalg.norm(landmarks[self.FACIAL_REGIONS['jaw'][0]] - 
                                 landmarks[self.FACIAL_REGIONS['jaw'][-1]])
        face_height = np.linalg.norm(landmarks[self.FACIAL_REGIONS['jaw'][8]] - 
                                   landmarks[self.FACIAL_REGIONS['nose_bridge'][0]])
        return face_height / jaw_width

    def _calculate_eye_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate symmetry between left and right eyes."""
        left_eye_width = np.linalg.norm(landmarks[self.FACIAL_REGIONS['left_eye'][0]] - 
                                      landmarks[self.FACIAL_REGIONS['left_eye'][3]])
        right_eye_width = np.linalg.norm(landmarks[self.FACIAL_REGIONS['right_eye'][0]] - 
                                       landmarks[self.FACIAL_REGIONS['right_eye'][3]])
        return left_eye_width / right_eye_width

    def _calculate_mouth_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate ratio of mouth height to width."""
        mouth_width = np.linalg.norm(landmarks[self.FACIAL_REGIONS['outer_mouth'][0]] - 
                                   landmarks[self.FACIAL_REGIONS['outer_mouth'][6]])
        mouth_height = np.linalg.norm(landmarks[self.FACIAL_REGIONS['outer_mouth'][3]] - 
                                    landmarks[self.FACIAL_REGIONS['outer_mouth'][9]])
        return mouth_height / mouth_width

    def _calculate_face_area(self, landmarks: np.ndarray) -> float:
        """Calculate area of face using jaw points."""
        jaw_points = landmarks[self.FACIAL_REGIONS['jaw']]
        return 0.5 * np.abs(np.dot(jaw_points[:, 0], np.roll(jaw_points[:, 1], 1)) - 
                           np.dot(jaw_points[:, 1], np.roll(jaw_points[:, 0], 1)))

    def _calculate_cheek_area(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate area of cheek region."""
        if side == 'left':
            points = np.array([
                landmarks[self.FACIAL_REGIONS['jaw'][0]],
                landmarks[self.FACIAL_REGIONS['left_eyebrow'][0]],
                landmarks[self.FACIAL_REGIONS['nose_bridge'][0]],
                landmarks[self.FACIAL_REGIONS['jaw'][8]]
            ])
        else:
            points = np.array([
                landmarks[self.FACIAL_REGIONS['jaw'][-1]],
                landmarks[self.FACIAL_REGIONS['right_eyebrow'][-1]],
                landmarks[self.FACIAL_REGIONS['nose_bridge'][0]],
                landmarks[self.FACIAL_REGIONS['jaw'][8]]
            ])
        return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - 
                           np.dot(points[:, 1], np.roll(points[:, 0], 1)))

    def _calculate_eye_area(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate area of eye region."""
        if side == 'left':
            points = landmarks[self.FACIAL_REGIONS['left_eye']]
        else:
            points = landmarks[self.FACIAL_REGIONS['right_eye']]
        return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - 
                           np.dot(points[:, 1], np.roll(points[:, 0], 1)))

    def _calculate_jaw_angle(self, landmarks: np.ndarray, side: str = None) -> float:
        """Calculate angle at jaw points."""
        if side == 'left':
            points = [landmarks[self.FACIAL_REGIONS['jaw'][0]],
                     landmarks[self.FACIAL_REGIONS['jaw'][4]],
                     landmarks[self.FACIAL_REGIONS['jaw'][8]]]
        elif side == 'right':
            points = [landmarks[self.FACIAL_REGIONS['jaw'][-1]],
                     landmarks[self.FACIAL_REGIONS['jaw'][-5]],
                     landmarks[self.FACIAL_REGIONS['jaw'][8]]]
        else:
            points = [landmarks[self.FACIAL_REGIONS['jaw'][0]],
                     landmarks[self.FACIAL_REGIONS['jaw'][8]],
                     landmarks[self.FACIAL_REGIONS['jaw'][-1]]]
        
        v1 = points[0] - points[1]
        v2 = points[2] - points[1]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    def _calculate_forehead_angle(self, landmarks: np.ndarray) -> float:
        """Calculate angle at forehead points."""
        points = np.array([
            landmarks[self.FACIAL_REGIONS['left_eyebrow'][0]],
            landmarks[self.FACIAL_REGIONS['nose_bridge'][0]],
            landmarks[self.FACIAL_REGIONS['right_eyebrow'][-1]]
        ])
        v1 = points[0] - points[1]
        v2 = points[2] - points[1]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate overall facial symmetry."""
        left_points = landmarks[:len(landmarks)//2]
        right_points = landmarks[len(landmarks)//2:]
        right_points = right_points.copy()
        right_points[:, 0] = -right_points[:, 0]
        return np.mean(np.abs(left_points - right_points))

    def _calculate_eye_mouth_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate symmetry between eye and mouth regions."""
        eye_symmetry = self._calculate_eye_symmetry(landmarks)
        mouth_symmetry = self._calculate_mouth_ratio(landmarks)
        return (eye_symmetry + mouth_symmetry) / 2

    def _calculate_curvature(self, points: np.ndarray) -> float:
        """Calculate curvature of a set of points."""
        return np.mean(np.abs(np.diff(points, axis=0))) 