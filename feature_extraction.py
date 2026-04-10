import math
from typing import Dict, List

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands


FEATURE_COLUMNS = ["aF1", "bF1", "cF1", "dF1", "eF1", "F2", "F3", "F4", "F5", "F6"]


def _triangle_area(a: float, b: float, c: float) -> float:
    s = (a + b + c) / 2.0
    # Guard against tiny negative values caused by floating-point precision.
    inside = max(s * (s - a) * (s - b) * (s - c), 0.0)
    return math.sqrt(inside)


def _pt(landmarks, hand_landmark, width: int, height: int) -> List[float]:
    return [
        landmarks.landmark[hand_landmark].x * width,
        landmarks.landmark[hand_landmark].y * height,
    ]


def extract_feature_vector(image_path: str) -> Dict[str, float]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        raise ValueError(f"No hand landmarks detected in image: {image_path}")

    image_height, image_width, _ = image.shape
    arr = results.multi_hand_landmarks[0]

    wrist = _pt(arr, mp_hands.HandLandmark.WRIST, image_width, image_height)
    thumb_mcp = _pt(arr, mp_hands.HandLandmark.THUMB_MCP, image_width, image_height)
    thumb_tip = _pt(arr, mp_hands.HandLandmark.THUMB_TIP, image_width, image_height)

    index_finger_mcp = _pt(arr, mp_hands.HandLandmark.INDEX_FINGER_MCP, image_width, image_height)
    index_finger_tip = _pt(arr, mp_hands.HandLandmark.INDEX_FINGER_TIP, image_width, image_height)

    middle_finger_mcp = _pt(arr, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, image_width, image_height)
    middle_finger_tip = _pt(arr, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, image_width, image_height)

    ring_finger_mcp = _pt(arr, mp_hands.HandLandmark.RING_FINGER_MCP, image_width, image_height)
    ring_finger_tip = _pt(arr, mp_hands.HandLandmark.RING_FINGER_TIP, image_width, image_height)

    pinky_mcp = _pt(arr, mp_hands.HandLandmark.PINKY_MCP, image_width, image_height)
    pinky_tip = _pt(arr, mp_hands.HandLandmark.PINKY_TIP, image_width, image_height)

    # Keep this definition consistent with loopcode.py for training/prediction parity.
    highest = [thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]

    d = math.dist(index_finger_mcp, pinky_mcp) / 2.0
    left = [wrist[0] - d, wrist[1]]
    right = [wrist[0] + d - 5, wrist[1]]
    width_of_wrist = math.dist(left, right)

    lengths = [
        math.dist(thumb_tip, thumb_mcp),
        math.dist(index_finger_tip, index_finger_mcp),
        math.dist(middle_finger_tip, middle_finger_mcp),
        math.dist(ring_finger_tip, ring_finger_mcp),
        math.dist(pinky_tip, pinky_mcp),
    ]
    longest = max(lengths)

    a = math.dist(wrist, index_finger_mcp)
    b = math.dist(wrist, pinky_mcp)
    c = math.dist(index_finger_mcp, pinky_mcp)
    area_on_palm = _triangle_area(a, b, c)
    peri_triangle = a + b + c

    a = math.dist(wrist, thumb_mcp)
    b = math.dist(thumb_mcp, index_finger_mcp)
    c = math.dist(index_finger_mcp, middle_finger_mcp)
    d = math.dist(middle_finger_mcp, ring_finger_mcp)
    e = math.dist(ring_finger_mcp, pinky_mcp)
    f = math.dist(pinky_mcp, wrist)
    poly_peri = a + b + c + d + e + f

    # Keeping highest[0] to mirror legacy script behavior exactly.
    a = math.dist(left, highest[0])
    b = math.dist(right, highest[0])
    c = math.dist(left, right)
    area_on_hand = _triangle_area(a, b, c)

    palm_len = math.dist(wrist, middle_finger_mcp)
    hand_len = math.dist(index_finger_tip, wrist)

    if longest == 0 or width_of_wrist == 0 or palm_len == 0 or peri_triangle == 0 or area_on_hand == 0:
        raise ValueError(f"Degenerate geometry for image: {image_path}")

    return {
        "aF1": lengths[0] / longest,
        "bF1": lengths[1] / longest,
        "cF1": lengths[2] / longest,
        "dF1": lengths[3] / longest,
        "eF1": lengths[4] / longest,
        "F2": area_on_palm / (width_of_wrist * palm_len),
        "F3": palm_len / longest,
        "F4": poly_peri / peri_triangle,
        "F5": area_on_palm / area_on_hand,
        "F6": hand_len / width_of_wrist,
    }
