"""Mission 1 - Reconnaissance de gestes (MediaPipe Hands)."""
from __future__ import annotations

import io
import threading
import time
import urllib.request
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

from utils import badges

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

MODEL_URL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"
DEFAULT_THRESHOLD = 1.0

GESTURE_LABELS = {
    "Thumb_Up": "Pouce levé",
    "Thumb_Down": "Pouce baissé",
    "Open_Palm": "Main ouverte",
    "Closed_Fist": "Poing fermé",
    "Pointing_Up": "Doigt pointé",
    "Victory": "Signe victoire (V)",
    "ILoveYou": "Je t'aime",
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _ensure_model_file() -> tuple[Path | None, str | None]:
    """Télécharge le modèle si absent."""
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            return MODEL_PATH, None
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            MODEL_PATH.write_bytes(response.read())
        return MODEL_PATH, None
    except Exception as exc:  # pragma: no cover - erreur réseau/IO
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def get_image_recognizer(model_path: str):
    """Reconnaisseur pour image fixe."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.GestureRecognizer.create_from_options(options)


def create_video_recognizer(model_path: str):
    """Reconnaisseur pour flux vidéo (timestamp requis)."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.GestureRecognizer.create_from_options(options)


def _load_image(uploaded_image) -> np.ndarray:
    """Convertit l'image uploadée en tableau RGB."""
    pil_image = Image.open(uploaded_image).convert("RGB")
    return np.array(pil_image)


def _draw_landmarks(image_bgr: np.ndarray, hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
        )
        mp_drawing.draw_landmarks(
            image_bgr,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )


def _extract_gestures(result, threshold: float) -> tuple[list[str], list[float], list[str]]:
    if not result.hand_landmarks:
        return ["Aucune main détectée"], [0.0], ["Aucune main détectée"]
    labels: list[str] = []
    scores: list[float] = []
    raw_labels: list[str] = []
    if result.gestures:
        for gesture_list in result.gestures:
            if not gesture_list:
                labels.append("Geste non reconnu")
                scores.append(0.0)
                raw_labels.append("Geste non reconnu")
                continue
            top = gesture_list[0]
            score = float(top.score)
            raw_label = GESTURE_LABELS.get(top.category_name, top.category_name)
            raw_labels.append(raw_label)
            scores.append(score)
            if score < threshold:
                labels.append("Geste non reconnu")
            else:
                labels.append(raw_label)
    else:
        labels = ["Geste non reconnu"] * len(result.hand_landmarks)
        scores = [0.0] * len(result.hand_landmarks)
        raw_labels = ["Geste non reconnu"] * len(result.hand_landmarks)
    return labels, scores, raw_labels


def _label_color(gesture: str):
    gesture_lower = gesture.lower()
    if "pouce" in gesture_lower:
        return (0, 200, 0)
    if gesture in ("Aucune main détectée", "Geste non reconnu"):
        return (0, 0, 255)
    return (0, 200, 200)


def _overlay_label(gesture: str) -> str:
    """OpenCV ne gère pas les accents : utiliser ASCII."""
    return (
        gesture.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ô", "o")
        .replace("î", "i")
    )


def _format_gesture_list(gestures: list[str]) -> str:
    if not gestures:
        return "—"
    if len(gestures) == 1:
        return gestures[0]
    return " | ".join(gestures)


def recognize_on_image(image_bytes: bytes, recognizer, threshold: float):
    """Reconnaissance sur image fixe + annotation."""
    image_rgb = _load_image(io.BytesIO(image_bytes))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = recognizer.recognize(mp_image)
    gestures, scores, _ = _extract_gestures(result, threshold)
    gesture_text = _format_gesture_list(gestures)
    confidence = max(scores) if scores else 0.0

    if result.hand_landmarks:
        _draw_landmarks(image_bgr, result.hand_landmarks)
    cv2.putText(
        image_bgr,
        _overlay_label(gesture_text),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _label_color(gesture_text),
        2,
        cv2.LINE_AA,
    )
    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return gesture_text, annotated_rgb, confidence


class HandGestureVideoProcessor(VideoProcessorBase):
    """Traitement vidéo en direct via MediaPipe Gesture Recognizer."""

    def __init__(self, model_path: str):
        self.recognizer = create_video_recognizer(model_path)
        self.last_gesture = "Aucune main détectée"
        self.last_confidence = 0.0
        self.threshold = DEFAULT_THRESHOLD
        self._lock = threading.Lock()
        self._last_ts = 0

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        with self._lock:
            threshold = self.threshold
        labels, scores, _ = _extract_gestures(result, threshold)
        label_text = _format_gesture_list(labels)
        confidence = max(scores) if scores else 0.0

        if result.hand_landmarks:
            _draw_landmarks(image_bgr, result.hand_landmarks)

        cv2.putText(
            image_bgr,
            _overlay_label(label_text),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _label_color(label_text),
            2,
            cv2.LINE_AA,
        )

        with self._lock:
            self.last_gesture = label_text
            self.last_confidence = confidence

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def get_last_gesture(self) -> str:
        with self._lock:
            return self.last_gesture

    def get_last_confidence(self) -> float:
        with self._lock:
            return self.last_confidence

    def set_threshold(self, threshold: float):
        with self._lock:
            self.threshold = threshold


def render_feedback(container, gesture: str):
    container.empty()
    if not gesture:
        container.info("Démarrez la caméra pour analyser votre geste.")
        return

    gesture_lower = gesture.lower()
    if "pouce" in gesture_lower:
        badges.unlock_badge("mission1")
        container.success("🎉 Bravo ! Geste validé : **pouce levé** 👍")
    elif gesture in ("Aucune main détectée", "Geste non reconnu"):
        container.error("Geste non reconnu ou main absente. Réessayez en montrant clairement un pouce levé.")
    else:
        container.warning(f"Geste détecté : **{gesture}**. La mission attend un **pouce levé** 👍.")


def run():
    st.title("Mission 1 : Découverte des gestes 🖐️")
    st.write(
        "**Objectif** : Faites un **geste de la main** devant la caméra (par ex. un pouce levé 👍). "
        "L'IA le reconnaît **en direct** grâce à MediaPipe Gesture Recognizer."
    )
    st.info(
        "🎯 **Défi** : Trouvez le **seuil le plus élevé** qui permet encore de détecter votre "
        "pouce levé sans faute. Augmentez le seuil jusqu'à ce que la détection devienne "
        "intermittente, puis revenez légèrement en arrière.\n\n"
        "👉 Seuil élevé = **précision** (moins de faux positifs), mais **rappel** plus faible.\n"
        "👉 Seuil bas = **rappel** élevé, mais plus de fausses alertes.\n\n"
        "Testez des gestes ambigus (poing, main ouverte) pour voir l'effet."
    )

    model_path, error = _ensure_model_file()
    if not model_path:
        st.error(f"Impossible de télécharger le modèle MediaPipe : {error}")
        st.stop()

    if "mission1_threshold" not in st.session_state:
        st.session_state.mission1_threshold = DEFAULT_THRESHOLD
    if "mission1_balloons" not in st.session_state:
        st.session_state.mission1_balloons = False

    if "mission1_completed" not in st.session_state:
        st.session_state.mission1_completed = False

    if WEBRTC_AVAILABLE:
        st.subheader("Détection en direct")
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_right:
            ctx = webrtc_streamer(
                key="mission1-live",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            }, "audio": False},
                video_processor_factory=lambda: HandGestureVideoProcessor(str(model_path)),
                async_processing=True,
            )
        with col_left:
            threshold = st.slider(
                "Seuil de confiance",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.mission1_threshold,
                step=0.01,
                key="mission1_threshold",
            )
            st.caption(f"Seuil de confiance : {threshold:.2f}")

            st.caption("Astuce : montrez un pouce levé pour valider la mission.")
            st.write("Dernier geste détecté :")

            status_placeholder = st.empty()
            confidence_placeholder = st.empty()
            feedback_box = st.empty()
            mission_box = st.empty()

            @st.fragment(run_every=0.2)
            def _live_status():
                gesture = None
                confidence = 0.0

                if ctx.state.playing and ctx.video_processor:
                    ctx.video_processor.set_threshold(st.session_state.mission1_threshold)
                    gesture = ctx.video_processor.get_last_gesture()
                    confidence = ctx.video_processor.get_last_confidence()

                status_placeholder.markdown(f"**{gesture or '—'}**")
                confidence_placeholder.write(f"Confiance actuelle : **{confidence:.2f}**")

                render_feedback(feedback_box, gesture)

                # ✅ ÉTAPE 1 : détecter la transition et forcer 1 rerun complet
                if st.session_state.badges.get("mission1") and not st.session_state.mission1_completed:
                    st.session_state.mission1_completed = True
                    st.rerun()


        _live_status()

        # ✅ Affichage persistant + 🎉 ballons one-shot (hors fragment)
        if st.session_state.mission1_completed:
            st.success("Mission 1 réussie ✅")
            st.markdown("<div style='font-size:40px'>👍</div>", unsafe_allow_html=True)

        if st.session_state.mission1_completed and not st.session_state.mission1_balloons:
            st.session_state.mission1_balloons = True
            st.balloons()

        # Bouton hors fragment (plus stable)
        if st.button("Accueil ➡️", key="introduction-next-live"):
            st.session_state.page = "introduction"
            st.rerun()