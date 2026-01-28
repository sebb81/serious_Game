"""Mission 2 - Face Mesh en temps réel (MediaPipe)."""
from __future__ import annotations

import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from utils import badges

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _dist(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def _overlay_label(text: str) -> str:
    """OpenCV ne gère pas les accents : utiliser ASCII."""
    return (
        text.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ô", "o")
        .replace("î", "i")
    )


def _overlay_color(emotion: str):
    if emotion == "Sourire":
        return (0, 200, 0)
    if emotion == "Surpris":
        return (0, 215, 255)
    if emotion == "Triste":
        return (80, 80, 255)
    if emotion == "Neutre":
        return (220, 220, 220)
    return (200, 200, 200)


def _estimate_emotion(landmarks) -> tuple[str, dict]:
    """Heuristique simple basée sur le ratio bouche/visage."""
    # Indices utiles (MediaPipe Face Mesh)
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    face_width = _dist(left_cheek, right_cheek)
    face_height = _dist(forehead, chin)
    mouth_width = _dist(mouth_left, mouth_right)
    mouth_height = _dist(upper_lip, lower_lip)
    mouth_center_y = (upper_lip.y + lower_lip.y) / 2
    corners_y = (mouth_left.y + mouth_right.y) / 2

    mouth_open_ratio = mouth_height / max(face_height, 1e-6)
    smile_width_ratio = mouth_width / max(face_width, 1e-6)
    corner_delta = corners_y - mouth_center_y  # >0: coins vers le bas

    if mouth_open_ratio > 0.065:
        label = "Surpris"
    elif corner_delta < -0.01 and smile_width_ratio > 0.43:
        label = "Sourire"
    elif corner_delta > 0.012:
        label = "Triste"
    else:
        label = "Neutre"

    metrics = {
        "mouth_open_ratio": mouth_open_ratio,
        "smile_width_ratio": smile_width_ratio,
        "corner_delta": corner_delta,
    }
    return label, metrics


class FaceMeshVideoProcessor(VideoProcessorBase):
    """Traitement vidéo en direct via MediaPipe Face Mesh."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_emotion = "Aucun visage détecté"
        self.last_metrics = {}
        self.threshold = 0.43  # seuil smile_width_ratio (valeur par défaut)
        self._lock = threading.Lock()

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        label = "Aucun visage détecté"
        metrics = {}

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            label, metrics = _estimate_emotion(face_landmarks.landmark)

            # Appliquer un seuil réglable sur le sourire (override de la décision)
            with self._lock:
                thr = self.threshold

            # On force "Sourire" uniquement si les métriques dépassent le seuil
            if label == "Sourire" and metrics.get("smile_width_ratio", 0.0) < thr:
                label = "Neutre"

            mp_drawing.draw_landmarks(
                image=image_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=image_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )

        cv2.putText(
            image_bgr,
            _overlay_label(label),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            _overlay_color(label),
            2,
            cv2.LINE_AA,
        )

        with self._lock:
            self.last_emotion = label
            self.last_metrics = metrics

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def get_last_emotion(self) -> str:
        with self._lock:
            return self.last_emotion

    def get_last_metrics(self) -> dict:
        with self._lock:
            return dict(self.last_metrics)

    def set_threshold(self, threshold: float):
        with self._lock:
            self.threshold = threshold

def run():
    st.title("Mission 2 : Face Mesh 😄")
    st.write(
        "**Objectif** : Utiliser la caméra en **temps réel** pour détecter les **points clés** "
        "du visage et afficher l'émotion estimée."
    )

    if not WEBRTC_AVAILABLE:
        st.warning(
            "Le mode live nécessite `streamlit-webrtc`. "
            "Installez-le puis relancez l'app."
        )
        return

    # États one-shot (ballons)
    if "mission2_balloons" not in st.session_state:
        st.session_state.mission2_balloons = False
    if "mission2_completed" not in st.session_state:
        st.session_state.mission2_completed = False

    st.subheader("Détection en direct")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_right:
        ctx = webrtc_streamer(
            key="mission2-live",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=FaceMeshVideoProcessor,
            async_processing=True,
        )
    with col_left:
        st.caption("Astuce : souriez pour valider la mission.")
        # Seuil réglable : smile_width_ratio
        threshold = st.slider(
            "Seuil de sourire (smile_width_ratio)",
            min_value=0.35,
            max_value=0.60,
            value=0.43,
            step=0.01,
            key="mission2_threshold",
        )
        st.caption(f"Seuil actuel : {threshold:.2f}")

        st.write("Émotion détectée :")

        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        feedback_placeholder = st.empty()

        @st.fragment(run_every=0.2)
        def _live_status():
            emotion = None
            metrics = {}

            if ctx.state.playing and ctx.video_processor:
                ctx.video_processor.set_threshold(st.session_state.mission2_threshold)
                emotion = ctx.video_processor.get_last_emotion()
                metrics = ctx.video_processor.get_last_metrics()
            
            status_placeholder.markdown(f"**{emotion or '—'}**")

            # Afficher 1-2 métriques utiles
            if metrics:
                metrics_placeholder.write(
                    f"smile_width_ratio: **{metrics.get('smile_width_ratio', 0.0):.3f}**  \n"
                    f"mouth_open_ratio: **{metrics.get('mouth_open_ratio', 0.0):.3f}**"
                )
            else:
                metrics_placeholder.write("smile_width_ratio: —  \nmouth_open_ratio: —")

            # Feedback
            if emotion == "Sourire":
                badges.unlock_badge("mission2")

                # Transition : on marque complété et on force un rerun complet 1 seule fois
                if not st.session_state.mission2_completed:
                    st.session_state.mission2_completed = True
                    st.rerun()

            if emotion == "Aucun visage détecté":
                feedback_placeholder.error("Aucun visage détecté. Placez-vous bien face à la caméra.")
            elif emotion == "Sourire":
                feedback_placeholder.success("Sourire détecté ✅")
            elif emotion:
                feedback_placeholder.info("Montrez un sourire pour réussir la mission.")
            else:
                feedback_placeholder.info("Démarrez la caméra pour analyser votre visage.")

        _live_status()

        # UI persistante + ballons one-shot (hors fragment)
        if st.session_state.mission2_completed:
            st.success("Mission 2 réussie ✅")
            st.markdown("<div style='font-size:40px'>😊</div>", unsafe_allow_html=True)

        if st.session_state.mission2_completed and not st.session_state.mission2_balloons:
            st.session_state.mission2_balloons = True
            st.balloons()

        if st.button("Accueil ➡️", key="mission2-next"):
            st.session_state.page = "introduction"
            st.rerun()