# =============================================================================
# EMER - RASPBERRY PI VISUAL VIDEO APPLICATION
# =============================================================================
#
# VIDEO FILE
#     ↓
# YuNet Face Detection
#     ↓
# EmotiEffNet-B0
#     ↓
# Per-face Emotion Recognition
#     ↓
# Classroom-level Emotion
#
# NO LIVE CAMERA
# NO MICROPHONE
# NO SOUNDDEVICE
# NO PORTAUDIO
# =============================================================================

import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)


# =============================================================================
# PATHS
# =============================================================================

EMER_ROOT = "/home/pi/Desktop/EMER"

MODELS_DIR = os.path.join(
    EMER_ROOT,
    "models"
)

VISUAL_CHECKPOINT = os.path.join(
    MODELS_DIR,
    "best_emotieffnet_b0_ferplus.pth"
)

YUNET_MODEL = os.path.join(
    MODELS_DIR,
    "face_detector",
    "face_detection_yunet_2026may.onnx"
)

# Optional default video directory
VIDEO_DIR = os.path.join(
    EMER_ROOT,
    "data"
)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# EMOTIONS
# =============================================================================

EMOTIONS = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


# =============================================================================
# TRANSFORM
# =============================================================================

VISUAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# MODEL
# =============================================================================

class EmotiEffNetB0(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):

        return self.model(x)


# =============================================================================
# LOAD VISUAL MODEL
# =============================================================================

def load_visual_model():

    print("=" * 70)
    print("EMER - LOADING VISUAL MODEL")
    print("=" * 70)

    print(
        "Checkpoint:"
    )

    print(
        VISUAL_CHECKPOINT
    )

    if not os.path.exists(
        VISUAL_CHECKPOINT
    ):

        raise FileNotFoundError(
            "Visual checkpoint not found:\n\n"
            + VISUAL_CHECKPOINT
        )

    model = EmotiEffNetB0(
        num_classes=8
    )

    checkpoint = torch.load(
        VISUAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    # ---------------------------------------------------------
    # Detect checkpoint format
    # ---------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif (
            "model" in checkpoint
            and isinstance(
                checkpoint["model"],
                dict
            )
        ):

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported visual checkpoint format."
        )

    # ---------------------------------------------------------
    # Clean prefixes
    # ---------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                len("model.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    # ---------------------------------------------------------
    # Load weights
    # ---------------------------------------------------------

    missing, unexpected = (
        model.model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing:

        print(
            "\nWARNING: Missing keys:"
        )

        for key in missing[:20]:

            print(
                " ",
                key
            )

    if unexpected:

        print(
            "\nWARNING: Unexpected keys:"
        )

        for key in unexpected[:20]:

            print(
                " ",
                key
            )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "\nVisual model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    return model


# =============================================================================
# LOAD YUNET
# =============================================================================

def load_yunet():

    print("=" * 70)
    print("EMER - LOADING YUNET")
    print("=" * 70)

    print(
        "YuNet:"
    )

    print(
        YUNET_MODEL
    )

    if not os.path.exists(
        YUNET_MODEL
    ):

        raise FileNotFoundError(
            "YuNet detector not found:\n\n"
            + YUNET_MODEL
        )

    if hasattr(
        cv2,
        "FaceDetectorYN"
    ):

        detector = (
            cv2.FaceDetectorYN.create(
                YUNET_MODEL,
                "",
                (320, 320),
                0.6,
                0.3,
                5000,
            )
        )

    elif hasattr(
        cv2,
        "FaceDetectorYN_create"
    ):

        detector = (
            cv2.FaceDetectorYN_create(
                YUNET_MODEL,
                "",
                (320, 320),
                0.6,
                0.3,
                5000,
            )
        )

    else:

        raise RuntimeError(
            "Your OpenCV installation does not "
            "support YuNet FaceDetectorYN.\n\n"
            f"OpenCV version: {cv2.__version__}"
        )

    print(
        "YuNet loaded successfully."
    )

    return detector


# =============================================================================
# VIDEO WORKER
# =============================================================================

class VideoWorker(QThread):

    frame_ready = Signal(QImage)

    status_signal = Signal(str)

    face_signal = Signal(int)

    classroom_signal = Signal(str)

    fps_signal = Signal(float)

    progress_signal = Signal(int)

    video_info_signal = Signal(str)

    error_signal = Signal(str)

    finished_signal = Signal()


    def __init__(
        self,
        video_path,
        visual_model,
        detector,
    ):

        super().__init__()

        self.video_path = (
            video_path
        )

        self.visual_model = (
            visual_model
        )

        self.detector = (
            detector
        )

        self.running = False

        self.target_fps = 8.0

        self.frame_interval = (
            1.0 / self.target_fps
        )


    # -------------------------------------------------------------------------
    # STOP
    # -------------------------------------------------------------------------

    def stop(self):

        self.running = False


    # -------------------------------------------------------------------------
    # DETECT FACES
    # -------------------------------------------------------------------------

    def detect_faces(
        self,
        frame
    ):

        height, width = (
            frame.shape[:2]
        )

        self.detector.setInputSize(
            (width, height)
        )

        result = self.detector.detect(
            frame
        )

        if result is None:

            return []

        _, faces = result

        if faces is None:

            return []

        return faces


    # -------------------------------------------------------------------------
    # PREDICT FACE
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict_face(
        self,
        face_crop
    ):

        if (
            face_crop is None
            or face_crop.size == 0
        ):

            return None, 0.0

        try:

            rgb = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                rgb
            )

            tensor = (
                VISUAL_TRANSFORM(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = (
                self.visual_model(
                    tensor
                )
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            emotion_index = int(
                prediction.item()
            )

            confidence_value = float(
                confidence.item()
            )

            emotion = EMOTIONS[
                emotion_index
            ]

            return (
                emotion,
                confidence_value
            )

        except Exception as exc:

            print(
                "Prediction error:",
                exc
            )

            return None, 0.0


    # -------------------------------------------------------------------------
    # DRAW FACE
    # -------------------------------------------------------------------------

    def draw_face(
        self,
        frame,
        x,
        y,
        w,
        h,
        emotion,
        confidence,
        person_number,
    ):

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x = max(
            0,
            min(
                x,
                frame_width - 1
            )
        )

        y = max(
            0,
            min(
                y,
                frame_height - 1
            )
        )

        x2 = max(
            x + 1,
            min(
                x + w,
                frame_width - 1
            )
        )

        y2 = max(
            y + 1,
            min(
                y + h,
                frame_height - 1
            )
        )

        # ---------------------------------------------------------
        # Face box
        # ---------------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        # ---------------------------------------------------------
        # Label
        # ---------------------------------------------------------

        if emotion is None:

            label = (
                f"Person {person_number}: Unknown"
            )

        else:

            label = (
                f"Person {person_number}: "
                f"{emotion} "
                f"{confidence * 100:.1f}%"
            )

        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        scale = 0.55

        thickness = 1

        text_size, baseline = (
            cv2.getTextSize(
                label,
                font,
                scale,
                thickness,
            )
        )

        text_width = (
            text_size[0]
        )

        text_height = (
            text_size[1]
        )

        label_y = max(
            text_height + 8,
            y
        )

        # Background

        cv2.rectangle(
            frame,
            (
                x,
                label_y
                - text_height
                - 8,
            ),
            (
                x
                + text_width
                + 8,
                label_y
                + baseline,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                x + 4,
                label_y - 3,
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    def run(self):

        self.running = True

        capture = None

        try:

            # =============================================================
            # OPEN VIDEO
            # =============================================================

            self.status_signal.emit(
                "Opening video..."
            )

            capture = cv2.VideoCapture(
                self.video_path
            )

            if not capture.isOpened():

                raise RuntimeError(
                    "Could not open video:\n\n"
                    + self.video_path
                )

            # =============================================================
            # VIDEO INFORMATION
            # =============================================================

            total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            source_fps = (
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if source_fps <= 0:

                source_fps = 25.0

            width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            duration = (
                total_frames
                / source_fps
                if total_frames > 0
                else 0
            )

            info = (
                f"{width} × {height} | "
                f"{source_fps:.1f} FPS | "
                f"{duration:.1f} sec"
            )

            self.video_info_signal.emit(
                info
            )

            print(
                "\nVideo information:"
            )

            print(
                info
            )

            # =============================================================
            # PROCESS VIDEO
            # =============================================================

            self.status_signal.emit(
                "Visual analysis running..."
            )

            processed_frames = 0

            fps_counter = 0

            fps_start = time.time()

            while self.running:

                loop_start = time.time()

                ret, frame = (
                    capture.read()
                )

                if not ret:

                    break

                processed_frames += 1

                # =========================================================
                # FACE DETECTION
                # =========================================================

                faces = (
                    self.detect_faces(
                        frame
                    )
                )

                face_count = 0

                emotion_results = []

                # =========================================================
                # PROCESS EACH FACE
                # =========================================================

                for face in faces:

                    if len(face) < 15:

                        continue

                    x = int(
                        face[0]
                    )

                    y = int(
                        face[1]
                    )

                    w = int(
                        face[2]
                    )

                    h = int(
                        face[3]
                    )

                    detector_confidence = float(
                        face[14]
                    )

                    if (
                        detector_confidence
                        < 0.50
                    ):

                        continue

                    # -----------------------------------------------------
                    # Face margin
                    # -----------------------------------------------------

                    margin_x = int(
                        w * 0.10
                    )

                    margin_y = int(
                        h * 0.10
                    )

                    x1 = max(
                        0,
                        x - margin_x
                    )

                    y1 = max(
                        0,
                        y - margin_y
                    )

                    x2 = min(
                        frame.shape[1],
                        x + w + margin_x
                    )

                    y2 = min(
                        frame.shape[0],
                        y + h + margin_y
                    )

                    face_crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    # -----------------------------------------------------
                    # Emotion prediction
                    # -----------------------------------------------------

                    emotion, confidence = (
                        self.predict_face(
                            face_crop
                        )
                    )

                    face_count += 1

                    emotion_results.append(
                        (
                            emotion,
                            confidence
                        )
                    )

                    # -----------------------------------------------------
                    # Draw
                    # -----------------------------------------------------

                    self.draw_face(
                        frame,
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        emotion,
                        confidence,
                        face_count,
                    )

                # =========================================================
                # CLASSROOM EMOTION
                # =========================================================

                valid_emotions = [
                    emotion
                    for emotion, confidence
                    in emotion_results
                    if emotion is not None
                ]

                if valid_emotions:

                    counts = Counter(
                        valid_emotions
                    )

                    classroom_emotion = (
                        counts.most_common(
                            1
                        )[0][0]
                    )

                    classroom_text = (
                        classroom_emotion.title()
                    )

                else:

                    classroom_text = (
                        "No faces detected"
                    )

                # =========================================================
                # OVERLAY
                # =========================================================

                overlay_height = 105

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (390, overlay_height),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Faces: {face_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Classroom: {classroom_text}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # =========================================================
                # PROGRESS
                # =========================================================

                if total_frames > 0:

                    progress = int(
                        (
                            processed_frames
                            / total_frames
                        )
                        * 100
                    )

                    progress = max(
                        0,
                        min(
                            100,
                            progress
                        )
                    )

                    self.progress_signal.emit(
                        progress
                    )

                # =========================================================
                # FPS
                # =========================================================

                fps_counter += 1

                now = time.time()

                elapsed = (
                    now
                    - fps_start
                )

                if elapsed >= 1.0:

                    current_fps = (
                        fps_counter
                        / elapsed
                    )

                    self.fps_signal.emit(
                        current_fps
                    )

                    fps_counter = 0

                    fps_start = now

                # =========================================================
                # SEND FRAME TO GUI
                # =========================================================

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_height, frame_width = (
                    rgb.shape[:2]
                )

                bytes_per_line = (
                    3 * frame_width
                )

                image = QImage(
                    rgb.data,
                    frame_width,
                    frame_height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )

                self.frame_ready.emit(
                    image.copy()
                )

                self.face_signal.emit(
                    face_count
                )

                self.classroom_signal.emit(
                    classroom_text
                )

                # =========================================================
                # TARGET PROCESSING RATE
                # =========================================================

                processing_time = (
                    time.time()
                    - loop_start
                )

                remaining = (
                    self.frame_interval
                    - processing_time
                )

                if remaining > 0:

                    time.sleep(
                        remaining
                    )

            # =============================================================
            # VIDEO FINISHED
            # =============================================================

            if self.running:

                self.status_signal.emit(
                    "Video processing completed"
                )

                self.progress_signal.emit(
                    100
                )

        except Exception as exc:

            message = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "VIDEO PROCESSING ERROR"
            )

            print(
                "=" * 70
            )

            print(
                message
            )

            self.error_signal.emit(
                message
            )

        finally:

            self.running = False

            if capture is not None:

                capture.release()

            self.finished_signal.emit()


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "EMER - Visual Emotion Recognition"
        )

        self.resize(
            1150,
            800
        )

        self.visual_model = None

        self.detector = None

        self.worker = None

        self.video_path = None

        self.setup_ui()


    # =========================================================================
    # UI
    # =========================================================================

    def setup_ui(self):

        layout = QVBoxLayout()

        # =====================================================================
        # TITLE
        # =====================================================================

        title = QLabel(
            "EMER - Smart Classroom Emotion Recognition"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title_font = QFont()

        title_font.setPointSize(
            18
        )

        title_font.setBold(
            True
        )

        title.setFont(
            title_font
        )

        layout.addWidget(
            title
        )

        subtitle = QLabel(
            "Visual Branch: YuNet + EmotiEffNet-B0"
        )

        subtitle.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            subtitle
        )

        # =====================================================================
        # VIDEO DISPLAY
        # =====================================================================

        self.video_label = QLabel(
            "Select a video to begin"
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setMinimumSize(
            720,
            450
        )

        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: black;
                color: white;
                border: 2px solid #555;
            }
            """
        )

        layout.addWidget(
            self.video_label,
            stretch=1
        )

        # =====================================================================
        # VIDEO SELECTION
        # =====================================================================

        video_group = QGroupBox(
            "Video Input"
        )

        video_layout = QHBoxLayout()

        self.video_path_label = QLabel(
            "No video selected"
        )

        self.video_path_label.setWordWrap(
            True
        )

        self.select_button = QPushButton(
            "SELECT VIDEO"
        )

        self.select_button.setMinimumHeight(
            45
        )

        video_layout.addWidget(
            self.video_path_label,
            stretch=1
        )

        video_layout.addWidget(
            self.select_button
        )

        video_group.setLayout(
            video_layout
        )

        layout.addWidget(
            video_group
        )

        # =====================================================================
        # INFORMATION
        # =====================================================================

        info_group = QGroupBox(
            "Visual Analysis"
        )

        info_layout = QGridLayout()

        # Faces

        info_layout.addWidget(
            QLabel("Detected Faces:"),
            0,
            0
        )

        self.faces_value = QLabel(
            "0"
        )

        self.faces_value.setFont(
            QFont(
                "",
                12,
                QFont.Bold
            )
        )

        info_layout.addWidget(
            self.faces_value,
            0,
            1
        )

        # Classroom

        info_layout.addWidget(
            QLabel("Classroom Emotion:"),
            0,
            2
        )

        self.classroom_value = QLabel(
            "--"
        )

        self.classroom_value.setFont(
            QFont(
                "",
                12,
                QFont.Bold
            )
        )

        info_layout.addWidget(
            self.classroom_value,
            0,
            3
        )

        # FPS

        info_layout.addWidget(
            QLabel("Processing FPS:"),
            1,
            0
        )

        self.fps_value = QLabel(
            "0.0"
        )

        info_layout.addWidget(
            self.fps_value,
            1,
            1
        )

        # Video information

        info_layout.addWidget(
            QLabel("Video:"),
            1,
            2
        )

        self.video_info_value = QLabel(
            "--"
        )

        info_layout.addWidget(
            self.video_info_value,
            1,
            3
        )

        # Device

        info_layout.addWidget(
            QLabel("Device:"),
            2,
            0
        )

        self.device_value = QLabel(
            str(DEVICE)
        )

        info_layout.addWidget(
            self.device_value,
            2,
            1
        )

        # Progress

        info_layout.addWidget(
            QLabel("Progress:"),
            2,
            2
        )

        self.progress_value = QLabel(
            "0%"
        )

        info_layout.addWidget(
            self.progress_value,
            2,
            3
        )

        info_group.setLayout(
            info_layout
        )

        layout.addWidget(
            info_group
        )

        # =====================================================================
        # STATUS
        # =====================================================================

        self.status_label = QLabel(
            "Status: Ready"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            self.status_label
        )

        # =====================================================================
        # CONTROL BUTTONS
        # =====================================================================

        button_layout = QHBoxLayout()

        self.start_button = QPushButton(
            "START PROCESSING"
        )

        self.stop_button = QPushButton(
            "STOP"
        )

        self.start_button.setMinimumHeight(
            45
        )

        self.stop_button.setMinimumHeight(
            45
        )

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            False
        )

        button_layout.addWidget(
            self.start_button
        )

        button_layout.addWidget(
            self.stop_button
        )

        layout.addLayout(
            button_layout
        )

        self.setLayout(
            layout
        )

        # =====================================================================
        # CONNECTIONS
        # =====================================================================

        self.select_button.clicked.connect(
            self.select_video
        )

        self.start_button.clicked.connect(
            self.start_processing
        )

        self.stop_button.clicked.connect(
            self.stop_processing
        )


    # =========================================================================
    # SELECT VIDEO
    # =========================================================================

    def select_video(self):

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            VIDEO_DIR,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV)"
        )

        if not video_path:

            return

        self.video_path = (
            video_path
        )

        self.video_path_label.setText(
            video_path
        )

        self.status_label.setText(
            "Status: Video selected"
        )

        self.start_button.setEnabled(
            True
        )

        self.progress_value.setText(
            "0%"
        )

        self.video_info_value.setText(
            "--"
        )

        self.faces_value.setText(
            "0"
        )

        self.classroom_value.setText(
            "--"
        )

        self.video_label.setText(
            "Video ready - click START PROCESSING"
        )


    # =========================================================================
    # START PROCESSING
    # =========================================================================

    def start_processing(self):

        if not self.video_path:

            QMessageBox.warning(
                self,
                "No Video",
                "Please select a video first."
            )

            return

        if self.worker is not None:

            if self.worker.isRunning():

                return

        try:

            self.start_button.setEnabled(
                False
            )

            self.select_button.setEnabled(
                False
            )

            self.stop_button.setEnabled(
                True
            )

            self.status_label.setText(
                "Status: Loading visual model..."
            )

            QApplication.processEvents()

            # -------------------------------------------------------------
            # Load model
            # -------------------------------------------------------------

            if self.visual_model is None:

                self.visual_model = (
                    load_visual_model()
                )

            # -------------------------------------------------------------
            # Load detector
            # -------------------------------------------------------------

            if self.detector is None:

                self.detector = (
                    load_yunet()
                )

            # -------------------------------------------------------------
            # Worker
            # -------------------------------------------------------------

            self.worker = VideoWorker(
                self.video_path,
                self.visual_model,
                self.detector,
            )

            self.worker.frame_ready.connect(
                self.update_frame
            )

            self.worker.status_signal.connect(
                self.update_status
            )

            self.worker.face_signal.connect(
                self.update_faces
            )

            self.worker.classroom_signal.connect(
                self.update_classroom
            )

            self.worker.fps_signal.connect(
                self.update_fps
            )

            self.worker.progress_signal.connect(
                self.update_progress
            )

            self.worker.video_info_signal.connect(
                self.update_video_info
            )

            self.worker.error_signal.connect(
                self.handle_error
            )

            self.worker.finished_signal.connect(
                self.processing_finished
            )

            self.worker.start()

        except Exception as exc:

            self.handle_error(
                f"{type(exc).__name__}: {exc}"
            )

            self.start_button.setEnabled(
                True
            )

            self.select_button.setEnabled(
                True
            )

            self.stop_button.setEnabled(
                False
            )


    # =========================================================================
    # STOP
    # =========================================================================

    def stop_processing(self):

        if self.worker is not None:

            if self.worker.isRunning():

                self.status_label.setText(
                    "Status: Stopping..."
                )

                self.worker.stop()

                self.stop_button.setEnabled(
                    False
                )


    # =========================================================================
    # FRAME
    # =========================================================================

    def update_frame(
        self,
        image
    ):

        pixmap = QPixmap.fromImage(
            image
        )

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(
            scaled
        )


    # =========================================================================
    # STATUS
    # =========================================================================

    def update_status(
        self,
        message
    ):

        self.status_label.setText(
            "Status: " + message
        )


    # =========================================================================
    # FACES
    # =========================================================================

    def update_faces(
        self,
        count
    ):

        self.faces_value.setText(
            str(count)
        )


    # =========================================================================
    # CLASSROOM
    # =========================================================================

    def update_classroom(
        self,
        emotion
    ):

        self.classroom_value.setText(
            emotion
        )


    # =========================================================================
    # FPS
    # =========================================================================

    def update_fps(
        self,
        fps
    ):

        self.fps_value.setText(
            f"{fps:.1f}"
        )


    # =========================================================================
    # PROGRESS
    # =========================================================================

    def update_progress(
        self,
        progress
    ):

        self.progress_value.setText(
            f"{progress}%"
        )


    # =========================================================================
    # VIDEO INFO
    # =========================================================================

    def update_video_info(
        self,
        info
    ):

        self.video_info_value.setText(
            info
        )


    # =========================================================================
    # ERROR
    # =========================================================================

    def handle_error(
        self,
        message
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            "EMER ERROR"
        )

        print(
            "=" * 70
        )

        print(
            message
        )

        self.status_label.setText(
            "Status: Error"
        )

        QMessageBox.critical(
            self,
            "EMER Error",
            message,
        )


    # =========================================================================
    # FINISHED
    # =========================================================================

    def processing_finished(
        self
    ):

        self.start_button.setEnabled(
            True
        )

        self.select_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Status: Processing finished"
        )

        self.worker = None


    # =========================================================================
    # CLOSE
    # =========================================================================

    def closeEvent(
        self,
        event
    ):

        if self.worker is not None:

            if self.worker.isRunning():

                self.worker.stop()

                self.worker.wait(
                    5000
                )

        event.accept()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )