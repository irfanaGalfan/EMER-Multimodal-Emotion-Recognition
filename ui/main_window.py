import os
import time
import threading
import subprocess
import shutil
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import sounddevice as sd
from PIL import Image
from torchvision import transforms
from transformers import Wav2Vec2Model

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QProgressBar,
    QMessageBox,
)


# ============================================================
# PATHS
# ============================================================

EMER_ROOT = r"C:\Users\User\Desktop\EMER"

VISUAL_CHECKPOINT = os.path.join(
    EMER_ROOT,
    "models",
    "best_emotieffnet_b0_ferplus.pth",
)

AUDIO_CHECKPOINT = os.path.join(
    EMER_ROOT,
    "models",
    "best_wav2vec2_ravdess_92acc.pth",
)

YUNET_MODEL = os.path.join(
    EMER_ROOT,
    "models",
    "face_detector",
    "face_detection_yunet_2026may.onnx",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EMER")
print("=" * 60)
print("Device:", DEVICE)
print("=" * 60)


# ============================================================
# CLASS NAMES
# ============================================================

VISUAL_CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

AUDIO_CLASSES = [
    "neutral",
    "calm",
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]


# ============================================================
# VISUAL TRANSFORM
# ============================================================

VISUAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# VISUAL MODEL
# ============================================================

def load_visual_model():

    print("Loading EmotiEffNet-B0...")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=8,
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 8)
    )

    checkpoint = torch.load(
        VISUAL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("EmotiEffNet loaded.")

    return model


# ============================================================
# AUDIO MODEL
# ============================================================

class Wav2Vec2EmotionModel(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.projector = nn.Linear(
            768,
            256,
        )

        self.classifier = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, input_values):

        outputs = self.wav2vec2(
            input_values=input_values
        )

        hidden = outputs.last_hidden_state

        features = hidden.mean(dim=1)

        projected = self.projector(
            features
        )

        logits = self.classifier(
            projected
        )

        return logits


def load_audio_model():

    print("Loading Wav2Vec2...")

    model = Wav2Vec2EmotionModel(
        num_classes=8
    )

    checkpoint = torch.load(
        AUDIO_CHECKPOINT,
        map_location=DEVICE,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Wav2Vec2 loaded.")

    return model


# ============================================================
# YUNET
# ============================================================

def create_yunet():

    print("Loading YuNet...")

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )

    print("YuNet loaded.")

    return detector


# ============================================================
# VIDEO / CAMERA WORKER
# ============================================================

class VideoWorker(QThread):

    frame_ready = Signal(QImage)

    statistics_ready = Signal(
        int,
        float,
        float,
        object,
        object,
    )

    error = Signal(str)

    finished_signal = Signal()

    def __init__(
        self,
        source_mode="camera",
        video_path=None,
        parent=None,
    ):

        super().__init__(parent)

        self.source_mode = source_mode
        self.video_path = video_path

        self.running = True

        self.model = None
        self.detector = None
        self.cap = None

        self.target_inference_fps = 8.0

        self.total_frames = 0

        self.last_fps_time = time.perf_counter()
        self.fps_counter = 0
        self.current_fps = 0.0

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

    # ========================================================
    # MODEL LOAD
    # ========================================================

    def load_models(self):

        self.model = load_visual_model()
        self.detector = create_yunet()

    # ========================================================
    # OPEN SOURCE
    # ========================================================

    def open_source(self):

        if self.source_mode == "camera":

            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():

                raise RuntimeError(
                    "Could not open the camera."
                )

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720,
            )

            return

        if not self.video_path:

            raise RuntimeError(
                "No video file selected."
            )

        self.cap = cv2.VideoCapture(
            self.video_path
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Could not open video:\n{self.video_path}"
            )

        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        print()
        print("=" * 60)
        print("VIDEO FILE")
        print("=" * 60)
        print("Path:", self.video_path)
        print("Resolution:", width, "x", height)
        print("FPS:", fps)
        print("Frames:", frame_count)
        print(
            "Duration:",
            round(duration, 2),
            "seconds",
        )
        print("=" * 60)
        print()

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        original_h, original_w = frame.shape[:2]

        detection_width = 960

        scale = (
            detection_width /
            float(original_w)
        )

        if scale < 1.0:

            detection_height = int(
                original_h * scale
            )

            detection_frame = cv2.resize(
                frame,
                (
                    detection_width,
                    detection_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:

            detection_frame = frame.copy()

            scale = 1.0

            detection_height = original_h
            detection_width = original_w

        self.detector.setInputSize(
            (
                detection_width,
                detection_height,
            )
        )

        _, faces = self.detector.detect(
            detection_frame
        )

        emotion_counts = Counter()

        inference_times = []

        face_count = 0

        probability_sum = np.zeros(
            len(VISUAL_CLASSES),
            dtype=np.float64,
        )

        if faces is None:

            return (
                frame,
                0,
                0.0,
                emotion_counts,
                probability_sum,
            )

        for face in faces:

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            detection_confidence = float(
                face[-1]
            )

            if w <= 0 or h <= 0:
                continue

            x_original = int(
                x / scale
            )

            y_original = int(
                y / scale
            )

            w_original = int(
                w / scale
            )

            h_original = int(
                h / scale
            )

            x_original = max(
                0,
                min(
                    x_original,
                    original_w - 1,
                ),
            )

            y_original = max(
                0,
                min(
                    y_original,
                    original_h - 1,
                ),
            )

            x2 = min(
                original_w,
                x_original + w_original,
            )

            y2 = min(
                original_h,
                y_original + h_original,
            )

            if x2 <= x_original:
                continue

            if y2 <= y_original:
                continue

            face_crop = frame[
                y_original:y2,
                x_original:x2,
            ]

            if face_crop.size == 0:
                continue

            start_inference = time.perf_counter()

            try:

                rgb_face = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB,
                )

                pil_face = Image.fromarray(
                    rgb_face
                )

                tensor = VISUAL_TRANSFORM(
                    pil_face
                )

                tensor = tensor.unsqueeze(0)

                tensor = tensor.to(
                    DEVICE
                )

                with torch.no_grad():

                    logits = self.model(
                        tensor
                    )

                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    confidence_tensor, prediction = (
                        torch.max(
                            probabilities,
                            dim=1,
                        )
                    )

                    probability_sum += (
                        probabilities[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                emotion_index = (
                    prediction.item()
                )

                emotion_confidence = (
                    confidence_tensor.item()
                )

                emotion = VISUAL_CLASSES[
                    emotion_index
                ]

            except Exception as exc:

                print(
                    "Emotion inference error:",
                    exc,
                )

                continue

            inference_ms = (
                time.perf_counter()
                - start_inference
            ) * 1000

            inference_times.append(
                inference_ms
            )

            face_count += 1

            emotion_counts[
                emotion
            ] += 1

            cv2.rectangle(
                frame,
                (
                    x_original,
                    y_original,
                ),
                (
                    x2,
                    y2,
                ),
                (0, 255, 0),
                3,
            )

            label = (
                f"{emotion.upper()} "
                f"{emotion_confidence * 100:.1f}%"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (
                text_width,
                text_height,
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            label_x = x_original

            label_y = (
                y_original
                - 10
            )

            if label_y < text_height:

                label_y = (
                    y_original
                    + text_height
                    + 10
                )

            cv2.rectangle(
                frame,
                (
                    label_x,
                    label_y - text_height - baseline,
                ),
                (
                    label_x + text_width + 8,
                    label_y + 4,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (
                    label_x + 4,
                    label_y,
                ),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

        average_inference = (
            float(
                np.mean(
                    inference_times
                )
            )
            if inference_times
            else 0.0
        )

        if face_count > 0:

            average_probabilities = (
                probability_sum
                / face_count
            )

        else:

            average_probabilities = np.zeros(
                len(VISUAL_CLASSES),
                dtype=np.float64,
            )

        return (
            frame,
            face_count,
            average_inference,
            emotion_counts,
            average_probabilities,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            print(
                "\nStarting video worker..."
            )

            self.load_models()

            self.open_source()

            last_processing_time = 0.0

            while self.running:

                ret, frame = self.cap.read()

                if not ret:

                    if self.source_mode == "video":

                        print(
                            "Video reached the end."
                        )

                    break

                self.total_frames += 1

                self.fps_counter += 1

                now = time.perf_counter()

                elapsed = (
                    now
                    - self.last_fps_time
                )

                if elapsed >= 1.0:

                    self.current_fps = (
                        self.fps_counter
                        / elapsed
                    )

                    self.fps_counter = 0

                    self.last_fps_time = now

                inference_interval = (
                    1.0
                    / self.target_inference_fps
                )

                if (
                    now
                    - last_processing_time
                    >= inference_interval
                ):

                    (
                        processed_frame,
                        face_count,
                        inference_ms,
                        emotion_counts,
                        average_probabilities,
                    ) = self.process_frame(
                        frame
                    )

                    last_processing_time = now

                    self.statistics_ready.emit(
                        face_count,
                        self.current_fps,
                        inference_ms,
                        dict(emotion_counts),
                        average_probabilities,
                    )

                    display_frame = (
                        processed_frame
                    )

                else:

                    display_frame = frame

                rgb = cv2.cvtColor(
                    display_frame,
                    cv2.COLOR_BGR2RGB,
                )

                h, w, ch = rgb.shape

                bytes_per_line = (
                    ch * w
                )

                qimage = QImage(
                    rgb.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format_RGB888,
                ).copy()

                self.frame_ready.emit(
                    qimage
                )

                if self.source_mode == "video":

                    video_fps = (
                        self.cap.get(
                            cv2.CAP_PROP_FPS
                        )
                    )

                    if (
                        video_fps <= 0
                        or video_fps > 120
                    ):

                        video_fps = 25.0

                    sleep_time = (
                        1.0
                        / video_fps
                    )

                    time.sleep(
                        max(
                            0.001,
                            sleep_time * 0.5,
                        )
                    )

                else:

                    time.sleep(
                        0.001
                    )

        except Exception as exc:

            import traceback

            traceback.print_exc()

            self.error.emit(
                str(exc)
            )

        finally:

            if self.cap is not None:

                self.cap.release()

            print(
                "Video worker stopped."
            )

            self.finished_signal.emit()


# ============================================================
# AUDIO WORKER
# ============================================================

class AudioWorker(QThread):

    audio_ready = Signal(
        str,
        float,
    )

    error = Signal(str)

    finished_signal = Signal()

    def __init__(
        self,
        audio_source="microphone",
        video_path=None,
        parent=None,
    ):

        super().__init__(parent)

        self.audio_source = audio_source
        self.video_path = video_path

        self.running = True

        self.model = None

        self.sample_rate = 16000

        self.audio_seconds = 3

        self.audio_samples = (
            self.sample_rate
            * self.audio_seconds
        )

        self.audio_buffer = np.zeros(
            self.audio_samples,
            dtype=np.float32,
        )

        self.audio_lock = threading.Lock()

        self.video_audio = None

        self.video_audio_position = 0

        self.smoothing_window = 5

        self.probability_history = []

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

    # ========================================================
    # FFMPEG
    # ========================================================

    def find_ffmpeg(self):

        ffmpeg_path = shutil.which(
            "ffmpeg"
        )

        if ffmpeg_path is not None:

            return ffmpeg_path

        fallback_path = (
            r"C:\Users\User\AppData\Local\Microsoft\WinGet"
            r"\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-9.0.1-full_build"
            r"\bin\ffmpeg.exe"
        )

        if os.path.isfile(
            fallback_path
        ):

            print(
                "FFmpeg found using fallback path:"
            )

            print(
                fallback_path
            )

            return fallback_path

        raise RuntimeError(
            "FFmpeg was not found.\n\n"
            "Please install FFmpeg and make sure "
            "it is available in your PATH."
        )

    # ========================================================
    # EXTRACT VIDEO AUDIO
    # ========================================================

    def extract_video_audio(self):

        if not self.video_path:

            raise RuntimeError(
                "No video file was selected."
            )

        ffmpeg_path = (
            self.find_ffmpeg()
        )

        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

        print(
            "Extracting audio from video..."
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_data, stderr_data = (
            process.communicate()
        )

        if process.returncode != 0:

            error_message = (
                stderr_data.decode(
                    errors="ignore"
                ).strip()
            )

            if (
                "Output file does not contain any stream"
                in error_message
            ):

                print(
                    "No audio stream found in video."
                )

                self.video_audio = None

                return

            raise RuntimeError(
                "Could not extract audio from video:\n\n"
                + error_message
            )

        if not stdout_data:

            print(
                "Video contains no usable audio."
            )

            self.video_audio = None

            return

        audio = np.frombuffer(
            stdout_data,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

        self.video_audio = audio

        self.video_audio_position = 0

        duration = (
            len(audio)
            / self.sample_rate
        )

        print(
            f"Video audio extracted: "
            f"{duration:.2f} seconds"
        )

    # ========================================================
    # GET VIDEO AUDIO WINDOW
    # ========================================================

    def get_video_audio_window(self):

        if self.video_audio is None:

            return None

        start = (
            self.video_audio_position
        )

        end = (
            start
            + self.audio_samples
        )

        if start >= len(
            self.video_audio
        ):

            return None

        window = self.video_audio[
            start:min(
                end,
                len(self.video_audio),
            )
        ]

        if len(window) < self.audio_samples:

            window = np.pad(
                window,
                (
                    0,
                    self.audio_samples
                    - len(window),
                ),
            )

        # Advance by one second.
        self.video_audio_position += (
            self.sample_rate
        )

        return window

    # ========================================================
    # RESET SMOOTHING
    # ========================================================

    def reset_smoothing(self):

        self.probability_history = []

    # ========================================================
    # AUDIO PREDICTION
    # ========================================================

    def predict_audio(self, audio):

        if audio is None:

            return

        # ----------------------------------------------------
        # MAKE SURE AUDIO IS A VALID FLOAT ARRAY
        # ----------------------------------------------------

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:

            self.reset_smoothing()

            self.audio_ready.emit(
                "neutral",
                0.0,
            )

            return

        # ----------------------------------------------------
        # REMOVE NaN / INF VALUES
        # ----------------------------------------------------

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ----------------------------------------------------
        # AUDIO LEVEL MEASUREMENT
        #
        # Peak alone is not enough.
        # RMS gives a better indication of whether there
        # is actual audio energy in the window.
        # ----------------------------------------------------

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

        rms_db = (
            20.0
            * np.log10(
                max(
                    rms,
                    1e-8,
                )
            )
        )

        print(
            f"Audio level: "
            f"RMS={rms:.6f}, "
            f"dBFS={rms_db:.2f}, "
            f"Peak={peak:.6f}"
        )

        # ----------------------------------------------------
        # SILENCE DETECTION
        #
        # Do NOT send silent audio to Wav2Vec2.
        #
        # The neural network will ALWAYS choose one of its
        # emotion classes, even if the input contains no
        # meaningful speech.
        # ----------------------------------------------------

        if (
            rms_db < -45.0
            and peak < 0.02
        ):

            print(
                "Audio window is SILENT."
            )

            self.reset_smoothing()

            self.audio_ready.emit(
                "neutral",
                0.0,
            )

            return

        # ----------------------------------------------------
        # VERY QUIET AUDIO
        #
        # This prevents extremely weak background noise,
        # microphone noise, etc. from being interpreted
        # as an emotion.
        # ----------------------------------------------------

        if (
            rms_db < -40.0
            and peak < 0.04
        ):

            print(
                "Audio window is too quiet "
                "for emotion classification."
            )

            self.reset_smoothing()

            self.audio_ready.emit(
                "neutral",
                0.0,
            )

            return

        # ----------------------------------------------------
        # WAV2VEC2 INFERENCE
        # ----------------------------------------------------

        waveform = torch.from_numpy(
            audio
        ).float()

        waveform = waveform.unsqueeze(
            0
        )

        waveform = waveform.to(
            DEVICE
        )

        with torch.no_grad():

            logits = self.model(
                waveform
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        current_probabilities = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

        self.probability_history.append(
            current_probabilities
        )

        if (
            len(
                self.probability_history
            )
            > self.smoothing_window
        ):

            self.probability_history.pop(
                0
            )

        smoothed_probabilities = (
            np.mean(
                self.probability_history,
                axis=0,
            )
        )

        predicted_index = int(
            np.argmax(
                smoothed_probabilities
            )
        )

        confidence = float(
            smoothed_probabilities[
                predicted_index
            ]
        )

        emotion = AUDIO_CLASSES[
            predicted_index
        ]

        print(
            "Audio:",
            emotion,
            f"{confidence * 100:.2f}%",
            "| smoothing:",
            f"{len(self.probability_history)}/"
            f"{self.smoothing_window}",
        )

        self.audio_ready.emit(
            emotion,
            confidence,
        )

    # ========================================================
    # MICROPHONE CALLBACK
    # ========================================================

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if not self.running:

            return

        if status:

            print(
                "Microphone:",
                status,
            )

        samples = (
            indata[:, 0]
            .astype(np.float32)
        )

        with self.audio_lock:

            self.audio_buffer = np.roll(
                self.audio_buffer,
                -len(samples),
            )

            self.audio_buffer[
                -len(samples):
            ] = samples

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            print(
                "Starting audio worker..."
            )

            self.model = (
                load_audio_model()
            )

            self.reset_smoothing()

            # =================================================
            # VIDEO AUDIO
            # =================================================

            if self.audio_source == "video":

                print(
                    "Using audio track from video."
                )

                self.extract_video_audio()

                if self.video_audio is None:

                    print(
                        "Audio analysis disabled because "
                        "the video has no audio track."
                    )

                    while self.running:

                        time.sleep(
                            0.5
                        )

                else:

                    while self.running:

                        audio_window = (
                            self.get_video_audio_window()
                        )

                        if audio_window is None:

                            break

                        self.predict_audio(
                            audio_window
                        )

                        time.sleep(
                            1.0
                        )

            # =================================================
            # MICROPHONE AUDIO
            # =================================================

            else:

                print(
                    "Using live microphone."
                )

                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self.audio_callback,
                    blocksize=1600,
                )

                stream.start()

                try:

                    while self.running:

                        time.sleep(
                            1.0
                        )

                        with self.audio_lock:

                            audio_window = (
                                self.audio_buffer.copy()
                            )

                        self.predict_audio(
                            audio_window
                        )

                finally:

                    stream.stop()
                    stream.close()

        except Exception as exc:

            import traceback

            traceback.print_exc()

            self.error.emit(
                str(exc)
            )

        finally:

            print(
                "Audio worker stopped."
            )

            self.finished_signal.emit()


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "EMER – Smart Classroom"
        )

        self.resize(
            1500,
            900,
        )

        self.video_worker = None
        self.audio_worker = None

        self.video_file = None

        self.latest_emotion_counts = {}

        # ====================================================
        # VIDEO AUDIO PLAYBACK
        # ====================================================

        self.audio_output = QAudioOutput(
            self
        )

        self.audio_output.setVolume(
            1.0
        )

        self.media_player = QMediaPlayer(
            self
        )

        self.media_player.setAudioOutput(
            self.audio_output
        )

        self.media_player.errorOccurred.connect(
            self.media_player_error
        )

        self.setup_ui()

    # ========================================================
    # UI
    # ========================================================

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        title = QLabel(
            "EMER – SMART CLASSROOM"
        )

        title.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                color: #ffffff;
                padding: 8px;
            }
            """
        )

        subtitle = QLabel(
            "Multimodal Emotion Recognition"
        )

        subtitle.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                color: #aab4c3;
                padding-bottom: 10px;
            }
            """
        )

        main_layout.addWidget(
            title
        )

        main_layout.addWidget(
            subtitle
        )

        # ----------------------------------------------------
        # INPUT GROUP
        # ----------------------------------------------------

        input_group = QGroupBox(
            "Input Sources"
        )

        input_layout = QGridLayout()

        # ----------------------------------------------------
        # VIDEO SOURCE
        # ----------------------------------------------------

        input_layout.addWidget(
            QLabel("Video Source:"),
            0,
            0,
        )

        self.video_source_combo = (
            QComboBox()
        )

        self.video_source_combo.addItems(
            [
                "Live Camera",
                "Video File",
            ]
        )

        self.video_source_combo.currentTextChanged.connect(
            self.video_source_changed
        )

        input_layout.addWidget(
            self.video_source_combo,
            0,
            1,
        )

        self.browse_video_button = (
            QPushButton(
                "Browse Video"
            )
        )

        self.browse_video_button.clicked.connect(
            self.browse_video
        )

        input_layout.addWidget(
            self.browse_video_button,
            0,
            2,
        )

        self.video_file_label = QLabel(
            "No video selected"
        )

        self.video_file_label.setStyleSheet(
            "color: #8f9aaa;"
        )

        input_layout.addWidget(
            self.video_file_label,
            0,
            3,
        )

        # ----------------------------------------------------
        # AUDIO SOURCE
        # ----------------------------------------------------

        input_layout.addWidget(
            QLabel("Audio Source:"),
            1,
            0,
        )

        self.audio_source_combo = (
            QComboBox()
        )

        self.audio_source_combo.addItems(
            [
                "Live Microphone",
                "None",
            ]
        )

        input_layout.addWidget(
            self.audio_source_combo,
            1,
            1,
        )

        input_group.setLayout(
            input_layout
        )

        main_layout.addWidget(
            input_group
        )

        # ----------------------------------------------------
        # CONTENT AREA
        # ----------------------------------------------------

        content_layout = QHBoxLayout()

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        video_group = QGroupBox(
            "Classroom Video"
        )

        video_layout = QVBoxLayout()

        self.video_label = QLabel(
            "Select an input and press START"
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setMinimumSize(
            900,
            500,
        )

        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #10151c;
                color: #687384;
                border: 1px solid #2b3440;
            }
            """
        )

        video_layout.addWidget(
            self.video_label
        )

        video_group.setLayout(
            video_layout
        )

        content_layout.addWidget(
            video_group,
            3,
        )

        # ----------------------------------------------------
        # RIGHT PANEL
        # ----------------------------------------------------

        right_layout = QVBoxLayout()

        # ----------------------------------------------------
        # MULTIMODAL ANALYSIS
        # ----------------------------------------------------

        analysis_group = QGroupBox(
            "Multimodal Analysis"
        )

        analysis_layout = QGridLayout()

        analysis_layout.addWidget(
            QLabel("Audio Emotion"),
            0,
            0,
        )

        self.audio_emotion_label = QLabel(
            "--"
        )

        analysis_layout.addWidget(
            self.audio_emotion_label,
            0,
            1,
        )

        analysis_layout.addWidget(
            QLabel("Active Speaker"),
            1,
            0,
        )

        self.active_speaker_label = QLabel(
            "--"
        )

        analysis_layout.addWidget(
            self.active_speaker_label,
            1,
            1,
        )

        analysis_layout.addWidget(
            QLabel("Fused Emotion"),
            2,
            0,
        )

        self.fused_emotion_label = QLabel(
            "--"
        )

        analysis_layout.addWidget(
            self.fused_emotion_label,
            2,
            1,
        )

        analysis_group.setLayout(
            analysis_layout
        )

        right_layout.addWidget(
            analysis_group
        )

        # ----------------------------------------------------
        # CLASSROOM OVERVIEW
        # ----------------------------------------------------

        overview_group = QGroupBox(
            "Classroom Overview"
        )

        overview_layout = QVBoxLayout()

        self.overall_classroom_emotion = QLabel(
            "Overall Classroom Emotion: --"
        )

        self.overall_classroom_emotion.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                padding: 6px;
            }
            """
        )

        overview_layout.addWidget(
            self.overall_classroom_emotion
        )

        self.overall_classroom_confidence = QLabel(
            "Confidence: --"
        )

        self.overall_classroom_confidence.setStyleSheet(
            """
            QLabel {
                color: #aab4c3;
                font-size: 13px;
                padding-bottom: 6px;
            }
            """
        )

        overview_layout.addWidget(
            self.overall_classroom_confidence
        )

        self.overview_bars = {}

        for emotion in VISUAL_CLASSES:

            if emotion == "contempt":
                continue

            row = QHBoxLayout()

            label = QLabel(
                emotion.capitalize()
            )

            label.setMinimumWidth(
                85
            )

            bar = QProgressBar()

            bar.setRange(
                0,
                100,
            )

            bar.setValue(
                0
            )

            bar.setFormat(
                "%p%"
            )

            row.addWidget(
                label
            )

            row.addWidget(
                bar
            )

            overview_layout.addLayout(
                row
            )

            self.overview_bars[
                emotion
            ] = bar

        overview_group.setLayout(
            overview_layout
        )

        right_layout.addWidget(
            overview_group
        )

        # ----------------------------------------------------
        # SYSTEM METRICS
        # ----------------------------------------------------

        metrics_group = QGroupBox(
            "System Metrics"
        )

        metrics_layout = QGridLayout()

        metrics_layout.addWidget(
            QLabel("Students Detected"),
            0,
            0,
        )

        self.students_label = QLabel(
            "0"
        )

        metrics_layout.addWidget(
            self.students_label,
            0,
            1,
        )

        metrics_layout.addWidget(
            QLabel("FPS"),
            1,
            0,
        )

        self.fps_label = QLabel(
            "0.0"
        )

        metrics_layout.addWidget(
            self.fps_label,
            1,
            1,
        )

        metrics_layout.addWidget(
            QLabel("Inference"),
            2,
            0,
        )

        self.inference_label = QLabel(
            "0 ms"
        )

        metrics_layout.addWidget(
            self.inference_label,
            2,
            1,
        )

        metrics_group.setLayout(
            metrics_layout
        )

        right_layout.addWidget(
            metrics_group
        )

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        status_group = QGroupBox(
            "System Status"
        )

        status_layout = QGridLayout()

        status_layout.addWidget(
            QLabel("Camera"),
            0,
            0,
        )

        self.camera_status = QLabel(
            "● Disconnected"
        )

        status_layout.addWidget(
            self.camera_status,
            0,
            1,
        )

        status_layout.addWidget(
            QLabel("Audio"),
            1,
            0,
        )

        self.audio_status = QLabel(
            "● Disconnected"
        )

        status_layout.addWidget(
            self.audio_status,
            1,
            1,
        )

        status_layout.addWidget(
            QLabel("Face Detection"),
            2,
            0,
        )

        self.face_status = QLabel(
            "● Inactive"
        )

        status_layout.addWidget(
            self.face_status,
            2,
            1,
        )

        status_layout.addWidget(
            QLabel("Fusion"),
            3,
            0,
        )

        self.fusion_status = QLabel(
            "● Inactive"
        )

        status_layout.addWidget(
            self.fusion_status,
            3,
            1,
        )

        status_group.setLayout(
            status_layout
        )

        right_layout.addWidget(
            status_group
        )

        right_layout.addStretch()

        content_layout.addLayout(
            right_layout,
            1,
        )

        main_layout.addLayout(
            content_layout
        )

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        button_layout = QHBoxLayout()

        self.start_button = QPushButton(
            "START"
        )

        self.start_button.clicked.connect(
            self.start_system
        )

        button_layout.addWidget(
            self.start_button
        )

        self.stop_button = QPushButton(
            "STOP"
        )

        self.stop_button.clicked.connect(
            self.stop_system
        )

        self.stop_button.setEnabled(
            False
        )

        button_layout.addWidget(
            self.stop_button
        )

        main_layout.addLayout(
            button_layout
        )

        # ----------------------------------------------------
        # STYLE
        # ----------------------------------------------------

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0b1016;
            }

            QGroupBox {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #293341;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QLabel {
                color: #d6dde7;
                font-size: 13px;
            }

            QComboBox {
                background-color: #171e27;
                color: white;
                border: 1px solid #35404d;
                padding: 7px;
                border-radius: 5px;
            }

            QPushButton {
                background-color: #1d6fe8;
                color: white;
                border: none;
                padding: 10px 22px;
                border-radius: 5px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #2b7df0;
            }

            QPushButton:disabled {
                background-color: #303943;
                color: #777f89;
            }

            QProgressBar {
                background-color: #171e27;
                border: none;
                border-radius: 4px;
                height: 18px;
                text-align: center;
                color: white;
            }

            QProgressBar::chunk {
                background-color: #1d8cf8;
                border-radius: 4px;
            }
            """
        )

    # ========================================================
    # VIDEO SOURCE CHANGED
    # ========================================================

    def video_source_changed(
        self,
        source,
    ):

        if source == "Video File":

            self.audio_source_combo.setEnabled(
                False
            )

            self.audio_source_combo.setCurrentText(
                "None"
            )

            self.browse_video_button.setEnabled(
                True
            )

        else:

            self.audio_source_combo.setEnabled(
                True
            )

            if (
                self.audio_source_combo.currentText()
                == "None"
            ):

                self.audio_source_combo.setCurrentText(
                    "Live Microphone"
                )

            self.browse_video_button.setEnabled(
                False
            )

    # ========================================================
    # BROWSE VIDEO
    # ========================================================

    def browse_video(self):

        file_path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Select Classroom Video",
                "",
                (
                    "Video Files "
                    "(*.mp4 *.avi *.mov *.mkv *.wmv);;"
                    "MP4 Files (*.mp4);;"
                    "All Files (*)"
                ),
            )
        )

        if not file_path:
            return

        self.video_file = file_path

        self.video_file_label.setText(
            os.path.basename(
                file_path
            )
        )

        self.video_source_combo.setCurrentText(
            "Video File"
        )

        print(
            "Selected video:",
            file_path,
        )

    # ========================================================
    # VIDEO AUDIO PLAYBACK
    # ========================================================

    def start_video_audio(
        self,
        video_path,
    ):

        if not video_path:
            return

        print()
        print("=" * 60)
        print("STARTING VIDEO AUDIO PLAYBACK")
        print("=" * 60)

        print(
            "Audio source:",
            video_path,
        )

        url = QUrl.fromLocalFile(
            os.path.abspath(
                video_path
            )
        )

        self.media_player.setSource(
            url
        )

        self.media_player.play()

        print(
            "Video audio playback started."
        )

        print("=" * 60)
        print()

    # ========================================================
    # STOP VIDEO AUDIO
    # ========================================================

    def stop_video_audio(self):

        if self.media_player is not None:

            self.media_player.stop()

            self.media_player.setSource(
                QUrl()
            )

            print(
                "Video audio playback stopped."
            )

    # ========================================================
    # MEDIA PLAYER ERROR
    # ========================================================

    def media_player_error(
        self,
        error,
        error_string,
    ):

        if error_string:

            print()
            print(
                "MEDIA PLAYER ERROR:"
            )

            print(
                error_string
            )

    # ========================================================
    # START SYSTEM
    # ========================================================

    def start_system(self):

        if self.video_worker is not None:

            QMessageBox.warning(
                self,
                "Already running",
                "The system is already running.",
            )

            return

        video_mode = (
            self.video_source_combo.currentText()
        )

        audio_mode = (
            self.audio_source_combo.currentText()
        )

        # ----------------------------------------------------
        # VIDEO SOURCE
        # ----------------------------------------------------

        if video_mode == "Live Camera":

            source_mode = "camera"
            video_path = None

        else:

            source_mode = "video"

            if not self.video_file:

                QMessageBox.warning(
                    self,
                    "No video selected",
                    "Please click Browse Video and select an MP4 file.",
                )

                return

            video_path = self.video_file

        # ----------------------------------------------------
        # CREATE VIDEO WORKER
        # ----------------------------------------------------

        self.video_worker = VideoWorker(
            source_mode=source_mode,
            video_path=video_path,
        )

        self.video_worker.frame_ready.connect(
            self.update_video
        )

        self.video_worker.statistics_ready.connect(
            self.update_statistics
        )

        self.video_worker.error.connect(
            self.show_worker_error
        )

        self.video_worker.finished_signal.connect(
            self.video_finished
        )

        # ----------------------------------------------------
        # START VIDEO
        # ----------------------------------------------------

        self.video_worker.start()

        self.camera_status.setText(
            "● Active"
        )

        self.face_status.setText(
            "● Starting..."
        )

        # ----------------------------------------------------
        # VIDEO AUDIO PLAYBACK
        # ----------------------------------------------------

        if source_mode == "video":

            self.start_video_audio(
                video_path
            )

        # ----------------------------------------------------
        # AUDIO ANALYSIS
        # ----------------------------------------------------

        if source_mode == "video":

            # For a video file, use the embedded
            # audio track automatically.

            self.audio_worker = AudioWorker(
                audio_source="video",
                video_path=video_path,
            )

        elif (
            audio_mode
            == "Live Microphone"
        ):

            self.audio_worker = AudioWorker(
                audio_source="microphone",
                video_path=None,
            )

        else:

            self.audio_worker = None

        if self.audio_worker is not None:

            self.audio_worker.audio_ready.connect(
                self.update_audio
            )

            self.audio_worker.error.connect(
                self.show_worker_error
            )

            self.audio_worker.finished_signal.connect(
                self.audio_finished
            )

            self.audio_worker.start()

            self.audio_status.setText(
                "● Starting..."
            )

        else:

            self.audio_status.setText(
                "● Disabled"
            )

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            True
        )

    # ========================================================
    # STOP SYSTEM
    # ========================================================

    def stop_system(self):

        # Stop speaker playback
        self.stop_video_audio()

        if self.video_worker is not None:

            self.video_worker.stop()

        if self.audio_worker is not None:

            self.audio_worker.stop()

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

    # ========================================================
    # VIDEO FRAME
    # ========================================================

    def update_video(
        self,
        image,
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

    # ========================================================
    # STATISTICS
    # ========================================================

    def update_statistics(
        self,
        face_count,
        fps,
        inference_ms,
        emotion_counts,
        average_probabilities,
    ):

        self.students_label.setText(
            str(face_count)
        )

        self.fps_label.setText(
            f"{fps:.1f}"
        )

        self.inference_label.setText(
            f"{inference_ms:.1f} ms"
        )

        self.latest_emotion_counts = (
            emotion_counts
        )

        total_faces = sum(
            emotion_counts.values()
        )

        if total_faces > 0:

            for emotion, bar in (
                self.overview_bars.items()
            ):

                count = emotion_counts.get(
                    emotion,
                    0,
                )

                percentage = (
                    count
                    / total_faces
                    * 100
                )

                bar.setValue(
                    int(
                        round(
                            percentage
                        )
                    )
                )

        else:

            for bar in (
                self.overview_bars.values()
            ):

                bar.setValue(
                    0
                )

        # ----------------------------------------------------
        # OVERALL CLASSROOM EMOTION
        # ----------------------------------------------------

        if face_count > 0:

            probabilities = np.asarray(
                average_probabilities,
                dtype=np.float64,
            )

            if (
                probabilities.size
                == len(VISUAL_CLASSES)
            ):

                overall_index = int(
                    np.argmax(
                        probabilities
                    )
                )

                overall_emotion = (
                    VISUAL_CLASSES[
                        overall_index
                    ]
                )

                overall_confidence = float(
                    probabilities[
                        overall_index
                    ]
                )

                self.overall_classroom_emotion.setText(
                    "Overall Classroom Emotion: "
                    f"{overall_emotion.capitalize()}"
                )

                self.overall_classroom_confidence.setText(
                    "Confidence: "
                    f"{overall_confidence * 100:.1f}%"
                )

            else:

                self.overall_classroom_emotion.setText(
                    "Overall Classroom Emotion: --"
                )

                self.overall_classroom_confidence.setText(
                    "Confidence: --"
                )

        else:

            self.overall_classroom_emotion.setText(
                "Overall Classroom Emotion: --"
            )

            self.overall_classroom_confidence.setText(
                "Confidence: --"
            )

        # ----------------------------------------------------
        # FACE STATUS
        # ----------------------------------------------------

        if face_count > 0:

            self.face_status.setText(
                "● Active"
            )

        else:

            self.face_status.setText(
                "● No faces detected"
            )

    # ========================================================
    # AUDIO UPDATE
    # ========================================================

    def update_audio(
        self,
        emotion,
        confidence,
    ):

        # ----------------------------------------------------
        # NO SPEECH / SILENCE
        #
        # AudioWorker sends confidence = 0.0 when the
        # audio window is silent or too quiet.
        #
        # IMPORTANT:
        # We deliberately do NOT display "Neutral".
        # Silence is not an emotion.
        # ----------------------------------------------------

        if confidence <= 0.0:

            self.audio_emotion_label.setText(
                "No speech"
            )

            self.audio_status.setText(
                "● Silent"
            )

            self.active_speaker_label.setText(
                "No audio activity"
            )

            self.fused_emotion_label.setText(
                "--"
            )

            self.fusion_status.setText(
                "● Waiting for audio"
            )

            return

        # ----------------------------------------------------
        # NORMAL AUDIO EMOTION
        # ----------------------------------------------------

        self.audio_emotion_label.setText(
            f"{emotion.capitalize()} "
            f"({confidence * 100:.1f}%)"
        )

        self.audio_status.setText(
            "● Active"
        )

        # Audio is classroom/audio-stream level.
        # It is not assigned to individual faces.

        self.active_speaker_label.setText(
            "Audio stream"
        )

        self.fused_emotion_label.setText(
            "--"
        )

        self.fusion_status.setText(
            "● Audio available"
        )

    # ========================================================
    # VIDEO FINISHED
    # ========================================================

    def video_finished(self):

        # Stop speaker playback when
        # the video reaches its end.

        self.stop_video_audio()

        self.camera_status.setText(
            "● Stopped"
        )

        self.face_status.setText(
            "● Inactive"
        )

        if self.video_worker is not None:

            self.video_worker.deleteLater()

            self.video_worker = None

        if self.audio_worker is None:

            self.start_button.setEnabled(
                True
            )

            self.stop_button.setEnabled(
                False
            )

    # ========================================================
    # AUDIO FINISHED
    # ========================================================

    def audio_finished(self):

        self.audio_status.setText(
            "● Stopped"
        )

        if self.audio_worker is not None:

            self.audio_worker.deleteLater()

            self.audio_worker = None

        if self.video_worker is None:

            self.start_button.setEnabled(
                True
            )

            self.stop_button.setEnabled(
                False
            )

    # ========================================================
    # ERROR
    # ========================================================

    def show_worker_error(
        self,
        message,
    ):

        print()
        print(
            "WORKER ERROR:"
        )

        print(
            message
        )

        QMessageBox.critical(
            self,
            "EMER Error",
            message,
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event,
    ):

        # Stop speaker playback
        self.stop_video_audio()

        if self.video_worker is not None:

            self.video_worker.stop()

            self.video_worker.wait(
                3000
            )

        if self.audio_worker is not None:

            self.audio_worker.stop()

            self.audio_worker.wait(
                3000
            )

        event.accept()