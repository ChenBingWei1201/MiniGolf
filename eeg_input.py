"""
EEG Input Abstraction Layer for MiniGolf (v2 - Hybrid Detection)
================================================================
Architecture Improvements:
- Blink: Detected via raw signal amplitude (~100ms delay, highly accurate)
- Focus/Relax: Classified using ML model
- Prediction frequency: Every 0.25 seconds

Usage:
    eeg = EEGInput(com_port="COM3")
    eeg.start()
    state, power, trigger_swing = eeg.update()
    eeg.close()
"""

import threading
import time
import traceback
import numpy as np
import joblib
from collections import deque
from enum import IntEnum
from typing import Tuple, Optional, List

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[EEG] Warning: pyserial not installed.")

try:
    from scipy.signal import butter, sosfiltfilt, welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[EEG] Warning: scipy not installed.")


class GameState(IntEnum):
    AIMING = 0
    CHARGING = 1
    FLYING = 2

class BCISignal(IntEnum):
    RELAX = 0
    FOCUS = 1
    BLINK = 2


# ============================================================
# BlinkDetector - Real-time amplitude-based blink detection
# ============================================================

class BlinkDetector:
    """
    Detect blinks using raw EEG signal amplitude (peak-to-peak).
    Blinks create large amplitude artifacts in EEG, which are faster and more
    accurate to detect directly than via the ML classifier.

    Mechanism:
    - Monitor peak-to-peak amplitude in a 0.3s sliding window
    - If it exceeds threshold -> detected as Blink
    - Cooldown period 1.5s -> prevent consecutive false triggers

    Auto-calibration:
    - First 3 seconds are used to calculate the baseline amplitude
    - Threshold = Baseline * Multiplier (default 4.0)
    """

    def __init__(self, sampling_rate: int = 512,
                 window_sec: float = 0.3,
                 cooldown_sec: float = 1.5,
                 calibration_sec: float = 3.0,
                 threshold_multiplier: float = 4.0,
                 min_threshold: float = 300,
                 max_threshold: float = 4000,
                 fallback_threshold: float = 800):
        self.sampling_rate = sampling_rate
        self.window_size = max(1, int(sampling_rate * window_sec))
        self.cooldown_samples = int(sampling_rate * cooldown_sec)
        self.calibration_samples = int(sampling_rate * calibration_sec)
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._buffer: deque = deque(maxlen=self.window_size)
        self._cooldown_counter: int = 0
        self._total_samples: int = 0

        # Calibration
        self._calibrating: bool = True
        self._calibration_ptps: List[float] = []
        self.threshold: float = fallback_threshold
        self._check_every: int = max(1, self.window_size // 3)

        # External toggle: pause detection (e.g., when CHARGING)
        self.enabled: bool = True

    def add_sample(self, raw_value: int) -> bool:
        """Add a raw EEG sample, returns True if blink detected."""
        self._buffer.append(raw_value)
        self._total_samples += 1

        # Cooldown: decrements even if enabled=False
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if len(self._buffer) < self.window_size:
            return False

        # Check periodically (~100ms) to save CPU
        if self._total_samples % self._check_every != 0:
            return False

        ptp = max(self._buffer) - min(self._buffer)

        # Calibration phase: must run to collect baseline ptps
        if self._calibrating:
            self._calibration_ptps.append(ptp)
            if self._total_samples >= self.calibration_samples:
                self._finish_calibration()
            return False

        # Skip detection if paused
        if not self.enabled:
            return False

        if self._cooldown_counter > 0:
            return False

        # Detection
        if ptp > self.threshold:
            self._cooldown_counter = self.cooldown_samples
            print(f"[BLINK] Detected! ptp={ptp:.0f} > threshold={self.threshold:.0f}")
            return True

        return False

    def _finish_calibration(self):
        """Complete calibration and set threshold"""
        if self._calibration_ptps:
            self.baseline_ptp = float(np.median(self._calibration_ptps))
            self.threshold = self.baseline_ptp * self.threshold_multiplier
            self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))
            print(f"[BLINK] Calibration done. baseline_ptp={self.baseline_ptp:.0f}, "
                  f"threshold={self.threshold:.0f}")
        else:
            self.baseline_ptp = 1000.0  # fallback
        self._calibrating = False

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating


# ============================================================
# EEGSerialReader - Background Thread
# ============================================================

class EEGSerialReader(threading.Thread):
    """
    Background thread for:
    1. Reading raw EEG data from Serial
    2. Real-time amplitude blink detection
    3. Rolling window ML predictions for Focus/Relax
    """

    SAMPLING_RATE = 512
    WINDOW_SIZE = 5 * SAMPLING_RATE  # Restore 5s window for 88.5% ML accuracy
    STEP_SEC = 0.25
    STEP_SIZE = int(SAMPLING_RATE * STEP_SEC)

    def __init__(self, com_port: str, baud_rate: int = 9600,
                 model_path: str = "enhanced_bci_classifier.pkl"):
        super().__init__(daemon=True)
        self.com_port = com_port
        self.baud_rate = baud_rate

        model_data = joblib.load(model_path)
        self._model = model_data['model']
        self._scaler = model_data['scaler']
        self._feature_selector = model_data['feature_selector']

        self._sos = butter(4, [1.0, 40.0], btype='bandpass',
                           fs=self.SAMPLING_RATE, output='sos')

        self._buffer: deque = deque()
        self._blink_detector = BlinkDetector(sampling_rate=self.SAMPLING_RATE)

        self._lock = threading.Lock()
        self._predictions: list = []
        self._running = False
        self._connected = False
        self._error_message: Optional[str] = None

    # --- Thread-safe getters ---

    def get_predictions_since(self, index: int) -> list:
        with self._lock:
            return list(self._predictions[index:])

    def get_prediction_count(self) -> int:
        with self._lock:
            return len(self._predictions)

    def get_latest_prediction(self) -> int:
        with self._lock:
            return self._predictions[-1] if self._predictions else BCISignal.RELAX

    def is_connected(self) -> bool:
        return self._connected

    def is_alive_and_running(self) -> bool:
        return self._running and self.is_alive()

    def get_error(self) -> Optional[str]:
        return self._error_message

    def get_buffer_fill(self) -> float:
        return min(len(self._buffer) / self.WINDOW_SIZE, 1.0)

    def is_calibrating(self) -> bool:
        return self._blink_detector.is_calibrating

    def is_ready(self) -> bool:
        """Returns True if EEG is connected and calibration is finished."""
        return self._connected and not self._blink_detector.is_calibrating

    def set_blink_detection(self, enabled: bool) -> None:
        """Enable/disable blink detection (e.g., disable during CHARGING to prevent false triggers)"""
        self._blink_detector.enabled = enabled

    def stop(self):
        self._running = False

    # --- Feature Extraction + Prediction ---

    def _extract_features(self, seg: np.ndarray) -> np.ndarray:
        """Extract 8 features (identical to train.py)"""
        seg_filtered = sosfiltfilt(self._sos, seg)
        var = np.var(seg_filtered)
        ptp = float(np.max(seg_filtered) - np.min(seg_filtered))
        rms = np.sqrt(np.mean(seg_filtered ** 2))
        freqs, psd = welch(seg_filtered, fs=self.SAMPLING_RATE,
                           nperseg=int(self.SAMPLING_RATE / 2))
        delta = np.sum(psd[(freqs >= 1) & (freqs < 4)])
        theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        total_power = delta + theta + alpha + beta
        if total_power == 0:
            total_power = 1e-10
        return np.array([
            var, ptp, rms,
            delta / total_power, theta / total_power,
            alpha / total_power, beta / total_power,
            alpha / beta if beta != 0 else 0.0
        ])

    def _predict_focus_relax(self, features: np.ndarray) -> Tuple[int, float, float]:
        """Returns (prediction, relax_p, blink_p).
        Model classes: Relax=low amplitude, Focus=mid, Blink=high amplitude.
        User's concentration also produces high amplitude (muscle artifacts),
        so blink_p > 0.5 effectively means high mental/physical arousal = FOCUS."""
        features = features.reshape(1, -1)
        scaled = self._scaler.transform(features)
        selected = self._feature_selector.transform(scaled)
        probas = self._model.predict_proba(selected)[0]
        relax_p = float(probas[0])
        blink_p = float(probas[2])

        # High amplitude (muscle artifacts during concentration) → Focus
        if blink_p > 0.5:
            return BCISignal.FOCUS, relax_p, blink_p

        # Low amplitude, calm EEG → Relax
        if relax_p >= 0.10:
            return BCISignal.RELAX, relax_p, blink_p

        return BCISignal.FOCUS, relax_p, blink_p

    # --- Thread Main Loop ---

    def run(self):
        self._running = True
        try:
            ser = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            self._connected = True
            print(f"[EEG] Connected to {self.com_port} (Baud: {self.baud_rate})")
            print(f"[EEG] Calibrating blink detector ({self._blink_detector.calibration_samples/self.SAMPLING_RATE:.0f}s)...")
        except serial.SerialException as e:
            self._error_message = f"Serial connection failed: {e}"
            print(f"[EEG] {self._error_message}")
            self._running = False
            return

        try:
            while self._running:
                packets_read = 0
                while ser.in_waiting >= 7 and packets_read < 50:
                    b1 = ser.read(1)
                    if b1 != b'\xaa':
                        continue
                    b2 = ser.read(1)
                    if b2 != b'\xaa':
                        continue
                    info = ser.read(3)
                    if len(info) < 3:
                        continue
                    if info[1] == 128 and info[2] == 2:
                        bs = ser.read(2)
                        if len(bs) == 2:
                            raw = int.from_bytes(bs, byteorder='big', signed=True)
                            self._buffer.append(raw)
                            packets_read += 1
                            if self._blink_detector.add_sample(raw):
                                with self._lock:
                                    self._predictions.append(BCISignal.BLINK)
                    else:
                        payload_len = info[0]
                        remaining = payload_len - 2
                        if remaining > 0:
                            ser.read(remaining)

                if len(self._buffer) >= self.WINDOW_SIZE:
                    try:
                        buf_list = list(self._buffer)
                        signals = np.array(buf_list[-self.WINDOW_SIZE:], dtype=np.float64)
                        features = self._extract_features(signals)
                        pred, relax_p, blink_p = self._predict_focus_relax(features)

                        # Dual-Window Override: Fast Relaxation Detection
                        # If the 5s ML model still outputs FOCUS (due to old artifacts)
                        # but the last 1 second of signal is extremely calm, force RELAX.
                        override_msg = ""
                        if pred == BCISignal.FOCUS and not self._blink_detector.is_calibrating:
                            short_window = int(1.0 * self.SAMPLING_RATE)
                            if len(self._buffer) >= short_window:
                                short_seg = list(self._buffer)[-short_window:]
                                short_ptp = max(short_seg) - min(short_seg)
                                baseline = getattr(self._blink_detector, 'baseline_ptp', 1000.0)
                                # If short-term amplitude is close to relaxed baseline (<= 1.5x)
                                if short_ptp <= baseline * 1.5:
                                    pred = BCISignal.RELAX
                                    override_msg = f" [Fast Override: ptp={short_ptp:.0f} <= {baseline*1.5:.0f}]"

                        with self._lock:
                            self._predictions.append(pred)

                        count = len(self._predictions)
                        label = ["Relax", "Focus"][pred]
                        print(f"[ML] #{count}: {label} (r={relax_p:.0%} b={blink_p:.0%}){override_msg}")

                        keep = self.WINDOW_SIZE - self.STEP_SIZE
                        while len(self._buffer) > keep:
                            self._buffer.popleft()

                    except Exception as e:
                        print(f"[EEG] Prediction error: {e}")
                        traceback.print_exc()
                        self._buffer.clear()

                time.sleep(0.001)

        except serial.SerialException as e:
            self._error_message = f"Serial error: {e}"
            print(f"[EEG] {self._error_message}")
        except Exception as e:
            self._error_message = f"Unexpected error: {e}"
            print(f"[EEG] {self._error_message}")
            traceback.print_exc()
        finally:
            self._running = False
            try:
                ser.close()
                self._connected = False
                print("[EEG] Serial disconnected.")
            except Exception:
                pass


# ============================================================
# BCIStateManager — 狀態機
# ============================================================

class BCIStateManager:
    """
    狀態轉換：
        AIMING  --[Blink]--> CHARGING    （振幅偵測，1 次即觸發）
        CHARGING --[Focus]--> power++
        CHARGING --[連續 N 次 Relax 且 power > min]--> FLYING（揮桿）
    """

    def __init__(self) -> None:
        self.current_state: GameState = GameState.AIMING
        self.power: float = 0.0

        self.blink_threshold: int = 1
        self.blink_counter: int = 0
        self._preds_since_last_blink: int = 0
        self._blink_timeout: int = 20

        # Need 5 consecutive Relax predictions (~1.25s) to swing.
        # This filters out brief noisy Relax readings without a voting buffer.
        self.relax_threshold: int = 5
        self.relax_counter: int = 0

        self.power_increment: float = 1.0
        self.max_power: float = 100.0
        self.min_swing_power: float = 10.0

    def update(self, raw_prediction: int, reader: 'EEGSerialReader') -> Tuple[GameState, float, bool]:
        trigger_swing = False

        if raw_prediction == BCISignal.BLINK:
            if self.current_state == GameState.AIMING:
                self.blink_counter += 1
                self._preds_since_last_blink = 0
                if self.blink_counter >= self.blink_threshold:
                    self.current_state = GameState.CHARGING
                    self.blink_counter = 0
                    reader.set_blink_detection(False)
                    print("[BCI] AIMING -> CHARGING")
            return self.current_state, self.power, trigger_swing

        if self.current_state == GameState.AIMING:
            self._preds_since_last_blink += 1
            if self.blink_counter > 0 and self._preds_since_last_blink >= self._blink_timeout:
                self.blink_counter = 0

        elif self.current_state == GameState.CHARGING:
            if raw_prediction == BCISignal.FOCUS:
                self.relax_counter = 0
                self.power = min(self.power + self.power_increment, self.max_power)
            elif raw_prediction == BCISignal.RELAX:
                self.relax_counter += 1
                if self.relax_counter >= self.relax_threshold and self.power > self.min_swing_power:
                    self.current_state = GameState.FLYING
                    trigger_swing = True
                    self.relax_counter = 0
                    reader.set_blink_detection(True)
                    print(f"[BCI] CHARGING -> FLYING (power={self.power:.0f})")

        return self.current_state, self.power, trigger_swing

    def reset_round(self, reader: Optional['EEGSerialReader'] = None) -> None:
        self.current_state = GameState.AIMING
        self.power = 0.0
        self.blink_counter = 0
        self.relax_counter = 0
        self._preds_since_last_blink = 0
        if reader is not None:
            reader.set_blink_detection(True)


# ============================================================
# EEGInput Main API
# ============================================================

class EEGInput:
    def __init__(self, com_port: str, baud_rate: int = 9600,
                 model_path: str = "enhanced_bci_classifier.pkl"):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed. Run: uv add pyserial")
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not installed. Run: uv add scipy")

        self.state_manager = BCIStateManager()
        self.reader = EEGSerialReader(com_port, baud_rate, model_path)
        self._last_pred_index = 0

        print(f"[EEG] EEGInput initialized. COM={com_port}")

    def start(self):
        """開始讀取 EEG"""
        self._last_pred_index = 0
        self.state_manager.reset_round()
        self.reader.start()
        print("[EEG] Reader started.")

    def sync(self):
        """同步到最新，丟棄所有累積的舊預測"""
        self._last_pred_index = self.reader.get_prediction_count()
        self.state_manager.reset_round()
        print(f"[EEG] Synced. Skipped {self._last_pred_index} old predictions.")

    def update(self) -> Tuple[GameState, float, bool]:
        """
        遊戲每幀呼叫。逐一處理所有新預測。

        Returns:
            (state, power_ratio 0~1, trigger_swing)
        """
        trigger_swing = False

        new_preds = self.reader.get_predictions_since(self._last_pred_index)
        if new_preds:
            self._last_pred_index += len(new_preds)
            for pred in new_preds:
                _, _, swung = self.state_manager.update(pred, self.reader)
                if swung:
                    trigger_swing = True

        state = self.state_manager.current_state
        power = self.state_manager.power
        power_ratio = power / self.state_manager.max_power
        return state, power_ratio, trigger_swing

    def reset_round(self) -> None:
        self.state_manager.reset_round(self.reader)

    def is_connected(self) -> bool:
        return self.reader.is_connected()

    def is_ready(self) -> bool:
        """Returns True if EEG is connected and calibration is finished."""
        return self.reader.is_ready()

    def close(self) -> None:
        self.reader.stop()
        self.reader.join(timeout=2)
        print("[EEG] Closed.")
