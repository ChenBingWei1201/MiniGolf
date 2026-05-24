"""
EEG Input Abstraction Layer for MiniGolf (v2 — Hybrid Detection)
================================================================
改進架構：
- 眨眼：用原始訊號振幅偵測（~100ms 延遲，非常準確）
- 專注/放鬆：用 ML 模型分類（忽略模型的 Blink 輸出）
- 預測頻率：每 0.25 秒一次（原本 0.5 秒）

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
# BlinkDetector — 振幅即時眨眼偵測
# ============================================================

class BlinkDetector:
    """
    用原始 EEG 訊號的振幅（peak-to-peak）偵測眨眼。
    眨眼會在 EEG 中產生非常明顯的大振幅人工偽影（artifact），
    比 ML 分類器更快、更準確。

    原理：
    - 監測 0.3 秒滑動窗口的 peak-to-peak 振幅
    - 超過門檻 → 判定為眨眼
    - 冷卻期 1 秒 → 防止連續誤觸發

    自動校準：
    - 前 3 秒為校準期，計算基線振幅
    - 門檻 = 基線 × 倍率（預設 3 倍）
    """

    def __init__(self, sampling_rate: int = 512,
                 window_sec: float = 0.3,
                 cooldown_sec: float = 1.0,
                 calibration_sec: float = 3.0,
                 threshold_multiplier: float = 3.0,
                 min_threshold: float = 300,
                 max_threshold: float = 2000,
                 fallback_threshold: float = 600):
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

        # 校準
        self._calibrating: bool = True
        self._calibration_ptps: List[float] = []
        self.threshold: float = fallback_threshold
        self._check_every: int = max(1, self.window_size // 3)

    def add_sample(self, raw_value: int) -> bool:
        """加入一個原始 EEG 樣本，回傳是否偵測到眨眼。"""
        self._buffer.append(raw_value)
        self._total_samples += 1

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False

        if len(self._buffer) < self.window_size:
            return False

        # 不要每個 sample 都算，每 ~100ms 算一次
        if self._total_samples % self._check_every != 0:
            return False

        ptp = max(self._buffer) - min(self._buffer)

        # 校準期：收集基線 ptp
        if self._calibrating:
            self._calibration_ptps.append(ptp)
            if self._total_samples >= self.calibration_samples:
                self._finish_calibration()
            return False

        # 正式偵測
        if ptp > self.threshold:
            self._cooldown_counter = self.cooldown_samples
            print(f"[BLINK] Detected! ptp={ptp:.0f} > threshold={self.threshold:.0f}")
            return True

        return False

    def _finish_calibration(self):
        """完成校準，設定門檻"""
        if self._calibration_ptps:
            baseline = float(np.median(self._calibration_ptps))
            self.threshold = baseline * self.threshold_multiplier
            self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))
            print(f"[BLINK] Calibration done. baseline_ptp={baseline:.0f}, "
                  f"threshold={self.threshold:.0f}")
        self._calibrating = False

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating


# ============================================================
# EEGSerialReader — 背景 Thread
# ============================================================

class EEGSerialReader(threading.Thread):
    """
    背景 thread：
    1. 讀取 BrainLink Serial 原始腦波
    2. 振幅偵測眨眼（即時）
    3. ML 模型分類專注/放鬆（每 0.25 秒）
    4. 合併結果到統一的預測串流
    """

    SAMPLING_RATE = 512
    WINDOW_SIZE = 5 * SAMPLING_RATE        # ML 模型窗口 5 秒
    STEP_SEC = 0.25                        # 每 0.25 秒預測一次
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
        self._last_ml_pred: int = BCISignal.RELAX

    # --- Thread-safe 讀取方法 ---

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

    def stop(self):
        self._running = False

    # --- 特徵提取 + 預測 ---

    def _extract_features(self, seg: np.ndarray) -> np.ndarray:
        """提取 8 個特徵（與 train.py 完全一致）"""
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

    def _predict_focus_relax(self, features: np.ndarray) -> Tuple[int, float]:
        """
        用 ML 模型預測，但只取 Focus/Relax（忽略 Blink 輸出）。
        使用 predict_proba 取得信心度。

        Returns:
            (prediction, confidence)
        """
        features = features.reshape(1, -1)
        scaled = self._scaler.transform(features)
        selected = self._feature_selector.transform(scaled)

        probas = self._model.predict_proba(selected)[0]
        # probas: [Relax概率, Focus概率, Blink概率]

        # 只看 Relax(0) vs Focus(1)，忽略 Blink(2)
        relax_p = probas[0]
        focus_p = probas[1]
        total = relax_p + focus_p

        if total > 0:
            relax_p /= total
            focus_p /= total

        if focus_p > relax_p:
            return BCISignal.FOCUS, focus_p
        else:
            return BCISignal.RELAX, relax_p

    # --- Thread 主迴圈 ---

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
                # --- 批次讀取 Serial 封包 ---
                packets_read = 0
                while ser.in_waiting >= 7 and packets_read < 200:
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

                            # --- 即時眨眼偵測 ---
                            if self._blink_detector.add_sample(raw):
                                with self._lock:
                                    self._predictions.append(BCISignal.BLINK)
                    else:
                        payload_len = info[0]
                        remaining = payload_len - 2
                        if remaining > 0:
                            ser.read(remaining)

                # --- ML 模型預測（Focus/Relax）---
                if len(self._buffer) >= self.WINDOW_SIZE:
                    try:
                        buf_list = list(self._buffer)
                        signals = np.array(buf_list[-self.WINDOW_SIZE:], dtype=np.float64)
                        features = self._extract_features(signals)
                        pred, confidence = self._predict_focus_relax(features)

                        # 信心度 > 55% 才採用，否則沿用上一次的預測
                        if confidence >= 0.55:
                            self._last_ml_pred = pred

                        with self._lock:
                            self._predictions.append(self._last_ml_pred)

                        count = len(self._predictions)
                        label = ["Relax", "Focus", "Blink"][self._last_ml_pred]
                        print(f"[ML] #{count}: {label} (conf={confidence:.0%})")

                        # 滑動窗口
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

        # 眨眼需 2 次（即使振幅偵測較準，仍保留防誤觸發）
        self.blink_threshold: int = 2
        self.blink_counter: int = 0
        self._preds_since_last_blink: int = 0  # 距離上次 Blink 經過幾個 ML 預測
        self._blink_timeout: int = 20  # 超過 20 個 ML 預測（~5 秒）沒 Blink 就重置計數

        # 放鬆需連續 3 次 ML 預測
        self.relax_threshold: int = 3
        self.relax_counter: int = 0

        # 蓄力參數
        self.power_increment: float = 2.0
        self.max_power: float = 100.0
        self.min_swing_power: float = 10.0

    def update(self, raw_prediction: int) -> Tuple[GameState, float, bool]:
        """
        Blink 事件和 ML 預測（Focus/Relax）分開處理，互不干擾：
        - Blink：只影響 blink_counter，不影響 relax_counter
        - Focus/Relax：只影響 power 和 relax_counter，不影響 blink_counter
        """
        trigger_swing = False

        # ========== Blink 事件處理 ==========
        if raw_prediction == BCISignal.BLINK:
            if self.current_state == GameState.AIMING:
                self.blink_counter += 1
                self._preds_since_last_blink = 0
                print(f"[BCI] Blink #{self.blink_counter}/{self.blink_threshold}")
                if self.blink_counter >= self.blink_threshold:
                    self.current_state = GameState.CHARGING
                    self.blink_counter = 0
                    print("[BCI] AIMING -> CHARGING (Blink confirmed!)")
            # Blink 不影響 CHARGING 狀態的 relax_counter
            return self.current_state, self.power, trigger_swing

        # ========== ML 預測處理（Focus / Relax）==========

        # 更新 blink 超時計數
        if self.current_state == GameState.AIMING:
            self._preds_since_last_blink += 1
            if self.blink_counter > 0 and self._preds_since_last_blink >= self._blink_timeout:
                print(f"[BCI] Blink counter timeout, reset ({self.blink_counter} -> 0)")
                self.blink_counter = 0

        # CHARGING 狀態：處理蓄力與揮桿
        elif self.current_state == GameState.CHARGING:
            if raw_prediction == BCISignal.FOCUS:
                self.power = min(self.power + self.power_increment,
                                 self.max_power)
                self.relax_counter = 0

            elif raw_prediction == BCISignal.RELAX:
                self.relax_counter += 1
                if (self.relax_counter >= self.relax_threshold
                        and self.power > self.min_swing_power):
                    self.current_state = GameState.FLYING
                    trigger_swing = True
                    self.relax_counter = 0
                    print(f"[BCI] CHARGING -> FLYING (Swing! power={self.power:.1f})")

        return self.current_state, self.power, trigger_swing

    def reset_round(self) -> None:
        self.current_state = GameState.AIMING
        self.power = 0.0
        self.blink_counter = 0
        self.relax_counter = 0
        self._preds_since_last_blink = 0


# ============================================================
# EEGInput — 統一介面
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
                _, _, swung = self.state_manager.update(pred)
                if swung:
                    trigger_swing = True

        state = self.state_manager.current_state
        power = self.state_manager.power
        power_ratio = power / self.state_manager.max_power
        return state, power_ratio, trigger_swing

    def reset_round(self) -> None:
        self.state_manager.reset_round()

    def is_connected(self) -> bool:
        return self.reader.is_connected()

    def close(self) -> None:
        self.reader.stop()
        self.reader.join(timeout=2)
        print("[EEG] Closed.")
