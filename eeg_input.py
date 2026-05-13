"""
EEG Input Abstraction Layer for MiniGolf
=========================================
整合 BrainLink EEG 穿戴裝置到遊戲中。

包含三個核心元件：
1. EEGSerialReader - 背景 thread，讀取 BrainLink Serial 資料並做即時預測
2. BCIStateManager - 狀態機，將連續預測轉為遊戲動作（瞄準→蓄力→揮桿）
3. EEGInput        - 統一介面，供 main.py 呼叫

Usage:
    eeg = EEGInput(com_port="COM3")
    eeg.start()   # 開始讀取 EEG
    # In game loop:
    state, power, trigger_swing = eeg.update()
    # On exit:
    eeg.close()
"""

import threading
import time
import traceback
import numpy as np
import joblib
from collections import deque
from enum import IntEnum
from typing import Tuple, Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[EEG] Warning: pyserial not installed. EEG mode unavailable.")

try:
    from scipy.signal import butter, sosfiltfilt, welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[EEG] Warning: scipy not installed. EEG mode unavailable.")


# ============================================================
# 遊戲狀態列舉
# ============================================================

class GameState(IntEnum):
    """遊戲操控狀態"""
    AIMING = 0      # 瞄準中（箭頭旋轉）
    CHARGING = 1    # 蓄力中（方向已鎖定，力道增長）
    FLYING = 2      # 球已擊出（等待靜止）


class BCISignal(IntEnum):
    """BCI 分類器輸出的腦波狀態"""
    RELAX = 0       # 放鬆
    FOCUS = 1       # 專注
    BLINK = 2       # 眨眼


# ============================================================
# EEGSerialReader - 背景 Thread 讀取 Serial + 模型預測
# ============================================================

class EEGSerialReader(threading.Thread):
    """
    背景 thread：連接 BrainLink EEG 裝置，即時讀取原始腦波，
    維護 5 秒滑動窗口，每 0.5 秒做一次特徵提取 + 模型預測。

    預測結果透過 get_predictions_since() 供遊戲主迴圈讀取。
    """

    # BrainLink 協議常數
    SAMPLING_RATE = 512
    WINDOW_SIZE = 5 * SAMPLING_RATE   # 5 秒 = 2560 samples
    STEP_SEC = 0.5
    STEP_SIZE = int(SAMPLING_RATE * STEP_SEC)  # 256 samples

    def __init__(self, com_port: str, baud_rate: int = 9600,
                 model_path: str = "enhanced_bci_classifier.pkl"):
        super().__init__(daemon=True)
        self.com_port = com_port
        self.baud_rate = baud_rate

        # 載入訓練好的模型
        model_data = joblib.load(model_path)
        self._model = model_data['model']
        self._scaler = model_data['scaler']
        self._feature_selector = model_data['feature_selector']

        # 預先計算帶通濾波器係數（避免每次預測時重複計算）
        self._sos = butter(4, [1.0, 40.0], btype='bandpass',
                           fs=self.SAMPLING_RATE, output='sos')

        # Serial buffer（不設 maxlen，手動管理大小）
        self._buffer = deque()

        # Thread-safe 共享狀態：用 list 儲存所有預測結果
        self._lock = threading.Lock()
        self._predictions: list = []  # 所有預測結果的歷史記錄
        self._running = False
        self._connected = False
        self._error_message: Optional[str] = None

    def get_predictions_since(self, index: int) -> list:
        """取得 index 之後的所有新預測結果"""
        with self._lock:
            return list(self._predictions[index:])

    def get_prediction_count(self) -> int:
        """取得總共做了幾次預測"""
        with self._lock:
            return len(self._predictions)

    def get_latest_prediction(self) -> int:
        """取得最新一次的 BCI 預測結果 (0/1/2)"""
        with self._lock:
            if self._predictions:
                return self._predictions[-1]
            return BCISignal.RELAX

    def is_connected(self) -> bool:
        """EEG 裝置是否已連線"""
        return self._connected

    def is_alive_and_running(self) -> bool:
        """Thread 是否仍在正常運行"""
        return self._running and self.is_alive()

    def get_error(self) -> Optional[str]:
        """取得錯誤訊息（如果有的話）"""
        return self._error_message

    def get_buffer_fill(self) -> float:
        """取得 buffer 填充百分比 (0.0 ~ 1.0)"""
        return min(len(self._buffer) / self.WINDOW_SIZE, 1.0)

    def stop(self):
        """停止 thread"""
        self._running = False

    def _extract_features(self, seg: np.ndarray) -> np.ndarray:
        """
        從一段 EEG 訊號提取 8 個特徵（與 train.py 完全一致）。
        """
        # 帶通濾波 1-40 Hz（與 train.py 一致）
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
            total_power = 1e-10  # 防止除以零

        return np.array([
            var, ptp, rms,
            delta / total_power, theta / total_power,
            alpha / total_power, beta / total_power,
            alpha / beta if beta != 0 else 0.0
        ])

    def _predict(self, features: np.ndarray) -> int:
        """用訓練好的模型做預測"""
        features = features.reshape(1, -1)
        scaled = self._scaler.transform(features)
        selected = self._feature_selector.transform(scaled)
        return int(self._model.predict(selected)[0])

    def run(self):
        """Thread 主迴圈：讀取 Serial → 填 buffer → 預測"""
        self._running = True

        try:
            ser = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            self._connected = True
            print(f"[EEG] Connected to {self.com_port} (Baud: {self.baud_rate})")
        except serial.SerialException as e:
            self._error_message = f"Serial connection failed: {e}"
            print(f"[EEG] {self._error_message}")
            self._running = False
            return

        try:
            while self._running:
                # 讀取所有可用的封包（批次讀取，避免一次只讀一個）
                packets_read = 0
                while ser.in_waiting >= 7 and packets_read < 100:
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
                    else:
                        # 非 raw data 封包，跳過剩餘的 payload
                        payload_len = info[0]
                        remaining = payload_len - 2  # 已讀了 info[1] 和 info[2]
                        if remaining > 0:
                            ser.read(remaining)

                # Buffer 夠滿 → 做預測
                if len(self._buffer) >= self.WINDOW_SIZE:
                    try:
                        # 取最新的 WINDOW_SIZE 個樣本
                        buf_list = list(self._buffer)
                        signals = np.array(buf_list[-self.WINDOW_SIZE:], dtype=np.float64)
                        features = self._extract_features(signals)
                        pred = self._predict(features)

                        with self._lock:
                            self._predictions.append(pred)

                        count = len(self._predictions)
                        label = ["Relax", "Focus", "Blink"][pred]
                        print(f"[EEG] Prediction #{count}: {label}")

                        # 滑動窗口：保留最新的 (WINDOW_SIZE - STEP_SIZE) 個樣本
                        keep = self.WINDOW_SIZE - self.STEP_SIZE
                        while len(self._buffer) > keep:
                            self._buffer.popleft()

                    except Exception as e:
                        print(f"[EEG] Prediction error (continuing): {e}")
                        traceback.print_exc()
                        # 清空 buffer 重新開始收集
                        self._buffer.clear()

                # 短暫讓出 CPU
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
# BCIStateManager - 狀態機
# ============================================================

class BCIStateManager:
    """
    將 BCI 分類器的原始預測序列轉換為遊戲操控狀態。

    狀態轉換：
        AIMING  --[連續 blink_threshold 次 Blink]--> CHARGING
        CHARGING --[收到 Focus]--> power += power_increment
        CHARGING --[連續 relax_threshold 次 Relax 且 power > min_swing_power]--> FLYING

    防誤觸發機制：
        - 眨眼需連續 N 次才觸發
        - 放鬆需連續 N 次才揮桿
        - 力道需超過最小門檻才擊出
    """

    def __init__(self) -> None:
        self.current_state: GameState = GameState.AIMING
        self.power: float = 0.0

        # 防誤觸發參數
        self.blink_threshold: int = 2
        self.blink_counter: int = 0

        self.relax_threshold: int = 3
        self.relax_counter: int = 0

        # 蓄力參數
        self.power_increment: float = 2.0
        self.max_power: float = 100.0
        self.min_swing_power: float = 10.0

    def update(self, raw_prediction: int) -> Tuple[GameState, float, bool]:
        """
        根據新的 BCI 預測更新狀態機。

        Returns:
            (current_state, power, trigger_swing)
        """
        trigger_swing = False

        if self.current_state == GameState.AIMING:
            if raw_prediction == BCISignal.BLINK:
                self.blink_counter += 1
                if self.blink_counter >= self.blink_threshold:
                    self.current_state = GameState.CHARGING
                    self.blink_counter = 0
                    print("[BCI] State: AIMING -> CHARGING (Blink detected)")
            else:
                self.blink_counter = 0

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
                    print(f"[BCI] State: CHARGING -> FLYING (Swing! power={self.power:.1f})")

            else:
                self.relax_counter = 0

        return self.current_state, self.power, trigger_swing

    def reset_round(self) -> None:
        """回合結束後重置狀態機"""
        self.current_state = GameState.AIMING
        self.power = 0.0
        self.blink_counter = 0
        self.relax_counter = 0


# ============================================================
# EEGInput - 統一介面
# ============================================================

class EEGInput:
    """
    EEG 輸入的統一介面。

    整合 EEGSerialReader（背景讀取 + 預測）和 BCIStateManager（狀態機），
    提供簡潔的 update() 方法供遊戲主迴圈呼叫。
    """

    def __init__(self, com_port: str, baud_rate: int = 9600,
                 model_path: str = "enhanced_bci_classifier.pkl"):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed. Run: uv add pyserial")
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is not installed. Run: uv add scipy")

        self.state_manager = BCIStateManager()
        self.reader = EEGSerialReader(com_port, baud_rate, model_path)
        self._last_pred_index = 0  # 追蹤已處理到第幾個預測

        print(f"[EEG] EEGInput initialized. COM={com_port}, Baud={baud_rate}")
        print("[EEG] Call eeg_input.start() to begin reading EEG data.")

    def start(self):
        """開始讀取 EEG 資料（啟動背景 thread）"""
        # 重置狀態，確保之前的預測不會影響遊戲
        self._last_pred_index = 0
        self.state_manager.reset_round()
        self.reader.start()
        print("[EEG] EEG reader started.")

    def sync(self):
        """同步預測索引到最新位置（丟棄所有累積的預測）。
        用於進入遊戲畫面時，忽略 start 畫面期間累積的預測。
        """
        self._last_pred_index = self.reader.get_prediction_count()
        self.state_manager.reset_round()
        print(f"[EEG] Synced. Skipping {self._last_pred_index} old predictions.")

    def update(self) -> Tuple[GameState, float, bool]:
        """
        遊戲每幀呼叫一次。
        逐一處理所有新預測（不會跳過），確保狀態機能正確追蹤連續事件。

        Returns:
            (state, power_ratio, trigger_swing)
            - state: GameState (AIMING / CHARGING / FLYING)
            - power_ratio: 力道比例 0.0 ~ 1.0
            - trigger_swing: 本幀是否觸發了揮桿
        """
        trigger_swing = False

        # 取得所有新預測，逐一餵給狀態機
        new_preds = self.reader.get_predictions_since(self._last_pred_index)
        if new_preds:
            self._last_pred_index += len(new_preds)
            for pred in new_preds:
                state, power, swung = self.state_manager.update(pred)
                if swung:
                    trigger_swing = True

        # 回傳目前狀態
        state = self.state_manager.current_state
        power = self.state_manager.power
        power_ratio = power / self.state_manager.max_power

        return state, power_ratio, trigger_swing

    def reset_round(self) -> None:
        """球進洞 / 出界後重置狀態機"""
        self.state_manager.reset_round()

    def is_connected(self) -> bool:
        """EEG 裝置是否已連線"""
        return self.reader.is_connected()

    def close(self) -> None:
        """關閉 EEG 連線，停止背景 thread"""
        self.reader.stop()
        self.reader.join(timeout=2)
        print("[EEG] EEGInput closed.")
