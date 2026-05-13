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
    # In game loop:
    state, power, trigger_swing = eeg.update()
    # On exit:
    eeg.close()
"""

import threading
import time
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

    預測結果透過 get_latest_prediction() 供遊戲主迴圈讀取。
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

        # Serial buffer
        self._buffer = deque(maxlen=self.WINDOW_SIZE)

        # Thread-safe 共享狀態
        self._lock = threading.Lock()
        self._latest_prediction: int = BCISignal.RELAX  # 預設放鬆
        self._prediction_count: int = 0
        self._running = True
        self._connected = False

    def get_latest_prediction(self) -> int:
        """供遊戲主迴圈呼叫，取得最新一次的 BCI 預測結果 (0/1/2)"""
        with self._lock:
            return self._latest_prediction

    def get_prediction_count(self) -> int:
        """取得總共做了幾次預測（debug 用）"""
        with self._lock:
            return self._prediction_count

    def is_connected(self) -> bool:
        """EEG 裝置是否已連線"""
        return self._connected

    def stop(self):
        """停止 thread"""
        self._running = False

    def _extract_features(self, seg: np.ndarray) -> np.ndarray:
        """
        從一段 EEG 訊號提取 8 個特徵（與 train.py 完全一致）。

        特徵：
        1. var         - 訊號變異數
        2. ptp         - Peak-to-peak
        3. rms         - 均方根
        4. delta/total - δ波佔比 (1-4 Hz)
        5. theta/total - θ波佔比 (4-8 Hz)
        6. alpha/total - α波佔比 (8-13 Hz)
        7. beta/total  - β波佔比 (13-30 Hz)
        8. alpha/beta  - α/β 比值
        """
        # 帶通濾波 1-40 Hz（與 train.py 一致）
        sos = butter(4, [1.0, 40.0], btype='bandpass',
                     fs=self.SAMPLING_RATE, output='sos')
        seg_filtered = sosfiltfilt(sos, seg)

        var = np.var(seg_filtered)
        ptp = np.ptp(seg_filtered)
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
        try:
            ser = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            self._connected = True
            print(f"[EEG] Connected to {self.com_port} (Baud: {self.baud_rate})")
            print("[EEG] Reading EEG signals... Waiting for buffer to fill.")
        except serial.SerialException as e:
            print(f"[EEG] Serial connection failed: {e}")
            return

        try:
            while self._running:
                # 解析 BrainLink 封包：0xAA 0xAA 開頭
                if ser.in_waiting >= 8:
                    if ser.read(1) == b'\xaa':
                        if ser.read(1) == b'\xaa':
                            info = ser.read(3)
                            if info[1] == 128 and info[2] == 2:
                                bs = ser.read(2)
                                if len(bs) == 2:
                                    raw = int.from_bytes(bs, byteorder='big',
                                                         signed=True)
                                    self._buffer.append(raw)

                # Buffer 滿了 → 做預測
                if len(self._buffer) == self.WINDOW_SIZE:
                    signals = np.array(self._buffer)
                    features = self._extract_features(signals)
                    pred = self._predict(features)

                    with self._lock:
                        self._latest_prediction = pred
                        self._prediction_count += 1

                    label = ["Relax", "Focus", "Blink"][pred]
                    print(f"[EEG] Prediction #{self._prediction_count}: {label}")

                    # 滑動窗口：移除最舊的 STEP_SIZE 個樣本
                    for _ in range(self.STEP_SIZE):
                        self._buffer.popleft()

                time.sleep(0.001)  # 避免 busy-wait

        except serial.SerialException as e:
            print(f"[EEG] Serial error: {e}")
        except Exception as e:
            print(f"[EEG] Unexpected error: {e}")
        finally:
            try:
                ser.close()
                self._connected = False
                print("[EEG] Serial disconnected.")
            except Exception:
                pass


# ============================================================
# BCIStateManager - 狀態機（從 integrating_classifier_output_with_games.py 搬來）
# ============================================================

class BCIStateManager:
    """
    將 BCI 分類器的原始預測序列轉換為遊戲操控狀態。

    狀態轉換：
        AIMING  --[連續 blink_threshold 次 Blink]--> CHARGING
        CHARGING --[收到 Focus]--> power += power_increment
        CHARGING --[連續 relax_threshold 次 Relax 且 power > min_swing_power]--> FLYING (觸發揮桿)

    防誤觸發機制：
        - 眨眼需連續 N 次才觸發（避免偶發雜訊）
        - 放鬆需連續 N 次才揮桿（避免蓄力中意外放鬆）
        - 力道需超過最小門檻才擊出（避免空揮）
    """

    def __init__(self) -> None:
        self.current_state: GameState = GameState.AIMING
        self.power: float = 0.0

        # 防誤觸發參數
        self.blink_threshold: int = 2   # 連續幾次 Blink 才觸發
        self.blink_counter: int = 0

        self.relax_threshold: int = 3   # 連續幾次 Relax 才揮桿
        self.relax_counter: int = 0

        # 蓄力參數
        self.power_increment: float = 2.0   # 每次 Focus 增加的力道
        self.max_power: float = 100.0       # 最大力道
        self.min_swing_power: float = 10.0  # 揮桿最低力道門檻

    def update(self, raw_prediction: int) -> Tuple[GameState, float, bool]:
        """
        根據新的 BCI 預測更新狀態機。

        Args:
            raw_prediction: BCI 分類器輸出 (0=Relax, 1=Focus, 2=Blink)

        Returns:
            (current_state, power, trigger_swing)
            - current_state: 目前遊戲狀態
            - power: 目前蓄力值 (0.0 ~ max_power)
            - trigger_swing: 是否剛觸發揮桿
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
                # Blink during charging = noise, ignore
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

    Usage:
        eeg = EEGInput(com_port="COM3")
        # Game loop:
        state, power_ratio, trigger_swing = eeg.update()
        # power_ratio is 0.0 ~ 1.0, ready for force calculation
    """

    def __init__(self, com_port: str, baud_rate: int = 9600,
                 model_path: str = "enhanced_bci_classifier.pkl"):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed. Run: uv add pyserial")
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is not installed. Run: uv add scipy")

        self.state_manager = BCIStateManager()
        self.reader = EEGSerialReader(com_port, baud_rate, model_path)
        self._last_prediction_count = 0

        # 啟動背景 thread
        self.reader.start()
        print(f"[EEG] EEGInput initialized. COM={com_port}, Baud={baud_rate}")

    def update(self) -> Tuple[GameState, float, bool]:
        """
        遊戲每幀呼叫一次。

        Returns:
            (state, power_ratio, trigger_swing)
            - state: GameState (AIMING / CHARGING / FLYING)
            - power_ratio: 力道比例 0.0 ~ 1.0（已從 BCIStateManager 的 0~100 映射）
            - trigger_swing: 是否剛觸發揮桿
        """
        # 檢查是否有新的預測
        current_count = self.reader.get_prediction_count()
        if current_count > self._last_prediction_count:
            # 有新預測 → 更新狀態機
            pred = self.reader.get_latest_prediction()
            self._last_prediction_count = current_count
            state, power, trigger_swing = self.state_manager.update(pred)
        else:
            # 沒有新預測 → 回傳當前狀態，不觸發任何事
            state = self.state_manager.current_state
            power = self.state_manager.power
            trigger_swing = False

        # 將 power (0~100) 映射到 0.0~1.0
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
