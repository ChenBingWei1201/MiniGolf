# EEG MiniGolf

基於 Pygame 的迷你高爾夫遊戲，最終目標是整合穿戴式 EEG 感測裝置，透過 **眨眼**、**專注**、**放鬆** 三種腦波狀態操控遊戲。

[![demo](https://img.youtube.com/vi/mGCUsP0w5WU/0.jpg)](https://www.youtube.com/watch?v=mGCUsP0w5WU)

## 操作方式

使用 BrainLink 穿戴式 EEG 裝置，透過真實腦波操控遊戲：

| 步驟 | 腦波狀態 | 說明 |
|------|----------|----- |
| 瞄準 | — | 紅色箭頭自動旋轉 |
| 鎖定方向 | 眨眼 ×1 | 利用原始振幅瞬間偵測眨眼 |
| 蓄力 | 持續專注 | 每次偵測到專注，力道 +1（最高 100） |
| 揮桿 | 放鬆或振幅驟降 | 結合 ML 預測與 < 1 秒的振幅快篩，達成瞬間揮桿 |

EEG 模型約每 0.25 秒做一次預測（5 秒滑動窗口、512 Hz 取樣率）。

## 專案結構

```
MiniGolf/
├── main.py              # 遊戲主迴圈、輸入處理、繪圖
├── eeg_input.py         # EEG 輸入抽象層（Serial reader + 狀態機 + 模型預測）
├── game_manager.py      # 關卡切換、生命值、計分、UI 文字
├── game_objects.py      # Ball / Hole / Wall / Cart 遊戲物件
├── physics.py           # 碰撞偵測、反彈、交點計算
├── scene.py             # 場景基底類別（地圖建構、牆壁生成）
├── levels.py            # 9 個關卡定義（Level1~9）
├── cell.py              # 地圖格子單元（地形類型 + 貼圖）
├── config.py            # 全域設定（螢幕大小、球半徑等）
├── rigidbody.py         # 剛體物理（速度、加速度、摩擦、反彈）
├── shapes.py            # Circle / Line 基礎幾何
├── sprite_sheet.py      # Sprite sheet 切圖 + Type 列舉
├── transformation.py    # 格子索引 ↔ 像素座標轉換
├── train.py             # EEG MLP 分類器訓練（離線，產生 .pkl 模型檔）
└── images/              # ball_golf.png / heart.png / sheet_map.png
```

## 各檔案說明

### `main.py`
遊戲入口，包含三階段迴圈：開始畫面 → 遊戲主迴圈 → 結束畫面。

輸入處理流程：
1. 球靜止時箭頭自動旋轉（`aim_angle += AIM_SPEED * dt`）
2. 鎖定方向（EEG 眨眼）
3. 蓄力（EEG 持續專注）
4. 揮桿（EEG 連續放鬆）

### `eeg_input.py`
EEG 輸入抽象層，包含三個核心元件：

- **`EEGSerialReader`**（`threading.Thread`）：背景 thread 連接 BrainLink Serial，解析 `0xAA 0xAA` 封包，維護 5 秒滑動窗口，每 0.25 秒做特徵提取 + 模型預測
- **`BCIStateManager`**：狀態機，將連續預測序列轉為遊戲動作（AIMING → CHARGING → FLYING），包含雙視窗攔截機制與防誤觸發
- **`EEGInput`**：統一介面，`update()` 回傳 `(state, power_ratio, trigger_swing)` 供遊戲主迴圈使用

### `train.py`
離線訓練 MLP 分類器。讀取 `bci_dataset_114-2_any/` 資料夾中的 EEG 資料，提取 8 個特徵（時域 + 頻域），使用 LOSO 交叉驗證，產出 `enhanced_bci_classifier.pkl`。

### `realtime.py`
獨立的 EEG 即時測試工具。連接 BrainLink 裝置，顯示即時波形圖 + 預測結果。可用於獨立測試裝置連線與模型效果。

### `game_manager.py`
繼承 `Levels`，管理 `throw_number`（揮桿數）和 `lives_number`（生命值，初始 3）。球出界或碰推車扣一命，命歸零重新開始。

### `game_objects.py`
- **Ball**：地形摩擦切換（草 0.98 / 沙 0.80 / 水 0.70）、靜止判斷
- **Hole**：黑色圓形球洞
- **Wall**：牆壁 + hitbox（平行偏移線段 + 端點圓形）
- **Cart**：來回移動的障礙推車

### `physics.py`
自製 2D 碰撞引擎：線段-圓碰撞、線段-線段相交、最近碰撞點搜尋、反彈法向量計算。

### `levels.py`
9 個關卡，三大主題：草地（L1-3）、沙漠（L4-6，含推車障礙）、天空（L7-9）。每關定義起終點、地形配置、牆壁與推車。

### `scene.py`
地圖建構工具：`horizontal_line` / `vertical_line` / `rectangle` 繪製地形並自動生成 Wall 碰撞體。

### `config.py`
螢幕 1440×960、球半徑 10、格子 16px、推車 40px。

## 安裝與執行

```bash
cd MiniGolf
uv sync
```

```bash
# 1. 先訓練模型（需要 bci_dataset_114-2_any/ 資料夾）
uv run train.py

# 2. 啟動 EEG 模式（替換 COM9 為你的裝置 port）
uv run main.py --eeg --port COM9
```

**環境需求：** Python ≥ 3.13、pygame ≥ 2.6.1、numpy ≥ 2.4.4、scikit-learn ≥ 1.6.0、scipy ≥ 1.15.0、joblib ≥ 1.5.0、pyserial ≥ 3.5

**原始repo：** [OlaPietka/MiniGolf](https://github.com/OlaPietka/MiniGolf)

**素材來源：** [Kenney.nl](https://www.kenney.nl/assets)
