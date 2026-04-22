# EEG MiniGolf

基於 Pygame 的迷你高爾夫遊戲，最終目標是整合穿戴式 EEG 感測裝置，透過 **眨眼**、**專注**、**放鬆** 三種腦波狀態操控遊戲。目前以滑鼠點擊與空白鍵模擬 EEG 輸入。

## 操作方式

| 步驟 | 操作 | 模擬的 EEG 狀態 |
|------|------|-----------------|
| 瞄準 | 球靜止時，紅色箭頭自動旋轉 | — |
| 鎖定方向 | 滑鼠點擊 | 眨眼 |
| 蓄力 | 按住空白鍵（箭頭由短變長） | 專注 |
| 揮桿 | 放開空白鍵 | 放鬆 |

力道與按住空白鍵的時長成正比，對應未來 EEG 的專注持續時間。

## 專案結構

```
MiniGolf/
├── main.py              # 遊戲主迴圈、輸入處理、繪圖
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
└── images/              # ball_golf.png / heart.png / sheet_map.png
```

## 各檔案說明

### `main.py`
遊戲入口，包含三階段迴圈：開始畫面 → 遊戲主迴圈 → 結束畫面。

輸入處理流程：
1. 球靜止時箭頭自動旋轉（`aim_angle += AIM_SPEED * dt`）
2. 滑鼠點擊鎖定方向（`aim_locked = True`）
3. 按住空白鍵蓄力（`charge_power` 隨時間增加，箭頭同步變長）
4. 放開空白鍵施加力道（`ball.apply_force(direction * charge_power * MAX_FORCE)`）

每幀執行：物理更新 → 地形摩擦 → 碰撞偵測 → 進洞/出界/碰推車判定 → 繪圖。

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

## 未來 EEG 整合

建立 `eeg_input.py` 輸入抽象層，將 `main.py` 中的事件判斷替換：

```python
# 滑鼠點擊 → eeg.blink_detected（眨眼偵測）
# 空白鍵按下 → eeg.attention_level > threshold（專注度超過閾值）
# 空白鍵放開 → eeg.relaxed（偵測到放鬆）
```

需修改的位置集中在 `main.py` 的事件處理區塊（約 L88-L105）。

## 安裝與執行

```bash
cd MiniGolf
uv sync
uv run python main.py
```

**環境需求：** Python ≥ 3.13、pygame ≥ 2.6.1、numpy ≥ 2.4.4

**原始repo：** [OlaPietka/MiniGolf](https://github.com/OlaPietka/MiniGolf)

**素材來源：** [Kenney.nl](https://www.kenney.nl/assets)
