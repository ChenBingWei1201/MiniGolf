import math
import sys
from game_manager import GameManager
import physics
import pygame
from pygame import Vector2
from game_objects import Ball, Hole
from shapes import Line
import config
from transformation import pixels_2_indexes
from eeg_input import EEGInput, GameState as EEGGameState


# ============================================================
# EEG 連線設定
# ============================================================
com_port = "COM3"  # 預設值
if "--port" in sys.argv:
    port_idx = sys.argv.index("--port") + 1
    if port_idx < len(sys.argv):
        com_port = sys.argv[port_idx]


pygame.init()
screen = pygame.display.set_mode(config.screen_size)
clock = pygame.time.Clock()

# 顯示載入畫面
screen.fill((255, 255, 255))
loading_font = pygame.font.SysFont('Comic Sans MS', 30)
loading_text = loading_font.render("Loading EEG model...", True, (100, 100, 100))
screen.blit(loading_text, (config.screen_size[0] // 2 - 130,
                           config.screen_size[1] // 2 - 20))
pygame.display.flip()

# 載入模型 + 初始化 EEG（但還不開始讀取 Serial）
print(f"[EEG] Initializing EEG on {com_port}...")
eeg_input = EEGInput(com_port=com_port)
print("[EEG] Model loaded. Waiting for game start...")

game_end = False
game_start = False

game_manager = GameManager()

ball = Ball(game_manager.current_level.start_point, config.ball_radius)
hole = Hole(game_manager.current_level.end_point, config.ball_radius + 5)

# === 瞄準 / 蓄力狀態 ===
aim_angle = 0.0             # 當前瞄準角度（弧度），持續旋轉
aim_locked = False           # 方向是否已被鎖定（眨眼確認）
charging = False             # 是否正在蓄力（專注狀態）
charge_power = 0.0           # 當前蓄力比例 (0.0 ~ 1.0)

AIM_SPEED = 1.5              # 方向指示線旋轉速度（弧度/秒）
MAX_FORCE = 800.0            # 最大力道
MIN_ARROW_LEN = 35.0         # 箭頭最短長度（預設短箭頭）
MAX_ARROW_LEN = 160.0        # 箭頭最長長度（蓄滿時）


def reset_aim_state():
    """重置瞄準與蓄力狀態"""
    global aim_angle, aim_locked, charging, charge_power
    aim_locked = False
    charging = False
    charge_power = 0.0
    eeg_input.reset_round()


def draw_arrow(surface, color, start, end, head_size=10, width=2):
    """繪製帶有箭頭頭部的箭頭線"""
    pygame.draw.line(surface, color, start, end, width)
    # 計算箭頭頭部
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    # 箭頭兩翼
    left_x = end[0] - head_size * math.cos(angle - math.pi / 6)
    left_y = end[1] - head_size * math.sin(angle - math.pi / 6)
    right_x = end[0] - head_size * math.cos(angle + math.pi / 6)
    right_y = end[1] - head_size * math.sin(angle + math.pi / 6)
    pygame.draw.polygon(surface, color, [(end[0], end[1]), (left_x, left_y), (right_x, right_y)])


def draw_eeg_status(surface):
    """在畫面左上角顯示 EEG 即時狀態資訊"""
    font = pygame.font.SysFont('Comic Sans MS', 16)
    y_offset = 50

    # 連線狀態
    connected = eeg_input.is_connected()
    thread_ok = eeg_input.reader.is_alive_and_running()
    if connected and thread_ok:
        conn_color = (0, 200, 0)
        conn_text = "EEG: Connected"
    elif connected and not thread_ok:
        conn_color = (255, 165, 0)
        conn_text = "EEG: Thread stopped!"
    else:
        conn_color = (200, 0, 0)
        conn_text = "EEG: Waiting..."
    text_surf = font.render(conn_text, True, conn_color)
    surface.blit(text_surf, (10, y_offset))

    # Buffer 填充度
    fill = eeg_input.reader.get_buffer_fill()
    fill_text = font.render(f"Buffer: {fill*100:.0f}%", True, (180, 180, 180))
    surface.blit(fill_text, (10, y_offset + 22))

    # 目前狀態
    state = eeg_input.state_manager.current_state
    state_names = {0: "AIMING", 1: "CHARGING", 2: "FLYING"}
    state_colors = {0: (100, 100, 255), 1: (255, 165, 0), 2: (0, 200, 0)}
    state_name = state_names.get(int(state), "UNKNOWN")
    state_text = font.render(f"State: {state_name}", True,
                             state_colors.get(int(state), (255, 255, 255)))
    surface.blit(state_text, (10, y_offset + 44))

    # 蓄力值
    power = eeg_input.state_manager.power
    power_text = font.render(f"Power: {power:.1f} / {eeg_input.state_manager.max_power:.0f}",
                             True, (255, 255, 255))
    surface.blit(power_text, (10, y_offset + 66))

    # 預測次數
    pred_count = eeg_input.reader.get_prediction_count()
    pred_text = font.render(f"Predictions: {pred_count}", True, (180, 180, 180))
    surface.blit(pred_text, (10, y_offset + 88))

    # 最新預測
    pred_val = eeg_input.reader.get_latest_prediction()
    pred_labels = {0: "Relax", 1: "Focus", 2: "Blink"}
    pred_label_colors = {0: (100, 200, 100), 1: (255, 200, 50), 2: (255, 100, 100)}
    label = pred_labels.get(pred_val, "?")
    label_text = font.render(f"Brain: {label}", True,
                             pred_label_colors.get(pred_val, (255, 255, 255)))
    surface.blit(label_text, (10, y_offset + 110))

    # 錯誤訊息
    err = eeg_input.reader.get_error()
    if err:
        err_text = font.render(f"Error: {err[:40]}", True, (255, 0, 0))
        surface.blit(err_text, (10, y_offset + 132))


if __name__ == "__main__":
    print("=" * 50)
    print("  EEG Mini Golf - Brain Control Mode")
    print("  Blink  → lock direction")
    print("  Focus  → charge power")
    print("  Relax  → swing!")
    print("=" * 50)

    # === 開始畫面 ===
    while not game_start:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                eeg_input.close()
                pygame.quit()
                exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = Vector2(pygame.mouse.get_pos())
                x = config.screen_size[0] // 2 - 140
                y = config.screen_size[1] // 2 - 70
                if physics.point_in_rect((x, y, x + 319, y + 85), mouse):
                    game_start = True

        game_manager.blit_start(screen)

        mode_font = pygame.font.SysFont('Comic Sans MS', 18)
        mode_text = mode_font.render("Mode: EEG Brain Control", True, (100, 100, 100))
        screen.blit(mode_text, (config.screen_size[0] // 2 - 100,
                                config.screen_size[1] // 2 + 30))

        pygame.display.flip()
        clock.tick(60)  # 限制開始畫面的幀率

    # === 進入遊戲：啟動 EEG 讀取並同步 ===
    eeg_input.start()
    # 等待一小段時間讓 Serial 連線穩定
    pygame.time.wait(500)
    # 同步預測索引 → 丟棄啟動期間的雜訊預測
    eeg_input.sync()

    game_manager.current_level.init()
    carts = game_manager.current_level.carts

    # === 遊戲主迴圈 ===
    while not game_end:
        screen.fill(game_manager.current_level.background_color)
        dt = clock.tick(140) / 1000.0  # 轉為秒

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                eeg_input.close()
                pygame.quit()
                exit(0)

        # ==============================================
        # EEG 輸入處理
        # ==============================================
        if ball.not_moving():
            state, power_ratio, trigger_swing = eeg_input.update()

            if state == EEGGameState.AIMING:
                # 瞄準中：箭頭持續旋轉
                aim_locked = False
                charging = False
                charge_power = 0.0

            elif state == EEGGameState.CHARGING:
                # 蓄力中：方向已鎖定，力道跟隨 EEG 專注程度
                aim_locked = True
                charging = True
                charge_power = power_ratio  # 0.0 ~ 1.0

            if trigger_swing:
                # 偵測到放鬆 → 觸發揮桿！
                direction = Vector2(math.cos(aim_angle), math.sin(aim_angle))
                force = direction * (charge_power * MAX_FORCE)
                ball.apply_force(force)
                game_manager.throw_number += 1
                reset_aim_state()

        # --- 每幀更新：方向旋轉 ---
        if ball.not_moving() and not aim_locked:
            aim_angle += AIM_SPEED * dt

        # --- 物理更新 ---
        ball.move(dt * 10)
        time_line = Line(ball.pos, ball.pos + ball.vel * dt * 10)

        i, j = pixels_2_indexes(*ball.pos)
        cell = game_manager.current_level.board[i][j]
        ball.ground_friction(cell.type)

        physics.check_collisions(game_manager.current_level.walls, time_line, ball)

        # --- 繪圖 ---
        game_manager.current_level.blit(screen)
        hole.blit(screen)
        ball.blit(screen)
        game_manager.blit_text(screen)
        game_manager.blit_lives(screen)

        if carts is not None:
            for cart in carts:
                cart.blit(screen)
                cart.move_cart(dt * 10)

                if game_manager.ball_touch_cart(ball, cart):
                    game_manager.reset_level(ball, hole)
                    reset_aim_state()

        # --- 繪製方向箭頭 ---
        if ball.not_moving():
            direction = Vector2(math.cos(aim_angle), math.sin(aim_angle))

            if charging:
                arrow_len = MIN_ARROW_LEN + (MAX_ARROW_LEN - MIN_ARROW_LEN) * charge_power
                end_point = ball.pos + direction * arrow_len
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            elif aim_locked:
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            else:
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)

        # --- 繪製 EEG 狀態資訊 ---
        draw_eeg_status(screen)

        pygame.display.flip()

        if physics.point_in_circle(hole, ball.pos):
            ball.vel += (hole.pos - ball.pos)

        if game_manager.ball_in_hole(ball, hole):
            carts, is_end = game_manager.new_level(ball, hole)
            reset_aim_state()

            if is_end:
                game_end = True
                continue

        if game_manager.ball_outside(cell.type.value):
            game_manager.reset_level(ball, hole)
            reset_aim_state()

        if game_manager.no_lives():
            carts = game_manager.new_game(ball, hole)
            reset_aim_state()

    exit_game = False
    font = pygame.font.SysFont('Comic Sans MS', 30)

    while not exit_game:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True

        pygame.draw.line(screen, (255, 255, 0), (10, 10), (100, 100))
        game_manager.blit_the_end(screen)

        pygame.display.flip()

    eeg_input.close()
    pygame.quit()
    exit(0)
