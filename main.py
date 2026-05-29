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
# EEG Connection Settings
# ============================================================
com_port = "COM3"
if "--port" in sys.argv:
    port_idx = sys.argv.index("--port") + 1
    if port_idx < len(sys.argv):
        com_port = sys.argv[port_idx]


pygame.init()

# Display window (scaled down, adjustable in config.py)
screen = pygame.display.set_mode(config.window_size)
# Game internal render surface (fixed at 1440x960)
game_surface = pygame.Surface(config.screen_size)
clock = pygame.time.Clock()

# Mouse coordinate scaling ratio
_sx = config.screen_size[0] / config.window_size[0]
_sy = config.screen_size[1] / config.window_size[1]


def scale_mouse(pos):
    return Vector2(pos[0] * _sx, pos[1] * _sy)


def flip_display():
    scaled = pygame.transform.scale(game_surface, config.window_size)
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


# Show loading screen
game_surface.fill((255, 255, 255))
loading_font = pygame.font.SysFont('Comic Sans MS', 30)
loading_text = loading_font.render("Loading EEG model...", True, (100, 100, 100))
game_surface.blit(loading_text, (config.screen_size[0] // 2 - 130,
                                 config.screen_size[1] // 2 - 20))
flip_display()

# Load model and initialize EEG
print(f"[EEG] Initializing EEG on {com_port}...")
eeg_input = EEGInput(com_port=com_port)
print("[EEG] Model loaded. Waiting for game start...")

game_end = False
game_start = False

game_manager = GameManager()

ball = Ball(game_manager.current_level.start_point, config.ball_radius)
hole = Hole(game_manager.current_level.end_point, config.ball_radius + 5)

# === Aiming / Charging State ===
aim_angle = 0.0
aim_locked = False
charging = False
charge_power = 0.0

AIM_SPEED = 1.5
MAX_FORCE = 800.0
MIN_ARROW_LEN = 35.0
MAX_ARROW_LEN = 160.0


def reset_aim_state():
    global aim_angle, aim_locked, charging, charge_power
    aim_locked = False
    charging = False
    charge_power = 0.0
    eeg_input.reset_round()


def draw_arrow(surface, color, start, end, head_size=10, width=2):
    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    left_x = end[0] - head_size * math.cos(angle - math.pi / 6)
    left_y = end[1] - head_size * math.sin(angle - math.pi / 6)
    right_x = end[0] - head_size * math.cos(angle + math.pi / 6)
    right_y = end[1] - head_size * math.sin(angle + math.pi / 6)
    pygame.draw.polygon(surface, color, [(end[0], end[1]), (left_x, left_y), (right_x, right_y)])


def draw_eeg_status(surface):
    font = pygame.font.SysFont('Comic Sans MS', 16)
    y_offset = 50

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

    if eeg_input.reader.is_calibrating():
        cal_text = font.render("Blink: Calibrating...", True, (255, 200, 50))
        surface.blit(cal_text, (10, y_offset + 22))
    else:
        threshold = eeg_input.reader._blink_detector.threshold
        fill = eeg_input.reader.get_buffer_fill()
        info_text = font.render(f"Blink threshold: {threshold:.0f} | Buffer: {fill*100:.0f}%",
                                True, (180, 180, 180))
        surface.blit(info_text, (10, y_offset + 22))

    state = eeg_input.state_manager.current_state
    state_names = {0: "AIMING", 1: "CHARGING", 2: "FLYING"}
    state_colors = {0: (100, 100, 255), 1: (255, 165, 0), 2: (0, 200, 0)}
    state_name = state_names.get(int(state), "UNKNOWN")
    state_text = font.render(f"State: {state_name}", True,
                             state_colors.get(int(state), (255, 255, 255)))
    surface.blit(state_text, (10, y_offset + 44))

    power = eeg_input.state_manager.power
    power_text = font.render(f"Power: {power:.1f} / {eeg_input.state_manager.max_power:.0f}",
                             True, (255, 255, 255))
    surface.blit(power_text, (10, y_offset + 66))

    pred_count = eeg_input.reader.get_prediction_count()
    pred_text = font.render(f"Predictions: {pred_count}", True, (180, 180, 180))
    surface.blit(pred_text, (10, y_offset + 88))

    pred_val = eeg_input.reader.get_latest_prediction()
    pred_labels = {0: "Relax", 1: "Focus", 2: "Blink"}
    pred_label_colors = {0: (100, 200, 100), 1: (255, 200, 50), 2: (255, 100, 100)}
    label = pred_labels.get(pred_val, "?")
    label_text = font.render(f"Brain: {label}", True,
                             pred_label_colors.get(pred_val, (255, 255, 255)))
    surface.blit(label_text, (10, y_offset + 110))

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

    # === Start Screen ===
    while not game_start:
        game_surface.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                eeg_input.close()
                pygame.quit()
                exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = scale_mouse(pygame.mouse.get_pos())
                x = config.screen_size[0] // 2 - 140
                y = config.screen_size[1] // 2 - 70
                if physics.point_in_rect((x, y, x + 319, y + 85), mouse):
                    game_start = True

        game_manager.blit_start(game_surface)

        mode_font = pygame.font.SysFont('Comic Sans MS', 18)
        mode_text = mode_font.render("Mode: EEG Brain Control", True, (100, 100, 100))
        game_surface.blit(mode_text, (config.screen_size[0] // 2 - 100,
                                      config.screen_size[1] // 2 + 30))

        flip_display()
        clock.tick(60)

    # === Game Setup (Start EEG after setup) ===
    game_manager.current_level.init()
    carts = game_manager.current_level.carts

    eeg_input.start()

    # === EEG Calibration Wait Screen ===
    _wait_font_big = pygame.font.SysFont('Comic Sans MS', 50)
    _wait_font_med = pygame.font.SysFont('Comic Sans MS', 26)
    _wait_font_sm  = pygame.font.SysFont('Comic Sans MS', 20)
    _dot_timer = 0.0
    _dot_count = 0

    while not eeg_input.is_ready():
        _dt_wait = clock.tick(60) / 1000.0
        _dot_timer += _dt_wait
        if _dot_timer >= 0.5:
            _dot_timer = 0.0
            _dot_count = (_dot_count + 1) % 4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                eeg_input.close()
                pygame.quit()
                exit(0)

        # 白色背景（與 Start 畫面一致）
        game_surface.fill((255, 255, 255))
        cx = config.screen_size[0] // 2
        cy = config.screen_size[1] // 2

        # Title
        _title_surf = _wait_font_big.render("EEG Setup" + "." * _dot_count, False, (0, 0, 0))
        game_surface.blit(_title_surf, _title_surf.get_rect(center=(cx, cy - 110)))

        # Connection Status
        _connected = eeg_input.is_connected()
        _conn_color = (50, 150, 50) if _connected else (180, 100, 0)
        _conn_label = "Serial: Connected" if _connected else "Serial: Connecting..."
        _conn_surf = _wait_font_med.render(_conn_label, False, _conn_color)
        game_surface.blit(_conn_surf, _conn_surf.get_rect(center=(cx, cy - 20)))

        # Calibration Status
        if eeg_input.reader.is_calibrating():
            _cal_surf = _wait_font_med.render("Calibrating blink detector (3s)...", False, (160, 110, 0))
        else:
            _cal_surf = _wait_font_med.render("Calibration done!", False, (50, 150, 50))
        game_surface.blit(_cal_surf, _cal_surf.get_rect(center=(cx, cy + 30)))

        # Hint Text
        _hint_surf = _wait_font_sm.render("Please keep still. Game will start automatically.", False, (100, 100, 100))
        game_surface.blit(_hint_surf, _hint_surf.get_rect(center=(cx, cy + 90)))

        # Calibration Progress Bar
        _total_cal = eeg_input.reader._blink_detector.calibration_samples
        _done_cal  = min(eeg_input.reader._blink_detector._total_samples, _total_cal)
        _bar_w, _bar_h = 500, 18
        _bar_x = cx - _bar_w // 2
        _bar_y = cy + 130
        pygame.draw.rect(game_surface, (210, 210, 210), (_bar_x, _bar_y, _bar_w, _bar_h), border_radius=9)
        _fill_w = int(_bar_w * _done_cal / max(_total_cal, 1))
        if _fill_w > 0:
            pygame.draw.rect(game_surface, (60, 60, 60), (_bar_x, _bar_y, _fill_w, _bar_h), border_radius=9)
        _bar_label = _wait_font_sm.render("Calibration progress", False, (120, 120, 120))
        game_surface.blit(_bar_label, _bar_label.get_rect(center=(cx, _bar_y + 38)))

        flip_display()

    # Sync predictions and enter level
    eeg_input.sync()

    # === Main Game Loop ===
    while not game_end:
        game_surface.fill(game_manager.current_level.background_color)
        dt = clock.tick(140) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                eeg_input.close()
                pygame.quit()
                exit(0)

        # ==============================================
        # EEG Input Handling
        # ==============================================
        if ball.not_moving():
            state, power_ratio, trigger_swing = eeg_input.update()

            if state == EEGGameState.AIMING:
                aim_locked = False
                charging = False
                charge_power = 0.0

            elif state == EEGGameState.CHARGING:
                aim_locked = True
                charging = True
                charge_power = power_ratio

            if trigger_swing:
                direction = Vector2(math.cos(aim_angle), math.sin(aim_angle))
                force = direction * (charge_power * MAX_FORCE)
                ball.apply_force(force)
                game_manager.throw_number += 1
                reset_aim_state()

        # --- Per-frame Update: Direction Rotation ---
        if ball.not_moving() and not aim_locked:
            aim_angle += AIM_SPEED * dt

        # --- Physics Update ---
        ball.move(dt * 10)
        time_line = Line(ball.pos, ball.pos + ball.vel * dt * 10)

        i, j = pixels_2_indexes(*ball.pos)
        cell = game_manager.current_level.board[i][j]
        ball.ground_friction(cell.type)

        physics.check_collisions(game_manager.current_level.walls, time_line, ball)

        # --- Draw (All drawn to game_surface) ---
        game_manager.current_level.blit(game_surface)
        hole.blit(game_surface)
        ball.blit(game_surface)
        game_manager.blit_text(game_surface)
        game_manager.blit_lives(game_surface)

        if carts is not None:
            for cart in carts:
                cart.blit(game_surface)
                cart.move_cart(dt * 10)

                if game_manager.ball_touch_cart(ball, cart):
                    game_manager.reset_level(ball, hole)
                    reset_aim_state()

        # --- Draw Direction Arrow ---
        if ball.not_moving():
            direction = Vector2(math.cos(aim_angle), math.sin(aim_angle))

            if charging:
                arrow_len = MIN_ARROW_LEN + (MAX_ARROW_LEN - MIN_ARROW_LEN) * charge_power
                end_point = ball.pos + direction * arrow_len
                draw_arrow(game_surface, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            elif aim_locked:
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(game_surface, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            else:
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(game_surface, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)

        # --- Draw EEG Status Info ---
        draw_eeg_status(game_surface)

        flip_display()

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
        game_surface.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True

        pygame.draw.line(game_surface, (255, 255, 0), (10, 10), (100, 100))
        game_manager.blit_the_end(game_surface)

        flip_display()

    eeg_input.close()
    pygame.quit()
    exit(0)
