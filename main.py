import math
from game_manager import GameManager
import physics
import pygame
from pygame import Vector2
from game_objects import Ball, Hole
from shapes import Line
import config
from transformation import pixels_2_indexes


pygame.init()
screen = pygame.display.set_mode(config.screen_size)
clock = pygame.time.Clock()

game_end = False
game_start = False

game_manager = GameManager()

ball = Ball(game_manager.current_level.start_point, config.ball_radius)
hole = Hole(game_manager.current_level.end_point, config.ball_radius + 5)

# === EEG 模擬操作狀態 ===
aim_angle = 0.0             # 當前瞄準角度（弧度），持續旋轉
aim_locked = False           # 方向是否已被鎖定（模擬眨眼確認）
charging = False             # 是否正在蓄力（模擬專注狀態）
charge_power = 0.0           # 當前蓄力比例 (0.0 ~ 1.0)

AIM_SPEED = 1.5              # 方向指示線旋轉速度（弧度/秒）
CHARGE_SPEED = 0.8           # 蓄力速度（每秒增加的比例）
MAX_FORCE = 800.0            # 最大力道
MIN_ARROW_LEN = 35.0         # 箭頭最短長度（預設短箭頭）
MAX_ARROW_LEN = 160.0        # 箭頭最長長度（蓄滿時）


def reset_aim_state():
    """重置瞄準與蓄力狀態"""
    global aim_angle, aim_locked, charging, charge_power
    aim_locked = False
    charging = False
    charge_power = 0.0


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


if __name__ == "__main__":
    while not game_start:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = Vector2(pygame.mouse.get_pos())
                x = config.screen_size[0] // 2 - 140
                y = config.screen_size[1] // 2 - 70
                if physics.point_in_rect((x, y, x + 319, y + 85), mouse):
                    game_start = True

        game_manager.blit_start(screen)
        pygame.display.flip()

    game_manager.current_level.init()
    carts = game_manager.current_level.carts

    while not game_end:
        screen.fill(game_manager.current_level.background_color)
        dt = clock.tick(140) / 1000.0  # 轉為秒

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            # --- 滑鼠點擊：鎖定方向（模擬眨眼） ---
            if event.type == pygame.MOUSEBUTTONDOWN:
                if ball.not_moving() and not aim_locked and not charging:
                    aim_locked = True

            # --- 空白鍵按下：開始蓄力（模擬進入專注狀態） ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if ball.not_moving() and aim_locked and not charging:
                        charging = True
                        charge_power = 0.0

            # --- 空白鍵放開：揮桿（模擬從專注轉為放鬆） ---
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    if charging:
                        direction = Vector2(math.cos(aim_angle), math.sin(aim_angle))
                        force = direction * (charge_power * MAX_FORCE)
                        ball.apply_force(force)
                        game_manager.throw_number += 1
                        reset_aim_state()

        # --- 每幀更新：方向旋轉 & 蓄力增長 ---
        if ball.not_moving() and not aim_locked:
            aim_angle += AIM_SPEED * dt  # 方向持續旋轉

        if charging:
            charge_power = min(charge_power + CHARGE_SPEED * dt, 1.0)

        # --- 物理更新 ---
        ball.move(dt * 10)  # 乘以 10 還原原始時間尺度 (原本 tick/100)
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
                # 蓄力中：箭頭隨蓄力由短變長
                arrow_len = MIN_ARROW_LEN + (MAX_ARROW_LEN - MIN_ARROW_LEN) * charge_power
                end_point = ball.pos + direction * arrow_len
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            elif aim_locked:
                # 方向已鎖定，等待按空白鍵：顯示紅色短箭頭
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)
            else:
                # 方向旋轉中：顯示紅色短箭頭
                end_point = ball.pos + direction * MIN_ARROW_LEN
                draw_arrow(screen, (255, 0, 0), ball.pos, end_point, head_size=20, width=7)

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

    pygame.quit()
    exit(0)
