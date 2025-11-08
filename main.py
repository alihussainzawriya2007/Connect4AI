import math
import random
import sys
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pygame

# Try to use gfxdraw; fall back gracefully if unavailable
try:
    import pygame.gfxdraw as gfx
    HAS_GFX = True
except Exception:
    gfx = None
    HAS_GFX = False

# Try to import python-docx for Word output
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    Document = None
    HAS_DOCX = False

# --------------------------
# Config & Constants
# --------------------------
BLUE   = (0, 0, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
YEL    = (255, 255, 0)
WHITE  = (255, 255, 255)
GREY   = (120, 120, 120)

ROW_COUNT     = 6
COLUMN_COUNT  = 7
WINDOW_LENGTH = 4

PLAYER = 0      # red
AI     = 1      # yellow (can be human in PvP)

EMPTY        = 0
PLAYER_PIECE = 1
AI_PIECE     = 2

# Depth slider (drag in the top bar)
SLIDER_MIN = 1
SLIDER_MAX = 9
SLIDER_W   = 220

WIN_SCORE = 10_000_000
LOSE_SCORE= -10_000_000
DRAW_SCORE= 0

# UI
SQUARESIZE = 100
RADIUS     = SQUARESIZE // 2 - 6
WIDTH      = COLUMN_COUNT * SQUARESIZE
TOPBAR_H   = SQUARESIZE
HEIGHT     = TOPBAR_H + ROW_COUNT * SQUARESIZE
SIZE       = (WIDTH, HEIGHT)

# --------------------------
# Board helpers
# --------------------------
def create_board() -> np.ndarray:
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)

def drop_piece(board: np.ndarray, row: int, col: int, piece: int) -> None:
    board[row, col] = piece

def is_valid_location(board: np.ndarray, col: int) -> bool:
    return board[ROW_COUNT - 1, col] == EMPTY

def get_next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    for r in range(ROW_COUNT):
        if board[r, col] == EMPTY:
            return r
    return None

def get_valid_locations(board: np.ndarray) -> List[int]:
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

def print_board(board: np.ndarray) -> None:
    print(np.flip(board, 0))

# --------------------------
# Game state checks
# --------------------------
def winning_move(board: np.ndarray, piece: int) -> bool:
    # Horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (board[r, c] == piece and board[r, c+1] == piece
                and board[r, c+2] == piece and board[r, c+3] == piece):
                return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (board[r, c] == piece and board[r+1, c] == piece
                and board[r+2, c] == piece and board[r+3, c] == piece):
                return True
    # Diagonal \
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (board[r, c] == piece and board[r+1, c+1] == piece
                and board[r+2, c+2] == piece and board[r+3, c+3] == piece):
                return True
    # Diagonal /
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (board[r, c] == piece and board[r-1, c+1] == piece
                and board[r-2, c+2] == piece and board[r-3, c+3] == piece):
                return True
    return False

def is_terminal_node(board: np.ndarray) -> bool:
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# --------------------------
# Heuristic scoring
# --------------------------
def evaluate_window(window: List[int], piece: int) -> int:
    score = 0
    opp = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 7           # a bit stronger than before
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    # punish letting opponent make a 3
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 8           # stronger block incentive
    return score

def score_position(board: np.ndarray, piece: int) -> int:
    score = 0
    # center control: weight center column a bit more
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    score += center_array.count(piece) * 4

    # rows
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c+WINDOW_LENGTH], piece)

    # cols
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r+WINDOW_LENGTH], piece)

    # positive diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([int(board[r+i, c+i]) for i in range(WINDOW_LENGTH)], piece)

    # negative diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window([int(board[r+3-i, c+i]) for i in range(WINDOW_LENGTH)], piece)
    return score

# --------------------------
# AI helpers: tactics + ordering + safety
# --------------------------
def simulate_drop(board: np.ndarray, col: int, piece: int):
    """Return (row, board_copy) after dropping piece; or (None, None) if invalid."""
    if not is_valid_location(board, col):
        return None, None
    row = get_next_open_row(board, col)
    if row is None:
        return None, None
    b_copy = board.copy()
    drop_piece(b_copy, row, col, piece)
    return row, b_copy

def immediate_wins(board: np.ndarray, piece: int) -> List[int]:
    wins = []
    for col in get_valid_locations(board):
        row, b2 = simulate_drop(board, col, piece)
        if b2 is not None and winning_move(b2, piece):
            wins.append(col)
    return wins

def is_unsafe_for_ai(board: np.ndarray, col: int) -> bool:
    """True if after AI plays at col, the player gets an immediate winning reply."""
    row, b2 = simulate_drop(board, col, AI_PIECE)
    if b2 is None:
        return True
    # if AI already wins, it's fine (not unsafe)
    if winning_move(b2, AI_PIECE):
        return False
    # can player win next move?
    return any(winning_move(simulate_drop(b2, oc, PLAYER_PIECE)[1], PLAYER_PIECE)
               for oc in get_valid_locations(b2))

def order_moves(board: np.ndarray, maximizing: bool) -> List[int]:
    """Sort moves by heuristic lookahead + center preference for strong pruning."""
    valid = get_valid_locations(board)
    center = COLUMN_COUNT // 2
    scored = []
    if maximizing:
        for col in valid:
            _, b2 = simulate_drop(board, col, AI_PIECE)
            if b2 is None: 
                continue
            s = score_position(b2, AI_PIECE)
            # small center bias
            s += (3 - abs(col - center))
            scored.append((s, col))
        scored.sort(reverse=True)  # best first
    else:
        # minimizing: prefer moves that are worst for AI (best for player)
        for col in valid:
            _, b2 = simulate_drop(board, col, PLAYER_PIECE)
            if b2 is None: 
                continue
            s = score_position(b2, AI_PIECE)
            s += (3 - abs(col - center))
            scored.append((s, col))
        scored.sort()  # worst for AI first
    return [c for _, c in scored]

# Transposition table (simple exact caching)
TRANSPOS: dict = {}

# --------------------------
# Minimax (with improved ordering + cache)
# --------------------------
def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[Optional[int], int]:
    key = (board.tobytes(), depth, maximizing)
    if key in TRANSPOS:
        return TRANSPOS[key]

    terminal = is_terminal_node(board)
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI_PIECE):       val = WIN_SCORE
            elif winning_move(board, PLAYER_PIECE): val = LOSE_SCORE
            else:                                    val = DRAW_SCORE
            TRANSPOS[key] = (None, val)
            return TRANSPOS[key]
        val = score_position(board, AI_PIECE)
        TRANSPOS[key] = (None, val)
        return TRANSPOS[key]

    best_col = None
    if maximizing:
        value = -math.inf
        for col in order_moves(board, True):
            row, b2 = simulate_drop(board, col, AI_PIECE)
            if b2 is None: 
                continue
            _, new_score = minimax(b2, depth-1, alpha, beta, False)
            if new_score > value:
                value, best_col = new_score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        result = (best_col, int(value))
    else:
        value = math.inf
        for col in order_moves(board, False):
            row, b2 = simulate_drop(board, col, PLAYER_PIECE)
            if b2 is None: 
                continue
            _, new_score = minimax(b2, depth-1, alpha, beta, True)
            if new_score < value:
                value, best_col = new_score, col
            beta = min(beta, value)
            if alpha >= beta:
                break
        result = (best_col, int(value))

    TRANSPOS[key] = result
    return result

# --------------------------
# Drawing helpers (centers, slider, glow, controls)
# --------------------------
def rc_to_screen(col: int, row: int) -> Tuple[int, int]:
    # board row 0 is bottom -> flip vertically for drawing
    x = int(col * SQUARESIZE + SQUARESIZE / 2)
    y = HEIGHT - int(row * SQUARESIZE + SQUARESIZE / 2)
    return x, y

def aa_ring(surface: pygame.Surface, center: Tuple[int,int], color: Tuple[int,int,int], radius: int, width: int=3):
    if HAS_GFX:
        for w in range(width):
            gfx.aacircle(surface, center[0], center[1], radius + w, color)
            gfx.circle(surface,  center[0], center[1], radius + w, color)
    else:
        pygame.draw.circle(surface, color, center, radius + width//2, width)

def glow_disc(surface: pygame.Surface, center: Tuple[int,int], base_color: Tuple[int,int,int]):
    glow = pygame.Surface((SQUARESIZE*2, SQUARESIZE*2), pygame.SRCALPHA)
    gx, gy = glow.get_width()//2, glow.get_height()//2
    for i in range(6, 0, -1):
        r = RADIUS + 3 + i*2
        alpha = 22 + i*6
        pygame.draw.circle(glow, (*base_color, alpha), (gx, gy), r)
    surface.blit(glow, glow.get_rect(center=center))
    aa_ring(surface, center, base_color, RADIUS + 4, width=3)

# Slider geometry
def slider_rects(ai_depth: int) -> Tuple[pygame.Rect, Tuple[int,int], int]:
    w = SLIDER_W
    x = WIDTH - w - 20
    y = TOPBAR_H - 28
    track = pygame.Rect(x, y, w, 8)
    handle_r = 10
    ratio = (ai_depth - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN)
    ratio = max(0.0, min(1.0, ratio))
    hx = x + int(ratio * w)
    hy = y + track.height // 2
    return track, (hx, hy), handle_r

def x_to_depth(x: int, track: pygame.Rect) -> int:
    ratio = (x - track.left) / max(1, track.width)
    val = SLIDER_MIN + round(max(0.0, min(1.0, ratio)) * (SLIDER_MAX - SLIDER_MIN))
    return max(SLIDER_MIN, min(SLIDER_MAX, val))

def draw_slider(surface: pygame.Surface, ai_depth: int, active: bool = True):
    track, (hx, hy), hr = slider_rects(ai_depth)
    # track + fill
    pygame.draw.rect(surface, (60,60,60) if active else (40,40,40), track, border_radius=4)
    fill = pygame.Rect(track.left, track.top, max(0, hx - track.left), track.height)
    pygame.draw.rect(surface, (200,200,200) if active else (90,90,90), fill, border_radius=4)
    # handle (fixed signatures)
    handle_col = WHITE if active else GREY
    if HAS_GFX:
        gfx.filled_circle(surface, hx, hy, hr, handle_col)
        gfx.aacircle(surface, hx, hy, hr, (220,220,220))
    else:
        pygame.draw.circle(surface, handle_col, (hx, hy), hr)
    # label
    font = pygame.font.SysFont("monospace", 18, bold=True)
    label_col = WHITE if active else GREY
    surface.blit(font.render(f"Depth: {ai_depth}", True, label_col), (track.left - 110, track.top - 4))

# Control buttons (mode + first)
def draw_button(surface: pygame.Surface, rect: pygame.Rect, text: str, enabled: bool=True):
    col = WHITE if enabled else GREY
    pygame.draw.rect(surface, (25,25,25), rect, border_radius=8)
    pygame.draw.rect(surface, col, rect, width=2, border_radius=8)
    font = pygame.font.SysFont("monospace", 20, bold=True)
    tw, th = font.size(text)
    surface.blit(font.render(text, True, col), (rect.x + (rect.w - tw)//2, rect.y + (rect.h - th)//2))

def get_control_rects():
    mode_rect  = pygame.Rect(12, TOPBAR_H - 36, 180, 28)
    first_rect = pygame.Rect(200, TOPBAR_H - 36, 160, 28)
    return mode_rect, first_rect

# --------------------------
# UI drawing
# --------------------------
def draw_board(surface: pygame.Surface,
               board: np.ndarray,
               last_user: Optional[Tuple[int,int]],
               last_ai: Optional[Tuple[int,int]],
               msg: str,
               ai_depth: int,
               mode: str,
               first: str) -> None:
    # grid
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(surface, BLUE, (c*SQUARESIZE, r*SQUARESIZE+TOPBAR_H, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(surface, BLACK,
                (int(c*SQUARESIZE + SQUARESIZE/2), int(r*SQUARESIZE + TOPBAR_H + SQUARESIZE/2)),
                RADIUS)

    # pieces
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r, c] == PLAYER_PIECE:
                pygame.draw.circle(surface, RED, rc_to_screen(c, r), RADIUS)
            elif board[r, c] == AI_PIECE:
                pygame.draw.circle(surface, YEL, rc_to_screen(c, r), RADIUS)

    # glow for last moves
    if last_user:
        glow_disc(surface, rc_to_screen(last_user[1], last_user[0]), RED)
    if last_ai:
        glow_disc(surface, rc_to_screen(last_ai[1], last_ai[0]), YEL)

    # top bar
    pygame.draw.rect(surface, BLACK, (0, 0, WIDTH, TOPBAR_H))
    font = pygame.font.SysFont("monospace", 32, bold=True)
    surface.blit(font.render(msg, True, WHITE if "Your" in msg or "New" in msg else YEL), (12, 14))

    # controls
    mode_rect, first_rect = get_control_rects()
    draw_button(surface, mode_rect, f"Mode: {mode}", True)
    first_enabled = (mode == "PvE")
    draw_button(surface, first_rect, f"First: {first}", first_enabled)

    # slider
    draw_slider(surface, ai_depth, active=(mode == "PvE"))

    pygame.display.update()

def show_message(surface: pygame.Surface, text: str, ai_depth: int, mode: str, first: str) -> None:
    pygame.draw.rect(surface, BLACK, (0, 0, WIDTH, TOPBAR_H))
    font = pygame.font.SysFont("monospace", 48, bold=True)
    label = font.render(text, True, YEL if "AI" in text else WHITE)
    rect = label.get_rect(midleft=(12, TOPBAR_H//2))
    surface.blit(label, rect)
    # controls
    mode_rect, first_rect = get_control_rects()
    draw_button(surface, mode_rect, f"Mode: {mode}", True)
    draw_button(surface, first_rect, f"First: {first}", mode == "PvE")
    draw_slider(surface, ai_depth, active=(mode == "PvE"))
    pygame.display.update()

# --------------------------
# Drop animation
# --------------------------
def draw_filled_disc(surface, x, y, color):
    if HAS_GFX:
        gfx.filled_circle(surface, x, y, RADIUS, color)
        gfx.aacircle(surface, x, y, RADIUS, color)
    else:
        pygame.draw.circle(surface, color, (x, y), RADIUS)

def animate_drop(screen: pygame.Surface,
                 board: np.ndarray,
                 col: int,
                 row: int,
                 color: Tuple[int,int,int],
                 last_user: Optional[Tuple[int,int]],
                 last_ai: Optional[Tuple[int,int]],
                 msg: str,
                 ai_depth: int,
                 mode: str,
                 first: str):
    clock = pygame.time.Clock()
    sx, sy = col * SQUARESIZE + SQUARESIZE//2, TOPBAR_H//2
    ex, ey = rc_to_screen(col, row)
    y = sy
    v = 0.0
    g = 1.25
    while y < ey - 1:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        v = min(v + g, 28)
        y = min(y + v, ey)
        draw_board(screen, board, last_user, last_ai, msg, ai_depth, mode, first)
        draw_filled_disc(screen, sx, int(y), color)
        pygame.display.update()
        clock.tick(120)

# --------------------------
# Word logging (Pandas -> DOCX)
# --------------------------
def save_game_result(board: np.ndarray, result: str, docx_path: str = "Connect4_Game_Logs.docx") -> None:
    """
    Append the final board to a Word document using Pandas.
    - Adds a table with symbols (R=Player, Y=AI, ·=empty), top row first.
    - Also adds a list-of-lists numeric matrix for reference.
    Falls back to CSV if python-docx is not installed.
    """
    df_num = pd.DataFrame(
        np.flip(board, 0).astype(int),
        columns=[f"C{c+1}" for c in range(COLUMN_COUNT)]
    )
    df_sym = df_num.replace({0: "·", 1: "R", 2: "Y"})
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not HAS_DOCX:
        csv_path = "Connect4_Game_Logs.csv"
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"# {stamp} — {result}\n")
        df_num.to_csv(csv_path, mode="a", index=False)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("\n")
        print("[INFO] python-docx not found; wrote to Connect4_Game_Logs.csv instead.")
        return

    doc = Document(docx_path) if os.path.exists(docx_path) else Document()
    doc.add_heading(f"Game Result — {result}", level=1)
    doc.add_paragraph(stamp)
    doc.add_paragraph("Legend: R = Player (Red), Y = AI/Player2 (Yellow), · = empty")

    rows, cols = df_sym.shape
    table = doc.add_table(rows=rows + 1, cols=cols + 1)
    hdr = table.rows[0].cells
    hdr[0].text = ""
    for j, cname in enumerate(df_sym.columns, start=1):
        hdr[j].text = cname
    for i in range(rows):
        cells = table.rows[i + 1].cells
        cells[0].text = f"R{ROW_COUNT - i}"
        for j in range(cols):
            cells[j + 1].text = str(df_sym.iat[i, j])

    doc.add_paragraph("Numeric matrix (top row first):")
    doc.add_paragraph(str(df_num.values.tolist()))
    doc.add_paragraph("")  # spacer

    doc.save(docx_path)
    print(f"[INFO] Saved board to {docx_path}")

# --------------------------
# Game loop
# --------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Connect Four (Player vs AI / PvP)")
    screen = pygame.display.set_mode(SIZE)
    clock  = pygame.time.Clock()

    board = create_board()
    last_user: Optional[Tuple[int,int]] = None  # (row, col) last red
    last_bot:  Optional[Tuple[int,int]] = None  # (row, col) last yellow

    mode = "PvE"      # "PvE" (Human vs AI)  or "PvP" (Human vs Human)
    first = "Human"   # In PvE: "Human" or "AI". In PvP: Red starts.
    ai_depth = 5      # default
    slider_drag = False

    # who starts?
    turn = (AI if (mode == "PvE" and first == "AI") else PLAYER)
    game_over = False

    # initialize hover to actual mouse x so first click works even without moving
    hover_x = pygame.mouse.get_pos()[0]

    draw_board(screen, board, last_user, last_bot,
               "Your turn — click a column" if turn == PLAYER else ("AI thinking…" if mode == "PvE" else "Yellow's turn"),
               ai_depth, mode, first)

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    # reset everything, also clear TT to avoid stale caches
                    board[:] = 0
                    TRANSPOS.clear()
                    last_user = last_bot = None
                    game_over = False
                    turn = (AI if (mode == "PvE" and first == "AI") else PLAYER)
                    draw_board(screen, board, last_user, last_bot,
                               "New game! Your turn" if turn == PLAYER else ("AI thinking…" if mode == "PvE" else "Yellow's turn"),
                               ai_depth, mode, first)

            # Top bar interactions (slider + buttons)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my < TOPBAR_H:
                    # Buttons
                    mode_rect, first_rect = get_control_rects()
                    if mode_rect.collidepoint(mx, my):
                        mode = "PvP" if mode == "PvE" else "PvE"
                        board[:] = 0
                        TRANSPOS.clear()
                        last_user = last_bot = None
                        game_over = False
                        turn = (AI if (mode == "PvE" and first == "AI") else PLAYER)
                        if mode == "PvP":
                            turn = PLAYER
                        draw_board(screen, board, last_user, last_bot,
                                   "New game! Your turn" if turn == PLAYER else ("AI thinking…" if mode == "PvE" else "Yellow's turn"),
                                   ai_depth, mode, first)
                        continue
                    if first_rect.collidepoint(mx, my) and mode == "PvE":
                        first = "AI" if first == "Human" else "Human"
                        board[:] = 0
                        TRANSPOS.clear()
                        last_user = last_bot = None
                        game_over = False
                        turn = (AI if first == "AI" else PLAYER)
                        draw_board(screen, board, last_user, last_bot,
                                   "New game! Your turn" if turn == PLAYER else "AI thinking…",
                                   ai_depth, mode, first)
                        continue

                    # Slider
                    track, (hx, hy), hr = slider_rects(ai_depth)
                    if mode == "PvE" and (track.inflate(20, 16).collidepoint(mx, my) or (mx - hx)*2 + (my - hy)2 <= (hr+6)*2):
                        slider_drag = True
                        ai_depth = x_to_depth(mx, track)
                        TRANSPOS.clear()  # new depth -> clear cache
                        draw_board(screen, board, last_user, last_bot, "Depth changed", ai_depth, mode, first)
                        continue  # don't treat as board click

            if event.type == pygame.MOUSEBUTTONUP:
                slider_drag = False

            if event.type == pygame.MOUSEMOTION and slider_drag and mode == "PvE":
                mx, my = event.pos
                track, _, _ = slider_rects(ai_depth)
                ai_depth = x_to_depth(mx, track)
                draw_board(screen, board, last_user, last_bot, "Depth changed", ai_depth, mode, first)
                continue

            if game_over:
                continue

            # Hover preview in top bar
            if event.type == pygame.MOUSEMOTION:
                hover_x = max(0, min(event.pos[0], WIDTH))
                if event.pos[1] < TOPBAR_H:
                    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, TOPBAR_H))
                    pygame.draw.circle(screen, RED if turn == PLAYER else YEL, (hover_x, TOPBAR_H // 2), RADIUS)
                    # controls & slider redraw
                    mode_rect, first_rect = get_control_rects()
                    draw_button(screen, mode_rect, f"Mode: {mode}", True)
                    draw_button(screen, first_rect, f"First: {first}", mode == "PvE")
                    draw_slider(screen, ai_depth, active=(mode == "PvE"))
                    pygame.display.update()

            # Human click on board
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my < TOPBAR_H:
                    continue  # click on top bar

                col = mx // SQUARESIZE

                if not is_valid_location(board, col):
                    show_message(screen, "Column full. Try another.", ai_depth, mode, first)
                    continue

                row = get_next_open_row(board, col)
                if row is None:
                    continue

                # Decide piece/color by whose turn (PvP or PvE-human)
                piece_color = RED if turn == PLAYER else YEL
                piece_code  = PLAYER_PIECE if turn == PLAYER else AI_PIECE

                animate_drop(screen, board, col, row, piece_color, last_user, last_bot,
                             "Your move…" if turn == PLAYER else ("Yellow move…" if mode=="PvP" else "AI thinking…"),
                             ai_depth, mode, first)
                drop_piece(board, row, col, piece_code)

                if turn == PLAYER:
                    last_user = (row, col)
                else:
                    last_bot = (row, col)

                print_board(board)
                draw_board(screen, board, last_user, last_bot, "…" , ai_depth, mode, first)

                # Win/Draw check
                if winning_move(board, piece_code):
                    if mode == "PvE":
                        if turn == PLAYER:
                            show_message(screen, "You win! 🎉", ai_depth, mode, first)
                            save_game_result(board, "Player (RED) wins vs AI")
                        else:
                            show_message(screen, "AI wins!", ai_depth, mode, first)
                            save_game_result(board, "AI (YELLOW) wins vs Player")
                    else:
                        # PvP: announce by color
                        msg = "Red wins! 🎉" if turn == PLAYER else "Yellow wins!"
                        show_message(screen, msg, ai_depth, mode, first)
                        save_game_result(board, msg)
                    game_over = True
                    continue

                if len(get_valid_locations(board)) == 0:
                    show_message(screen, "Draw!", ai_depth, mode, first)
                    save_game_result(board, "Draw")
                    game_over = True
                    continue

                # Next turn
                if mode == "PvE":
                    if turn == PLAYER:
                        turn = AI
                        show_message(screen, "AI thinking…", ai_depth, mode, first)
                    else:
                        turn = PLAYER
                        show_message(screen, "Your turn", ai_depth, mode, first)
                else:
                    # PvP: toggle between players
                    turn = PLAYER if turn == AI else AI
                    who = "Red's turn" if turn == PLAYER else "Yellow's turn"
                    show_message(screen, who, ai_depth, mode, first)

        # --------------------------
        # AI move (only in PvE) with smart tactics
        # --------------------------
        if not game_over and mode == "PvE" and turn == AI:
            # 1) take instant win
            wins = immediate_wins(board, AI_PIECE)
            if wins:
                col = max(wins, key=lambda c: -abs(c - COLUMN_COUNT//2))
            else:
                # 2) block player's instant win
                opp_wins = immediate_wins(board, PLAYER_PIECE)
                if opp_wins:
                    col = max(opp_wins, key=lambda c: -abs(c - COLUMN_COUNT//2))
                else:
                    # 3) prefer safe moves that don't allow immediate reply win
                    valids = get_valid_locations(board)
                    safe_moves = [c for c in valids if not is_unsafe_for_ai(board, c)]
                    candidate_moves = safe_moves if safe_moves else valids

                    # 4) strong search with ordering + cache
                    target_ms = 150 + 110 * ai_depth
                    start = pygame.time.get_ticks()
                    # Try best-ordered candidate as a hint before full minimax (move ordering)
                    ordered = sorted(candidate_moves,
                                     key=lambda c: score_position(simulate_drop(board, c, AI_PIECE)[1], AI_PIECE) if simulate_drop(board, c, AI_PIECE)[1] is not None else -10_000,
                                     reverse=True)
                    # kick alpha-beta from the (likely) best
                    col_guess = ordered[0] if ordered else None
                    # full minimax from position
                    col_mm, _ = minimax(board, ai_depth, -math.inf, math.inf, True)
                    col = col_mm if col_mm is not None else (col_guess if col_guess is not None else (ordered_valid := get_valid_locations(board))[0])

                    compute_ms = pygame.time.get_ticks() - start
                    wait_ms = max(0, target_ms - compute_ms)
                    if wait_ms > 0:
                        pygame.time.wait(wait_ms)

            if col is None:
                show_message(screen, "Draw!", ai_depth, mode, first)
                save_game_result(board, "Draw")
                game_over = True
            else:
                row = get_next_open_row(board, col)
                if row is not None:
                    animate_drop(screen, board, col, row, YEL, last_user, last_bot, "AI thinking…", ai_depth, mode, first)
                    drop_piece(board, row, col, AI_PIECE)
                    last_bot = (row, col)
                    print_board(board)
                    draw_board(screen, board, last_user, last_bot, "AI moved", ai_depth, mode, first)

                    if winning_move(board, AI_PIECE):
                        show_message(screen, "AI wins!", ai_depth, mode, first)
                        save_game_result(board, "AI (YELLOW) wins vs Player")
                        game_over = True
                    elif len(get_valid_locations(board)) == 0:
                        show_message(screen, "Draw!", ai_depth, mode, first)
                        save_game_result(board, "Draw")
                        game_over = True
                    else:
                        turn = PLAYER
                        show_message(screen, "Your turn", ai_depth, mode, first)

if _name_ == "_main_":
    main()
