
import cv2
import json
import numpy as np
import chess
import chess.engine
import chess.svg
from PIL import Image
import io
import random
import os
import sys
import cairosvg
import time

# === CONFIG ===
CALIB_JSON = "sqdict.json"
ENGINE_PATH = r"stockfish-windows-x86-64-avx2.exe"
MOVE_THRESHOLD = 10
MIN_CONTOUR_AREA = 200
CAM_INDEX = 0
BOARD_ORIENTATION = "TOP"  # "TOP", "BOTTOM", "SIDE_L", "SIDE_R"

DEBUG_MODE = False  # Tekan 'd' untuk toggle ON/OFF

# === ENGINE ===
if not os.path.exists(ENGINE_PATH):
    print(f"[ERROR] File engine tidak ditemukan: {ENGINE_PATH}")
    sys.exit(1)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
print(f"[INFO] Stockfish dijalankan dari {ENGINE_PATH}")

# === LOAD JSON ===
if not os.path.exists(CALIB_JSON):
    print(f"[ERROR] File kalibrasi tidak ditemukan: {CALIB_JSON}")
    engine.quit()
    sys.exit(1)

with open(CALIB_JSON, "r") as f:
    sq_points = json.load(f)
print(f"[INFO] Memuat {len(sq_points)} kotak dari {CALIB_JSON}")

# === ORIENTATION ===
files = 'abcdefgh'
ranks = '12345678'

def remap_square(square_name: str) -> str:
    f = square_name[0]
    r = square_name[1]
    fi = files.index(f)
    ri = ranks.index(r)
    if BOARD_ORIENTATION == "TOP":
        return square_name
    elif BOARD_ORIENTATION == "BOTTOM":
        return f"{files[7 - fi]}{ranks[7 - ri]}"
    elif BOARD_ORIENTATION == "SIDE_L":
        return f"{files[ri]}{ranks[7 - fi]}"
    elif BOARD_ORIENTATION == "SIDE_R":
        return f"{files[7 - ri]}{ranks[fi]}"
    else:
        return square_name

# === HELPERS ===
def poly_center(pts):
    a = np.array(pts, np.int32)
    M = cv2.moments(a)
    if M["m00"] == 0:
        return int(a[:, 0].mean()), int(a[:, 1].mean())
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def find_square(x, y):
    for sq, pts in sq_points.items():
        poly = np.array(pts, np.int32)
        if cv2.pointPolygonTest(poly, (x, y), False) > 0:
            return remap_square(sq)
    return None

def overlay_poly(frame, poly_pts, color, alpha=0.45):
    overlay = frame.copy()
    pts = np.array(poly_pts, np.int32)
    cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_board_labels(base_frame):
    overlay = base_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for sq, pts in sq_points.items():
        p = np.array(pts, np.int32)
        cv2.polylines(overlay, [p], True, (255, 255, 255), 1)
        if sq == "a1":
            cx, cy = poly_center(pts)
            mapped = remap_square(sq)
            cv2.putText(overlay, mapped, (cx - 12, cy + 5), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return overlay

def pick_top_two_contours(contours):
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    return [c for c in contours_sorted if cv2.contourArea(c) > MIN_CONTOUR_AREA][:2]

def show_board(board, last_move=None):
    svg = chess.svg.board(board=board, lastmove=last_move, coordinates=True, size=450)
    png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(png_data))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Board State", img_cv)
    cv2.waitKey(1)

def draw_contours_debug(frame, contours):
    dbg = frame.copy()
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        cx, cy = x + w // 2, y + int(0.6 * h)
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(dbg, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(dbg, f"A:{int(area)}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return dbg

# === CAMERA ===
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Kamera tidak bisa dibuka.")
    engine.quit()
    sys.exit(1)

board = chess.Board()
ref_frame = None
last_move = None
comp_turn = False
move_history = []

print("[INFO] Tekan 'r' dua kali untuk langkah, 'u'=undo, 'U'=undo 2 langkah, 'd'=toggle debug, 'q'=keluar.")
show_board(board)

try:
    while not board.is_game_over():
        ret, frame_raw = cap.read()
        if not ret:
            continue

        display = draw_board_labels(frame_raw.copy())
        cv2.imshow("Chess Tracker", display)
        key = cv2.waitKey(1) & 0xFF

        # === Toggle debug mode ===
        if key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            state = "ON" if DEBUG_MODE else "OFF"
            print(f"[INFO] Debug mode: {state}")
            if not DEBUG_MODE:
                try:
                    if cv2.getWindowProperty("Diff", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("Diff")
                except cv2.error:
                    pass

                try:
                    if cv2.getWindowProperty("Contours", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("Contours")
                except cv2.error:
                    pass

        # === Rekam langkah pemain (dua kali tekan 'r') ===
        if key == ord('r'):
            if ref_frame is None:
                ref_frame = frame_raw.copy()
                print("[DEBUG] Frame awal disimpan.")
            else:
                print("[DEBUG] Frame akhir direkam, memproses...")
                # versi lebih universal (untuk putih gading dan hitam/coklat)
                g1 = 0.5 * ref_frame[:, :, 2] + 0.4 * ref_frame[:, :, 1] + 0.1 * ref_frame[:, :, 0]
                g2 = 0.5 * frame_raw[:, :, 2] + 0.4 * frame_raw[:, :, 1] + 0.1 * frame_raw[:, :, 0]
                g1 = g1.astype(np.uint8)
                g2 = g2.astype(np.uint8)

                g1 = cv2.GaussianBlur(g1, (5, 5), 0)
                g2 = cv2.GaussianBlur(g2, (5, 5), 0)
                diff = cv2.absdiff(g1, g2)
                diff = cv2.GaussianBlur(diff, (3,3), 0)
                diff = cv2.convertScaleAbs(diff, alpha=1.3, beta=0)  # memperkuat kontras perbedaan
                _, diff_thresh = cv2.threshold(diff, MOVE_THRESHOLD, 255, cv2.THRESH_BINARY)
                diff_m = cv2.dilate(diff_thresh, None, iterations=4)
                diff_m = cv2.erode(diff_m, None, iterations=2)

                if DEBUG_MODE:
                    cv2.imshow("Diff", diff_m)

                contours, _ = cv2.findContours(diff_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                top_contours = pick_top_two_contours(contours)
                detected = set()

                for c in top_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w // 2, y + int(0.6 * h)
                    sq = find_square(cx, cy)
                    if sq:
                        detected.add(sq)
                    if DEBUG_MODE:
                        print(f"[DEBUG] Kontur di ({cx},{cy}) -> {sq}")

                if DEBUG_MODE:
                    dbg = draw_contours_debug(frame_raw, top_contours)
                    cv2.imshow("Contours", dbg)

                print(f"[DEBUG] Kotak terdeteksi: {detected}")

                # === interpretasi langkah ===
                from_sq, to_sq = None, None
                if len(detected) == 2:
                    a, b = list(detected)

                    # gunakan snapshot board sebelum perubahan
                    prev_board = board.copy()

                    # coba kedua urutan (a+b) dan (b+a) — aman jika salah satu bukan UCI
                    mv_ab = None
                    mv_ba = None
                    try:
                        mv_ab = chess.Move.from_uci(a + b)
                    except Exception:
                        mv_ab = None
                    try:
                        mv_ba = chess.Move.from_uci(b + a)
                    except Exception:
                        mv_ba = None

                    legal_ab = (mv_ab in board.legal_moves) if mv_ab else False
                    legal_ba = (mv_ba in board.legal_moves) if mv_ba else False

                    if legal_ab and not legal_ba:
                        from_sq, to_sq = a, b
                    elif legal_ba and not legal_ab:
                        from_sq, to_sq = b, a
                    elif legal_ab and legal_ba:
                        # kedua legal: prefer move yang melakukan capture
                        is_cap_ab = board.is_capture(mv_ab)
                        is_cap_ba = board.is_capture(mv_ba)
                        if is_cap_ab and not is_cap_ba:
                            from_sq, to_sq = a, b
                        elif is_cap_ba and not is_cap_ab:
                            from_sq, to_sq = b, a
                        else:
                            # prefer kotak yang sebelumnya berisi bidak
                            piece_a = prev_board.piece_at(chess.parse_square(a))
                            piece_b = prev_board.piece_at(chess.parse_square(b))
                            if piece_a and not piece_b:
                                from_sq, to_sq = a, b
                            elif piece_b and not piece_a:
                                from_sq, to_sq = b, a
                            else:
                                # fallback lama: urut berdasarkan rank
                                def rank_idx(s): return int(s[1])
                                if board.turn == chess.WHITE:
                                    from_sq, to_sq = sorted([a, b], key=rank_idx)
                                else:
                                    from_sq, to_sq = sorted([a, b], key=rank_idx, reverse=True)
                    else:
                        # tidak ada yang legal — pakai fallback prev_board atau rank
                        piece_a = prev_board.piece_at(chess.parse_square(a))
                        piece_b = prev_board.piece_at(chess.parse_square(b))
                        if piece_a and not piece_b:
                            from_sq, to_sq = a, b
                        elif piece_b and not piece_a:
                            from_sq, to_sq = b, a
                        else:
                            def rank_idx(s): return int(s[1])
                            if board.turn == chess.WHITE:
                                from_sq, to_sq = sorted([a, b], key=rank_idx)
                            else:
                                from_sq, to_sq = sorted([a, b], key=rank_idx, reverse=True)

                elif len(detected) == 1:
                    # hanya satu kotak berubah — coba cara lebih andal untuk cari from_sq
                    to_sq = list(detected)[0]
                    prev_board = board.copy()  # snapshot posisi sebelum langkah
                    piece_now = board.piece_at(chess.parse_square(to_sq))

                    # 1) Jika kotak sekarang terisi, coba cari legal move yang berakhir di sini
                    if piece_now:
                        # filter kandidat yang asalnya memang punya piece pada prev_board
                        candidates = [m for m in board.legal_moves if m.uci()[2:] == to_sq]
                        chosen = None
                        for m in candidates:
                            src = m.uci()[:2]
                            if prev_board.piece_at(chess.parse_square(src)):
                                chosen = m
                                break
                        # jika tidak ketemu yang asalnya terisi, fallback ke first candidate
                        if not chosen and candidates:
                            chosen = candidates[0]
                        if chosen:
                            from_sq = chosen.uci()[:2]
                            to_sq = chosen.uci()[2:]
                    else:
                        # 2) kalau kotak akhir kosong -> kemungkinan bidak pindah dari kotak sekeliling
                        file = to_sq[0]
                        rank = int(to_sq[1])
                        fi = files.index(file)

                        # buat urutan pencarian yang prioritas: vertikal (sesuai giliran), horizontal, diagonal, 2-langkah
                        search_offsets = []

                        if board.turn == chess.WHITE:
                            # prefer datang dari bawah (rank-1), lalu left/right, lalu diagonals, then two-step from rank-2
                            search_offsets += [(0, -1), (-1, 0), (1, 0), (-1, -1), (1, -1), (0, -2)]
                        else:
                            # black moves downward in rank numbers (from higher rank to lower)
                            search_offsets += [(0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1), (0, 2)]

                        # ensure we also consider all orthogonals/diagonals if needed
                        search_offsets += [(-1, 1), (1, 1), (-1, -1), (1, -1)]

                        found = False
                        for df, dr in search_offsets:
                            f_idx = fi + df
                            r_idx = rank + dr
                            if 0 <= f_idx < 8 and 1 <= r_idx <= 8:
                                adj = f"{files[f_idx]}{r_idx}"
                                if prev_board.piece_at(chess.parse_square(adj)):
                                    # verify move adj -> to_sq is legal
                                    try_mv = chess.Move.from_uci(adj + to_sq)
                                    if try_mv in board.legal_moves:
                                        from_sq = adj
                                        found = True
                                        break
                        # 3) jika belum juga ketemu, fallback: cari *any* neighbor yang punya piece (tanpa cek legal)
                        if not found:
                            for df in (-1, 0, 1):
                                for dr in (-1, 0, 1):
                                    if df == 0 and dr == 0:
                                        continue
                                    f_idx = fi + df
                                    r_idx = rank + dr
                                    if 0 <= f_idx < 8 and 1 <= r_idx <= 8:
                                        adj = f"{files[f_idx]}{r_idx}"
                                        if prev_board.piece_at(chess.parse_square(adj)):
                                            # jika move adj->to_sq legal gunakan, kalau tidak, tetap simpan adj sebagai last-resort
                                            try_mv = None
                                            try:
                                                try_mv = chess.Move.from_uci(adj + to_sq)
                                            except Exception:
                                                try_mv = None
                                            if try_mv and try_mv in board.legal_moves:
                                                from_sq = adj
                                                found = True
                                                break
                                            if not from_sq:
                                                from_sq = adj
                                if found:
                                    break
                        # jika masih None, from_sq tetap None dan akan dianggap invalid

                else:
                    print("[WARN] Deteksi tidak valid.")

                # === eksekusi langkah ===
                if from_sq and to_sq:
                    move = from_sq + to_sq
                    try:
                        mv = chess.Move.from_uci(move)
                        if mv in board.legal_moves:
                            board.push(mv)
                            move_history.append(mv)
                            last_move = mv
                            print(f"[YOU] Kamu main: {move}")
                            show_board(board, last_move)

                            # === HIGHLIGHT langkah pemain (FROM hijau, TO merah) ===
                            try:
                                frame_high = overlay_poly(frame_raw.copy(), sq_points[from_sq], (0, 255, 0), 0.5)
                                frame_high = overlay_poly(frame_high, sq_points[to_sq], (0, 0, 255), 0.5)
                                frame_high = draw_board_labels(frame_high)
                                cv2.imshow("Chess Tracker", frame_high)
                                cv2.waitKey(700)  # tampilkan sebentar
                            except Exception as e:
                                # jangan crash kalau key tidak ada di sq_points (safety)
                                if DEBUG_MODE:
                                    print(f"[DEBUG] Gagal highlight pemain: {e}")

                            comp_turn = True
                        else:
                            print(f"[!] Invalid move: {move}")
                    except Exception as e:
                        print(f"[!] Error interpretasi langkah: {e}")

                ref_frame = None

        # === Undo 1 langkah ===
        if key == ord('u'):
            if move_history:
                mv = move_history.pop()
                board.pop()
                print(f"[UNDO] Menghapus langkah terakhir: {mv}")
                show_board(board)
            else:
                print("[INFO] Tidak ada langkah untuk di-undo.")

        # === Undo 2 langkah ===
        if key == ord('U'):
            if len(move_history) >= 2:
                mv2 = move_history.pop()
                mv1 = move_history.pop()
                board.pop()
                board.pop()
                print(f"[UNDO] Menghapus 2 langkah terakhir: {mv1}, {mv2}")
                show_board(board)
            else:
                print("[INFO] Tidak cukup langkah untuk undo 2 kali.")

        # === COMPUTER TURN ===
        if comp_turn:
            result = engine.play(board, chess.engine.Limit(time=random.uniform(0.4, 0.9)))
            mv = result.move
            board.push(mv)
            move_history.append(mv)
            last_move = mv
            print(f"[AI] Komputer main: {mv.uci()}")
            show_board(board, last_move)

            # === HIGHLIGHT langkah AI (FROM kuning, TO oranye) ===
            try:
                move_str = mv.uci()
                frame_ai = overlay_poly(frame_raw.copy(), sq_points[move_str[:2]], (0, 255, 255), 0.45)  # kuning
                frame_ai = overlay_poly(frame_ai, sq_points[move_str[2:]], (0, 165, 255), 0.45)  # oranye-ish
                frame_ai = draw_board_labels(frame_ai)
                cv2.imshow("Chess Tracker", frame_ai)
                cv2.waitKey(900)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[DEBUG] Gagal highlight AI: {e}")

            comp_turn = False

        if key == ord('q'):
            print("[INFO] Keluar.")
            break

    print("[INFO] Permainan selesai.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    engine.quit()
