import cv2
import mediapipe as mp
import vizdoom as vzd
import os
import threading
import time
import ctypes
import sys

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Helper pra resolução da tela (posicionamento webcam e game)
def get_screen_size() -> tuple[int, int]:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        try:
            w = ctypes.windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
            h = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

MODEL_PATH = "pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# Thresholds p/ precisão
EXTENSION_THRESHOLD = 0.08
CAM_DEAD_ZONE = 0.14
PIP_MCP_POINTING = 0.06
POSE_VIZ_THRESHOLD = 0.5

# Marcação dos pontos dedo indicador e ombro e pulso direito
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
RIGHT_SHOULDER = 12
RIGHT_WRIST = 16

# Threading
_latest_result = None
_result_lock = threading.Lock()

_latest_hand_result = None
_hand_result_lock = threading.Lock()

# Forward->stop timeout
_last_state = "STOP"
_state_change_time = None

# Trigger
_last_trigger_time = 0
TRIGGER_COOLDOWN = 0.35
TRIGGER_STATE = ""
PIP_MCP_THRESHOLD = 0.10

# Tracking
_show_landmarks = False

def _on_result(result, output_image, timestamp_ms: int):
    global _latest_result
    with _result_lock:
        _latest_result = result

def _on_hand_result(result, output_image, timestamp_ms: int):
    global _latest_hand_result
    with _hand_result_lock:
        _latest_hand_result = result

def get_latest_result():
    with _result_lock:
        return _latest_result

def get_latest_hand_result():
    with _hand_result_lock:
        return _latest_hand_result
    

# Verificações para o gatilho/disparo
def _is_pointing(lm) -> bool:
    return abs(lm[INDEX_PIP].x - lm[INDEX_MCP].x) < PIP_MCP_POINTING

def detect_trigger_pull(hand_result, wrist_x: float, movement_state: str) -> bool:

    global _last_trigger_time, TRIGGER_STATE
    current_time = time.time()
    
    if not hand_result or not hand_result.hand_landmarks:
        TRIGGER_STATE = "READY"
        return False
    
    # cooldown
    if current_time - _last_trigger_time < TRIGGER_COOLDOWN:
        return False
    
    lm = hand_result.hand_landmarks[0]
    
    # Caso STOP
    if movement_state == "STOP":
        tip_y = lm[INDEX_TIP].y
        pip_y = lm[INDEX_PIP].y
        if tip_y > pip_y:  # tip abaixo do pip
            _last_trigger_time = current_time
            TRIGGER_STATE = "PULLED_TRIGGER_S"
            return True
        TRIGGER_STATE = "STOPPED"
        return False
    
    mcp_x = lm[INDEX_MCP].x
    pip_x = lm[INDEX_PIP].x
    dip_x = lm[INDEX_DIP].x
    tip_x = lm[INDEX_TIP].x
    
    # Mais pontos no centro -> CENTER
    center_points = sum(1 for x in [mcp_x, pip_x, dip_x, tip_x] if 0.4 <= x <= 0.6)
    
    if center_points >= 2:
        if _is_pointing(lm):
            # mcp---pip
            distance = abs(pip_x - mcp_x)
            if distance > PIP_MCP_THRESHOLD:
                _last_trigger_time = current_time
                TRIGGER_STATE = "PULLED_TRIGGER_C"
                return True
        TRIGGER_STATE = "CENTER"
        return False
    
    if tip_x >= 0.6:
        if tip_x < pip_x:
            _last_trigger_time = current_time
            TRIGGER_STATE = "PULLED_TRIGGER_L"
            return True
        TRIGGER_STATE = "LEFT"
        return False
    
    elif tip_x <= 0.4:
        if tip_x > pip_x:
            _last_trigger_time = current_time
            TRIGGER_STATE = "PULLED_TRIGGER_R"
            return True
        TRIGGER_STATE = "RIGHT"
        return False

# Verificações para movimento e direção
def compute_arm_extension(shoulder_y: float, tip_y: float) -> float:
    return float(tip_y - shoulder_y)

def classify_arm_state(extension: float) -> str:
    if extension >= EXTENSION_THRESHOLD:
        return "FORWARD"
    return "STOP"

def classify_turn(wrist_x: float) -> tuple[str, float]:

    x = 1.0 - wrist_x # Display flipado
    center = 0.5
    
    if x < center - CAM_DEAD_ZONE:
        # Esquerda - velocity aumenta com distância do centro
        velocity = min(1.0, (center - CAM_DEAD_ZONE - x) / (center - CAM_DEAD_ZONE))
        return "TURN_LEFT", velocity
    elif x > center + CAM_DEAD_ZONE:
        # Direita - velocity aumenta com distância do centro
        velocity = min(1.0, (x - (center + CAM_DEAD_ZONE)) / (1.0 - center - CAM_DEAD_ZONE))
        return "TURN_RIGHT", velocity
    
    return "STRAIGHT", 0.0

def draw_hand_landmarks(frame, hand_result, movement_state: str, turn_state: tuple[str, float]):

    if not hand_result or not hand_result.hand_landmarks:
        return frame
    
    lm = hand_result.hand_landmarks[0]
    h, w = frame.shape[:2]
    
    INDEX_CONNECTIONS = [(0, 5), (5, 6), (6, 7), (7, 8)]
    
    for start, end in INDEX_CONNECTIONS:
        start_pos = lm[start]
        end_pos = lm[end]
        x1, y1 = int((1.0 - start_pos.x) * w), int(start_pos.y * h)
        x2, y2 = int((1.0 - end_pos.x) * w), int(end_pos.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2) 
    
    for i in [0, 5, 6, 7, 8]:
        landmark = lm[i]
        x = int((1.0 - landmark.x) * w)
        y = int(landmark.y * h)
        
        if i == 8:
            color = (0, 0, 255) 
        else:
            color = (0, 255, 0)
        
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    turn_direction, turn_velocity = turn_state
    
    color = (0, 255, 0)
    cv2.putText(frame, f"Mov: {movement_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Dir: {turn_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Trig: {TRIGGER_STATE}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # X e Y
    axis_color = (200, 200, 200) 
    tick_length = 5
    text_color = (200, 200, 200)
    
    for i in range(11): 
        x_normalized = i / 10.0
        x_pixel = int(x_normalized * w)
        y_pos = h - 15
        # tick
        cv2.line(frame, (x_pixel, h - 2), (x_pixel, h - tick_length - 2), axis_color, 1)
        # label
        cv2.putText(frame, f"{x_normalized:.1f}", (x_pixel - 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    
    for i in range(11):
        y_normalized = i / 10.0
        y_pixel = int(y_normalized * h)
        x_pos = 20
        cv2.line(frame, (0, y_pixel), (tick_length, y_pixel), axis_color, 1)
        cv2.putText(frame, f"{y_normalized:.1f}", (x_pos, y_pixel + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    
    return frame

def get_action(move_state: str, turn_state: tuple[str, float], shoot: bool, apply_backward: bool) -> list[bool]:

    forward = move_state == "FORWARD" and not apply_backward
    backward = apply_backward
    turn_direction, turn_velocity = turn_state
    turn_left = turn_direction == "TURN_LEFT"
    turn_right = turn_direction == "TURN_RIGHT"
    
    # lista boolean, padrão do vizdoom
    return [forward, backward, turn_left, turn_right, shoot]


def init_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "freedoom1.cfg"))
    # Desativar buffer (freedoom ativa por padrão e bloqueia o audio)
    game.set_audio_buffer_enabled(False)
    game.set_sound_enabled(True)
    game.set_console_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X1024)

    game.set_doom_map("E1M2")  # Fase (1 - elevador confuso)
    game.set_doom_skill(1)  # BABY!!
    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.ATTACK,
    ])
    game.init()
    return game

def run():      
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=_on_result,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=_on_hand_result,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    SCREEN_W, SCREEN_H = get_screen_size()

    CAM_W = max(360, SCREEN_W // 4)

    cap = cv2.VideoCapture(0)

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAM_H = int(CAM_W * native_h / max(native_w, 1))

    cam_x = SCREEN_W - CAM_W

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam", CAM_W, CAM_H)
    cv2.moveWindow("Webcam", cam_x-40, 30)
    
    # Iniciar depois do opencv pra ganhar window focus e permitir audio
    game = init_game()

    with PoseLandmarker.create_from_options(options) as landmarker, \
         HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        global _last_state, _state_change_time, _show_landmarks
        timestamp_ms = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            landmarker.detect_async(mp_image, timestamp_ms)
            hand_landmarker.detect_async(mp_image, timestamp_ms)

            display_frame = cv2.resize(cv2.flip(frame, 1), (CAM_W, CAM_H))
            
            result = get_latest_result()
            hand_result = get_latest_hand_result()
            
            state = "STOP"
            turn_state = ("STRAIGHT", 0.0)
            wrist_x = 0.5  # Default center

            if result and result.pose_landmarks and hand_result and hand_result.hand_landmarks:
                pose_landmarks = result.pose_landmarks[0]
                hand_landmarks = hand_result.hand_landmarks[0]
                
                shoulder = pose_landmarks[RIGHT_SHOULDER]
                wrist = pose_landmarks[RIGHT_WRIST]
                tip = hand_landmarks[INDEX_TIP]
                
                if wrist.visibility >= POSE_VIZ_THRESHOLD and shoulder.visibility >= POSE_VIZ_THRESHOLD:
                    wrist_x = wrist.x
                    extension = compute_arm_extension(shoulder.y, tip.y)
                    state = classify_arm_state(extension)
                    turn_state = classify_turn(wrist.x)
            
            if _show_landmarks and hand_result:
                display_frame = draw_hand_landmarks(display_frame, hand_result, state, turn_state)
            
            cv2.imshow("Webcam", display_frame)
            cv2.waitKey(1)  # Frame display update

            if state != _last_state:
                _last_state = state
                _state_change_time = time.time()
            
            apply_backward = False
            if _last_state == "FORWARD" and state == "STOP" and _state_change_time is not None:
                elapsed = time.time() - _state_change_time
                if elapsed > 1.0:
                    apply_backward = True

            shoot = detect_trigger_pull(hand_result, wrist_x, state)
            action = get_action(state, turn_state, shoot, apply_backward)

            if not game.is_episode_finished():
                game.make_action(action)

            # 0x1B = ESC para sair
            if ctypes.windll.user32.GetAsyncKeyState(0x1B):
                break
            
            # 0x20 = SPACE para trackers
            if ctypes.windll.user32.GetAsyncKeyState(0x20):
                _show_landmarks = not _show_landmarks

            if game.is_player_dead():
                game.respawn_player()

    cap.release()
    cv2.destroyAllWindows()
    game.close()

if __name__ == "__main__":
    run()