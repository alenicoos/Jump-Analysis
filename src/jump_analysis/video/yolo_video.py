"""YOLO video pipeline.

Questo file contiene tutta la parte "video":
- legge frame da webcam o video;
- usa YOLO pose per stimare i keypoint del corpo;
- controlla che si vedano i punti necessari;
- fa il setup terra/box senza chiudere la webcam;
- registra il drop jump quando rileva la discesa dal box;
- trova i keyframe biomeccanici principali;
- converte i keypoint YOLO nelle 37 feature frontali confrontabili col dataset.

Le parti cliniche/modello restano fuori da qui: questo file prepara i dati.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from jump_analysis.data import FRONT_2D_FEATURE_COLUMNS
from jump_analysis.feedback import AudioFeedback
from jump_analysis.features.front_2d_features import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    FrontKeyframes,
    angle,
    body_keypoint,
    build_front_2d_feature_row,
)
from jump_analysis.validation import (
    DropJumpProtocolValidator,
    StablePoseBuffer,
    SetupCalibration,
    SetupValidator,
)


# Soglia minima di confidenza YOLO per considerare un keypoint "visibile".
CONFIDENCE_THRESHOLD = 0.30

# Per le 37 feature frontali non serve vedere la testa.
# Servono pero' spalle, anche, ginocchia e caviglie: senza questi punti
# alcune feature sarebbero inventate o troppo rumorose.
REQUIRED_POSE_KEYPOINTS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

# Connessioni usate solo per disegnare lo scheletro nella finestra OpenCV.
SKELETON = [
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class YoloPoseFrame:
    """Un frame valido gia' normalizzato e pronto per feature/keyframe."""

    frame_index: int
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    box_xyxy: np.ndarray | None


@dataclass
class StablePoseCapture:
    """Posa stabile acquisita durante il setup + altezza corpo in pixel."""

    keypoints_xy: np.ndarray
    body_height_px: float


def visible(conf: np.ndarray, index: int) -> bool:
    """Dice se un keypoint YOLO supera la soglia minima di confidenza."""

    return conf[index] >= CONFIDENCE_THRESHOLD


def required_visible(conf: np.ndarray) -> bool:
    """Controlla se tutti i keypoint necessari sono visibili nel frame."""

    return all(visible(conf, index) for index in REQUIRED_POSE_KEYPOINTS)


def missing_required(conf: np.ndarray) -> list[int]:
    """Restituisce gli indici COCO dei keypoint necessari che mancano."""

    return [index for index in REQUIRED_POSE_KEYPOINTS if not visible(conf, index)]


def select_pose(result, locked_track_id):
    """Sceglie quale persona seguire nel frame YOLO.

    Se YOLO tracking ha gia' assegnato un track id, continuiamo a usare
    sempre lo stesso individuo. Questo evita di passare a un'altra persona
    se qualcuno entra nell'inquadratura.

    Se non abbiamo ancora un track id, scegliamo la persona con bounding box
    piu' grande, cioe' di solito quella piu' vicina e centrale.
    """

    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None

    # YOLO restituisce tensori; li portiamo in numpy per fare calcoli semplici.
    kpts_xy = result.keypoints.xy.cpu().numpy()
    kpts_conf = result.keypoints.conf.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    track_ids = None
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    if locked_track_id is not None and track_ids is not None:
        matches = np.where(track_ids == locked_track_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])
            box = boxes_xyxy[idx] if boxes_xyxy is not None else None
            return idx, locked_track_id, kpts_xy[idx], kpts_conf[idx], box

    idx = 0
    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        idx = int(np.argmax(areas))

    track_id = None
    if track_ids is not None and len(track_ids) > idx:
        track_id = int(track_ids[idx])
    box = boxes_xyxy[idx] if boxes_xyxy is not None else None
    return idx, track_id, kpts_xy[idx], kpts_conf[idx], box


def draw_pose(frame: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> None:
    """Disegna punti e segmenti dello scheletro sul frame mostrato a schermo."""

    for i, j in SKELETON:
        if visible(kpts_conf, i) and visible(kpts_conf, j):
            cv2.line(
                frame,
                tuple(kpts_xy[i].astype(int).tolist()),
                tuple(kpts_xy[j].astype(int).tolist()),
                (255, 0, 0),
                2,
            )
    for index, (x, y) in enumerate(kpts_xy):
        if visible(kpts_conf, index):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


def put_lines(frame: np.ndarray, lines: list[str]) -> None:
    """Scrive messaggi leggibili sopra la webcam.

    OpenCV disegna testo direttamente sull'immagine; qui aggiungiamo anche
    un rettangolo scuro dietro, cosi' il testo resta leggibile.
    """

    x, y = 16, 20
    line_height = 28
    width = max(520, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0] for line in lines) + 32)
    height = 18 + line_height * len(lines)
    cv2.rectangle(frame, (x - 8, y - 8), (x + width, y + height), (36, 36, 36), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + line_height * (idx + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def shoulder_width_px(kpts_xy: np.ndarray) -> float:
    """Larghezza spalle in pixel tra keypoint sinistro e destro."""

    return float(np.linalg.norm(kpts_xy[LEFT_SHOULDER] - kpts_xy[RIGHT_SHOULDER]))


def normalize_keypoints_to_mocap_scale(
    kpts_xy: np.ndarray,
    shoulder_width_m: float,
) -> np.ndarray:
    """Converte coordinate YOLO in una scala simile al mocap.

    YOLO lavora in pixel: se l'utente si avvicina alla camera, le distanze
    aumentano anche se il corpo e' lo stesso. Per confrontare i salti col
    dataset, riscalo i keypoint usando la larghezza spalle misurata durante il
    setup: altezza reale inserita / altezza in pixel osservata a terra.

    Il risultato non e' una ricostruzione 3D: e' una normalizzazione 2D utile
    per rendere le feature piu' confrontabili tra video diversi.
    """

    scale = shoulder_width_m / max(shoulder_width_px(kpts_xy), 1.0)
    return kpts_xy * scale


def estimate_person_height_px(box_xyxy: np.ndarray | None, kpts_xy: np.ndarray) -> float:
    """Stima l'altezza della persona in pixel.

    Se YOLO fornisce la bounding box della persona, usiamo quella perché
    rappresenta meglio la figura intera. Se manca, usiamo il range verticale
    dei keypoint disponibili.
    """

    if box_xyxy is not None:
        return float(max(box_xyxy[3] - box_xyxy[1], 1.0))
    return float(max(kpts_xy[:, 1].max() - kpts_xy[:, 1].min(), 1.0))


def capture_stable_setup_pose(
    cap: cv2.VideoCapture,
    model: YOLO,
    prompt_lines: list[str],
    show: bool = True,
    timeout_seconds: float = 30.0,
    stable_frames: int = 12,
) -> StablePoseCapture:
    """Acquisisce una posa stabile durante una fase di setup.

    Viene usata sia quando l'utente e' a terra sia nella modalita' manuale sul
    box. La posa non viene presa subito: accumuliamo diversi frame e accettiamo
    la mediana solo quando il corpo e' rimasto abbastanza fermo.
    """

    # StablePoseBuffer evita di calibrare mentre l'utente sta ancora entrando
    # nell'inquadratura, spostando il telefono o cercando la posizione.
    buffer = StablePoseBuffer(min_frames=stable_frames)
    body_height_buffer: deque[float] = deque(maxlen=stable_frames)
    started = time.monotonic()
    locked_track_id = None
    last_debug_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame during setup.")
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError("Setup timeout: non ho visto una posa stabile in tempo.")

        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        lines = list(prompt_lines)

        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, box = selection
            # Dopo il primo riconoscimento buono, proviamo a seguire sempre
            # lo stesso track id per non cambiare persona durante il setup.
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if required_visible(kpts_conf):
                buffer.add(kpts_xy)
                body_height_buffer.append(estimate_person_height_px(box, kpts_xy))
                stable = buffer.stable_pose()
                if stable is not None:
                    # Restituiamo una posa mediana/stabile, non un singolo frame.
                    # Questo rende meno rumorosa la stima del pavimento e del box.
                    lines.append("Posa stabile acquisita")
                    if show:
                        put_lines(display, lines)
                        cv2.imshow("YOLO setup", display)
                        cv2.waitKey(250)
                    body_height_px = float(np.median(body_height_buffer))
                    return StablePoseCapture(stable, body_height_px)
                lines.append("Resta fermo...")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Figura incompleta: missing {missing}")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Setup incompleto. Keypoint mancanti: {missing}", flush=True)
                    last_debug_print = now
                # Se mancano punti necessari, svuotiamo il buffer: non vogliamo
                # mischiare frame validi vecchi con una posa nuova/incompleta.
                buffer.clear()
                body_height_buffer.clear()
        else:
            lines.append("Nessuna persona")
            buffer.clear()
            body_height_buffer.clear()

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO setup", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Setup interrotto dall'utente.")


def capture_stable_box_pose_after_floor(
    cap: cv2.VideoCapture,
    model: YOLO,
    floor_pose: np.ndarray,
    show: bool = True,
    timeout_seconds: float = 45.0,
    stable_frames: int = 12,
    min_box_height_ratio: float = 0.05,
) -> np.ndarray:
    """Acquisisce automaticamente la posa sul box dopo la posa a terra.

    Non chiediamo all'utente di premere invio. Lo script capisce che e' salito
    sul box quando le caviglie risultano piu' alte rispetto alla posa a terra.
    """

    buffer = StablePoseBuffer(min_frames=stable_frames)
    started = time.monotonic()
    locked_track_id = None
    last_debug_print = 0.0
    floor_ankle_y = float((floor_pose[LEFT_ANKLE][1] + floor_pose[RIGHT_ANKLE][1]) / 2.0)
    floor_scale = max(float(np.linalg.norm(floor_pose[LEFT_SHOULDER] - floor_pose[RIGHT_SHOULDER])), 1.0)
    # La soglia e' relativa alla larghezza spalle: cosi' non dipende troppo
    # dalla risoluzione della webcam.
    min_box_height_px = min_box_height_ratio * floor_scale

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame during box setup.")
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeError("Setup timeout: non ho visto l'utente salire e stare fermo sul box.")

        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        lines = [
            "SETUP 2/2: sali sul box",
            "Resta fermo quando sei sopra il rialzo",
        ]

        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, _ = selection
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if required_visible(kpts_conf):
                ankle_y = float((kpts_xy[LEFT_ANKLE][1] + kpts_xy[RIGHT_ANKLE][1]) / 2.0)
                # In coordinate immagine, y cresce verso il basso. Se l'utente
                # sale sul box, le caviglie salgono nell'immagine e quindi y
                # diventa piu' piccolo: floor_ankle_y - ankle_y e' positivo.
                box_height_px = floor_ankle_y - ankle_y
                lines.append(f"Altezza rilevata: {box_height_px:.1f}px / {min_box_height_px:.1f}px")
                if box_height_px > min_box_height_px:
                    buffer.add(kpts_xy)
                    stable = buffer.stable_pose()
                    if stable is not None:
                        lines.append("Posa sul box acquisita")
                        if show:
                            put_lines(display, lines)
                            cv2.imshow("YOLO setup", display)
                            cv2.waitKey(250)
                        return stable
                    lines.append("Resta fermo sul box...")
                else:
                    buffer.clear()
                    lines.append("Non sei ancora abbastanza sopra il pavimento")
            else:
                missing = missing_required(kpts_conf)
                lines.append(f"Figura incompleta: missing {missing}")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Setup box incompleto. Keypoint mancanti: {missing}", flush=True)
                    last_debug_print = now
                buffer.clear()
        else:
            lines.append("Nessuna persona")
            buffer.clear()

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO setup", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Setup interrotto dall'utente.")


def run_floor_box_setup_with_open_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    height_cm: float,
    show: bool = True,
    audio: bool = False,
    auto_detect_box: bool = True,
) -> SetupCalibration:
    """Setup terra/box usando una webcam gia' aperta.

    Questa e' la funzione usata da `scripts/main.py`: prima calibra l'utente a
    terra, poi sul box, valida geometria/altezza e infine lascia la webcam
    aperta per iniziare subito il drop jump.
    """

    feedback = AudioFeedback(enabled=audio)
    feedback.speak("Setup. Stand on the floor and stay still.", force=True)
    print("\nSETUP 1/2: mettiti a terra, fermo, frontale alla camera.")
    floor_sample = capture_stable_setup_pose(
        cap,
        model,
        ["SETUP 1/2: stai a terra", "Figura intera visibile per misurare l'altezza"],
        show=show,
    )
    floor_pose = floor_sample.keypoints_xy

    feedback.speak("Floor pose acquired.", force=True)
    print("Posa a terra acquisita.")
    print(f"Altezza corpo in pixel: {floor_sample.body_height_px:.1f}px")
    print("\nSETUP 2/2: sali sul box/rialzo mantenendo la stessa distanza dalla camera.")
    feedback.speak("Now stand on the box and stay still.", force=True)
    if auto_detect_box:
        box_pose = capture_stable_box_pose_after_floor(
            cap,
            model,
            floor_pose,
            show=show,
            min_box_height_ratio=SetupValidator().min_box_height_ratio,
        )
    else:
        box_sample = capture_stable_setup_pose(
            cap,
            model,
            ["SETUP 2/2: sali sul box", "Resta nella stessa distanza dalla camera"],
            show=show,
        )
        box_pose = box_sample.keypoints_xy

    result = SetupValidator().validate_floor_and_box(
        floor_pose,
        box_pose,
        height_cm=height_cm,
        floor_body_height_px=floor_sample.body_height_px,
    )
    for message in result.messages:
        print(message, flush=True)
    if result.calibration and result.calibration.estimated_box_height_cm is not None:
        feedback.speak(
            f"Estimated box height {result.calibration.estimated_box_height_cm:.1f} centimeters.",
            force=True,
        )
        print(f"Pixel scale: {result.calibration.meters_per_pixel:.6f} m/px")
        print(f"Measured shoulder width: {result.calibration.measured_shoulder_width_m * 100:.1f} cm")
        print(f"Estimated box height: {result.calibration.estimated_box_height_cm:.1f} cm")
    if not result.passed:
        raise RuntimeError("Setup non valido: correggi gli errori e riprova.")
    return result.calibration


def capture_yolo_pose_frames_with_open_capture(
    cap: cv2.VideoCapture,
    model: YOLO,
    seconds: float,
    shoulder_width_m: float,
    show: bool = True,
    prepare_seconds: float = 2.0,
    min_drop_ratio: float = 0.06,
    max_wait_seconds: float = 30.0,
) -> list[YoloPoseFrame]:
    """Registra i frame validi del drop jump.

    La funzione non parte subito a registrare. Prima aspetta che l'utente sia
    fermo sul box, costruisce una baseline della posizione delle caviglie e poi
    avvia la registrazione quando rileva una discesa sufficiente.

    In pratica:
    1. utente fermo sul box;
    2. baseline caviglie;
    3. caviglie scendono nell'immagine;
    4. il drop e' iniziato;
    5. salviamo i frame normalizzati per estrarre le feature.
    """

    frames: list[YoloPoseFrame] = []

    # Buffer di preparazione: contiene la posizione media dei piedi e l'altezza
    # del corpo prima della discesa. Serve per capire quando il drop inizia.
    prep_feet_y = deque(maxlen=45)
    prep_body_height = deque(maxlen=45)

    # Teniamo qualche frame precedente al trigger, cosi' quando parte il drop
    # non perdiamo l'inizio del movimento.
    pre_roll_frames: deque[YoloPoseFrame] = deque(maxlen=60)

    locked_track_id = None
    opened_at = time.monotonic()
    recording_started_at = None
    frame_index = 0
    detected_frames = 0
    incomplete_frames = 0
    last_debug_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        if recording_started_at is None and now - opened_at > max_wait_seconds:
            raise RuntimeError("Timeout: non ho visto una partenza valida da una altezza entro il tempo massimo.")
        if recording_started_at is not None and seconds and now - recording_started_at > seconds:
            break

        # `persist=True` mantiene il tracking tra frame consecutivi.
        result = model.track(frame, stream=False, persist=True, verbose=False)[0]
        selection = select_pose(result, locked_track_id)
        display = frame.copy()
        if recording_started_at is None:
            lines = [
                "Preparazione: sali sul box/sedia e resta fermo",
                "La registrazione parte quando inizi a scendere",
            ]
        else:
            lines = ["Registrazione drop jump", "Atterra e fai subito il secondo salto"]
        if selection is not None:
            _, detected_track_id, kpts_xy, kpts_conf, box = selection
            detected_frames += 1
            if locked_track_id is None and detected_track_id is not None:
                locked_track_id = detected_track_id
            draw_pose(display, kpts_xy, kpts_conf)
            if required_visible(kpts_conf):
                # Coordinate y delle caviglie: in OpenCV y aumenta verso il
                # basso, quindi una discesa fa aumentare questo valore.
                feet_y = float((kpts_xy[LEFT_ANKLE][1] + kpts_xy[RIGHT_ANKLE][1]) / 2.0)
                body_height = estimate_person_height_px(box, kpts_xy)

                # Da qui in poi salviamo coordinate normalizzate, non pixel raw.
                # La scala usa la larghezza spalle misurata dal setup:
                # altezza reale / altezza in pixel -> metri per pixel.
                normalized = normalize_keypoints_to_mocap_scale(kpts_xy, shoulder_width_m)

                if recording_started_at is None:
                    if len(prep_feet_y) >= 8:
                        baseline = float(np.median(prep_feet_y))
                        required_drop_px = min_drop_ratio * float(np.median(prep_body_height))
                        current_drop_px = feet_y - baseline
                        lines.append(f"Pronto quando scendi: {current_drop_px:.1f}px / {required_drop_px:.1f}px")
                        # Trigger del drop: abbastanza frame di preparazione
                        # + caviglie scese oltre soglia relativa all'altezza.
                        if len(prep_feet_y) >= int(prepare_seconds * 15) and current_drop_px > required_drop_px:
                            recording_started_at = now
                            frames = list(pre_roll_frames)
                            print("Drop rilevato: inizio registrazione.", flush=True)
                            frames.append(YoloPoseFrame(frame_index, normalized, kpts_conf, box))
                        else:
                            # Ancora non e' iniziato il drop: aggiorniamo sia
                            # baseline sia pre-roll.
                            pre_roll_frames.append(YoloPoseFrame(frame_index, normalized, kpts_conf, box))
                            prep_feet_y.append(feet_y)
                            prep_body_height.append(body_height)
                    else:
                        # Fase iniziale: accumuliamo abbastanza frame per una
                        # baseline stabile.
                        pre_roll_frames.append(YoloPoseFrame(frame_index, normalized, kpts_conf, box))
                        prep_feet_y.append(feet_y)
                        prep_body_height.append(body_height)
                        lines.append("Resta fermo sul rialzo...")
                else:
                    # Registrazione gia' partita: ogni frame valido viene usato
                    # per keyframe e feature.
                    frames.append(YoloPoseFrame(frame_index, normalized, kpts_conf, box))
                    lines.append(f"Frame validi: {len(frames)}")
            else:
                incomplete_frames += 1
                missing = missing_required(kpts_conf)
                lines.append(f"Figura incompleta: missing {missing}")
                now = time.monotonic()
                if now - last_debug_print > 1.0:
                    print(f"Figura incompleta. Keypoint mancanti: {missing}", flush=True)
                    last_debug_print = now
        else:
            lines.append("Nessuna persona")

        if show:
            put_lines(display, lines)
            cv2.imshow("YOLO front capture", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    if show:
        cv2.destroyWindow("YOLO front capture")
    if len(frames) < 10:
        # Con pochi frame validi non possiamo stimare keyframe e feature in modo
        # affidabile: meglio fermarsi e chiedere di riprovare.
        raise RuntimeError(
            f"Too few valid pose frames: {len(frames)}. "
            f"Detected frames: {detected_frames}, incomplete frames: {incomplete_frames}. "
            "Prova con --model yolo26n-pose.pt o --model yolo11n-pose.pt, "
            "allontanati finche' spalle, anche, ginocchia e caviglie sono sempre visibili, "
            "e usa buona luce."
        )
    return frames


def ankle_mean_y(frame: YoloPoseFrame) -> float:
    """Media verticale tra caviglia sinistra e destra.
    
    Meglio di usare il centro del corpo perche' e' piu' stabile,
    ma in alcuni video potrebbe essere l'unico punto affidabile
    per capire quando l'utente atterra.
    """

    k = frame.keypoints_xy
    return float((k[LEFT_ANKLE][1] + k[RIGHT_ANKLE][1]) / 2.0)


def knee_flexion_proxy(frame: YoloPoseFrame) -> float:
    """Proxy semplice della flessione delle ginocchia.

    Usiamo 180 - angolo anca-ginocchio-caviglia: piu' il ginocchio si piega,
    piu' questo valore cresce. Serve per trovare il frame di massima flessione.
    """

    k = frame.keypoints_xy
    left = 180.0 - angle(k[LEFT_HIP], k[LEFT_KNEE], k[LEFT_ANKLE])
    right = 180.0 - angle(k[RIGHT_HIP], k[RIGHT_KNEE], k[RIGHT_ANKLE])
    return float(np.nanmean([left, right]))


def find_yolo_keyframes(frames: list[YoloPoseFrame]) -> tuple[int, int]:
    """Trova initial contact e massima flessione ginocchio.

    `ic` e' stimato cercando quando le caviglie sono piu' basse nell'immagine,
    cioe' quando l'utente atterra.

    `kf` e' stimato combinando due indizi:
    - massimo proxy di flessione ginocchia;
    - massimo abbassamento del centro corpo dopo il contatto.
    """

    ankle_y = np.array([ankle_mean_y(frame) for frame in frames])
    body_y = np.array([body_keypoint(frame.keypoints_xy)[1] for frame in frames])
    knee_flex = np.array([knee_flexion_proxy(frame) for frame in frames])

    landing_level = float(np.percentile(ankle_y, 90))
    # Prendiamo il primo frame vicino al livello di atterraggio, non per forza
    # il massimo assoluto: cosi' evitiamo di scegliere troppo tardi.
    candidates = np.where(ankle_y >= landing_level - 0.04 * max(np.ptp(ankle_y), 1.0))[0]
    ic = int(candidates[0]) if len(candidates) else int(np.argmax(ankle_y))
    end = min(len(frames), ic + 60)
    if end <= ic + 2:
        end = len(frames)

    kf_by_knee = ic + int(np.nanargmax(knee_flex[ic:end]))
    kf_by_body = ic + int(np.argmax(body_y[ic:end]))
    kf = int(round((kf_by_knee + kf_by_body) / 2))
    return ic, kf


def extract_front_features_from_yolo_frames(frames: list[YoloPoseFrame]) -> tuple[dict[str, float], dict[str, int]]:
    """Estrae feature frontali e metadati da una registrazione YOLO."""

    ic, kf = find_yolo_keyframes(frames)

    # Il validatore controlla che il movimento registrato sembri davvero un
    # drop jump: partenza da altezza, contatto a due piedi, secondo salto.
    protocol = DropJumpProtocolValidator().validate(frames, ic, kf)

    # Le 37 feature del dataset vengono calcolate solo su due keyframe:
    # initial contact e massima flessione del ginocchio.
    keyframes = FrontKeyframes(
        initial_contact=frames[ic].keypoints_xy,
        max_knee_flexion=frames[kf].keypoints_xy,
        crop_length_frames=frames[kf].frame_index - frames[ic].frame_index + 1,
    )
    metadata = {
        "valid_pose_frames": len(frames),
        "ic_valid_frame": ic,
        "kfmax_valid_frame": kf,
        "ic_raw_frame": frames[ic].frame_index,
        "kfmax_raw_frame": frames[kf].frame_index,
    }
    metadata.update(protocol.as_metadata())
    return build_front_2d_feature_row(keyframes), metadata


def compare_to_reference(features: dict[str, float], reference_csv: str | Path) -> pd.DataFrame:
    """Confronta una riga di feature YOLO con il CSV mocap di riferimento."""

    reference = pd.read_csv(reference_csv)
    rows = []
    for column in FRONT_2D_FEATURE_COLUMNS:
        # Per ogni feature calcoliamo statistiche semplici: media, deviazione
        # standard, z-score e percentile empirico nel dataset.
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        value = float(features[column])
        mean = float(ref.mean())
        std = float(ref.std(ddof=0))
        percentile = float((ref <= value).mean() * 100.0)
        z_score = (value - mean) / std if std > 0 else 0.0
        rows.append(
            {
                "feature": column,
                "value": value,
                "reference_mean": mean,
                "reference_std": std,
                "z_score": z_score,
                "percentile": percentile,
            }
        )
    return pd.DataFrame(rows)
