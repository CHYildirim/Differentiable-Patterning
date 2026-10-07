"""
signal_threshold_scan.py
========================

Laesst sich ein FERTIG GEWACHSENES Muster eines Multi-Attraktor-gNCA mit einem
untrainierten Reiz in einen anderen Attraktor umschalten - und wenn ja: mit
welchem Reiz, ab welcher Reizfeldgroesse, bei welcher Dauer und wo?

Das Skript testet nacheinander die Modelle in MODEL_BUNDLES (2, 3 und 4
Attraktoren, je CIRCULAR und REPLICATE).

ABLAUF JE MODELL
----------------
1. Jeder Attraktor waechst aus seinem Startpixel. Zu den Zeitpunkten T_SIGNALS
   (waehrend der Entwicklung, z.B. t = 8, 32, 128) und beim fertigen Muster
   (t = GROW_STEPS) wird der Zustand festgehalten. Waechst ein Attraktor bis
   GROW_STEPS nicht zu sich selbst, werden seine Richtungen uebersprungen.
2. Ab einem dieser Zeitpunkte wird in einer Scheibe mit Radius r um eine
   Position D Schritte lang (vor jedem NCA-Schritt neu) ein Reiz angelegt,
   danach laufen FREE_STEPS freie Schritte.
3. Bewertung (zwei Urteile, siehe unten), Schwellensuche ueber den Radius.
   Ergebnis je Zeitpunkt: bis wann in der Entwicklung laesst sich das Muster
   noch umlenken, und geht es am fertigen Muster noch?

REIZTYPEN (SIGNALS)
-------------------
Jeder Reiz ist je Kanal  x_neu = a * x + b + g * G (+ Rauschen).
"Ziel" = ein anderer Attraktor; zielgerichtete Reize werden fuer jede Richtung
Quelle -> Ziel getestet, neutrale Reize einmal je Quelle (dann zaehlt das
Umschalten in irgendeinen anderen Attraktor).

  nur sichtbare Kanaele (RGBA), Farbe kodiert die Identitaet:
    seed_rgba         RGBA = Startpixel-Farbe des Ziels, versteckte Kanaele bleiben
    seed_cell         wie seed_rgba, versteckte Kanaele = 0 (frischer Startpixel)
  uncodierte Pixel (neutral):
    black_rgba        RGBA = schwarz (0, 0, 0, 1), versteckte Kanaele bleiben
    black_zero        RGBA = schwarz, versteckte Kanaele = 0
    transparent_rgba  RGBA = 0 (durchsichtig), versteckte Kanaele bleiben
    transparent_zero  ALLE Kanaele = 0 (Loch / Schaden)
  alle 64 Kanaele:
    graft             alle Kanaele = Zustand des fertigen Zielmusters an derselben
                      Stelle (Gewebe-Transplantat)
    graft_hidden      nur die versteckten Kanaele = Zielmuster an derselben Stelle,
                      Farbe bleibt (unsichtbarer Identitaets-Code)
    target_mean       alle Kanaele = mittlerer Zustand der Zellen des Zielmusters
                      (einheitlicher Code, ohne Ortsinformation)
    push1 / push2     alle Kanaele = Zustand vor dem Reiz + 1x / 2x (Mittelwert
                      Ziel - Mittelwert Quelle): verschiebt die Zellen in Richtung
                      Ziel-Identitaet, behaelt aber ihre Ortsinformation
    noise             alle Kanaele += Rauschen mit der Streuung des Musters je Kanal
                      (neutral; Kontrolle: schaltet jede starke Stoerung um?)

Variiert werden ausserdem Reizbeginn (T_SIGNALS), Position (POSITIONS: Mitte +
Rasterpunkte auf dem fertigen Koerper), Radius (RADIUS_LADDER) und Dauer
(DURATIONS).

BEWERTUNG: ZWEI URTEILE
-----------------------
Etikett (wie bisher): naechstes Emoji-Zielbild per MSE (bestmoeglich
  verschoben), INTERMEDIATE wenn mind. zwei Attraktoren je >= 20 % der
  sichtbaren Pixel stellen, NO_VISIBLE_PATTERN wenn nichts sichtbar ist.
  Schwaeche (im ersten Scan gesehen): eine grosse, einfarbig uebermalte Scheibe
  kann als Mikrobe/Alien gelten, obwohl keine neue Form waechst.

Echt (streng): ein Lauf zaehlt nur als umgeschaltet zu Attraktor A, wenn
  1. das Etikett am Ende A ist,
  2. A durch Wachstum entstanden ist: direkt nach dem Reiz war das Etikett
     noch nicht A - oder, falls doch, hat sich das Muster danach noch deutlich
     veraendert (sichtbare MSE zwischen "direkt nach dem Reiz" und "Ende"
     >= CHANGE_MIN). Das schliesst stehen gebliebene Farbflecken und reine
     Ersetzung (grosses Transplantat) aus, laesst aber das Umlenken eines
     noch kleinen Musters frueh in der Entwicklung zu,
  3. das Endmuster dem vom Modell selbst gewachsenen fertigen A am
     aehnlichsten ist (MSE, bestmoeglich verschoben), und
  4. die Form passt: Ueberlappung der sichtbaren Flaechen (IoU) mit dem
     gewachsenen A >= IOU_MIN.

Die Schwellensuche (adaptiv) laeuft ueber das strenge Urteil: erst die Radien
SCREEN_RADII mit REP_SCREEN Wiederholungen, und nur wo mindestens ein Lauf echt
umgeschaltet hat, alle Radien dazwischen mit REP_CONFIRM Wiederholungen.
r* = kleinster Radius mit echter Umschaltrate >= SWITCH_THRESHOLD. Das
Etikett-Urteil wird zusaetzlich mit ausgegeben (r* aus dem Screen).

START
-----
Aus dem Repo-Stammordner (dort, wo Experiments/, Common/ und models/ liegen):

    export DATA_PATH_BASE=demo/demo_data     # Ordner, der "Emojis/" enthaelt
    python signal_threshold_scan.py --dry-run          # Aufwand schaetzen
    nohup python signal_threshold_scan.py > threshold_scan.log 2>&1 &
    tail -f threshold_scan.log

Optionen:
    --signals graft,black_zero   nur diese Reiztypen
    --times 8,256                nur diese Reizzeitpunkte
    --models 2attr_circular,4attr_replicate   nur diese Modelle
    --bundle PFAD          statt MODEL_BUNDLES genau dieses eine Modell
    --batch-size N         gleichzeitig simulierte Laeufe (Standard 64)
    --quick                sehr kleiner Testlauf (CPU-tauglich)
    --no-resume            alles neu rechnen
    --dry-run              nur zaehlen (Obergrenze, ohne Verfeinerung)

Ergebnisse: inference_outputs/signal_scan_reize_entwicklung/<modell-tag>/ (report.md, ...) und
OUTPUT_ROOT/summary_all_models.md + .csv + overview_all_models.png.
Wiederaufnahme: ein Neustart rechnet nur fehlende Laeufe (rollouts.csv).
Es darf nur ein Lauf gleichzeitig laufen (Sperrdatei OUTPUT_ROOT/.lock).
"""

from pathlib import Path
import argparse
import csv
import itertools
import json
import math
import os
import sys
import gc
import time
import traceback
from collections import Counter, defaultdict
from types import SimpleNamespace


# ======================================================================
# CPU / GPU  (muss vor "import jax" stehen)
# ======================================================================
FORCE_CPU = False

if FORCE_CPU or "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# GPU-Speicher nur nach Bedarf belegen (sonst reserviert JAX 75 % am Stueck,
# dann fehlt Platz fuer cuFFT und neu kompilierte Programme).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from omegaconf import OmegaConf
from PIL import Image

from Experiments.config_helpers import build_model
from Common.dataloader.emoji import load_emoji_sequence


# ######################################################################
# >>>>>>>>>>  HIER DIE MODELLE EINTRAGEN  <<<<<<<<<<
# ######################################################################
#
# Bundle-Ordner (mit model.eqx und config.yaml), relativ zum Repo-Stammordner
# oder absolut. Attraktorzahl und Padding liest das Skript aus config.yaml.
#
# Zwei dieser Modelle sind laut manifest.yaml nicht fertig trainiert
# (completed: false): 20260909T215800Z (2 Attr. CIRCULAR, Iteration 1546/16000)
# und 20260922T142806Z (4 Attr. REPLICATE, Iteration 8862/16000). Das Skript
# rechnet sie trotzdem, gibt aber eine Warnung aus. Fertige Alternativen:
#   2 Attr. CIRCULAR   20260929T132309Z-cfg372a0c3f14b5-e84bd156
#   4 Attr. REPLICATE  20260927T020111Z-cfg1be8b5068de7-6ada72d2

MODEL_BUNDLES = [
    "models/local/emoji_multi_attractor_example_v2/bundles/nca-emoji/emoji-multi-attractor-example-v2/20260830T174447Z-cfga63e288e1e9e-db1f45fc",   # gNCA, 2 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260909T215800Z-cfg049eedbd4068-d2916c1d",   # gNCA, 2 Attraktoren, CIRCULAR
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260921T212103Z-cfg9b176dd6be39-b584de1c",   # gNCA, 3 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260909T232154Z-cfg23d2f219b302-df559ea2",   # gNCA, 3 Attraktoren, CIRCULAR
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260922T142806Z-cfgee7db3859486-86b9651a",   # gNCA, 4 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260910T173700Z-cfgf1ae05f3a10d-be54beee",   # gNCA, 4 Attraktoren, CIRCULAR
]

# Erwartete Kombinationen (Attraktorzahl, Padding). Abweichungen -> Warnung.
EXPECTED_MODELS = [
    (2, "REPLICATE"), (2, "CIRCULAR"),
    (3, "REPLICATE"), (3, "CIRCULAR"),
    (4, "REPLICATE"), (4, "CIRCULAR"),
]

# ######################################################################


# Ausgabeordner. Die Ergebnisse frueherer Scans (signal_threshold_scan,
# signal_threshold_scan_v2) bleiben dadurch unberuehrt. Aendern hier oder beim
# Start mit --out ORDNER.
OUTPUT_ROOT = Path("inference_outputs/signal_scan_reize_entwicklung")

# Ordner mit den Emoji-PNGs. None -> $DATA_PATH_BASE/Emojis, sonst
# demo/demo_data/Emojis.
EMOJI_DIR = None

SEED = 0

# Ausgangsmuster: so lange wachsen lassen, bevor der Reiz kommt (fertig).
GROW_STEPS = 256
# Freie Schritte nach dem ENDE des Reizes bis zur Bewertung.
FREE_STEPS = 192

# ----------------------------- Reiztypen -------------------------------
# (Art, Beschreibung). Art "ziel" = je Richtung Quelle -> Ziel,
# "neutral" = einmal je Quelle.
SIGNAL_INFO = {
    "seed_rgba":        ("ziel", "RGBA = Startpixel-Farbe des Ziels, versteckte Kanaele bleiben"),
    "seed_cell":        ("ziel", "RGBA = Startpixel-Farbe des Ziels, versteckte Kanaele = 0"),
    "black_rgba":       ("neutral", "RGBA = schwarz (0,0,0,1), versteckte Kanaele bleiben"),
    "black_zero":       ("neutral", "RGBA = schwarz (0,0,0,1), versteckte Kanaele = 0"),
    "transparent_rgba": ("neutral", "RGBA = 0 (durchsichtig), versteckte Kanaele bleiben"),
    "transparent_zero": ("neutral", "alle Kanaele = 0 (Loch)"),
    "graft":            ("ziel", "alle Kanaele = fertiges Zielmuster an derselben Stelle"),
    "graft_hidden":     ("ziel", "nur versteckte Kanaele = Zielmuster an derselben Stelle"),
    "target_mean":      ("ziel", "alle Kanaele = mittlerer Zustand der Zielmuster-Zellen"),
    "push1":            ("ziel", "alle Kanaele + 1 x (Mittel Ziel - Mittel Quelle)"),
    "push2":            ("ziel", "alle Kanaele + 2 x (Mittel Ziel - Mittel Quelle)"),
    "noise":            ("neutral", "alle Kanaele + Rauschen (Streuung des Musters je Kanal)"),
}
SIGNALS = list(SIGNAL_INFO)

# ---------------------------- Reizparameter ----------------------------
# Reizbeginn (Wachstumsschritt). GROW_STEPS = fertiges Muster; die kleineren
# Werte liegen in der Entwicklung (Training-t der Modelle: 128).
T_SIGNALS = [8, 32, 128, GROW_STEPS]
# Positionen: Mitte + Rasterpunkte (Abstand 40 px) auf dem FERTIGEN Koerper -
# fuer fruehe Zeitpunkte also dort, wo der Koerper hinwachsen wird.
POSITIONS = ["seed", "body:40"]
# Im ersten Scan machte die Dauer (1 bis 128) keinen Unterschied; fuer mehr
# Dauern z.B. [1, 8, 32, 128] (kostet entsprechend mehr Zeit).
DURATIONS = [1, 32]
RADIUS_LADDER = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
# Screen-Radien: r = 64 ist weggefallen (uebermalt fast die ganze Leinwand).
SCREEN_RADII = [4, 16, 48]

# ------------------------- Schwellensuche ------------------------------
REP_SCREEN = 2
REP_CONFIRM = 8
SWITCH_THRESHOLD = 0.5
# Formueberlappung (IoU) mit dem selbst gewachsenen Zielmuster fuer "echt".
# Zwei fertige Muster desselben Attraktors liegen deutlich darueber; eine
# uebermalte Scheibe (r = 48-64) gegen die Mikrobe bei ~0.5-0.6.
IOU_MIN = 0.75
# Mindest-Veraenderung (mittlere quadratische Abweichung der sichtbaren Kanaele
# zwischen "direkt nach dem Reiz" und "Ende"), damit ein sofort erkanntes
# Ziel trotzdem als nachgewachsen gilt (siehe Kriterium 2).
CHANGE_MIN = 0.01
# Verfeinern, wenn im Screen etwas "echt" ("strict") oder nach Etikett
# ("label") umgeschaltet hat. "label" kostet viel mehr (grosse Reize schalten
# nach Etikett fast immer "um").
REFINE_ON = "strict"

# Richtungen aus Attraktoren, die schon ohne Reiz nicht sie selbst sind,
# ueberspringen (ihr "Umschalten" waere nicht interpretierbar).
SKIP_UNSTABLE_SOURCES = True

# Gleichzeitig simulierte Laeufe. 64 passt auf eine 8-GB-GPU. Laeuft der
# Speicher voll, halbiert das Skript die Batchgroesse selbst (bis MIN_BATCH_SIZE).
BATCH_SIZE = 64
MIN_BATCH_SIZE = 4

# Klassifikation (wie in spatial_position_scan_multi.py)
ACTIVE_ALPHA_THRESHOLD = 0.10
MIXED_MIN_FRACTION = 0.20

# Karten je Modell hoechstens (nur Gruppen mit echtem Umschalten)
MAX_MAPS_PER_MODEL = 40
# Beispielbilder je Modell
N_EXAMPLES_PER_MODEL = 6
EXAMPLE_SNAPSHOT_STEPS = [1, 8, 32, 64, 128, "end"]
SAVE_GIFS = True

# Sehr kleiner Testlauf (--quick)
QUICK = dict(
    GROW_STEPS=192, FREE_STEPS=128, T_SIGNALS=[8, 192],
    POSITIONS=["seed"], DURATIONS=[1], SCREEN_RADII=[8, 48],
    RADIUS_LADDER=[0, 2, 4, 8, 16, 32, 48], REP_SCREEN=1, REP_CONFIRM=2,
    BATCH_SIZE=8, N_EXAMPLES_PER_MODEL=2, EXAMPLE_SNAPSHOT_STEPS=[1, 32, "end"],
    SAVE_GIFS=False,
)


# ======================================================================
# NAMEN, FARBEN
# ======================================================================

INTERMEDIATE = "INTERMEDIATE"
NO_PATTERN = "NO_VISIBLE_PATTERN"
ANY = "*"          # Ziel bei neutralen Reizen: irgendein anderer Attraktor

SEQ_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e4e3df"


# ======================================================================
# MODELL / ZIELBILDER LADEN
# ======================================================================

def cfg_value(cfg, *paths, default=None):
    for path in paths:
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None:
            return value
    return default


def plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def parse_pad(pad):
    if pad is None:
        return (0, 0, 0, 0)
    if isinstance(pad, (int, float)):
        return (int(pad),) * 4
    pad = [int(p) for p in pad]
    if len(pad) == 1:
        return (pad[0],) * 4
    if len(pad) == 2:
        return (pad[0], pad[0], pad[1], pad[1])
    if len(pad) == 4:
        return tuple(pad)
    raise ValueError(f"Unerwartetes Padding: {pad}")


def resolve_bundle(entry):
    path = Path(str(entry)).expanduser()
    if path.name in {"model.eqx", "config.yaml", "manifest.yaml"}:
        path = path.parent
    for name in ("model.eqx", "config.yaml"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"{path / name} fehlt")
    return path


def resolve_emoji_dir():
    if EMOJI_DIR is not None:
        return Path(EMOJI_DIR)
    if os.environ.get("DATA_PATH_BASE"):
        return Path(os.environ["DATA_PATH_BASE"]) / "Emojis"
    return Path("demo/demo_data/Emojis")


def load_emoji(filename, emoji_dir, downsample, crop_square):
    ds = load_emoji_sequence([filename], impath_emojis=emoji_dir,
                             downsample=downsample, crop_square=crop_square)
    data = ds.data if hasattr(ds, "data") else ds
    return np.asarray(data[0, 0], dtype=np.float32)


def initial_seed(initial, image):
    """Startpixel wie beim Training (full / patch / pixel) -> (Pixel, (y, x))."""
    mode = initial.get("mode", "full")
    _, height, width = image.shape
    if mode == "full":
        yx = ((height - 1) // 2, (width - 1) // 2)
        return image[:, yx[0], yx[1]].copy(), yx, "full image"
    if mode == "patch":
        size = int(initial.get("size", 12))
        top, left = (height - size) // 2, (width - size) // 2
        yx = (top + (size - 1) // 2, left + (size - 1) // 2)
        return image[:, yx[0], yx[1]].copy(), yx, f"patch {size}x{size}"
    if mode == "pixel":
        pixel = np.zeros(image.shape[0], np.float32)
        pixel[int(initial.get("channel", 0))] = float(initial.get("value", 1.0))
        return pixel, (height // 2, width // 2), "pixel"
    raise ValueError(f"Unbekannter initial-Modus {mode!r}")


def name_from_file(filename, used):
    base = Path(str(filename)).stem.upper().replace("-", "_") or "ATTRACTOR"
    name, i = base, 2
    while name in used or name in (INTERMEDIATE, NO_PATTERN):
        name, i = f"{base}_{i}", i + 1
    used.add(name)
    return name


def describe_bundle(bundle):
    """Liest nur config/manifest (schnell, ohne Modell) fuer Planung/Warnungen."""
    cfg = OmegaConf.load(bundle / "config.yaml")
    task = cfg_value(cfg, "data.augmentation.task", "data.emoji.task")
    if task != "multi_attractor":
        raise ValueError(f"{bundle}: task ist {task!r}, erwartet 'multi_attractor'")
    pairs = plain(cfg_value(cfg, "data.augmentation.pairs", "data.emoji.pairs"))
    padding = str(plain(cfg_value(cfg, "model.padding"))).upper()
    family = str(cfg_value(cfg, "model.family", default="?"))
    info = SimpleNamespace(bundle=bundle, cfg=cfg, pairs=pairs, K=len(pairs),
                           padding=padding, family=family, warnings=[])
    manifest = bundle / "manifest.yaml"
    if manifest.is_file():
        m = OmegaConf.load(manifest)
        if OmegaConf.select(m, "training.completed", default=None) is False:
            info.warnings.append(
                "Training NICHT abgeschlossen (completed: false, bester Checkpoint "
                f"Iteration {OmegaConf.select(m, 'training.best_iteration')} von "
                f"{cfg_value(cfg, 'training.loop.iterations')}, Loss "
                f"{OmegaConf.select(m, 'training.best_loss')}).")
    info.tag = f"{info.K}attr_{padding.lower()}_{family}"
    return info


def prepare(info, emoji_dir):
    cfg = info.cfg
    downsample = int(cfg_value(cfg, "data.preprocessing.downsample",
                               "data.downsample", default=1))
    crop_square = bool(cfg_value(cfg, "data.augmentation.crop_square", default=False))
    pad = parse_pad(plain(cfg_value(cfg, "data.augmentation.pad",
                                    "data.emoji.pad", default=0)))
    dc = int(cfg_value(cfg, "data.augmentation.data_channels",
                       "data.data_channels", default=4))

    used, attractors = set(), []
    for index, pair in enumerate(info.pairs):
        initial = pair["initial"]
        if isinstance(initial, str):
            initial = {"image": initial, "mode": "full"}
        target = load_emoji(pair["target"], emoji_dir, downsample, crop_square)
        init_img = load_emoji(initial["image"], emoji_dir, downsample, crop_square)
        pixel, yx, desc = initial_seed(initial, init_img)
        attractors.append(SimpleNamespace(
            index=index, name=name_from_file(pair["target"], used),
            target=target, seed_pixel=pixel[:dc].astype(np.float32),
            seed_yx=yx, initial_description=desc))

    _, H, W = attractors[0].target.shape
    top, bottom, left, right = pad
    seed_positions = {a.seed_yx for a in attractors}
    if len(seed_positions) > 1:
        info.warnings.append(f"Attraktoren haben verschiedene Seed-Positionen "
                             f"{sorted(seed_positions)}; benutzt wird die erste.")
    for a, b in itertools.combinations(attractors, 2):
        if np.allclose(a.seed_pixel, b.seed_pixel, atol=0.5 / 255):
            info.warnings.append(f"{a.name} und {b.name} haben denselben Startpixel.")

    model = build_model(cfg.model, key=jr.PRNGKey(0))
    model = model[0] if isinstance(model, tuple) else model
    model = model.load(info.bundle / "model.eqx")

    return SimpleNamespace(
        info=info, bundle=info.bundle, tag=info.tag, cfg=cfg, model=model,
        padding=info.padding, circular=(info.padding == "CIRCULAR"),
        family=info.family, n_channels=int(cfg.model.channels), dc=dc,
        attractors=attractors, names=[a.name for a in attractors],
        class_names=[a.name for a in attractors] + [INTERMEDIATE, NO_PATTERN],
        K=len(attractors),
        targets=np.stack([np.pad(a.target[:dc], ((0, 0), (top, bottom), (left, right)))
                          for a in attractors]).astype(np.float32),
        H=H + top + bottom, W=W + left + right,
        seed_y=attractors[0].seed_yx[0] + top, seed_x=attractors[0].seed_yx[1] + left,
        train_t=cfg_value(cfg, "training.loop.t", default="?"),
        warnings=list(info.warnings),
    )


# ======================================================================
# AUSRICHTUNG + KLASSIFIKATION (JAX, verschiebungsinvariant)
# ======================================================================

def make_aligner(targets, circular):
    """Gibt f(vis) -> (mse[K], verschobene Zielbilder[K, dc, H, W]) zurueck.
    Jedes Zielbild wird so verschoben, dass es am besten passt.
    CIRCULAR: mit Wrap-around. Sonst ohne Wrap-around (herausgeschobene Teile
    = 0, Verschiebung hoechstens eine halbe Leinwand)."""
    T = jnp.asarray(targets)                       # (K, dc, H, W)
    K, dc, H, W = T.shape
    N = dc * H * W
    if circular:
        PH, PW = H, W
        Tp = T
        energy = jnp.broadcast_to(jnp.sum(T ** 2, axis=(1, 2, 3))[:, None, None], (K, PH, PW))
        valid = None
    else:
        PH, PW = 2 * H, 2 * W
        Tp = jnp.pad(T, ((0, 0), (0, 0), (0, H), (0, W)))
        mask = jnp.zeros((PH, PW)).at[:H, :W].set(1.0)
        e_fft = jnp.fft.rfft2(jnp.sum(Tp ** 2, axis=1))
        energy = jnp.fft.irfft2(jnp.fft.rfft2(mask)[None] * jnp.conj(e_fft), s=(PH, PW))
        dy = jnp.minimum(jnp.arange(PH), PH - jnp.arange(PH))
        dx = jnp.minimum(jnp.arange(PW), PW - jnp.arange(PW))
        valid = (dy[:, None] <= H // 2) & (dx[None, :] <= W // 2)
    T_fft = jnp.fft.rfft2(Tp)

    def align(vis):
        Xp = vis if circular else jnp.pad(vis, ((0, 0), (0, H), (0, W)))
        corr = jnp.fft.irfft2(jnp.sum(jnp.fft.rfft2(Xp)[None] * jnp.conj(T_fft), axis=1),
                              s=(PH, PW))
        mse_map = (jnp.sum(vis ** 2) + energy - 2 * corr) / N
        if valid is not None:
            mse_map = jnp.where(valid[None], mse_map, jnp.inf)
        flat = mse_map.reshape(K, -1)
        best = jnp.argmin(flat, axis=1)
        mse = jnp.maximum(jnp.min(flat, axis=1), 0.0)
        shifted = jnp.stack([
            jnp.roll(Tp[k], (best[k] // PW, best[k] % PW), axis=(-2, -1))[:, :H, :W]
            for k in range(K)])
        return mse, shifted

    return align


def _active(vis, dc):
    return (vis[3] > ACTIVE_ALPHA_THRESHOLD) if dc >= 4 else \
        (jnp.max(jnp.abs(vis[:3]), axis=0) > 0.10)


def make_classifier(ctx):
    """Etikett-Urteil gegen die Emoji-Zielbilder (wie bisher)."""
    align = make_aligner(ctx.targets, ctx.circular)
    K, dc = ctx.K, ctx.dc

    def classify_one(vis):
        mse, shifted = align(vis)
        err = jnp.mean((vis[None, :3] - shifted[:, :3]) ** 2, axis=1)   # (K,H,W)
        active = _active(vis, dc)
        n_active = jnp.sum(active)
        if K == 1:
            fractions = jnp.where(n_active > 0, 1.0, 0.0)[None]
        else:
            ordered = jnp.sort(err, axis=0)
            counted = (ordered[0] < ordered[1]) & active
            nearest = jnp.argmin(err, axis=0)
            fractions = jnp.stack([jnp.sum(counted & (nearest == k)) for k in range(K)]
                                  ) / jnp.maximum(n_active, 1)
        mixed = jnp.sum(fractions >= MIXED_MIN_FRACTION) >= 2
        label = jnp.where(mixed, K, jnp.argmin(mse))
        label = jnp.where(n_active == 0, K + 1, label).astype(jnp.int32)
        return label, fractions, mse, n_active

    return jax.vmap(classify_one)


def make_ref_compare(ctx, refs):
    """Vergleich mit den vom Modell selbst gewachsenen Mustern (fuer 'echt'):
    MSE und Formueberlappung (IoU der sichtbaren Flaechen) je Attraktor."""
    align = make_aligner(refs, ctx.circular)
    dc = ctx.dc

    def one(vis):
        mse, shifted = align(vis)
        a = _active(vis, dc)
        m = jax.vmap(lambda s: _active(s, dc))(shifted)            # (K,H,W)
        inter = jnp.sum(a[None] & m, axis=(1, 2))
        union = jnp.sum(a[None] | m, axis=(1, 2))
        return mse, inter / jnp.maximum(union, 1)

    return eqx.filter_jit(jax.vmap(one))


# ======================================================================
# REIZ UND ROLLOUT
# ======================================================================

def make_step_functions(ctx):
    H, W, dc = ctx.H, ctx.W, ctx.dc
    circular = ctx.circular
    classify = make_classifier(ctx)
    yy = jnp.arange(H)[None, :, None]
    xx = jnp.arange(W)[None, None, :]

    def apply_signal(x, k, P):
        dy = jnp.abs(yy - P["cy"][:, None, None])
        dx = jnp.abs(xx - P["cx"][:, None, None])
        if circular:
            dy = jnp.minimum(dy, H - dy)
            dx = jnp.minimum(dx, W - dx)
        inside = (dy ** 2 + dx ** 2) <= (P["radius"][:, None, None] ** 2)
        on = inside & (k < P["duration"])[:, None, None] & P["active"][:, None, None]
        e = lambda v: v[:, :, None, None]
        x_new = e(P["a"]) * x + e(P["b"]) + e(P["g"]) * P["G"]
        # Rauschen nur erzeugen, wenn ein Lauf im Batch es braucht
        noise = jax.lax.cond(
            jnp.any(P["ns"] > 0),
            lambda: jax.vmap(lambda kk: jr.normal(jr.fold_in(kk, k + 1_000_000),
                                                   x.shape[1:]))(P["key"]),
            lambda: jnp.zeros_like(x))
        x_new = x_new + e(P["ns"]) * noise
        return jnp.where(on[:, None], x_new, x)

    @eqx.filter_jit
    def signal_step(model, x, k, P):
        x = apply_signal(x, k, P)
        keys = jax.vmap(jr.fold_in, in_axes=(0, None))(P["key"], k)
        x = jax.vmap(lambda s, kk: model(s, key=kk))(x, keys)
        label, fractions, mse, n_active = classify(x[:, :dc])
        return x, label, fractions, mse, n_active

    @eqx.filter_jit
    def grow_step(model, x, keys, t):
        keys = jax.vmap(jr.fold_in, in_axes=(0, None))(keys, t)
        return jax.vmap(lambda s, kk: model(s, key=kk))(x, keys)

    @eqx.filter_jit
    def classify_batch(x):
        return classify(x[:, :dc])

    return signal_step, grow_step, classify_batch


def seed_state(ctx, attractor):
    x = np.zeros((ctx.n_channels, ctx.H, ctx.W), np.float32)
    x[:ctx.dc, ctx.seed_y, ctx.seed_x] = attractor.seed_pixel
    return x


def growth_key(a_index, repeat):
    return jr.fold_in(jr.fold_in(jr.PRNGKey(SEED), 1), a_index * 100003 + repeat)


def post_key(repeat):
    # gleiche Rauschfolge fuer alle Bedingungen mit derselben Wiederholungsnummer
    return jr.fold_in(jr.fold_in(jr.PRNGKey(SEED), 2), repeat)


def is_oom(exc):
    text = f"{type(exc).__name__} {exc}"
    return any(m in text for m in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY", "out of memory",
                                   "cuFFT", "CUBIN"))


def free_gpu_memory():
    gc.collect()
    jax.clear_caches()
    gc.collect()


def grow_states(ctx, grow_step, sources, repeats, times):
    """Waechst jeden Attraktor in 'sources' 'repeats'-mal und speichert die
    Zustaende zu den Zeitpunkten 'times' -> dict[(a, r, t)]."""
    global BATCH_SIZE
    times = sorted(set(int(t) for t in times))
    jobs = [(a, r) for a in sources for r in range(repeats)]
    out = {}
    start = 0
    while start < len(jobs):
        chunk = jobs[start:start + BATCH_SIZE]
        try:
            x = jnp.asarray(np.stack([seed_state(ctx, ctx.attractors[a]) for a, _ in chunk]))
            keys = jnp.stack([growth_key(a, r) for a, r in chunk])
            snaps, t = {}, 0
            for target_t in times:
                while t < target_t:
                    x = grow_step(ctx.model, x, keys, jnp.int32(t))
                    t += 1
                snaps[target_t] = np.asarray(x)
        except Exception as exc:
            if not is_oom(exc) or BATCH_SIZE <= MIN_BATCH_SIZE:
                raise
            x = None
            free_gpu_memory()
            BATCH_SIZE = max(MIN_BATCH_SIZE, BATCH_SIZE // 2)
            print(f"    GPU-Speicher voll -> Batchgroesse jetzt {BATCH_SIZE}", flush=True)
            continue
        for target_t, host in snaps.items():
            for i, (a, r) in enumerate(chunk):
                out[(a, r, target_t)] = host[i]
        start += len(chunk)
    return out


@jax.jit
def _capture(fin, x, k, ends):
    hit = (k + 1 == ends)[:, None, None, None]
    return jnp.where(hit, x[:, :fin.shape[1]], fin)


def body_stats(ctx, state):
    """Mittelwert und Streuung je Kanal ueber die lebenden Zellen."""
    alive = state[3] > ACTIVE_ALPHA_THRESHOLD if ctx.dc >= 4 else \
        np.max(np.abs(state[:3]), axis=0) > 0.1
    cells = state[:, alive]
    if cells.shape[1] == 0:
        return np.zeros(state.shape[0], np.float32), np.zeros(state.shape[0], np.float32)
    return cells.mean(axis=1).astype(np.float32), cells.std(axis=1).astype(np.float32)


def signal_coefficients(ctx, c, stats):
    """Je Kanal a, b, g, Rauschstaerke und Quelle der Zusatzkarte G fuer
    x_neu = a*x + b + g*G + ns*Rauschen."""
    C, dc = ctx.n_channels, ctx.dc
    a = np.ones(C, np.float32)
    b = np.zeros(C, np.float32)
    g = np.zeros(C, np.float32)
    ns = np.zeros(C, np.float32)
    G = None                                   # None | "target" | "onset"
    sig = c["signal"]
    S, T, t = c["source_index"], c["target_index"], c["t_signal"]
    black = np.zeros(dc, np.float32)
    if dc >= 4:
        black[3] = 1.0
    if sig == "none":
        pass
    elif sig == "seed_rgba":
        a[:dc] = 0; b[:dc] = ctx.attractors[T].seed_pixel
    elif sig == "seed_cell":
        a[:] = 0; b[:dc] = ctx.attractors[T].seed_pixel
    elif sig == "black_rgba":
        a[:dc] = 0; b[:dc] = black
    elif sig == "black_zero":
        a[:] = 0; b[:dc] = black
    elif sig == "transparent_rgba":
        a[:dc] = 0
    elif sig == "transparent_zero":
        a[:] = 0
    elif sig == "graft":
        a[:] = 0; g[:] = 1; G = "target"
    elif sig == "graft_hidden":
        a[dc:] = 0; g[dc:] = 1; G = "target"
    elif sig == "target_mean":
        a[:] = 0; b[:] = stats[(T, t)][0]
    elif sig in ("push1", "push2"):
        s = 1.0 if sig == "push1" else 2.0
        a[:] = 0; g[:] = 1; G = "onset"
        b[:] = s * (stats[(T, t)][0] - stats[(S, t)][0])
    elif sig == "noise":
        ns[:] = stats[(S, t)][1]
    else:
        raise ValueError(f"unbekannter Reiztyp {sig!r}")
    return a, b, g, ns, G


# ======================================================================
# BEDINGUNGEN
# ======================================================================

def disc_mask(ctx, y, x, r):
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W]
    dy, dx = np.abs(yy - y), np.abs(xx - x)
    if ctx.circular:
        dy, dx = np.minimum(dy, ctx.H - dy), np.minimum(dx, ctx.W - dx)
    return (dy ** 2 + dx ** 2) <= r ** 2


def positions_for(ctx, ref_state):
    """Positionen (name, y, x) fuer ein Ausgangsmuster."""
    out, seen = [], set()
    alive = ref_state[3] > ACTIVE_ALPHA_THRESHOLD if ctx.dc >= 4 else \
        np.max(np.abs(ref_state[:3]), axis=0) > 0.1
    for p in POSITIONS:
        if p == "seed":
            cand = [("seed", ctx.seed_y, ctx.seed_x)]
        elif isinstance(p, str) and p.startswith(("body:", "grid:")):
            step = int(p.split(":")[1])
            cand = []
            for y in range(ctx.seed_y % step, ctx.H, step):
                for x in range(ctx.seed_x % step, ctx.W, step):
                    if p.startswith("body:") and not alive[y, x]:
                        continue
                    cand.append((f"dy{y - ctx.seed_y:+d}_dx{x - ctx.seed_x:+d}", y, x))
        else:
            y, x = int(p[0]), int(p[1])
            cand = [(f"dy{y - ctx.seed_y:+d}_dx{x - ctx.seed_x:+d}", y, x)]
        for name, y, x in cand:
            if (y, x) not in seen:
                seen.add((y, x))
                out.append((name, y, x))
    return out


def make_cond(ctx, src, tgt, signal, pos, r, d, t, cache):
    """tgt: Attraktor oder None (neutraler Reiz: irgendein Ziel).
    signal 'none' = Kontrolle ohne Reiz. t = Wachstumsschritt beim Reizbeginn."""
    if signal == "none":
        c = dict(source=src.name, source_index=src.index, target="", target_index=-1,
                 signal="none", position="none", y=ctx.seed_y, x=ctx.seed_x,
                 radius=0, n_pixels=0, duration=1)
    else:
        name, y, x = pos
        k = (y, x, r)
        if k not in cache:
            cache[k] = int(disc_mask(ctx, y, x, r).sum())
        c = dict(source=src.name, source_index=src.index,
                 target=(tgt.name if tgt is not None else ANY),
                 target_index=(tgt.index if tgt is not None else -1),
                 signal=signal, position=name, y=y, x=x, radius=int(r),
                 n_pixels=cache[k], duration=int(d))
    c["t_signal"] = int(t)
    c["dose"] = c["n_pixels"] * c["duration"]
    c["key"] = (f"{c['source']}|{c['target']}|{c['y']},{c['x']}|r{c['radius']}|"
                f"d{c['duration']}|{c['signal']}|t{c['t_signal']}")
    return c


def group_key(c):
    return (c["signal"], c["t_signal"], c["source"], c["target"], c["position"], c["duration"])


# ======================================================================
# ERGEBNIS-SPEICHER + RECHNEN
# ======================================================================

ROLL_FIELDS = ["key", "repeat", "final_label", "outcome", "switched_to", "strict_switch",
               "label_after_stimulus", "change_after_stimulus", "ref_nearest",
               "ref_iou_switched", "ref_iou_source",
               "steps_to_switch", "max_target_fraction", "final_target_fraction",
               "final_source_fraction", "final_n_active"]


class Store:
    """rollouts.csv eines Modells: anhaengen nach jedem Batch, wiederaufnehmbar."""

    def __init__(self, path, resume):
        self.path = path
        self.rows = defaultdict(dict)          # key -> {repeat: row}
        if resume and path.is_file():
            with open(path) as f:
                rd = csv.DictReader(f)
                if "strict_switch" in (rd.fieldnames or []):
                    for r in rd:
                        self.rows[r["key"]][int(r["repeat"])] = r
                else:
                    print("  alte rollouts.csv ohne strenges Urteil -> wird neu gerechnet")
            if not self.rows and path.is_file():
                path.rename(path.with_name(f"rollouts_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
        elif path.is_file():
            path.unlink()
        new = not path.is_file()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=ROLL_FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def has(self, key, rep):
        return rep in self.rows.get(key, {})

    def add(self, row):
        row = {k: ("" if row.get(k) is None else row.get(k)) for k in ROLL_FIELDS}
        self.w.writerow(row)
        self.rows[row["key"]][int(row["repeat"])] = {k: str(v) for k, v in row.items()}

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

    def stats(self, key):
        rs = list(self.rows.get(key, {}).values())
        n = len(rs)
        if n == 0:
            return None
        oc = Counter(r["outcome"] for r in rs)
        num = lambda k: [float(r[k]) for r in rs if r[k] not in ("", None)]
        sts = num("steps_to_switch")
        tf, mt, iou = num("final_target_fraction"), num("max_target_fraction"), num("ref_iou_switched")
        n_strict = sum(r["strict_switch"] == "True" for r in rs)
        to = Counter(r["switched_to"] for r in rs if r["switched_to"])
        return dict(runs=n, n_switched=oc["switched"], switch_rate=oc["switched"] / n,
                    n_strict=n_strict, strict_rate=n_strict / n,
                    stay_rate=oc["stayed"] / n, intermediate_rate=oc["intermediate"] / n,
                    other_attractor_rate=oc["other_attractor"] / n,
                    no_pattern_rate=oc["no_pattern"] / n,
                    switched_to=" ".join(f"{k}:{v}" for k, v in to.most_common()),
                    median_steps_to_switch=(float(np.median(sts)) if sts else ""),
                    mean_final_target_fraction=(float(np.mean(tf)) if tf else ""),
                    mean_max_target_fraction=(float(np.mean(mt)) if mt else ""),
                    mean_iou_switched=(float(np.mean(iou)) if iou else ""))


def analyse_rollout(labels, fracs, n_active, end, c, ctx, ref_mse, ref_iou, change):
    K = ctx.K
    labels = labels[:end]
    final = int(labels[-1])
    S, T = c["source_index"], c["target_index"]
    neutral = (T < 0 and c["signal"] != "none")
    # Ziel fuer Anteile: festes Ziel, bei neutralen Reizen der staerkste andere Attraktor
    if neutral:
        others = [k for k in range(K) if k != S]
        T_eff = others[int(np.argmax(fracs[end - 1, others]))] if others else -1
    else:
        T_eff = T
    after = int(labels[min(c["duration"], end) - 1])

    if final == K + 1:
        outcome = "no_pattern"
    elif final == K:
        outcome = "intermediate"
    elif final == S:
        outcome = "stayed"
    elif final == T or (neutral and final != S):
        outcome = "switched"
    else:
        outcome = "other_attractor"

    row = dict(final_label=ctx.class_names[final], outcome=outcome,
               label_after_stimulus=ctx.class_names[after],
               change_after_stimulus=float(change),
               final_n_active=int(n_active[end - 1]),
               final_source_fraction=float(fracs[end - 1, S]),
               ref_nearest=ctx.names[int(np.argmin(ref_mse))],
               ref_iou_source=float(ref_iou[S]))
    if outcome == "switched":
        A = final
        row["switched_to"] = ctx.names[A]
        row["ref_iou_switched"] = float(ref_iou[A])
        # echt: Form passt zum selbst gewachsenen A, und A ist durch Wachstum
        # entstanden (nicht schon direkt nach dem Reiz da - oder, falls doch,
        # hat sich das Muster danach noch deutlich veraendert, z.B. frueh in
        # der Entwicklung). Schliesst stehen gebliebene Farbflecken und reine
        # Ersetzung (grosses Transplantat) aus.
        rebuilt = (after != A) or (change >= CHANGE_MIN)
        row["strict_switch"] = bool(rebuilt and int(np.argmin(ref_mse)) == A
                                    and ref_iou[A] >= IOU_MIN)
        not_a = np.nonzero(labels != A)[0]
        row["steps_to_switch"] = int(not_a[-1] + 2) if len(not_a) else 1
    else:
        row["strict_switch"] = False
    if T_eff >= 0:
        row["max_target_fraction"] = float(np.max(fracs[:end, T_eff]))
        row["final_target_fraction"] = float(fracs[end - 1, T_eff])
    return row


def run_jobs(ctx, jobs, grown, stats, signal_step, ref_compare, store, label=""):
    """jobs: Liste (cond, repeat). Rechnet nur, was noch nicht im Store ist.
    Laeuft der GPU-Speicher voll, wird die Batchgroesse halbiert."""
    global BATCH_SIZE
    jobs = [(c, r) for c, r in jobs if not store.has(c["key"], r)]
    if not jobs:
        print(f"  {label}: nichts zu rechnen (alles vorhanden)")
        return
    # Batches moeglichst gleichartig (gleiche Dauer -> gleiche Laenge)
    jobs.sort(key=lambda j: (j[0]["duration"], j[0]["t_signal"], j[0]["signal"],
                             j[0]["source_index"], j[1]))
    total = sum(c["duration"] + FREE_STEPS for c, _ in jobs)
    print(f"  {label}: {len(jobs)} Laeufe, {total:,} Laufschritte, "
          f"Batchgroesse {BATCH_SIZE}", flush=True)
    C = ctx.n_channels
    coef_cache = {}
    t0, done_steps, start = time.time(), 0, 0
    while start < len(jobs):
        bs = BATCH_SIZE
        batch = jobs[start:start + bs]
        n_real = len(batch)
        padded = batch + [batch[-1]] * (bs - n_real) if len(jobs) > bs else batch
        try:
            coefs = []
            for c, r in padded:
                ck = (c["signal"], c["source_index"], c["target_index"], c["t_signal"])
                if ck not in coef_cache:
                    coef_cache[ck] = signal_coefficients(ctx, c, stats)
                coefs.append(coef_cache[ck])
            x0 = np.stack([grown[(c["source_index"], r, c["t_signal"])] for c, r in padded])
            need_G = any(cf[4] is not None for cf in coefs)
            if need_G:
                G = np.zeros_like(x0)
                for i, ((c, r), cf) in enumerate(zip(padded, coefs)):
                    if cf[4] == "target":
                        G[i] = grown[(c["target_index"], r, c["t_signal"])]
                    elif cf[4] == "onset":
                        G[i] = x0[i]
                G = jnp.asarray(G)
            else:
                G = jnp.zeros((len(padded), C, ctx.H, ctx.W), jnp.float32)
            P = dict(
                cy=jnp.asarray([c["y"] for c, _ in padded], jnp.int32),
                cx=jnp.asarray([c["x"] for c, _ in padded], jnp.int32),
                radius=jnp.asarray([c["radius"] for c, _ in padded], jnp.float32),
                duration=jnp.asarray([c["duration"] for c, _ in padded], jnp.int32),
                active=jnp.asarray([c["signal"] != "none" for c, _ in padded]),
                a=jnp.asarray(np.stack([cf[0] for cf in coefs])),
                b=jnp.asarray(np.stack([cf[1] for cf in coefs])),
                g=jnp.asarray(np.stack([cf[2] for cf in coefs])),
                ns=jnp.asarray(np.stack([cf[3] for cf in coefs])),
                G=G,
                key=jnp.stack([post_key(r) for _, r in padded]),
            )
            ends = np.array([c["duration"] + FREE_STEPS for c, _ in padded])
            ends_j = jnp.asarray(ends, jnp.int32)
            dur_j = jnp.asarray([c["duration"] for c, _ in padded], jnp.int32)
            x = jnp.asarray(x0)
            fin = x[:, :ctx.dc]
            aft = x[:, :ctx.dc]
            series = []
            for k in range(int(ends.max())):
                x, lab, frac, _, nact = signal_step(ctx.model, x, jnp.int32(k), P)
                fin = _capture(fin, x, jnp.int32(k), ends_j)
                aft = _capture(aft, x, jnp.int32(k), dur_j)
                series.append((lab, frac, nact))
            labels = np.stack([np.asarray(s[0]) for s in series], axis=1)
            fracs = np.stack([np.asarray(s[1]) for s in series], axis=1)
            nacts = np.stack([np.asarray(s[2]) for s in series], axis=1)
            rmse, riou = ref_compare(fin)
            rmse, riou = np.asarray(rmse), np.asarray(riou)
            change = np.asarray(jnp.mean((fin - aft) ** 2, axis=(1, 2, 3)))
        except Exception as exc:
            if not is_oom(exc) or BATCH_SIZE <= MIN_BATCH_SIZE:
                raise
            x = fin = aft = series = P = G = None
            free_gpu_memory()
            BATCH_SIZE = max(MIN_BATCH_SIZE, BATCH_SIZE // 2)
            print(f"    GPU-Speicher voll -> Batchgroesse jetzt {BATCH_SIZE}, "
                  f"Batch wird wiederholt", flush=True)
            continue
        x = fin = aft = series = P = G = None
        for i, (c, r) in enumerate(batch):
            row = analyse_rollout(labels[i], fracs[i], nacts[i], int(ends[i]), c, ctx,
                                  rmse[i], riou[i], change[i])
            row.update(key=c["key"], repeat=r)
            store.add(row)
        store.flush()
        start += n_real
        done_steps += int(ends[:n_real].sum())
        el = time.time() - t0
        print(f"    {start}/{len(jobs)} Laeufe: {el / 60:.1f} min, "
              f"Rest ~{el / done_steps * (total - done_steps) / 60:.1f} min", flush=True)


# ======================================================================
# SCHWELLENSUCHE
# ======================================================================

def _hit(s):
    if not s:
        return False
    return (s["n_strict"] if REFINE_ON == "strict" else s["n_switched"]) > 0


def refine_radii(screen_stats):
    """Radien, die mit REP_CONFIRM Wiederholungen gerechnet werden sollen."""
    screen = sorted(screen_stats)
    hits = [r for r in screen if _hit(screen_stats[r])]
    if not hits:
        return []
    r_hi = hits[0]
    lower = [r for r in screen if r < r_hi]
    r_lo = lower[-1] if lower else -1
    between = [r for r in RADIUS_LADDER if r_lo < r < r_hi]
    return sorted(set(([r_lo] if r_lo >= 0 else []) + between +
                      [r for r in screen if r >= r_hi]))


def threshold_of(radius_stats, rate_key="strict_rate", min_runs=None):
    min_runs = REP_CONFIRM if min_runs is None else min_runs
    ok = {r: s for r, s in radius_stats.items() if s and s["runs"] >= min_runs}
    for r in sorted(ok):
        if ok[r][rate_key] >= SWITCH_THRESHOLD:
            return r
    return None


# ======================================================================
# EIN MODELL
# ======================================================================

def run_model(info, emoji_dir, args):
    ctx = prepare(info, emoji_dir)
    out_dir = OUTPUT_ROOT / ctx.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {ctx.tag} ====================")
    print(f"Bundle: {ctx.bundle}")
    print(f"{ctx.family}, Padding {ctx.padding}, Attraktoren {ctx.names}, "
          f"Leinwand {ctx.H}x{ctx.W}, Seed ({ctx.seed_y},{ctx.seed_x})")

    run_settings = dict(bundle=str(ctx.bundle.resolve()), GROW_STEPS=GROW_STEPS,
                        FREE_STEPS=FREE_STEPS, SEED=SEED, IOU_MIN=IOU_MIN,
                        CHANGE_MIN=CHANGE_MIN,
                        ACTIVE_ALPHA_THRESHOLD=ACTIVE_ALPHA_THRESHOLD,
                        MIXED_MIN_FRACTION=MIXED_MIN_FRACTION)
    sp = out_dir / "settings.json"
    resume = not args.no_resume
    if resume and sp.is_file() and (out_dir / "rollouts.csv").is_file():
        if json.loads(sp.read_text()).get("run_settings") != run_settings:
            bak = out_dir / f"rollouts_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            (out_dir / "rollouts.csv").rename(bak)
            print(f"  Einstellungen geaendert -> alte Ergebnisse nach {bak.name}")
    sp.write_text(json.dumps(dict(
        run_settings=run_settings, SIGNALS=SIGNALS, T_SIGNALS=T_SIGNALS, POSITIONS=POSITIONS,
        DURATIONS=DURATIONS, RADIUS_LADDER=RADIUS_LADDER, SCREEN_RADII=SCREEN_RADII,
        REP_SCREEN=REP_SCREEN, REP_CONFIRM=REP_CONFIRM, REFINE_ON=REFINE_ON,
        SWITCH_THRESHOLD=SWITCH_THRESHOLD), indent=2))

    t_model = time.time()
    signal_step, grow_step, classify_batch = make_step_functions(ctx)

    # ---- Ausgangsmuster wachsen lassen + pruefen ----
    reps = max(REP_SCREEN, REP_CONFIRM)
    times = sorted(set(T_SIGNALS) | {GROW_STEPS})
    print(f"Wachse {ctx.K} Attraktoren x {reps} Wiederholungen, Zeitpunkte {times} ...")
    t0 = time.time()
    grown = grow_states(ctx, grow_step, [a.index for a in ctx.attractors], reps, times)
    print(f"  fertig in {time.time() - t0:.1f} s")
    ref = {a.index: grown[(a.index, 0, GROW_STEPS)] for a in ctx.attractors}
    refs_rgba = np.stack([ref[a.index][:ctx.dc] for a in ctx.attractors])
    ref_compare = make_ref_compare(ctx, refs_rgba)
    stats = {(a.index, t): body_stats(ctx, grown[(a.index, 0, t)])
             for a in ctx.attractors for t in times}
    save_development_strip(ctx, grown, times, out_dir)

    lab, fr, _, _ = classify_batch(jnp.asarray(np.stack([ref[a.index] for a in ctx.attractors])))
    # natuerliche Streuung: Formueberlappung zwischen Wiederholungen desselben Attraktors
    others = np.stack([grown[(a.index, 1, GROW_STEPS)][:ctx.dc] for a in ctx.attractors]) \
        if reps > 1 else refs_rgba
    _, self_iou = ref_compare(jnp.asarray(others))
    self_iou = np.asarray(self_iou)
    base, valid_sources = {}, []
    for i, a in enumerate(ctx.attractors):
        own = float(np.asarray(fr[i])[a.index])
        label = ctx.class_names[int(lab[i])]
        base[a.name] = dict(label=label, own_fraction=own, ok=(int(lab[i]) == a.index),
                            self_iou=float(self_iou[i, a.index]))
        print(f"  {a.name} ohne Reiz: {label} (eigener Anteil {own:.2f}, Form-IoU "
              f"zwischen zwei Wiederholungen {self_iou[i, a.index]:.2f})")
        Image.fromarray(to_rgb_uint8(ref[a.index][:ctx.dc])).resize(
            (ctx.W * 2, ctx.H * 2), Image.NEAREST).save(out_dir / f"grown_{a.name}.png")
        if base[a.name]["ok"] or not SKIP_UNSTABLE_SOURCES:
            valid_sources.append(a)
        else:
            ctx.warnings.append(f"{a.name} waechst ohne Reiz nicht zu sich selbst "
                                f"({label}); Richtungen aus {a.name} werden uebersprungen.")
        if base[a.name]["self_iou"] < IOU_MIN:
            ctx.warnings.append(f"{a.name}: zwei Wiederholungen ueberlappen nur mit IoU "
                                f"{base[a.name]['self_iou']:.2f} < IOU_MIN={IOU_MIN}; das "
                                "strenge Urteil ist fuer dieses Ziel zu streng.")
    for w in ctx.warnings:
        print("  WARNUNG:", w)

    store = Store(out_dir / "rollouts.csv", resume)
    cache = {}
    positions = {a.index: positions_for(ctx, ref[a.index]) for a in valid_sources}
    for a in valid_sources:
        print(f"  {a.name}: {len(positions[a.index])} Reizpositionen")

    # ---- Stufe 0: Kontrollen ohne Reiz ----
    controls = [make_cond(ctx, a, None, "none", None, 0, 1, GROW_STEPS, cache)
                for a in ctx.attractors]
    run_jobs(ctx, [(c, r) for c in controls for r in range(REP_CONFIRM)], grown, stats,
             signal_step, ref_compare, store, "Kontrollen ohne Reiz")

    # ---- Stufe 1: Screen ----
    groups = {}
    for sig in SIGNALS:
        kind = SIGNAL_INFO[sig][0]
        for s in valid_sources:
            tgts = [t for t in ctx.attractors if t.index != s.index] if kind == "ziel" else [None]
            for t in tgts:
                for ts, pos, d in itertools.product(T_SIGNALS, positions[s.index], DURATIONS):
                    conds = {r: make_cond(ctx, s, t, sig, pos, r, d, ts, cache)
                             for r in SCREEN_RADII}
                    groups[group_key(conds[SCREEN_RADII[0]])] = conds
    run_jobs(ctx, [(c, r) for g in groups.values() for c in g.values()
                   for r in range(REP_SCREEN)], grown, stats, signal_step, ref_compare,
             store, "Screen")

    # ---- Stufe 2: Verfeinerung ----
    refine_jobs = []
    for gk, conds in groups.items():
        radii = refine_radii({r: store.stats(c["key"]) for r, c in conds.items()})
        if not radii:
            continue
        c0 = next(iter(conds.values()))
        src = ctx.attractors[c0["source_index"]]
        tgt = ctx.attractors[c0["target_index"]] if c0["target_index"] >= 0 else None
        pos = (c0["position"], c0["y"], c0["x"])
        for r in radii:
            if r not in conds:
                conds[r] = make_cond(ctx, src, tgt, c0["signal"], pos, r, c0["duration"],
                                     c0["t_signal"], cache)
            refine_jobs += [(conds[r], rep) for rep in range(REP_CONFIRM)]
    n_ref = len({group_key(c) for c, _ in refine_jobs})
    print(f"  Verfeinerung: {n_ref} von {len(groups)} Gruppen haben im Screen "
          f"{'echt' if REFINE_ON == 'strict' else 'nach Etikett'} umgeschaltet")
    run_jobs(ctx, refine_jobs, grown, stats, signal_step, ref_compare, store, "Verfeinerung")
    store.close()

    # ---- Auswertung ----
    group_rows, cond_rows = [], []
    for gk, conds in groups.items():
        rstats = {r: store.stats(c["key"]) for r, c in conds.items()}
        for r, c in sorted(conds.items()):
            if rstats[r]:
                cond_rows.append({**{k: c[k] for k in ("signal", "t_signal", "source", "target",
                                                       "position",
                                                       "y", "x", "radius", "n_pixels",
                                                       "duration", "dose")}, **rstats[r]})
        c0 = conds[SCREEN_RADII[0]]
        r_strict = threshold_of(rstats, "strict_rate")
        r_label = threshold_of(rstats, "switch_rate", min_runs=1)
        max_strict = max((s["strict_rate"] for s in rstats.values() if s), default=0.0)
        max_label = max((s["switch_rate"] for s in rstats.values() if s), default=0.0)
        frac = [(s["mean_final_target_fraction"], r) for r, s in rstats.items()
                if s and s["mean_final_target_fraction"] != ""]
        best_frac, best_frac_r = max(frac) if frac else ("", "")
        status = ("schaltbar" if r_strict is not None else
                  "sporadisch" if max_strict > 0 else "kein_umschalten")
        to = Counter()
        for s in rstats.values():
            if s and s["switched_to"]:
                for part in s["switched_to"].split():
                    n, v = part.split(":")
                    to[n] += int(v)
        row = dict(signal=c0["signal"], t_signal=c0["t_signal"], source=c0["source"],
                   target=c0["target"],
                   position=c0["position"], y=c0["y"], x=c0["x"], duration=c0["duration"],
                   status=status,
                   r_threshold=("" if r_strict is None else r_strict),
                   n_pixels_threshold=("" if r_strict is None else conds[r_strict]["n_pixels"]),
                   dose_threshold=("" if r_strict is None else conds[r_strict]["dose"]),
                   strict_rate_at_threshold=("" if r_strict is None else rstats[r_strict]["strict_rate"]),
                   steps_to_switch_at_threshold=("" if r_strict is None else
                                                 rstats[r_strict]["median_steps_to_switch"]),
                   switched_to_at_threshold=("" if r_strict is None else rstats[r_strict]["switched_to"]),
                   max_strict_rate=max_strict,
                   label_r_threshold=("" if r_label is None else r_label),
                   label_max_switch_rate=max_label,
                   label_switched_to=" ".join(f"{k}:{v}" for k, v in to.most_common()),
                   best_final_target_fraction=best_frac, radius_of_best_fraction=best_frac_r,
                   radii_tested=" ".join(str(r) for r in sorted(conds)))
        s_max = rstats.get(max(SCREEN_RADII))
        if s_max:
            outcomes = {"umgeschaltet (Etikett)": s_max["switch_rate"], "bleibt": s_max["stay_rate"],
                        "Mischung": s_max["intermediate_rate"],
                        "dritter Attraktor": s_max["other_attractor_rate"],
                        "leer": s_max["no_pattern_rate"]}
            row["outcome_at_max_screen_radius"] = max(outcomes, key=outcomes.get)
        group_rows.append(row)

    control_rows = []
    for c in controls:
        s = store.stats(c["key"])
        control_rows.append(dict(source=c["source"], **(s or {})))

    write_csv(out_dir / "thresholds.csv", group_rows)
    write_csv(out_dir / "conditions_summary.csv", cond_rows)
    write_csv(out_dir / "controls.csv", control_rows)
    sig_rows = signal_summary(ctx, group_rows, base)
    write_csv(out_dir / "signals_summary.csv", sig_rows)

    figures = []

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as exc:
            print(f"  WARNUNG: {fn.__name__} fehlgeschlagen: {type(exc).__name__}: {exc}")
            return []

    figures += safe(plot_model_overview, ctx, group_rows, out_dir)
    figures += safe(plot_threshold_maps, ctx, group_rows, ref, out_dir)
    figures += safe(run_examples, ctx, choose_examples(ctx, group_rows),
                    grown, stats, signal_step, out_dir)

    runtime = time.time() - t_model
    write_model_report(ctx, base, control_rows, sig_rows, group_rows, figures, out_dir, runtime)
    print(f"  {ctx.tag} fertig in {runtime / 60:.1f} min -> {out_dir / 'report.md'}")
    return dict(ctx=ctx, sig_rows=sig_rows, base=base, group_rows=group_rows,
                out_dir=out_dir, runtime=runtime)


def signal_summary(ctx, group_rows, base):
    """Je (Reiztyp, Quelle, Ziel): wie oft schaltbar, bester Reiz."""
    rows = []
    by = defaultdict(list)
    for g in group_rows:
        by[(g["signal"], g["t_signal"], g["source"], g["target"])].append(g)
    for (sig, ts, src, tgt), gs in sorted(by.items(), key=lambda kv: (SIGNALS.index(kv[0][0]),) + kv[0][1:]):
        sw = [g for g in gs if g["status"] == "schaltbar"]
        best = min(sw, key=lambda g: (g["dose_threshold"], g["r_threshold"], g["duration"])) if sw else None
        row = dict(model=ctx.tag, signal=sig, t_signal=ts, source=src, target=tgt,
                   n_groups=len(gs), n_switchable=len(sw),
                   frac_switchable=len(sw) / len(gs),
                   n_sporadic=sum(g["status"] == "sporadisch" for g in gs),
                   n_label_switch=sum(g["label_max_switch_rate"] > 0 for g in gs),
                   label_switched_to=" ".join(sorted({p.split(":")[0] for g in gs
                                                      for p in g["label_switched_to"].split()})))
        if best:
            row.update(best_position=best["position"], best_radius=best["r_threshold"],
                       best_duration=best["duration"], best_dose=best["dose_threshold"],
                       best_rate=best["strict_rate_at_threshold"],
                       best_steps=best["steps_to_switch_at_threshold"],
                       best_switched_to=best["switched_to_at_threshold"])
        rows.append(row)
    for a in ctx.attractors:
        if not base[a.name]["ok"] and SKIP_UNSTABLE_SOURCES:
            rows.append(dict(model=ctx.tag, signal="alle", t_signal="", source=a.name, target="",
                             n_groups=0, n_switchable=0, frac_switchable="",
                             note="nicht auswertbar (Quelle instabil)"))
    return rows


def write_csv(path, rows, fields=None):
    if not rows:
        return
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_rgb_uint8(rgba):
    v = np.clip(np.asarray(rgba, np.float32), 0, 1)
    if v.shape[0] >= 4:
        a = v[3:4]
        rgb = v[:3] * a + (1 - a)
    else:
        rgb = v[:3]
    return (np.moveaxis(rgb, 0, -1) * 255).round().astype(np.uint8)


# ======================================================================
# ABBILDUNGEN
# ======================================================================

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.titlesize": 9, "figure.facecolor": "white",
    })
    return plt, LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)


def save_development_strip(ctx, grown, times, out_dir):
    """Die Ausgangsmuster zu den Reizzeitpunkten (Wiederholung 0)."""
    rows = [np.concatenate([to_rgb_uint8(grown[(a.index, 0, t)][:ctx.dc]) for t in times], 1)
            for a in ctx.attractors]
    Image.fromarray(np.concatenate(rows, 0)).save(out_dir / "development_at_signal_times.png")


def _grid_cells(group_rows, row_key, col_key):
    cell = defaultdict(list)
    for g in group_rows:
        cell[(row_key(g), col_key(g))].append(g)
    return cell


def _heat(ax, cmap, M, texts, xt, yt, title):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9)
    return im


def plot_model_overview(ctx, group_rows, out_dir):
    """Zeilen = Reiztypen, Spalten = Reizzeitpunkte. Zelle = Anteil der
    getesteten (Richtung, Ort, Dauer)-Kombinationen, die ECHT umschalten.
    Text: echt-Anteil, kleinstes r*, und in Klammern wie viele nur nach
    Etikett umschalten."""
    plt, cmap = _mpl()
    sigs = [s for s in SIGNALS if any(g["signal"] == s for g in group_rows)]
    times = sorted({g["t_signal"] for g in group_rows})
    if not sigs:
        return []
    cell = _grid_cells(group_rows, lambda g: g["signal"], lambda g: g["t_signal"])
    M = np.full((len(sigs), len(times)), np.nan)
    T = [["" for _ in times] for _ in sigs]
    for i, s in enumerate(sigs):
        for j, t in enumerate(times):
            gs = cell.get((s, t), [])
            if not gs:
                continue
            sw = [g for g in gs if g["status"] == "schaltbar"]
            lab = sum(g["label_max_switch_rate"] > 0 for g in gs)
            M[i, j] = len(sw) / len(gs)
            T[i][j] = (f"{M[i, j] * 100:.0f}%" + (f" r*>={min(g['r_threshold'] for g in sw)}" if sw else "")
                       + f"\n(Etikett {lab}/{len(gs)})")
    fig, ax = plt.subplots(figsize=(1.6 * len(times) + 3.2, 0.5 * len(sigs) + 1.4),
                           constrained_layout=True)
    im = _heat(ax, cmap, M, T, [f"t={t}" + (" fertig" if t == GROW_STEPS else "") for t in times],
               sigs, f"{ctx.tag}: Anteil echt umschaltender Reize je Reiztyp und Reizbeginn")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="echt schaltbar [%]")
    cb.set_ticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    p = out_dir / "overview.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


def plot_threshold_maps(ctx, group_rows, ref, out_dir):
    """Karte je (Reiztyp, Zeitpunkt, Quelle, Ziel) mit mindestens einem echten
    Umschalten: Punkt = schaltbar ab r* (Zahl), Ring = sporadisch, x = nie."""
    plt, _ = _mpl()
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("thr", SEQ_RAMP[2:][::-1])
    ladder = sorted(set(RADIUS_LADDER) | set(SCREEN_RADII))
    norm = Normalize(vmin=0, vmax=len(ladder) - 1)
    by = defaultdict(list)
    for g in group_rows:
        by[(g["signal"], g["t_signal"], g["source"], g["target"])].append(g)
    keys = [k for k, gs in by.items() if any(g["status"] != "kein_umschalten" for g in gs)]
    keys.sort(key=lambda k: (-k[1], SIGNALS.index(k[0]), k[2], k[3]))
    map_dir = out_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    paths = []
    for (sig, ts, src, tgt) in keys[:MAX_MAPS_PER_MODEL]:
        gs = by[(sig, ts, src, tgt)]
        fig, axes = plt.subplots(1, len(DURATIONS), figsize=(2.4 * len(DURATIONS) + 1.2, 3.1),
                                 squeeze=False, constrained_layout=True)
        bg = to_rgb_uint8(ref[ctx.names.index(src)][:ctx.dc])
        sc = None
        for j, d in enumerate(DURATIONS):
            ax = axes[0][j]
            ax.imshow(bg, alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            sel = [g for g in gs if g["duration"] == d]
            for st, kw in (("kein_umschalten", dict(marker="x", s=18, c=MUTED, linewidths=1)),
                           ("sporadisch", dict(marker="o", s=40, facecolors="none",
                                               edgecolors=CAT[1], linewidths=1.3))):
                pts = [g for g in sel if g["status"] == st]
                if pts:
                    ax.scatter([g["x"] for g in pts], [g["y"] for g in pts], **kw)
            sw = [g for g in sel if g["status"] == "schaltbar"]
            if sw:
                sc = ax.scatter([g["x"] for g in sw], [g["y"] for g in sw],
                                c=[ladder.index(g["r_threshold"]) for g in sw], cmap=cmap,
                                norm=norm, s=60, edgecolors="white", linewidths=1.5)
                for g in sw:
                    ax.annotate(str(g["r_threshold"]), (g["x"], g["y"]), fontsize=6,
                                xytext=(4, 4), textcoords="offset points")
            ax.set_title(f"D = {d}")
        if sc is not None:
            cb = fig.colorbar(sc, ax=axes, shrink=0.75, label="r* [px]")
            cb.set_ticks(range(len(ladder)), labels=[str(r) for r in ladder])
            cb.outline.set_visible(False)
        fig.suptitle(f"{ctx.tag}: {src} -> {tgt}, Reiz {sig}, Beginn t={ts}\n"
                     f"Punkt = echt schaltbar ab r*, Ring = sporadisch, x = nie "
                     f"(Hintergrund: fertiges {src})", fontsize=8)
        p = map_dir / f"map_{sig}_t{ts}_{src}_to_{tgt.replace('*', 'ANY')}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    if len(keys) > MAX_MAPS_PER_MODEL:
        print(f"  Hinweis: {len(keys)} Karten moeglich, {MAX_MAPS_PER_MODEL} gezeichnet "
              "(spaete Zeitpunkte zuerst; MAX_MAPS_PER_MODEL erhoehen fuer mehr)")
    return paths


def plot_all_models_overview(results, path):
    """Ein Feld je Modell: Reiztypen x Reizzeitpunkte, Anteil echt schaltbar."""
    plt, cmap = _mpl()
    results = [r for r in results if r["group_rows"]]
    if not results:
        return
    n = len(results)
    cols = 3 if n > 2 else n
    rows_n = math.ceil(n / cols)
    times = sorted({g["t_signal"] for r in results for g in r["group_rows"]})
    sigs = [s for s in SIGNALS if any(g["signal"] == s for r in results for g in r["group_rows"])]
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * (1.0 * len(times) + 2.6),
                                                    rows_n * (0.34 * len(sigs) + 1.3)),
                             squeeze=False, constrained_layout=True)
    im = None
    for idx, res in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        cell = _grid_cells(res["group_rows"], lambda g: g["signal"], lambda g: g["t_signal"])
        M = np.full((len(sigs), len(times)), np.nan)
        T = [["" for _ in times] for _ in sigs]
        for i, s in enumerate(sigs):
            for j, t in enumerate(times):
                gs = cell.get((s, t), [])
                if gs:
                    M[i, j] = sum(g["status"] == "schaltbar" for g in gs) / len(gs)
                    T[i][j] = f"{M[i, j] * 100:.0f}"
        im = _heat(ax, cmap, M, T, [str(t) for t in times],
                   sigs if idx % cols == 0 else [""] * len(sigs), res["ctx"].tag)
        ax.set_xlabel("Reizbeginn t")
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label="echt schaltbar [%]")
    cb.set_ticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    fig.suptitle(f"Echtes Umschalten je Reiztyp und Reizbeginn (t={GROW_STEPS} = fertiges Muster)",
                 fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def choose_examples(ctx, group_rows):
    """Je Reiztyp der spaeteste Zeitpunkt mit echtem Umschalten (kleinste Dosis),
    bis N_EXAMPLES_PER_MODEL. Gibt es keins, die Reize mit dem groessten
    Zielanteil am fertigen Muster."""
    picks = []
    for sig in SIGNALS:
        sw = [g for g in group_rows if g["signal"] == sig and g["status"] == "schaltbar"]
        if sw:
            t_max = max(g["t_signal"] for g in sw)
            g = min([g for g in sw if g["t_signal"] == t_max],
                    key=lambda g: (g["dose_threshold"], g["r_threshold"]))
            picks.append((-t_max, g["dose_threshold"],
                          f"{sig}_t{t_max}_{g['source']}_to_{g['target'].replace('*', 'ANY')}_echt",
                          g, g["r_threshold"]))
    if not picks:
        cand = [g for g in group_rows if g["t_signal"] == GROW_STEPS and
                g["best_final_target_fraction"] != ""]
        cand.sort(key=lambda g: -g["best_final_target_fraction"])
        for g in cand[:2]:
            picks.append((0, 0, f"{g['signal']}_t{g['t_signal']}_{g['source']}_to_"
                                f"{g['target'].replace('*', 'ANY')}_am_naechsten", g,
                          g["radius_of_best_fraction"]))
    picks.sort(key=lambda p: (p[0], p[1]))
    return [(tag, g, r) for _, _, tag, g, r in picks[:N_EXAMPLES_PER_MODEL]]


def run_examples(ctx, examples, grown, stats, signal_step, out_dir):
    if not examples:
        return []
    plt, _ = _mpl()
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    paths = []
    for tag, g, r in examples:
        src = ctx.attractors[ctx.names.index(g["source"])]
        tgt = ctx.attractors[ctx.names.index(g["target"])] if g["target"] != ANY else None
        c = make_cond(ctx, src, tgt, g["signal"], (g["position"], g["y"], g["x"]), int(r),
                      g["duration"], g["t_signal"], {})
        a, b, gg, ns, Gk = signal_coefficients(ctx, c, stats)
        x0 = grown[(src.index, 0, c["t_signal"])]
        if Gk == "target":
            G = grown[(tgt.index, 0, c["t_signal"])]
        elif Gk == "onset":
            G = x0
        else:
            G = np.zeros_like(x0)
        P = dict(cy=jnp.asarray([c["y"]], jnp.int32), cx=jnp.asarray([c["x"]], jnp.int32),
                 radius=jnp.asarray([c["radius"]], jnp.float32),
                 duration=jnp.asarray([c["duration"]], jnp.int32),
                 active=jnp.asarray([True]), a=jnp.asarray(a[None]), b=jnp.asarray(b[None]),
                 g=jnp.asarray(gg[None]), ns=jnp.asarray(ns[None]), G=jnp.asarray(G[None]),
                 key=jnp.stack([post_key(0)]))
        end = c["duration"] + FREE_STEPS
        wanted = sorted({end if s == "end" else int(s) for s in EXAMPLE_SNAPSHOT_STEPS
                         if s == "end" or 1 <= int(s) <= end})
        x = jnp.asarray(x0[None])
        frames, gif = [(0, np.asarray(x[0, :ctx.dc]), None)], []
        for k in range(end):
            x, lab, *_ = signal_step(ctx.model, x, jnp.int32(k), P)
            if k + 1 in wanted:
                frames.append((k + 1, np.asarray(x[0, :ctx.dc]), int(lab[0])))
            if SAVE_GIFS and k % 2 == 0:
                gif.append(to_rgb_uint8(np.asarray(x[0, :ctx.dc])))
        fig, axes = plt.subplots(1, len(frames), figsize=(1.6 * len(frames), 2.1))
        for ax, (k, rgba, lab) in zip(np.atleast_1d(axes), frames):
            ax.imshow(to_rgb_uint8(rgba))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(("vor Reiz" if k == 0 else f"+{k}") +
                         ("" if lab is None else f"\n{ctx.class_names[lab]}"), fontsize=7)
        np.atleast_1d(axes)[0].add_patch(plt.Circle((c["x"], c["y"]), max(c["radius"], 0.7),
                                                    fill=False, color=CAT[1], lw=1))
        fig.suptitle(f"{ctx.tag} {tag}: r={c['radius']} D={c['duration']} @ {c['position']}, "
                     f"Beginn t={c['t_signal']}", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        p = ex_dir / f"{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
        if SAVE_GIFS and gif:
            imgs = [Image.fromarray(f).resize((280, 280), Image.NEAREST) for f in gif]
            imgs[0].save(ex_dir / f"{tag}.gif", save_all=True, append_images=imgs[1:],
                         duration=60, loop=0)
    return paths


# ======================================================================
# BERICHTE
# ======================================================================

def pct(v):
    return "" if v in ("", None) else f"{100 * float(v):.0f} %"


def latest_switch_time(group_rows, signal):
    ts = [g["t_signal"] for g in group_rows if g["signal"] == signal and g["status"] == "schaltbar"]
    return max(ts) if ts else None


def write_model_report(ctx, base, control_rows, sig_rows, group_rows, figures, out_dir, runtime):
    L = [f"# Signal-Schwellen-Scan: {ctx.tag}\n",
         f"Modell: `{ctx.bundle}`  ",
         f"{ctx.family}, Padding {ctx.padding}, Training-t {ctx.train_t}, Leinwand "
         f"{ctx.H}x{ctx.W}, Seed ({ctx.seed_y}, {ctx.seed_x})  ",
         f"Reizbeginn {T_SIGNALS} (fertig = {GROW_STEPS}), Dauern {DURATIONS}, Screen-Radien "
         f"{SCREEN_RADII}, Wdh. {REP_SCREEN}/{REP_CONFIRM}, IOU_MIN {IOU_MIN}, CHANGE_MIN "
         f"{CHANGE_MIN}, Laufzeit {runtime / 60:.1f} min\n"]
    for w in ctx.warnings:
        L.append(f"> **WARNUNG:** {w}\n")

    L.append("## Ausgangsmuster und Kontrollen ohne Reiz\n")
    L.append("| Attraktor | ohne Reiz | eigener Anteil | Form-IoU zweier Wdh. | Kontrolle: bleibt | "
             "Mischung | anderer Attr. |")
    L.append("|---|---|---|---|---|---|---|")
    ctrl = {c["source"]: c for c in control_rows}
    for a in ctx.attractors:
        c = ctrl.get(a.name, {})
        L.append(f"| {a.name} | {base[a.name]['label']} | {base[a.name]['own_fraction']:.2f} | "
                 f"{base[a.name]['self_iou']:.2f} | {pct(c.get('stay_rate'))} | "
                 f"{pct(c.get('intermediate_rate'))} | {pct(c.get('other_attractor_rate'))} |")
    L.append("")

    L.append("## Echtes Umschalten je Reiztyp und Reizbeginn\n")
    L.append("Zelle: Anteil der getesteten (Richtung, Ort, Dauer)-Kombinationen, die echt "
             f"umschalten (Rate >= {SWITCH_THRESHOLD:g} bei einem Radius <= {max(SCREEN_RADII)}), "
             "dahinter das kleinste r*. In Klammern: Kombinationen, die nur nach dem Etikett "
             "(mindestens einmal) umschalten.\n")
    times = sorted({g["t_signal"] for g in group_rows})
    L.append("| Reiztyp | " + " | ".join(f"t={t}" + (" (fertig)" if t == GROW_STEPS else "")
                                         for t in times) + " |")
    L.append("|---|" + "---|" * len(times))
    for sig in SIGNALS:
        cells = []
        for t in times:
            gs = [g for g in group_rows if g["signal"] == sig and g["t_signal"] == t]
            if not gs:
                cells.append("")
                continue
            sw = [g for g in gs if g["status"] == "schaltbar"]
            lab = sum(g["label_max_switch_rate"] > 0 for g in gs)
            cells.append(f"{len(sw)}/{len(gs)}" + (f", r*>={min(g['r_threshold'] for g in sw)}" if sw else "")
                         + f" ({lab})")
        if any(cells):
            L.append(f"| {sig} | " + " | ".join(cells) + " |")
    L.append("")

    grown_sw = [g for g in group_rows if g["t_signal"] == GROW_STEPS and g["status"] == "schaltbar"]
    L.append(f"## Fertiges Muster (t = {GROW_STEPS})\n")
    if grown_sw:
        L.append("Echt schaltbare Reize am fertigen Muster:\n")
        L.append("| Reiz | Quelle -> Ziel | Ort | D | r* | Pixel | Rate | Schritte | geworden |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for g in sorted(grown_sw, key=lambda g: (SIGNALS.index(g["signal"]), g["dose_threshold"])):
            L.append(f"| {g['signal']} | {g['source']} -> {g['target']} | {g['position']} | "
                     f"{g['duration']} | {g['r_threshold']} | {g['n_pixels_threshold']} | "
                     f"{pct(g['strict_rate_at_threshold'])} | {g['steps_to_switch_at_threshold']} | "
                     f"{g['switched_to_at_threshold']} |")
    else:
        lab = [g for g in group_rows if g["t_signal"] == GROW_STEPS and g["label_max_switch_rate"] > 0]
        L.append("Kein Reiz schaltet das fertige Muster echt um." +
                 (f" Nach dem Etikett allein schalten {len(lab)} Kombinationen um; das sind "
                  "uebermalte oder ersetzte Flaechen ohne neu gewachsene Form (Kriterien 2-4)."
                  if lab else ""))
    L.append("")

    L.append("## Entwicklungsfenster\n")
    L.append("Spaetester Reizbeginn, zu dem der Reiztyp noch echt umschalten kann:\n")
    for sig in SIGNALS:
        if not any(g["signal"] == sig for g in group_rows):
            continue
        t = latest_switch_time(group_rows, sig)
        L.append(f"- {sig}: " + ("nie" if t is None else
                                 f"t = {t}" + (" (auch fertiges Muster)" if t == GROW_STEPS else "")))
    L.append("")

    outc = defaultdict(Counter)
    for g in group_rows:
        if g["t_signal"] == GROW_STEPS:
            outc[g["signal"]][g.get("outcome_at_max_screen_radius", "")] += 1
    if outc:
        L.append(f"## Fertiges Muster: Ergebnis beim groessten Screen-Radius (r = {max(SCREEN_RADII)})\n")
        for sig in SIGNALS:
            if sig in outc:
                L.append(f"- {sig}: " + ", ".join(f"{k} {v}" for k, v in outc[sig].most_common()))
        L.append("")

    L.append("## Dateien\n")
    L.append("- `thresholds.csv` - je (Reiztyp, Zeitpunkt, Richtung, Ort, Dauer): Status, r*, "
             "Etikett-Urteil, ...")
    L.append("- `signals_summary.csv` - je (Reiztyp, Zeitpunkt, Richtung)")
    L.append("- `conditions_summary.csv` - je getestetem Radius; `controls.csv`; `rollouts.csv`")
    L.append("- `development_at_signal_times.png` - Ausgangsmuster zu den Reizzeitpunkten")
    for p in figures:
        L.append(f"- `{p.relative_to(out_dir)}`")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def write_summary(results, failed, path_md, path_csv):
    write_csv(path_csv, [r for res in results for r in res["sig_rows"]])
    L = ["# Signal-Schwellen-Scan: alle Modelle\n",
         f"Reizbeginn {T_SIGNALS} (fertig = {GROW_STEPS}), Positionen {POSITIONS}, Dauern "
         f"{DURATIONS}, Radien bis {max(SCREEN_RADII)}, {FREE_STEPS} freie Schritte. "
         f"Echt = Umschaltrate >= {SWITCH_THRESHOLD:g} ({REP_CONFIRM} Wdh.) mit nachgewachsener "
         f"Zielform (IoU >= {IOU_MIN}).\n"]
    L.append("Zelle: spaetester Reizbeginn t, zu dem der Reiztyp in diesem Modell noch echt "
             "umschaltet (`-` = nie, **fett** = auch das fertige Muster).\n")
    L.append("| Reiztyp | " + " | ".join(res["ctx"].tag for res in results) + " |")
    L.append("|---|" + "---|" * len(results))
    for sig in SIGNALS:
        cells = []
        for res in results:
            if not any(g["signal"] == sig for g in res["group_rows"]):
                cells.append("")
                continue
            t = latest_switch_time(res["group_rows"], sig)
            cells.append("-" if t is None else (f"**{t}**" if t == GROW_STEPS else str(t)))
        if any(cells):
            L.append(f"| {sig} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("| Modell | Bundle | Warnungen |")
    L.append("|---|---|---|")
    for res in results:
        ctx = res["ctx"]
        L.append(f"| {ctx.tag} | `{ctx.bundle.name}` | {'; '.join(ctx.warnings)} |")
    for tag, err in failed:
        L.append(f"| {tag} | FEHLER | {err} |")
    L.append("")
    L.append("Details je Modell: `<modell-tag>/report.md`. Uebersicht: `overview_all_models.png`.\n")
    path_md.write_text("\n".join(L) + "\n")


# ======================================================================
# MAIN
# ======================================================================

def check_expected(infos):
    if not EXPECTED_MODELS:
        return
    exp = Counter((n, p.upper()) for n, p in EXPECTED_MODELS)
    got = Counter((i.K, i.padding) for i in infos)
    warn = [f"fehlt: {n} Attraktoren {p}" for (n, p) in sorted(exp - got)] + \
           [f"unerwartet/doppelt: {n} Attraktoren {p}" for (n, p) in sorted(got - exp)]
    if warn:
        print("WARNUNG: Modelle passen nicht zu EXPECTED_MODELS:")
        for w in warn:
            print("  !", w)


def estimate(infos):
    """Obergrenze des Aufwands im Screen (Positionen geschaetzt)."""
    pos_guess = sum(1 for p in POSITIONS if p == "seed") + \
        sum(max(1, int(7000 / int(p.split(":")[1]) ** 2) - 1) for p in POSITIONS
            if isinstance(p, str) and p.startswith("body:"))
    per = sum(d + FREE_STEPS for d in DURATIONS) * len(SCREEN_RADII) * REP_SCREEN * len(T_SIGNALS)
    total = 0
    for i in infos:
        units = sum((i.K * (i.K - 1) if SIGNAL_INFO[s][0] == "ziel" else i.K) for s in SIGNALS)
        steps = units * pos_guess * per
        total += steps
        print(f"  {i.tag:28s} {units:3d} (Reiztyp x Richtung) x ~{pos_guess} Positionen x "
              f"{len(T_SIGNALS)} Zeitpunkte -> ~{steps:,} Laufschritte")
    print(f"  SUMME ~{total:,} Laufschritte im Screen (+ Wachstum, + Verfeinerung wo etwas "
          "echt umschaltet)")
    print(f"  Grob: GPU ~0.9 ms je Laufschritt -> ~{total * 0.9e-3 / 3600:.1f} h")


def main():
    g = globals()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--signals", default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.quick:
        for k, v in QUICK.items():
            g[k] = v
        g["OUTPUT_ROOT"] = Path(str(OUTPUT_ROOT) + "_quick")
    if args.out:
        g["OUTPUT_ROOT"] = Path(args.out)
    if args.batch_size:
        g["BATCH_SIZE"] = args.batch_size
    if args.signals:
        wanted = [s.strip() for s in args.signals.split(",")]
        bad = [s for s in wanted if s not in SIGNAL_INFO]
        if bad:
            raise SystemExit(f"Unbekannte Reiztypen {bad}. Vorhanden: {list(SIGNAL_INFO)}")
        g["SIGNALS"] = wanted
    if args.times:
        g["T_SIGNALS"] = sorted({int(t) for t in args.times.split(",")})
    if max(T_SIGNALS) > GROW_STEPS:
        raise SystemExit(f"T_SIGNALS darf nicht groesser als GROW_STEPS={GROW_STEPS} sein")

    bundles = [args.bundle] if args.bundle else MODEL_BUNDLES
    infos, failed = [], []
    for b in bundles:
        try:
            infos.append(describe_bundle(resolve_bundle(b)))
        except Exception as exc:
            failed.append((str(b), f"{type(exc).__name__}: {exc}"))
            print(f"FEHLER beim Lesen von {b}: {exc}")
    used = Counter()
    for i in infos:
        used[i.tag] += 1
        if used[i.tag] > 1:
            i.tag = f"{i.tag}_{used[i.tag]}"
    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        infos = [i for i in infos if any(i.tag.startswith(w) for w in wanted)]
    if not args.bundle:
        check_expected(infos)

    print("JAX-Geraete:", jax.devices())
    print("Reiztypen:", SIGNALS)
    print("Reizbeginn:", T_SIGNALS, f"(fertig = {GROW_STEPS})")
    print("Modelle:")
    for i in infos:
        print(f"  {i.tag:28s} {i.bundle}")
        for w in i.warnings:
            print(f"      WARNUNG: {w}")
    if args.dry_run:
        estimate(infos)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Nur ein Lauf gleichzeitig: zwei Prozesse auf einer GPU nehmen sich
    # gegenseitig den Speicher weg und schreiben in dieselben Dateien.
    lock = OUTPUT_ROOT / ".lock"
    if lock.is_file():
        try:
            other = int(lock.read_text().strip())
            cmd = Path(f"/proc/{other}/cmdline").read_bytes().decode(errors="ignore")
        except (ValueError, OSError):
            cmd = ""                 # alte Sperrdatei eines beendeten Laufs
        if "python" in cmd and other != os.getpid():
            raise SystemExit(
                f"Es laeuft schon ein Scan (Prozess {other}), der nach {OUTPUT_ROOT} "
                f"schreibt. Erst warten oder ihn beenden:  kill {other}")
    lock.write_text(str(os.getpid()))
    try:
        _run_all(infos, failed, args)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_all(infos, failed, args):
    emoji_dir = resolve_emoji_dir()
    print("Emojis:", emoji_dir)
    results = []
    t_all = time.time()
    for info in infos:
        try:
            results.append(run_model(info, emoji_dir, args))
        except Exception as exc:
            traceback.print_exc()
            failed.append((info.tag, f"{type(exc).__name__}: {str(exc)[:200]}"))
            print(f"  {info.tag} uebersprungen wegen Fehler; weiter mit dem naechsten Modell.")
        free_gpu_memory()
        write_summary(results, failed, OUTPUT_ROOT / "summary_all_models.md",
                      OUTPUT_ROOT / "summary_all_models.csv")
    try:
        plot_all_models_overview(results, OUTPUT_ROOT / "overview_all_models.png")
    except Exception as exc:
        print(f"WARNUNG: Uebersichtsbild fehlgeschlagen: {exc}")
    print(f"\nFertig in {(time.time() - t_all) / 60:.1f} min. "
          f"Zusammenfassung: {OUTPUT_ROOT / 'summary_all_models.md'}")


if __name__ == "__main__":
    main()
