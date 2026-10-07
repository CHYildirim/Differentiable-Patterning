"""
signal_switch_scan.py
=====================

Untrainiertes Signalling in einem Multi-Attraktor-gNCA.

FRAGE
-----
Ein gNCA wurde auf zwei Attraktoren trainiert (z.B. Krabbe aus einem rosa
Startpixel, Mikrobe aus einem tuerkisen Startpixel, CIRCULAR-Padding). Ein
Signal wurde NICHT trainiert. Kann man ein fertig gewachsenes Muster trotzdem
mit einem Reiz in den anderen Attraktor umschalten - und unter welchem Reiz?

REIZ ("Farb-Signal")
--------------------
In einer Scheibe mit Radius r um eine Position (y, x) wird der Zustand fuer D
Schritte in Richtung der Startpixel-Farbe eines Attraktors gezogen:

    rgba_neu = rgba + s * (seed_rgba - rgba)            s = Staerke (strength)

Zwei Varianten (Spalte "mode"):

    "rgba"      Nur die sichtbaren Kanaele (RGBA) werden veraendert, die
                versteckten Kanaele bleiben wie sie sind. Reines Farbsignal,
                vergleichbar mit Stovolds 1-Pixel-Signal, aber ohne eigenen
                Umweltkanal.
    "seedcell"  Zusaetzlich werden die versteckten Kanaele in Richtung 0
                gezogen: hidden_neu = (1 - s) * hidden. Bei s = 1 ist jede
                Zelle in der Scheibe exakt ein frischer Startpixel.

Farbe des Reizes (Spalte "colour"):

    "other"     Startpixel-Farbe des ANDEREN Attraktors (der eigentliche Test)
    "own"       Startpixel-Farbe des EIGENEN Attraktors (Kontrolle: schaltet
                schon jede Stoerung um, oder wirklich die "fremde" Farbe?)
    "none"      kein Reiz (Kontrolle: spontanes Umschalten)

Form des Reizes (Spalte "region"):

    "disc"      die ganze Scheibe (auch Hintergrund-Pixel)
    "body"      nur lebende Zellen des Organismus in der Scheibe (alpha > 0.1);
                mit grossem Radius = der ganze Organismus wird umgefaerbt

Zeitlicher Ablauf je Lauf:

    Startpixel --GROW--> fertiges Muster (t_signal Schritte)
               --> D Schritte mit Reiz (vor jedem NCA-Schritt neu angelegt)
               --> FREE_STEPS Schritte frei
               --> Bewertung

D = 1 ist ein einzelner Puls. D > 1 haelt die Scheibe D Schritte fest. Jeder
Lauf bekommt nach Reizende gleich viele freie Schritte, damit lange Reize
nicht weniger Zeit zum Ausheilen haben als kurze.

BEWERTUNG
---------
Nach jedem Schritt wird der Zustand klassifiziert wie in
spatial_position_scan_multi.py (naechstes Zielbild per MSE, INTERMEDIATE wenn
mindestens zwei Attraktoren je >= 20 % der sichtbaren Pixel stellen,
NO_VISIBLE_PATTERN wenn nichts sichtbar ist). Neu: Bei CIRCULAR wird jedes
Zielbild vorher per FFT optimal (mit Wrap-around) verschoben. Ein Muster, das
nach dem Reiz versetzt neu waechst, wird also trotzdem erkannt.

    switched      Endzustand = der Attraktor, dessen Farbe der Reiz hatte
    stayed        Endzustand = Ausgangsattraktor
    intermediate  Mischung / zwei Organismen
    no_pattern    alles verschwunden
    transient     war zwischendurch der Ziel-Attraktor, ist es am Ende nicht
    steps_to_switch   ab welchem Schritt nach Reizbeginn der Zustand bis zum
                      Ende durchgehend der Ziel-Attraktor ist

SCANS (alle in SCANS unten einstellbar)
---------------------------------------
    A_dose         Radius x Dauer x Staerke x Modus, Reiz am Seed-Ort
    B_controls     eigene Farbe und kein Reiz
    C_location     wo im/um das Muster muss der Reiz sitzen? (Raster)
    D_timing       Reiz waehrend des Wachstums statt danach
    E_body         nur die Zellen des Organismus umfaerben (bis ganzer Koerper)

ERSTER BEFUND (Testlauf, Modell 0929, CPU, je 1 Wiederholung)
--------------------------------------------------------------
1. Fertiges Muster (t = 256): KEIN Reiz am Seed-Ort schaltet um, weder
   Krabbe -> Mikrobe noch umgekehrt (r = 0..48, D = 1..64, s = 1, rgba und
   seedcell). Flecken bis r ~ 2 werden repariert; groessere bleiben als
   fremdfarbiger Fleck stehen ("Chimaere"), der weder ueberschrieben wird
   noch sich ausbreitet. Dauer und Modus aendern daran praktisch nichts.
2. Waehrend des Wachstums gibt es ein kurzes Fenster: Ein einzelner Puls mit
   r = 8 (197 Pixel) bei t = 8 schaltet Krabbe -> Mikrobe um (rgba und
   seedcell) und Mikrobe -> Krabbe nur mit seedcell. Mit rgba wird die
   Mikrobe kurz zur Krabbe und kehrt dann zurueck -> die Identitaet der
   Mikrobe steckt zum Teil in den versteckten Kanaelen. Ab t = 16 schaltet
   nichts mehr um (hoechstens Mischung), ab t = 32 wird ein r = 8-Fleck
   wieder vollstaendig repariert.
Der volle Scan prueft das mit 8 Wiederholungen, feinerem Zeitraster, Ort,
Staerke und Ganzkoerper-Umfaerbung.

START
-----
Aus dem Repo-Stammordner (dort, wo Experiments/, Common/ und models/ liegen):

    export DATA_PATH_BASE=demo/demo_data     # Ordner, der "Emojis/" enthaelt
    nohup python signal_switch_scan.py > signal_scan.log 2>&1 &
    tail -f signal_scan.log

Optionen:
    --bundle PFAD     anderes Modell als MODEL_BUNDLE unten
    --scans A_dose,B_controls   nur diese Scans
    --quick           sehr kleiner Testlauf (CPU-tauglich)
    --batch-size N    gleichzeitig simulierte Laeufe
    --no-resume       bereits gerechnete Laeufe NICHT ueberspringen
    --dry-run         nur zaehlen, wie viele Laeufe es werden (und grob wie lange)

Der Lauf ist wiederaufnehmbar: Ergebnisse werden nach jedem Batch in
rollouts.csv angehaengt; ein Neustart mit denselben Einstellungen rechnet nur
die fehlenden Laeufe.
"""

from pathlib import Path
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace


# ======================================================================
# CPU / GPU  (muss vor "import jax" stehen)
# ======================================================================
FORCE_CPU = False

if FORCE_CPU or "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
# >>>>>>>>>>  HIER DAS MODELL EINTRAGEN, DAS GETESTET WERDEN SOLL  <<<<<<
# ######################################################################
#
# Bundle-Ordner (der Ordner mit model.eqx und config.yaml), relativ zum
# Repo-Stammordner oder absolut. Zeigt der Pfad auf model.eqx/config.yaml,
# wird der Ordner darueber genommen. Die drei 2-Attraktor-CIRCULAR-gNCAs
# unter models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/
# emoji-multi-attractor-2-3-4/ :
#
#   20260909T215800Z-cfg049eedbd4068-d2916c1d   ABGEBROCHEN (completed: false,
#        bester Checkpoint Iteration 1546, Loss 0.011) - verwaschene Muster,
#        Mikrobe schon ohne Reiz INTERMEDIATE. Nicht fuer Signalling nehmen.
#   20260925T030524Z-cfgb93723a3e5bd-eb28218c   fertig, Loss 0.0027
#   20260929T132309Z-cfg372a0c3f14b5-e84bd156   fertig, Loss 0.0018 (repeat=1)
#
# Funktioniert auch mit 3 oder 4 Attraktoren: dann wird jeder Attraktor mit
# der Farbe jedes anderen gereizt.

MODEL_BUNDLE = (
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/"
    "emoji-multi-attractor-2-3-4/20260929T132309Z-cfg372a0c3f14b5-e84bd156"
)

# ######################################################################


OUTPUT_ROOT = Path("inference_outputs/signal_switch_scan")

# Ordner mit den Emoji-PNGs. None -> $DATA_PATH_BASE/Emojis, sonst
# demo/demo_data/Emojis.
EMOJI_DIR = None

SEED = 0

# Wie lange das Ausgangsmuster waechst, bevor der Reiz kommt (Scans A-C).
# Deutlich ueber dem Training-t (128), damit das Muster wirklich fertig ist.
GROW_STEPS = 256

# Freie Schritte nach dem ENDE des Reizes bis zur Bewertung.
FREE_STEPS = 192

# Wiederholungen je Bedingung (unterschiedliches fire_rate-Rauschen beim
# Wachsen und danach). Die Rauschfolge nach dem Reiz ist fuer alle
# Bedingungen mit derselben Wiederholungsnummer gleich ("common random
# numbers"), dadurch sind Unterschiede zwischen Bedingungen weniger verrauscht.
REPEATS = 8

# Gleichzeitig simulierte Laeufe. Bei Speicherproblemen verkleinern.
BATCH_SIZE = 128

# Klassifikation (wie in spatial_position_scan_multi.py)
ACTIVE_ALPHA_THRESHOLD = 0.10
MIXED_MIN_FRACTION = 0.20

# Bewertungsschwellen fuer die Zusammenfassung
SWITCH_THRESHOLDS = (0.5, 0.9)

# Fuer die Beispielbilder: Schritte nach Reizbeginn, die als Bildstreifen
# gespeichert werden (zusaetzlich zum Zustand direkt vor dem Reiz).
EXAMPLE_SNAPSHOT_STEPS = [1, 4, 16, 32, 64, 128, 256, "end"]
# Beispiele laufen zusaetzlich so viele Schritte weiter, um das Langzeit-
# Schicksal zu zeigen (bleibt eine Mischung stehen oder loest sie sich auf?).
EXAMPLE_EXTRA_STEPS = 256
N_EXAMPLES_PER_DIRECTION = 3
SAVE_GIFS = True


# ----------------------------------------------------------------------
# SCANS
# ----------------------------------------------------------------------
# Jeder Scan ist ein kartesisches Produkt seiner Listen. Fuer jedes Paar
# (Ausgangsattraktor, Zielfarbe) wird jede Kombination gerechnet.
#
#   position  "seed"    = Trainings-Seed-Position (Mitte)
#             "grid:N"  = Raster mit Abstand N Pixel ueber die ganze Leinwand
#             "grid:N:R" = dasselbe, aber nur bis Abstand R vom Seed
#             [y, x]    = feste Position (Leinwandkoordinaten)
#   radius    Scheibenradius in Pixeln (0 = genau ein Pixel)
#   duration  Anzahl Schritte, in denen der Reiz angelegt wird (>= 1)
#   strength  s in [0, 1] (Werte > 1 sind erlaubt = "uebersteuert")
#   mode      "rgba" | "seedcell"
#   colour    "other" | "own" | "none"
#   region    optional, ["disc"] (Standard) | ["body"]
#   t_signal  Schritt des Wachstums, an dem der Reiz beginnt
#   repeats   optional, ueberschreibt REPEATS fuer diesen Scan

SCANS = {
    "A_dose": dict(
        position=["seed"],
        radius=[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100],
        duration=[1, 2, 4, 8, 16, 32, 64],
        strength=[0.25, 0.5, 1.0],
        mode=["rgba", "seedcell"],
        colour=["other"],
        t_signal=[GROW_STEPS],
    ),
    "B_controls": dict(
        position=["seed"],
        radius=[0, 4, 16, 32],
        duration=[1, 16, 64],
        strength=[1.0],
        mode=["rgba", "seedcell"],
        colour=["own", "none"],
        t_signal=[GROW_STEPS],
    ),
    # Wo muss der Reiz sitzen? Frueh (im Zeitfenster, in dem Umschalten im
    # Testlauf moeglich war) fein um den Seed herum ...
    "C_location_early": dict(
        position=["grid:4:24"],
        radius=[2, 8],
        duration=[1],
        strength=[1.0],
        mode=["rgba", "seedcell"],
        colour=["other"],
        t_signal=[8],
        repeats=4,
    ),
    # ... und am fertigen Muster grob ueber die ganze Leinwand.
    "C_location_grown": dict(
        position=["grid:16"],
        radius=[8],
        duration=[16],
        strength=[1.0],
        mode=["seedcell"],
        colour=["other"],
        t_signal=[GROW_STEPS],
        repeats=4,
    ),
    "D_timing": dict(
        position=["seed"],
        radius=[2, 4, 8, 16],
        duration=[1, 16],
        strength=[1.0],
        mode=["rgba", "seedcell"],
        colour=["other"],
        t_signal=[2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 128, 256],
    ),
    "E_body": dict(
        position=["seed"],
        radius=[12, 24, 48, 100],
        duration=[1, 16, 64],
        strength=[0.5, 1.0],
        mode=["rgba", "seedcell"],
        colour=["other"],
        region=["body"],
        t_signal=[GROW_STEPS],
    ),
}

# Sehr kleiner Lauf zum Testen (--quick), z.B. auf der CPU.
QUICK = dict(
    GROW_STEPS=256,
    FREE_STEPS=160,
    REPEATS=1,
    BATCH_SIZE=8,
    SCANS={
        "A_dose": dict(
            position=["seed"],
            radius=[0, 2, 6, 12, 24, 48],
            duration=[1, 16, 64],
            strength=[1.0],
            mode=["rgba", "seedcell"],
            colour=["other"],
            t_signal=[256],
        ),
        "B_controls": dict(
            position=["seed"],
            radius=[12],
            duration=[16],
            strength=[1.0],
            mode=["rgba", "seedcell"],
            colour=["own", "none"],
            t_signal=[256],
        ),
        "D_timing": dict(
            position=["seed"],
            radius=[8],
            duration=[1],
            strength=[1.0],
            mode=["seedcell"],
            colour=["other"],
            t_signal=[8, 16],
        ),
    },
)


# ======================================================================
# KLASSEN-NAMEN UND FARBEN
# ======================================================================

INTERMEDIATE = "INTERMEDIATE"
NO_PATTERN = "NO_VISIBLE_PATTERN"

# Sequentielle Farbskala (eine Farbe, hell -> dunkel) fuer Raten 0..1
SEQ_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
# Kategoriale Farben (feste Reihenfolge) fuer Linien
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


# ======================================================================
# EINSTELLUNGEN / MODELL / ZIELBILDER LADEN
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
        p = int(pad)
        return (p, p, p, p)
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
            raise FileNotFoundError(f"{path / name} fehlt (MODEL_BUNDLE pruefen)")
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
    """Startpixel genau wie beim Training (Modus full / patch / pixel).
    Gibt (Pixelwert, (y, x) im ungepaddeten Bild) zurueck."""
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


def prepare(bundle, emoji_dir):
    cfg = OmegaConf.load(bundle / "config.yaml")

    task = cfg_value(cfg, "data.augmentation.task", "data.emoji.task")
    if task != "multi_attractor":
        raise ValueError(f"task ist {task!r}, erwartet 'multi_attractor'")

    padding = str(plain(cfg_value(cfg, "model.padding"))).upper()
    downsample = int(cfg_value(cfg, "data.preprocessing.downsample",
                               "data.downsample", default=1))
    crop_square = bool(cfg_value(cfg, "data.augmentation.crop_square",
                                 default=False))
    pad = parse_pad(plain(cfg_value(cfg, "data.augmentation.pad",
                                    "data.emoji.pad", default=0)))
    dc = int(cfg_value(cfg, "data.augmentation.data_channels",
                       "data.data_channels", default=4))
    pairs = plain(cfg_value(cfg, "data.augmentation.pairs", "data.emoji.pairs"))

    used = set()
    attractors = []
    for index, pair in enumerate(pairs):
        initial = pair["initial"]
        if isinstance(initial, str):
            initial = {"image": initial, "mode": "full"}
        target = load_emoji(pair["target"], emoji_dir, downsample, crop_square)
        init_img = load_emoji(initial["image"], emoji_dir, downsample, crop_square)
        pixel, yx, desc = initial_seed(initial, init_img)
        attractors.append(SimpleNamespace(
            index=index, name=name_from_file(pair["target"], used),
            target_file=pair["target"], target=target,
            seed_pixel=pixel[:dc].astype(np.float32), seed_yx=yx,
            initial_description=desc))

    channels, H, W = attractors[0].target.shape
    top, bottom, left, right = pad
    CH, CW = H + top + bottom, W + left + right
    seed_y = attractors[0].seed_yx[0] + top
    seed_x = attractors[0].seed_yx[1] + left

    targets = np.stack([
        np.pad(a.target[:dc], ((0, 0), (top, bottom), (left, right)))
        for a in attractors]).astype(np.float32)

    model = build_model(cfg.model, key=jr.PRNGKey(0))
    model = model[0] if isinstance(model, tuple) else model
    model = model.load(bundle / "model.eqx")

    warnings = []
    manifest = bundle / "manifest.yaml"
    if manifest.is_file():
        m = OmegaConf.load(manifest)
        completed = OmegaConf.select(m, "training.completed", default=None)
        best_it = OmegaConf.select(m, "training.best_iteration", default=None)
        best_loss = OmegaConf.select(m, "training.best_loss", default=None)
        iters = cfg_value(cfg, "training.loop.iterations", default=None)
        if completed is False:
            warnings.append(
                f"Training NICHT abgeschlossen (completed: false, bester Checkpoint "
                f"Iteration {best_it} von {iters}, Loss {best_loss}).")

    return SimpleNamespace(
        warnings=warnings,
        bundle=bundle, cfg=cfg, model=model, padding=padding,
        circular=(padding == "CIRCULAR"),
        family=str(cfg_value(cfg, "model.family", default="?")),
        n_channels=int(cfg.model.channels), dc=dc,
        attractors=attractors, names=[a.name for a in attractors],
        class_names=[a.name for a in attractors] + [INTERMEDIATE, NO_PATTERN],
        K=len(attractors), targets=targets, H=CH, W=CW,
        seed_y=seed_y, seed_x=seed_x,
        train_t=cfg_value(cfg, "training.loop.t", default="?"),
    )


# ======================================================================
# KLASSIFIKATION (JAX, verschiebungsinvariant bei CIRCULAR)
# ======================================================================

def make_classifier(ctx):
    T = jnp.asarray(ctx.targets)                       # (K, dc, H, W)
    K, dc, H, W = T.shape
    T_fft = jnp.fft.rfft2(T)
    T_sq = jnp.sum(T ** 2, axis=(1, 2, 3))
    N = dc * H * W
    align = ctx.circular

    def classify_one(vis):
        """vis: (dc, H, W) -> (label, fractions[K], mse[K], n_active)"""
        if align:
            X_fft = jnp.fft.rfft2(vis)
            # corr[k, s] = sum_p vis[p] * T_k[p - s]  (Zielbild um s verschoben)
            corr = jnp.fft.irfft2(jnp.sum(X_fft[None] * jnp.conj(T_fft), axis=1),
                                  s=(H, W))
            mse_map = (jnp.sum(vis ** 2) + T_sq[:, None, None] - 2 * corr) / N
            flat = mse_map.reshape(K, -1)
            best = jnp.argmin(flat, axis=1)
            mse = jnp.maximum(jnp.min(flat, axis=1), 0.0)
            shifted = jnp.stack([
                jnp.roll(T[k], (best[k] // W, best[k] % W), axis=(-2, -1))
                for k in range(K)])
        else:
            shifted = T
            mse = jnp.mean((vis[None] - T) ** 2, axis=(1, 2, 3))

        err = jnp.mean((vis[None, :3] - shifted[:, :3]) ** 2, axis=1)  # (K,H,W)
        if dc >= 4:
            active = vis[3] > ACTIVE_ALPHA_THRESHOLD
        else:
            active = jnp.max(jnp.abs(vis[:3]), axis=0) > 0.10
        n_active = jnp.sum(active)

        if K == 1:
            fractions = jnp.where(n_active > 0, 1.0, 0.0)[None]
        else:
            ordered = jnp.sort(err, axis=0)
            unique = ordered[0] < ordered[1]
            nearest = jnp.argmin(err, axis=0)
            counted = unique & active
            fractions = jnp.stack([jnp.sum(counted & (nearest == k))
                                   for k in range(K)]) / jnp.maximum(n_active, 1)

        mixed = jnp.sum(fractions >= MIXED_MIN_FRACTION) >= 2
        label = jnp.where(mixed, K, jnp.argmin(mse))
        label = jnp.where(n_active == 0, K + 1, label).astype(jnp.int32)
        return label, fractions, mse, n_active

    return jax.vmap(classify_one)


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
        # region "body": nur lebende Zellen des Organismus (alpha > Schwelle)
        alive = x[:, 3] > ACTIVE_ALPHA_THRESHOLD if dc >= 4 else jnp.ones_like(inside)
        inside = inside & jnp.where(P["body"][:, None, None], alive, True)
        on = inside & (k < P["duration"])[:, None, None] & P["active"][:, None, None]
        s = P["strength"][:, None, None, None]
        rgba = x[:, :dc]
        rgba_new = rgba + s * (P["colour"][:, :, None, None] - rgba)
        hidden = x[:, dc:]
        hidden_new = jnp.where(P["reset_hidden"][:, None, None, None],
                               (1.0 - s) * hidden, hidden)
        x_new = jnp.concatenate([rgba_new, hidden_new], axis=1)
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


def growth_keys(a_index, repeat):
    return jr.fold_in(jr.fold_in(jr.PRNGKey(SEED), 1), a_index * 100003 + repeat)


def post_key(repeat):
    return jr.fold_in(jr.fold_in(jr.PRNGKey(SEED), 2), repeat)


def grow_snapshots(ctx, grow_step, times, repeats):
    """Waechst jeden Attraktor 'repeats'-mal und speichert die Zustaende zu
    den Zeiten in 'times'. Rueckgabe: dict[(a, r, t)] -> np.ndarray."""
    times = sorted(set(int(t) for t in times))
    jobs = [(a.index, r) for a in ctx.attractors for r in range(repeats)]
    snaps = {}
    for start in range(0, len(jobs), BATCH_SIZE):
        chunk = jobs[start:start + BATCH_SIZE]
        x = jnp.asarray(np.stack([seed_state(ctx, ctx.attractors[a]) for a, _ in chunk]))
        keys = jnp.stack([growth_keys(a, r) for a, r in chunk])
        t = 0
        for target_t in times:
            while t < target_t:
                x = grow_step(ctx.model, x, keys, jnp.int32(t))
                t += 1
            host = np.asarray(x)
            for i, (a, r) in enumerate(chunk):
                snaps[(a, r, target_t)] = host[i]
    return snaps


# ======================================================================
# BEDINGUNGEN AUFBAUEN
# ======================================================================

def expand_positions(spec, ctx):
    out = []
    for p in spec:
        if p == "seed":
            out.append(("seed", ctx.seed_y, ctx.seed_x))
        elif isinstance(p, str) and p.startswith("grid:"):
            # "grid:N" = ganze Leinwand, "grid:N:R" = nur bis Abstand R vom Seed
            parts = p.split(":")
            step = int(parts[1])
            limit = float(parts[2]) if len(parts) > 2 else None
            # Raster so legen, dass der Seed ein Rasterpunkt ist
            ys = range(ctx.seed_y % step, ctx.H, step)
            xs = range(ctx.seed_x % step, ctx.W, step)
            for y in ys:
                for x in xs:
                    if limit is not None:
                        dy, dx = abs(y - ctx.seed_y), abs(x - ctx.seed_x)
                        if ctx.circular:
                            dy, dx = min(dy, ctx.H - dy), min(dx, ctx.W - dx)
                        if dy * dy + dx * dx > limit * limit:
                            continue
                    out.append((f"grid{step}", y, x))
        else:
            y, x = int(p[0]), int(p[1])
            out.append((f"yx_{y}_{x}", y, x))
    return out


def disc_mask(ctx, y, x, r):
    yy, xx = np.mgrid[0:ctx.H, 0:ctx.W]
    dy, dx = np.abs(yy - y), np.abs(xx - x)
    if ctx.circular:
        dy, dx = np.minimum(dy, ctx.H - dy), np.minimum(dx, ctx.W - dx)
    return (dy ** 2 + dx ** 2) <= r ** 2


def build_conditions(ctx, scans, ref_states=None):
    conditions = {}
    pixels_cache = {}
    for scan_name, spec in scans.items():
        positions = expand_positions(spec["position"], ctx)
        repeats = int(spec.get("repeats", REPEATS))
        for src in ctx.attractors:
            for colour_kind in spec["colour"]:
                if colour_kind == "other":
                    sig_targets = [a for a in ctx.attractors if a.index != src.index]
                elif colour_kind == "own":
                    sig_targets = [src]
                elif colour_kind == "none":
                    sig_targets = [None]
                else:
                    raise ValueError(f"unbekannte Farbe {colour_kind!r}")
                for sig in sig_targets:
                    for (pos_name, y, x), r, d, s, mode, ts, region in itertools.product(
                            positions, spec["radius"], spec["duration"],
                            spec["strength"], spec["mode"], spec["t_signal"],
                            spec.get("region", ["disc"])):
                        if colour_kind == "none":
                            # ohne Reiz sind Radius/Dauer/Modus bedeutungslos
                            pos_name, y, x = "seed", ctx.seed_y, ctx.seed_x
                            r, d, s, mode, region = 0, 1, 0.0, "rgba", "disc"
                        r, d, ts = int(r), int(d), int(ts)
                        if region not in ("disc", "body"):
                            raise ValueError(f"unbekannte region {region!r}")
                        pk = (r, region, src.index, y, x)
                        if pk not in pixels_cache:
                            mask = disc_mask(ctx, y, x, r)
                            if region == "body" and ref_states is not None:
                                mask &= ref_states[src.index][3] > ACTIVE_ALPHA_THRESHOLD
                            pixels_cache[pk] = int(mask.sum())
                        cond = dict(
                            scan=scan_name,
                            source=src.name, source_index=src.index,
                            colour=colour_kind,
                            signal_target=(sig.name if sig else ""),
                            signal_target_index=(sig.index if sig else -1),
                            position=pos_name, y=y, x=x,
                            radius=r, region=region, n_pixels=pixels_cache[pk],
                            duration=d, strength=float(s), mode=mode,
                            t_signal=ts, repeats=repeats,
                        )
                        cond["dose"] = cond["n_pixels"] * d * float(s)
                        key = condition_key(cond)
                        if key not in conditions:
                            cond["key"] = key
                            conditions[key] = cond
    return list(conditions.values())


def condition_key(c):
    return (f"{c['scan']}|{c['source']}|{c['colour']}|{c['signal_target']}|"
            f"{c['y']},{c['x']}|r{c['radius']}|d{c['duration']}|"
            f"s{c['strength']:g}|{c['mode']}|t{c['t_signal']}"
            + ("" if c.get("region", "disc") == "disc" else f"|{c['region']}"))


# ======================================================================
# BATCHES RECHNEN
# ======================================================================

ROLL_FIELDS = [
    "key", "repeat", "final_label", "outcome", "ever_target", "transient",
    "steps_to_switch", "first_target_step", "label_switches",
    "max_target_fraction", "target_frac_trend_last32", "final_n_active",
]


def outcome_of(label, cond, ctx):
    K = ctx.K
    if label == K + 1:
        return "no_pattern"
    if label == K:
        return "intermediate"
    if label == cond["source_index"]:
        return "stayed"
    if cond["signal_target_index"] >= 0 and label == cond["signal_target_index"]:
        return "switched"
    return "other_attractor"


def analyse_rollout(labels, fractions, mse, n_active, end, cond, ctx):
    """labels etc. sind Reihen ueber die Schritte nach Reizbeginn (Index k =
    Zustand nach k+1 Schritten). end = Anzahl gueltiger Schritte."""
    labels = labels[:end]
    final = int(labels[-1])
    tgt = cond["signal_target_index"]
    row = dict(final_label=ctx.class_names[final],
               outcome=outcome_of(final, cond, ctx),
               final_n_active=int(n_active[end - 1]))
    for k, name in enumerate(ctx.names):
        row[f"final_frac_{name}"] = float(fractions[end - 1, k])
        row[f"final_mse_{name}"] = float(mse[end - 1, k])
    row["label_switches"] = int(np.sum(labels[1:] != labels[:-1]))
    if tgt >= 0:
        is_t = labels == tgt
        row["ever_target"] = bool(is_t.any())
        row["first_target_step"] = int(np.argmax(is_t) + 1) if is_t.any() else ""
        row["max_target_fraction"] = float(np.max(fractions[:end, tgt]))
        # Aendert sich der Anteil am Ende noch? (> 0: Umbau laeuft noch Richtung Ziel)
        back = max(end - 33, 0)
        row["target_frac_trend_last32"] = float(fractions[end - 1, tgt] - fractions[back, tgt])
        if final == tgt:
            not_t = np.nonzero(~is_t)[0]
            row["steps_to_switch"] = int(not_t[-1] + 2) if len(not_t) else 1
        else:
            row["steps_to_switch"] = ""
        row["transient"] = bool(is_t.any() and final != tgt)
    else:
        row.update(ever_target="", first_target_step="", max_target_fraction="",
                   target_frac_trend_last32="",
                   steps_to_switch="", transient="")
    return row


def run_jobs(ctx, conditions, signal_step, grow_step, out_dir, resume):
    roll_path = out_dir / "rollouts.csv"
    fields = ROLL_FIELDS + [f"final_frac_{n}" for n in ctx.names] + \
        [f"final_mse_{n}" for n in ctx.names]

    done = set()
    if resume and roll_path.is_file():
        with open(roll_path) as f:
            for r in csv.DictReader(f):
                done.add((r["key"], int(r["repeat"])))
        print(f"Wiederaufnahme: {len(done)} Laeufe schon vorhanden.")
    elif roll_path.is_file():
        roll_path.unlink()

    jobs = [(c, r) for c in conditions for r in range(c["repeats"])
            if (c["key"], r) not in done]
    # Gleich lange Laeufe zusammen rechnen
    jobs.sort(key=lambda j: (j[0]["t_signal"], j[0]["duration"],
                             j[0]["source_index"], j[1]))

    total_steps = sum(c["duration"] + FREE_STEPS for c, _ in jobs)
    print(f"\n{len(jobs)} Laeufe zu rechnen, {total_steps:,} Laufschritte "
          f"(+ Wachstum).")
    if not jobs:
        return

    need = {(c["source_index"], r, c["t_signal"]) for c, r in jobs}
    max_rep = max(r for _, r in jobs) + 1
    times = sorted({t for _, _, t in need})
    print(f"Wachse Ausgangsmuster: {ctx.K} Attraktoren x {max_rep} "
          f"Wiederholungen, Zeiten {times} ...")
    t0 = time.time()
    snaps = grow_snapshots(ctx, grow_step, times, max_rep)
    print(f"  fertig in {time.time() - t0:.1f} s")

    write_header = not roll_path.is_file()
    f = open(roll_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    colours = {a.index: a.seed_pixel for a in ctx.attractors}
    zero_col = np.zeros(ctx.dc, np.float32)
    t_start = time.time()
    steps_done = 0

    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        n_real = len(batch)
        # Letzten Batch auffuellen, damit JAX nicht neu kompilieren muss
        if len(jobs) > BATCH_SIZE:
            padded = batch + [batch[-1]] * (BATCH_SIZE - n_real)
        else:
            padded = batch
        x0 = np.stack([snaps[(c["source_index"], r, c["t_signal"])] for c, r in padded])
        P = dict(
            cy=jnp.asarray([c["y"] for c, _ in padded], jnp.int32),
            cx=jnp.asarray([c["x"] for c, _ in padded], jnp.int32),
            radius=jnp.asarray([c["radius"] for c, _ in padded], jnp.float32),
            duration=jnp.asarray([c["duration"] for c, _ in padded], jnp.int32),
            strength=jnp.asarray([c["strength"] for c, _ in padded], jnp.float32),
            reset_hidden=jnp.asarray([c["mode"] == "seedcell" for c, _ in padded]),
            body=jnp.asarray([c.get("region", "disc") == "body" for c, _ in padded]),
            active=jnp.asarray([c["colour"] != "none" for c, _ in padded]),
            colour=jnp.asarray(np.stack([
                colours[c["signal_target_index"]] if c["signal_target_index"] >= 0
                else zero_col for c, _ in padded])),
            key=jnp.stack([post_key(r) for _, r in padded]),
        )
        ends = np.array([c["duration"] + FREE_STEPS for c, _ in padded])
        T = int(ends.max())

        x = jnp.asarray(x0)
        ends_j = jnp.asarray(ends, jnp.int32)
        fin = x[:, :ctx.dc]
        series = []
        for k in range(T):
            x, lab, frac, mse, nact = signal_step(ctx.model, x, jnp.int32(k), P)
            fin = capture(fin, x, jnp.int32(k), ends_j)
            series.append((lab, frac, mse, nact))
        labels = np.stack([np.asarray(s[0]) for s in series], axis=1)
        fracs = np.stack([np.asarray(s[1]) for s in series], axis=1)
        mses = np.stack([np.asarray(s[2]) for s in series], axis=1)
        nacts = np.stack([np.asarray(s[3]) for s in series], axis=1)

        final_rgba = np.asarray(fin)
        for i, (c, r) in enumerate(batch):
            row = analyse_rollout(labels[i], fracs[i], mses[i], nacts[i],
                                  int(ends[i]), c, ctx)
            row.update(key=c["key"], repeat=r)
            writer.writerow(row)
            if r == 0:
                save_thumb(out_dir, c, final_rgba[i])
        f.flush()

        steps_done += sum(int(e) for e in ends[:n_real])
        el = time.time() - t_start
        eta = el / steps_done * (total_steps - steps_done)
        print(f"  Batch {start // BATCH_SIZE + 1}/{math.ceil(len(jobs) / BATCH_SIZE)}"
              f"  ({start + n_real}/{len(jobs)} Laeufe)  "
              f"{el / 60:.1f} min, Rest ~{eta / 60:.1f} min", flush=True)

    f.close()


@jax.jit
def capture(fin, x, k, ends):
    """Haelt fuer jeden Lauf den sichtbaren Zustand nach genau 'end' Schritten fest."""
    hit = (k + 1 == ends)[:, None, None, None]
    return jnp.where(hit, x[:, :fin.shape[1]], fin)


def thumb_path(out_dir, cond):
    safe = cond["key"].replace("|", "__").replace(",", "_")
    return out_dir / "final_states" / cond["scan"] / f"{safe}.png"


def save_thumb(out_dir, cond, rgba):
    path = thumb_path(out_dir, cond)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb_uint8(rgba)).save(path)


def to_rgb_uint8(rgba):
    """RGBA-Zustand (4,H,W) ueber Weiss gelegt -> (H,W,3) uint8."""
    v = np.clip(np.asarray(rgba, np.float32), 0, 1)
    if v.shape[0] >= 4:
        a = v[3:4]
        rgb = v[:3] * a + (1 - a)
    else:
        rgb = v[:3]
    return (np.moveaxis(rgb, 0, -1) * 255).round().astype(np.uint8)


# ======================================================================
# ZUSAMMENFASSUNG
# ======================================================================

SUMMARY_FIELDS = [
    "scan", "source", "colour", "signal_target", "position", "y", "x",
    "radius", "region", "n_pixels", "duration", "strength", "mode", "t_signal", "dose",
    "runs", "switch_rate", "stay_rate", "intermediate_rate", "no_pattern_rate",
    "other_attractor_rate", "transient_rate", "ever_target_rate",
    "median_steps_to_switch", "mean_final_target_fraction",
    "mean_max_target_fraction", "mean_target_frac_trend_last32", "key",
]


def summarise(ctx, conditions, out_dir):
    rows = defaultdict(list)
    with open(out_dir / "rollouts.csv") as f:
        for r in csv.DictReader(f):
            rows[r["key"]].append(r)
    summary = []
    for c in conditions:
        rs = rows.get(c["key"], [])
        if not rs:
            continue
        n = len(rs)
        oc = defaultdict(int)
        for r in rs:
            oc[r["outcome"]] += 1
        sts = [int(r["steps_to_switch"]) for r in rs if r["steps_to_switch"] not in ("", None)]
        tgt = c["signal_target"]
        s = {k: c[k] for k in SUMMARY_FIELDS if k in c}
        s.update(
            runs=n,
            switch_rate=oc["switched"] / n, stay_rate=oc["stayed"] / n,
            intermediate_rate=oc["intermediate"] / n,
            no_pattern_rate=oc["no_pattern"] / n,
            other_attractor_rate=oc["other_attractor"] / n,
            transient_rate=(sum(r["transient"] == "True" for r in rs) / n) if tgt else "",
            ever_target_rate=(sum(r["ever_target"] == "True" for r in rs) / n) if tgt else "",
            median_steps_to_switch=(float(np.median(sts)) if sts else ""),
            mean_final_target_fraction=(float(np.mean([float(r[f"final_frac_{tgt}"]) for r in rs])) if tgt else ""),
            mean_max_target_fraction=(float(np.mean([float(r["max_target_fraction"]) for r in rs])) if tgt else ""),
            mean_target_frac_trend_last32=(float(np.mean([float(r["target_frac_trend_last32"]) for r in rs]))
                                           if tgt and all(r.get("target_frac_trend_last32") not in ("", None) for r in rs) else ""),
        )
        summary.append(s)
    write_csv(out_dir / "conditions_summary.csv", summary, SUMMARY_FIELDS)
    return summary


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def thresholds(summary):
    """Kleinster Radius je (Scan, Richtung, Modus, Staerke, Dauer, t), ab dem die
    Umschaltrate die Schwellen erreicht."""
    groups = defaultdict(list)
    for s in summary:
        if s["colour"] != "other" or s["position"] != "seed":
            continue
        groups[(s["scan"], s["source"], s["signal_target"], s["mode"],
                s["strength"], s["duration"], s["t_signal"], s.get("region", "disc"))].append(s)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        items.sort(key=lambda s: s["radius"])
        row = dict(zip(["scan", "source", "signal_target", "mode", "strength",
                        "duration", "t_signal", "region"], key))
        row["max_switch_rate"] = max(s["switch_rate"] for s in items)
        not_rep = [s for s in items if s["stay_rate"] < 0.5]
        row["min_radius_not_repaired"] = not_rep[0]["radius"] if not_rep else ""
        fr = [s for s in items if s["mean_final_target_fraction"] != ""]
        row["max_final_target_fraction"] = max(s["mean_final_target_fraction"] for s in fr) if fr else ""
        row["radius_at_max"] = max(items, key=lambda s: (s["switch_rate"], -s["radius"]))["radius"]
        for th in SWITCH_THRESHOLDS:
            hit = [s for s in items if s["switch_rate"] >= th]
            row[f"min_radius_switch_ge_{th:g}"] = hit[0]["radius"] if hit else ""
        out.append(row)
    return out


def best_stimuli(summary, ctx, n=5):
    """Je Richtung die Reize mit der kleinsten 'Dosis' (Pixel x Dauer x
    Staerke), die in >= 50 % der Laeufe umschalten."""
    out = {}
    for src in ctx.attractors:
        for tgt in ctx.attractors:
            if tgt.index == src.index:
                continue
            cand = [s for s in summary
                    if s["colour"] == "other" and s["source"] == src.name
                    and s["signal_target"] == tgt.name
                    and s["switch_rate"] >= SWITCH_THRESHOLDS[0]]
            cand.sort(key=lambda s: (s["dose"], -s["switch_rate"]))
            out[(src.name, tgt.name)] = cand[:n]
    return out


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
        "axes.titlesize": 10, "figure.facecolor": "white",
    })
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    return plt, cmap


def _reg(region):
    return "" if region in (None, "", "disc") else f"_{region}"


def _reg_txt(region):
    return "" if region in (None, "", "disc") else f" [nur {region}]"


HEATMAP_METRICS = {
    "switch_rate": "Umschaltrate",
    "mean_final_target_fraction": "Anteil Ziel-Attraktor am Ende",
}


def plot_dose_heatmaps(summary, ctx, out_dir, metric="switch_rate"):
    plt, cmap = _mpl()
    by_scan = defaultdict(list)
    for s in summary:
        if s["colour"] == "other" and s["position"] == "seed":
            by_scan[(s["scan"], s["t_signal"], s.get("region", "disc"))].append(s)
    paths = []
    for (scan, ts, region), items in by_scan.items():
        radii = sorted({s["radius"] for s in items})
        durs = sorted({s["duration"] for s in items})
        if len(radii) < 2 or len(durs) < 2:
            continue
        pairs = sorted({(s["source"], s["signal_target"]) for s in items})
        modes = [m for m in ("rgba", "seedcell") if any(s["mode"] == m for s in items)]
        for strength in sorted({s["strength"] for s in items}):
            fig, axes = plt.subplots(len(pairs), len(modes),
                                     figsize=(2.2 + 0.55 * len(durs) * len(modes),
                                              0.9 + 0.32 * len(radii) * len(pairs)),
                                     squeeze=False, constrained_layout=True)
            for i, (src, tgt) in enumerate(pairs):
                for j, mode in enumerate(modes):
                    ax = axes[i][j]
                    M = np.full((len(radii), len(durs)), np.nan)
                    for s in items:
                        if (s["source"], s["signal_target"], s["mode"], s["strength"]) == (src, tgt, mode, strength):
                            M[radii.index(s["radius"]), durs.index(s["duration"])] = s[metric]
                    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                                   origin="lower")
                    for (a, b), v in np.ndenumerate(M):
                        if np.isfinite(v):
                            ax.text(b, a, f"{v * 100:.0f}", ha="center", va="center",
                                    fontsize=7, color="white" if v > 0.55 else INK)
                    ax.set_xticks(range(len(durs)), [str(d) for d in durs])
                    ax.set_yticks(range(len(radii)), [str(r) for r in radii])
                    ax.set_xlabel("Dauer D [Schritte]")
                    if j == 0:
                        ax.set_ylabel(f"{src} -> {tgt}\nRadius r [px]")
                    ax.set_title(f"Modus {mode}")
                    for sp in ax.spines.values():
                        sp.set_visible(False)
            cb = fig.colorbar(im, ax=axes, shrink=0.6, label=HEATMAP_METRICS[metric] + " [%]")
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25", "50", "75", "100"])
            cb.outline.set_visible(False)
            fig.suptitle(f"{scan}: {HEATMAP_METRICS[metric]} in %{_reg_txt(region)}\nReiz am Seed-Ort, Staerke {strength:g}, "
                         f"t_signal={ts}, {FREE_STEPS} freie Schritte nach Reizende", fontsize=10)
            p = out_dir / f"heatmap_{metric}_{scan}_t{ts}_s{strength:g}{_reg(region)}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
    return paths


def plot_montages(summary, ctx, out_dir):
    """Endzustaende (Wiederholung 0) als Raster Radius x Dauer."""
    plt, _ = _mpl()
    paths = []
    groups = defaultdict(list)
    for s in summary:
        if s["colour"] == "other" and s["position"] == "seed":
            groups[(s["scan"], s["t_signal"], s["source"], s["signal_target"],
                    s["mode"], s["strength"], s.get("region", "disc"))].append(s)
    for (scan, ts, src, tgt, mode, strength, region), items in groups.items():
        radii = sorted({s["radius"] for s in items})
        durs = sorted({s["duration"] for s in items})
        if len(radii) < 2 or len(durs) < 2:
            continue
        fig, axes = plt.subplots(len(radii), len(durs),
                                 figsize=(max(1.1 * len(durs) + 0.8, 3.6), 1.15 * len(radii) + 0.9),
                                 squeeze=False)
        lookup = {(s["radius"], s["duration"]): s for s in items}
        for a, r in enumerate(reversed(radii)):
            for b, d in enumerate(durs):
                ax = axes[a][b]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                s = lookup.get((r, d))
                if s is None:
                    continue
                p = thumb_path(out_dir, s)
                if p.is_file():
                    ax.imshow(Image.open(p))
                ax.set_title(f"{s['switch_rate'] * 100:.0f}%", fontsize=7, pad=1,
                             color=INK if s["switch_rate"] < 0.5 else CAT[0])
                if b == 0:
                    ax.set_ylabel(f"r={r}", fontsize=7)
                if a == len(radii) - 1:
                    ax.set_xlabel(f"D={d}", fontsize=7)
        fig.suptitle(f"{scan}: {src} + Farbe {tgt}, Modus {mode}, s={strength:g}{_reg_txt(region)}\n"
                     f"Endzustand (Wdh. 0); ueber jedem Bild die Umschaltrate", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, wspace=0.05, hspace=0.3,
                            top=1 - 0.75 / fig.get_figheight())
        p = out_dir / f"montage_{scan}_t{ts}_{src}_to_{tgt}_{mode}_s{strength:g}{_reg(region)}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_location_maps(summary, ctx, snaps_ref, out_dir):
    plt, cmap = _mpl()
    paths = []
    groups = defaultdict(list)
    for s in summary:
        if s["position"].startswith("grid"):
            groups[(s["scan"], s["source"], s["signal_target"], s["radius"],
                    s["duration"], s["strength"], s["mode"], s["t_signal"],
                    s["position"])].append(s)
    for (scan, src, tgt, r, d, st, mode, ts, pos), items in groups.items():
        step = int(pos.replace("grid", ""))
        src_index = ctx.names.index(src)
        bg_state = snaps_ref.get((src_index, ts), snaps_ref.get((src_index, GROW_STEPS)))
        bg = to_rgb_uint8(bg_state[:ctx.dc])
        fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
        ax.imshow(bg)
        M = np.full((ctx.H, ctx.W), np.nan)
        for s in items:
            y0, x0 = s["y"] - step // 2, s["x"] - step // 2
            M[max(y0, 0):y0 + step, max(x0, 0):x0 + step] = s["switch_rate"]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, alpha=0.6)
        ax.plot(ctx.seed_x, ctx.seed_y, marker="+", color=INK, ms=8, mew=1.5)
        ys = [s["y"] for s in items]; xs = [s["x"] for s in items]
        if max(ys) - min(ys) < ctx.H - 2 * step:   # lokales Raster -> hineinzoomen
            ax.set_xlim(min(xs) - step, max(xs) + step)
            ax.set_ylim(max(ys) + step, min(ys) - step)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Umschaltrate [%]")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25", "50", "75", "100"])
        cb.outline.set_visible(False)
        ax.set_title(f"{src} -> {tgt}: Reizort  (r={r}, D={d}, s={st:g}, {mode}, t={ts})\n"
                     f"Hintergrund: {src} bei t={ts}, + = Seed", fontsize=8)
        p = out_dir / f"location_{scan}_{src}_to_{tgt}_r{r}_d{d}_s{st:g}_{mode}_t{ts}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_timing(summary, ctx, out_dir):
    plt, _ = _mpl()
    paths = []
    by_scan = defaultdict(list)
    for s in summary:
        if s["colour"] == "other" and s["position"] == "seed":
            by_scan[s["scan"]].append(s)
    for scan, items in by_scan.items():
        times = sorted({s["t_signal"] for s in items})
        if len(times) < 3:
            continue
        pairs = sorted({(s["source"], s["signal_target"]) for s in items})
        modes = [m for m in ("rgba", "seedcell") if any(s["mode"] == m for s in items)]
        stimuli = sorted({(s["radius"], s["duration"], s["strength"]) for s in items})
        fig, axes = plt.subplots(len(pairs), len(modes), figsize=(3.4 * len(modes), 2.6 * len(pairs)),
                                 squeeze=False, sharey=True, constrained_layout=True)
        for i, (src, tgt) in enumerate(pairs):
            for j, mode in enumerate(modes):
                ax = axes[i][j]
                for c, (r, d, st) in enumerate(stimuli[:len(CAT)]):
                    pts = sorted((s["t_signal"], s["switch_rate"]) for s in items
                                 if (s["source"], s["signal_target"], s["mode"],
                                     s["radius"], s["duration"], s["strength"]) == (src, tgt, mode, r, d, st))
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=CAT[c], lw=2, marker="o", ms=4,
                            label=f"r={r}, D={d}, s={st:g}")
                ax.axvline(ctx.train_t if isinstance(ctx.train_t, int) else 128,
                           color=GRID, lw=1, ls="--")
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel("t_signal (Wachstumsschritt beim Reiz)")
                if j == 0:
                    ax.set_ylabel(f"{src} -> {tgt}\nUmschaltrate")
                ax.set_title(f"Modus {mode}")
                ax.grid(color=GRID, lw=0.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        axes[0][-1].legend(fontsize=7, frameon=False, loc="best")
        fig.suptitle(f"{scan}: Reiz waehrend des Wachstums (gestrichelt = Training-t)",
                     fontsize=10)
        p = out_dir / f"timing_{scan}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def run_examples(ctx, examples, signal_step, snaps, out_dir):
    """Rechnet ausgewaehlte Bedingungen (Wdh. 0) noch einmal und speichert
    Bildstreifen (und GIFs)."""
    if not examples:
        return []
    plt, _ = _mpl()
    colours = {a.index: a.seed_pixel for a in ctx.attractors}
    zero_col = np.zeros(ctx.dc, np.float32)
    paths = []
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    for tag, c in examples:
        P = dict(
            cy=jnp.asarray([c["y"]], jnp.int32), cx=jnp.asarray([c["x"]], jnp.int32),
            radius=jnp.asarray([c["radius"]], jnp.float32),
            duration=jnp.asarray([c["duration"]], jnp.int32),
            strength=jnp.asarray([c["strength"]], jnp.float32),
            reset_hidden=jnp.asarray([c["mode"] == "seedcell"]),
            body=jnp.asarray([c.get("region", "disc") == "body"]),
            active=jnp.asarray([c["colour"] != "none"]),
            colour=jnp.asarray([colours[c["signal_target_index"]]
                                if c["signal_target_index"] >= 0 else zero_col]),
            key=jnp.stack([post_key(0)]),
        )
        end = c["duration"] + FREE_STEPS + EXAMPLE_EXTRA_STEPS
        wanted = [end if s == "end" else int(s) for s in EXAMPLE_SNAPSHOT_STEPS]
        wanted = sorted({w for w in wanted if 1 <= w <= end})
        x = jnp.asarray(snaps[(c["source_index"], 0, c["t_signal"])][None])
        frames, gif = [(0, np.asarray(x[0, :ctx.dc]), None)], []
        for k in range(end):
            x, lab, *_ = signal_step(ctx.model, x, jnp.int32(k), P)
            if (k + 1) in wanted:
                frames.append((k + 1, np.asarray(x[0, :ctx.dc]), int(lab[0])))
            if SAVE_GIFS and k % 2 == 0:
                gif.append(to_rgb_uint8(np.asarray(x[0, :ctx.dc])))
        fig, axes = plt.subplots(1, len(frames), figsize=(1.6 * len(frames), 2.0))
        for ax, (k, rgba, lab) in zip(axes, frames):
            ax.imshow(to_rgb_uint8(rgba))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            title = "vor Reiz" if k == 0 else f"+{k}"
            if lab is not None:
                title += f"\n{ctx.class_names[lab]}"
            ax.set_title(title, fontsize=7)
        if c["radius"] >= 0:
            circ = plt.Circle((c["x"], c["y"]), max(c["radius"], 0.7), fill=False,
                              color=CAT[1], lw=1)
            axes[0].add_patch(circ)
        fig.suptitle(f"{tag}: {c['source']} + Farbe {c['signal_target'] or '-'}  "
                     f"r={c['radius']} D={c['duration']} s={c['strength']:g} "
                     f"{c['mode']}{_reg_txt(c.get('region', 'disc'))} @({c['y']},{c['x']}) t={c['t_signal']}"
                     f"   (Bewertung im Scan bei +{c['duration'] + FREE_STEPS})", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        safe = tag.replace(" ", "_")
        p = ex_dir / f"{safe}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
        if SAVE_GIFS and gif:
            imgs = [Image.fromarray(g).resize((280, 280), Image.NEAREST) for g in gif]
            imgs[0].save(ex_dir / f"{safe}.gif", save_all=True,
                         append_images=imgs[1:], duration=60, loop=0)
    return paths


def choose_examples(summary, best, ctx):
    ex = []
    for (src, tgt), items in best.items():
        for i, s in enumerate(items[:N_EXAMPLES_PER_DIRECTION]):
            ex.append((f"{src}_to_{tgt}_best{i + 1}", s))
        # knapp daneben: gleiche Gruppe, naechstkleinerer Radius
        if items:
            b = items[0]
            smaller = [s for s in summary if s["scan"] == b["scan"] and s["source"] == src
                       and s["signal_target"] == tgt and s["mode"] == b["mode"]
                       and s["strength"] == b["strength"] and s["duration"] == b["duration"]
                       and s["t_signal"] == b["t_signal"] and s["position"] == b["position"]
                       and s["y"] == b["y"] and s["x"] == b["x"] and s["radius"] < b["radius"]
                       and s.get("region") == b.get("region")]
            if smaller:
                ex.append((f"{src}_to_{tgt}_near_miss", max(smaller, key=lambda s: s["radius"])))
        if not items:
            # nichts schaltet: zeige den Reiz mit der hoechsten Rate
            cand = [s for s in summary if s["source"] == src and s["signal_target"] == tgt
                    and s["colour"] == "other"]
            if cand:
                ex.append((f"{src}_to_{tgt}_highest", max(cand, key=lambda s: (s["switch_rate"], s["mean_final_target_fraction"] or 0))))
    return ex


# ======================================================================
# BERICHT
# ======================================================================

def fmt_pct(v):
    return "" if v in ("", None) else f"{100 * float(v):.0f} %"


def write_report(ctx, summary, thr, best, base_check, out_dir, figures, runtime):
    L = []
    L.append(f"# Signal-Scan: {ctx.bundle.name}\n")
    L.append(f"Modell: `{ctx.bundle}`  ")
    L.append(f"Familie {ctx.family}, Padding {ctx.padding}, {ctx.n_channels} Kanaele, "
             f"Training-t {ctx.train_t}, Leinwand {ctx.H}x{ctx.W}, Seed ({ctx.seed_y}, {ctx.seed_x})  ")
    L.append("Attraktoren: " + ", ".join(
        f"{a.name} (Seed-RGBA {np.round(a.seed_pixel * 255).astype(int).tolist()})"
        for a in ctx.attractors) + "  ")
    L.append(f"GROW_STEPS {GROW_STEPS}, FREE_STEPS {FREE_STEPS}, Laufzeit {runtime / 60:.1f} min\n")
    for w in ctx.warnings:
        L.append(f"> **WARNUNG:** {w}\n")

    L.append("## Ausgangsmuster ohne Reiz\n")
    for name, info in base_check.items():
        L.append(f"- {name}: nach {GROW_STEPS} Schritten klassifiziert als {info}")
    L.append("")

    ctrl = [s for s in summary if s["colour"] in ("none", "own")]
    if ctrl:
        L.append("## Kontrollen\n")
        L.append("| Scan | Quelle | Farbe | r | D | Modus | bleibt | Mischung | leer | anderer Attraktor |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(ctrl, key=lambda s: (s["source"], s["colour"], s["mode"], s["radius"], s["duration"])):
            L.append(f"| {s['scan']} | {s['source']} | {s['colour']} | {s['radius']} | {s['duration']} | "
                     f"{s['mode']} | {fmt_pct(s['stay_rate'])} | {fmt_pct(s['intermediate_rate'])} | "
                     f"{fmt_pct(s['no_pattern_rate'])} | {fmt_pct(s['other_attractor_rate'])} |")
        L.append("")

    L.append(f"## Kleinste Reize mit Umschaltrate >= {SWITCH_THRESHOLDS[0]:g}\n")
    L.append("Dosis = Pixel x Dauer x Staerke.\n")
    for (src, tgt), items in best.items():
        L.append(f"### {src} -> {tgt}\n")
        if not items:
            n_any = [s for s in summary if s["source"] == src and s["signal_target"] == tgt
                     and s["colour"] == "other"]
            mx = max(n_any, key=lambda s: (s["switch_rate"], s["mean_final_target_fraction"] or 0)) if n_any else None
            L.append("Kein getesteter Reiz schaltet in >= 50 % der Laeufe um." +
                     (f" Hoechste Rate: {fmt_pct(mx['switch_rate'])} bei r={mx['radius']}, "
                      f"D={mx['duration']}, s={mx['strength']:g}, {mx['mode']}, "
                      f"Position {mx['position']} ({mx['y']},{mx['x']}), t={mx['t_signal']} "
                      f"(Anteil Ziel-Attraktor am Ende {fmt_pct(mx['mean_final_target_fraction'])})."
                      if mx else "") + "\n")
            continue
        L.append("| Scan | r | Pixel | D | s | Modus | Ort | t | Dosis | Umschalten | Mischung | Schritte bis Umschalten |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in items:
            L.append(f"| {s['scan']} | {s['radius']} | {s['n_pixels']} | {s['duration']} | {s['strength']:g} | "
                     f"{s['mode']} | {s['position']} ({s['y']},{s['x']}) | {s['t_signal']} | {s['dose']:g} | "
                     f"{fmt_pct(s['switch_rate'])} | {fmt_pct(s['intermediate_rate'])} | {s['median_steps_to_switch']} |")
        L.append("")

    if thr:
        L.append("## Schwellen-Radius je Dauer (Reiz am Seed-Ort)\n")
        L.append("Kleinster Radius, ab dem die Umschaltrate >= 50 % bzw. >= 90 % ist (leer = nie).\n")
        L.append("r(Mischung) = kleinster Radius, ab dem der Endzustand in mehr als 50 % der Laeufe "
                 "nicht mehr als Ausgangsattraktor gilt (meist: Mischung/Chimaere).\n")
        L.append("| Scan | Quelle -> Ziel | Modus | s | D | t | r(Mischung) | r(50 %) | r(90 %) | max. Rate (bei r) | max. Ziel-Anteil |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in thr:
            L.append(f"| {t['scan']} | {t['source']} -> {t['signal_target']} | {t['mode']}{_reg_txt(t['region'])} | {t['strength']:g} | "
                     f"{t['duration']} | {t['t_signal']} | {t['min_radius_not_repaired']} | "
                     f"{t['min_radius_switch_ge_0.5']} | {t['min_radius_switch_ge_0.9']} | "
                     f"{fmt_pct(t['max_switch_rate'])} ({t['radius_at_max']}) | "
                     f"{fmt_pct(t['max_final_target_fraction'])} |")
        L.append("")

        # Was passiert stattdessen? Ergebnis je Radius
        L.append("## Ergebnis je Radius (Reiz am Seed-Ort)\n")
        L.append("U = umgeschaltet, B = bleibt Ausgangsattraktor (ein kleiner fremdfarbiger Fleck kann "
                 "trotzdem stehen bleiben), M = Mischung/Chimaere, L = leer. Zahl = mittlerer Anteil "
                 "der sichtbaren Pixel, die dem Ziel-Attraktor am naechsten sind (Ausgangsmuster ohne "
                 "Reiz: siehe oben, ~1-4 %).\n")
        groups = defaultdict(list)
        for s in summary:
            if s["colour"] == "other" and s["position"] == "seed":
                groups[(s["scan"], s["source"], s["signal_target"], s["mode"],
                        s["strength"], s["duration"], s["t_signal"], s.get("region", "disc"))].append(s)
        radii = sorted({s["radius"] for g in groups.values() for s in g})
        L.append("| Scan | Quelle -> Ziel | Modus | s | D | t | " + " | ".join(f"r={r}" for r in radii) + " |")
        L.append("|---|---|---|---|---|---|" + "---|" * len(radii))
        for key in sorted(groups):
            by_r = {s["radius"]: s for s in groups[key]}
            cells = []
            for r in radii:
                s = by_r.get(r)
                if s is None:
                    cells.append("")
                    continue
                rates = {"U": s["switch_rate"], "B": s["stay_rate"],
                         "M": s["intermediate_rate"], "L": s["no_pattern_rate"]}
                code = max(rates, key=rates.get)
                cells.append(f"{code} {fmt_pct(s['mean_final_target_fraction'])}")
            scan, src, tgt, mode, st, d, ts, region = key
            L.append(f"| {scan} | {src} -> {tgt} | {mode}{_reg_txt(region)} | {st:g} | {d} | {ts} | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## Dateien\n")
    L.append("- `conditions_summary.csv` - eine Zeile je Bedingung (Raten ueber die Wiederholungen)")
    L.append("- `rollouts.csv` - eine Zeile je Lauf")
    L.append("- `thresholds.csv` - Schwellen-Radien")
    for p in figures:
        L.append(f"- `{p.relative_to(out_dir)}`")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


# ======================================================================
# MAIN
# ======================================================================

def main():
    global GROW_STEPS, FREE_STEPS, REPEATS, BATCH_SIZE, SCANS

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="nur Bedingungen zaehlen und Aufwand schaetzen, nichts rechnen")
    args = ap.parse_args()

    if args.quick:
        GROW_STEPS = QUICK["GROW_STEPS"]
        FREE_STEPS = QUICK["FREE_STEPS"]
        REPEATS = QUICK["REPEATS"]
        BATCH_SIZE = QUICK["BATCH_SIZE"]
        SCANS = QUICK["SCANS"]
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.repeats:
        REPEATS = args.repeats
        for spec in SCANS.values():
            spec.pop("repeats", None)
    scans = SCANS
    if args.scans:
        wanted = [s.strip() for s in args.scans.split(",")]
        scans = {k: v for k, v in SCANS.items() if k in wanted}
        missing = set(wanted) - set(scans)
        if missing:
            raise SystemExit(f"Unbekannte Scans: {missing}. Vorhanden: {list(SCANS)}")

    bundle = resolve_bundle(args.bundle or MODEL_BUNDLE)
    emoji_dir = resolve_emoji_dir()
    print("JAX-Geraete:", jax.devices())
    print("Modell:", bundle)
    print("Emojis:", emoji_dir)

    ctx = prepare(bundle, emoji_dir)
    print(f"{ctx.family}, Padding {ctx.padding}, {ctx.n_channels} Kanaele, "
          f"Attraktoren {ctx.names}, Leinwand {ctx.H}x{ctx.W}, Seed ({ctx.seed_y},{ctx.seed_x})")
    for a in ctx.attractors:
        print(f"  {a.name}: Seed-RGBA {np.round(a.seed_pixel * 255).astype(int).tolist()} "
              f"({a.initial_description})")

    tag = f"{ctx.K}attr_{ctx.padding.lower()}_{ctx.family}_{bundle.name.split('-')[0]}"
    out_dir = Path(args.out) if args.out else OUTPUT_ROOT / (tag + ("_quick" if args.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Nur Einstellungen, die ein einzelnes Laufergebnis veraendern. Andere
    # Scans hinzuzufuegen ist erlaubt (es werden nur die fehlenden gerechnet).
    run_settings = dict(
        bundle=str(bundle.resolve()), GROW_STEPS=GROW_STEPS, FREE_STEPS=FREE_STEPS,
        SEED=SEED, ACTIVE_ALPHA_THRESHOLD=ACTIVE_ALPHA_THRESHOLD,
        MIXED_MIN_FRACTION=MIXED_MIN_FRACTION)
    settings_path = out_dir / "settings.json"
    if settings_path.is_file() and (out_dir / "rollouts.csv").is_file():
        old = json.loads(settings_path.read_text()).get("run_settings")
        if old != run_settings:
            backup = out_dir / f"rollouts_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            (out_dir / "rollouts.csv").rename(backup)
            print(f"Einstellungen haben sich geaendert -> alte Ergebnisse nach {backup.name} "
                  f"verschoben, alles wird neu gerechnet.")
    settings_path.write_text(json.dumps(dict(
        run_settings=run_settings, REPEATS=REPEATS, BATCH_SIZE=BATCH_SIZE,
        SCANS=scans), indent=2))
    print("Ausgabe:", out_dir)

    if args.dry_run:
        conds = build_conditions(ctx, scans)
        per = defaultdict(lambda: [0, 0])
        for c in conds:
            per[c["scan"]][0] += c["repeats"]
            per[c["scan"]][1] += c["repeats"] * (c["duration"] + FREE_STEPS)
        tot = sum(v[1] for v in per.values())
        for k, (n, st) in per.items():
            print(f"  {k:20s} {n:7d} Laeufe  {st:12,d} Laufschritte")
        print(f"  {'SUMME':20s} {sum(v[0] for v in per.values()):7d} Laeufe  {tot:12,d} Laufschritte")
        print("  Grob: GPU ~1 ms je Laufschritt -> "
              f"~{tot * 1e-3 / 60:.0f} min (CPU ~150 ms -> ~{tot * 0.15 / 3600:.0f} h)")
        return

    t_all = time.time()
    signal_step, grow_step, classify_batch = make_step_functions(ctx)

    # Kontrolle: Ausgangsmuster ohne Reiz (Wdh. 0) und Referenzbilder
    ref = grow_snapshots(ctx, grow_step, [GROW_STEPS], 1)
    ref_states = {a.index: ref[(a.index, 0, GROW_STEPS)] for a in ctx.attractors}
    lab, fr, mse, nact = classify_batch(jnp.asarray(np.stack(list(ref_states.values()))))
    base_check = {}
    for i, a in enumerate(ctx.attractors):
        base_check[a.name] = (f"{ctx.class_names[int(lab[i])]} "
                              f"(Anteile {dict(zip(ctx.names, np.round(np.asarray(fr[i]), 3).tolist()))}, "
                              f"MSE {dict(zip(ctx.names, np.round(np.asarray(mse[i]), 4).tolist()))})")
        print(f"  {a.name} nach {GROW_STEPS} Schritten ohne Reiz: {base_check[a.name]}")
        if int(lab[i]) != a.index:
            ctx.warnings.append(
                f"{a.name} ist nach {GROW_STEPS} Schritten ohne Reiz nicht als {a.name} "
                f"klassifiziert, sondern als {ctx.class_names[int(lab[i])]}. "
                f"'Umschalten' aus diesem Attraktor ist dann nicht interpretierbar.")
        Image.fromarray(to_rgb_uint8(ref_states[a.index][:ctx.dc])).resize(
            (ctx.W * 2, ctx.H * 2), Image.NEAREST).save(out_dir / f"grown_{a.name}.png")

    conditions = build_conditions(ctx, scans, ref_states)
    counts = defaultdict(int)
    for c in conditions:
        counts[c["scan"]] += c["repeats"]
    print("Laeufe je Scan:", dict(counts))
    write_csv(out_dir / "conditions.csv", conditions)

    if ctx.warnings:
        print("\n**************************************************")
        for w in ctx.warnings:
            print("  WARNUNG:", w)
        print("**************************************************\n")

    run_jobs(ctx, conditions, signal_step, grow_step, out_dir, resume=not args.no_resume)

    summary = summarise(ctx, conditions, out_dir)
    thr = thresholds(summary)
    write_csv(out_dir / "thresholds.csv", thr)
    best = best_stimuli(summary, ctx)

    figures = []

    def safe(fn, *a):
        # Eine fehlerhafte Abbildung soll nicht den ganzen Bericht verhindern
        try:
            return fn(*a)
        except Exception as exc:
            print(f"  WARNUNG: {fn.__name__} fehlgeschlagen: {type(exc).__name__}: {exc}")
            return []

    for metric in HEATMAP_METRICS:
        figures += safe(plot_dose_heatmaps, summary, ctx, out_dir, metric)
    figures += safe(plot_montages, summary, ctx, out_dir)
    loc_times = sorted({s["t_signal"] for s in summary if s["position"].startswith("grid")})
    if loc_times:
        snaps_loc = grow_snapshots(ctx, grow_step, loc_times, 1)
        bg_states = {(a, t): v for (a, r, t), v in snaps_loc.items()}
        figures += safe(plot_location_maps, summary, ctx, bg_states, out_dir)
    figures += safe(plot_timing, summary, ctx, out_dir)

    cond_by_key = {c["key"]: c for c in conditions}
    examples = [(tag, cond_by_key[s["key"]]) for tag, s in choose_examples(summary, best, ctx)]
    if examples:
        need_t = sorted({c["t_signal"] for _, c in examples})
        snaps = grow_snapshots(ctx, grow_step, need_t, 1)
        figures += safe(run_examples, ctx, examples, signal_step, snaps, out_dir)

    write_report(ctx, summary, thr, best, base_check, out_dir, figures, time.time() - t_all)
    print(f"\nFertig in {(time.time() - t_all) / 60:.1f} min. Bericht: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
