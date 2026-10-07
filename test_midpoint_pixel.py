from pathlib import Path
import os
import csv

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from Experiments.config_helpers import build_model
from Common.dataloader.emoji import load_emoji_sequence


# ============================================================
# SETTINGS
# ============================================================

BUNDLE = Path(
    "models/local/emoji_multi_attractor_example_v2/"
    "bundles/nca-emoji/emoji-multi-attractor-example-v2/"
    "20260830T174447Z-cfga63e288e1e9e-db1f45fc"
)

MODEL_PATH = BUNDLE / "model.eqx"
CONFIG_PATH = BUNDLE / "config.yaml"

OUTPUT_ROOT = Path(
    "inference_outputs/expanded_attractor_scan"
)

NORMAL_DIR = OUTPUT_ROOT / "normal_colour_scan"
COMPLEMENT_DIR = OUTPUT_ROOT / "complementary_colour_scan"
SPECIAL_DIR = OUTPUT_ROOT / "special_pixels"

for directory in [
    OUTPUT_ROOT,
    NORMAL_DIR,
    COMPLEMENT_DIR,
    SPECIAL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# NCA settings
# ------------------------------------------------------------

STEPS = 128

SEED = 0

PATCH_SIZE = 1

# 0%, 1%, ..., 100%
ALPHAS = np.arange(
    0,
    101
) / 100.0


# ------------------------------------------------------------
# Mixed / intermediate classification
# ------------------------------------------------------------

MIXED_MIN_FRACTION = 0.20

ACTIVE_ALPHA_THRESHOLD = 0.10


# ============================================================
# 1. CONFIG LADEN
# ============================================================

cfg = OmegaConf.load(
    CONFIG_PATH
)


def cfg_value(*paths, default=None):

    for path in paths:

        value = OmegaConf.select(
            cfg,
            path,
            default=None
        )

        if value is not None:
            return value

    return default


downsample = int(
    cfg_value(
        "data.preprocessing.downsample",
        "data.downsample",
        "data.emoji.downsample",
        default=1
    )
)


crop_square = bool(
    cfg_value(
        "data.augmentation.crop_square",
        "data.emoji.crop_square",
        "data.preprocessing.crop_square",
        default=False
    )
)


pad = cfg_value(
    "data.augmentation.pad",
    "data.emoji.pad",
    default=0
)


data_channels = int(
    cfg_value(
        "data.augmentation.data_channels",
        "data.emoji.data_channels",
        "data.data_channels",
        default=4
    )
)


iterations = cfg_value(
    "run.iterations",
    "training.loop.iterations",
    default="?"
)


run_t = cfg_value(
    "run.t",
    "training.loop.t",
    default="?"
)


print("\n======================================")
print("CONFIG")
print("======================================")

print("Bundle:", BUNDLE)
print("Family:", cfg.model.family)
print("Channels:", cfg.model.channels)
print("Fire rate:", cfg.model.fire_rate)
print("Downsample:", downsample)
print("Data channels:", data_channels)
print("Iterations:", iterations)
print("Training t:", run_t)


# ============================================================
# 2. MODELL LADEN
# ============================================================

model_result = build_model(
    cfg.model,
    key=jr.PRNGKey(0)
)


if isinstance(
    model_result,
    tuple
):

    model = model_result[0]

else:

    model = model_result


model = model.load(
    MODEL_PATH
)


print("\n======================================")
print("MODEL LOADED")
print("======================================")

print(
    model.get_config()
)


# ============================================================
# 3. EMOJIS LADEN
# ============================================================

if "DATA_PATH_BASE" not in os.environ:

    raise RuntimeError(
        "DATA_PATH_BASE ist nicht gesetzt."
    )


EMOJI_DIR = (
    Path(
        os.environ[
            "DATA_PATH_BASE"
        ]
    )
    /
    "Emojis"
)


def load_emoji(filename):

    dataset = load_emoji_sequence(
        [filename],
        impath_emojis=EMOJI_DIR,
        downsample=downsample,
        crop_square=crop_square
    )

    return jnp.asarray(
        dataset.data[0, 0],
        dtype=jnp.float32
    )


crab = load_emoji(
    "crab.png"
)

microbe = load_emoji(
    "microbe.png"
)


if crab.shape != microbe.shape:

    raise RuntimeError(
        "Crab and microbe have different shapes."
    )


C, H, W = crab.shape


# ============================================================
# 4. PATCH POSITION
# ============================================================

patch_y = (
    H - PATCH_SIZE
) // 2

patch_x = (
    W - PATCH_SIZE
) // 2


crab_pixel = crab[
    :data_channels,
    patch_y,
    patch_x
]


microbe_pixel = microbe[
    :data_channels,
    patch_y,
    patch_x
]


print("\n======================================")
print("ORIGINAL ENDPOINT PIXELS")
print("======================================")

print(
    "Crab:",
    np.round(
        np.asarray(
            crab_pixel
        ) * 255
    ).astype(int)
)

print(
    "Microbe:",
    np.round(
        np.asarray(
            microbe_pixel
        ) * 255
    ).astype(int)
)


# ============================================================
# 5. KOMPLEMENTÄRFARBEN
# ============================================================

def complementary_pixel(pixel):

    result = pixel.copy()

    result = result.at[
        :3
    ].set(
        1.0 - pixel[:3]
    )

    # Alpha bleibt unverändert

    return result


complement_crab = complementary_pixel(
    crab_pixel
)

complement_microbe = complementary_pixel(
    microbe_pixel
)


print("\n======================================")
print("COMPLEMENTARY ENDPOINTS")
print("======================================")

print(
    "Complement Crab:",
    np.round(
        np.asarray(
            complement_crab
        ) * 255
    ).astype(int)
)

print(
    "Complement Microbe:",
    np.round(
        np.asarray(
            complement_microbe
        ) * 255
    ).astype(int)
)


# ============================================================
# 6. SPECIAL PIXELS
# ============================================================

# vollständig transparent:
# kein RGB und Alpha = 0
transparent_pixel = jnp.array(
    [0.0, 0.0, 0.0, 0.0],
    dtype=jnp.float32
)


# opakes Schwarz:
# RGB = 0, Alpha = 1
black_pixel = jnp.array(
    [0.0, 0.0, 0.0, 1.0],
    dtype=jnp.float32
)


# ============================================================
# 7. PADDING
# ============================================================

if pad is None:

    pad = 0


if isinstance(
    pad,
    (int, float)
):

    pad_top = int(pad)
    pad_bottom = int(pad)
    pad_left = int(pad)
    pad_right = int(pad)

else:

    pad_list = list(pad)

    if len(pad_list) == 1:

        p = int(
            pad_list[0]
        )

        pad_top = p
        pad_bottom = p
        pad_left = p
        pad_right = p

    elif len(pad_list) == 2:

        pad_top = int(
            pad_list[0]
        )

        pad_bottom = int(
            pad_list[0]
        )

        pad_left = int(
            pad_list[1]
        )

        pad_right = int(
            pad_list[1]
        )

    elif len(pad_list) == 4:

        pad_top = int(
            pad_list[0]
        )

        pad_bottom = int(
            pad_list[1]
        )

        pad_left = int(
            pad_list[2]
        )

        pad_right = int(
            pad_list[3]
        )

    else:

        raise ValueError(
            f"Unexpected padding: {pad_list}"
        )


canvas_H = (
    H
    + pad_top
    + pad_bottom
)

canvas_W = (
    W
    + pad_left
    + pad_right
)


y = (
    patch_y
    + pad_top
)

x = (
    patch_x
    + pad_left
)


N_CHANNELS = int(
    cfg.model.channels
)


# ============================================================
# 8. STATE -> IMAGE
# ============================================================

def state_to_rgba(state):

    image = np.asarray(
        jax.device_get(
            state[:4]
        )
    )

    image = np.moveaxis(
        image,
        0,
        -1
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return np.round(
        image * 255
    ).astype(
        np.uint8
    )


# ============================================================
# 9. NCA STEP
# ============================================================

@eqx.filter_jit
def nca_step(
    model,
    state,
    key
):

    return model(
        state,
        key=key
    )


# ============================================================
# 10. ENDZUSTAND KLASSIFIZIEREN
# ============================================================

def classify_state(state):

    # --------------------------------------------------------
    # Bereich ohne Padding
    # --------------------------------------------------------

    final_crop = state[
        :data_channels,
        pad_top:pad_top + H,
        pad_left:pad_left + W
    ]


    # --------------------------------------------------------
    # globale MSE
    # --------------------------------------------------------

    mse_crab = float(
        jnp.mean(
            (
                final_crop
                -
                crab[:data_channels]
            ) ** 2
        )
    )


    mse_microbe = float(
        jnp.mean(
            (
                final_crop
                -
                microbe[:data_channels]
            ) ** 2
        )
    )


    # --------------------------------------------------------
    # lokaler RGB-Vergleich
    # --------------------------------------------------------

    final_rgb = final_crop[
        :3
    ]


    crab_rgb = crab[
        :3
    ]


    microbe_rgb = microbe[
        :3
    ]


    error_crab = jnp.mean(
        (
            final_rgb
            -
            crab_rgb
        ) ** 2,
        axis=0
    )


    error_microbe = jnp.mean(
        (
            final_rgb
            -
            microbe_rgb
        ) ** 2,
        axis=0
    )


    # --------------------------------------------------------
    # sichtbare Pixel
    # --------------------------------------------------------

    if data_channels >= 4:

        active_mask = (
            final_crop[3]
            >
            ACTIVE_ALPHA_THRESHOLD
        )

    else:

        active_mask = (
            jnp.max(
                jnp.abs(
                    final_rgb
                ),
                axis=0
            )
            >
            0.10
        )


    active_pixels = int(
        jnp.sum(
            active_mask
        )
    )


    # --------------------------------------------------------
    # keine sichtbare Struktur
    # --------------------------------------------------------

    if active_pixels == 0:

        return {
            "classification":
                "NO_VISIBLE_PATTERN",

            "mse_crab":
                mse_crab,

            "mse_microbe":
                mse_microbe,

            "active_pixels":
                0,

            "crab_fraction":
                0.0,

            "microbe_fraction":
                0.0,
        }


    crab_like = (
        (
            error_crab
            <
            error_microbe
        )
        &
        active_mask
    )


    microbe_like = (
        (
            error_microbe
            <
            error_crab
        )
        &
        active_mask
    )


    crab_like_pixels = int(
        jnp.sum(
            crab_like
        )
    )


    microbe_like_pixels = int(
        jnp.sum(
            microbe_like
        )
    )


    crab_fraction = (
        crab_like_pixels
        /
        active_pixels
    )


    microbe_fraction = (
        microbe_like_pixels
        /
        active_pixels
    )


    # --------------------------------------------------------
    # mixed / intermediate
    # --------------------------------------------------------

    if (
        crab_fraction
        >=
        MIXED_MIN_FRACTION
        and
        microbe_fraction
        >=
        MIXED_MIN_FRACTION
    ):

        classification = (
            "INTERMEDIATE"
        )

    elif (
        mse_crab
        <
        mse_microbe
    ):

        classification = (
            "CRAB"
        )

    else:

        classification = (
            "MICROBE"
        )


    return {

        "classification":
            classification,

        "mse_crab":
            mse_crab,

        "mse_microbe":
            mse_microbe,

        "active_pixels":
            active_pixels,

        "crab_fraction":
            crab_fraction,

        "microbe_fraction":
            microbe_fraction,
    }


# ============================================================
# 11. EINEN BELIEBIGEN SEED-PIXEL LAUFEN LASSEN
# ============================================================

def run_pixel(
    pixel,
    collect_frames=False
):

    state = jnp.zeros(
        (
            N_CHANNELS,
            canvas_H,
            canvas_W
        ),
        dtype=jnp.float32
    )


    state = state.at[
        :data_channels,
        y,
        x
    ].set(
        pixel
    )


    key = jr.PRNGKey(
        SEED
    )


    frames = []


    if collect_frames:

        frames.append(
            Image.fromarray(
                state_to_rgba(
                    state
                ),
                mode="RGBA"
            )
        )


    for t in range(
        1,
        STEPS + 1
    ):

        key, subkey = jr.split(
            key
        )

        state = nca_step(
            model,
            state,
            subkey
        )


        if (
            collect_frames
            and
            t % 2 == 0
        ):

            frames.append(
                Image.fromarray(
                    state_to_rgba(
                        state
                    ),
                    mode="RGBA"
                )
            )


    classification = classify_state(
        state
    )


    return (
        state,
        classification,
        frames
    )


# ============================================================
# 12. 1%-SCAN FUNKTION
# ============================================================

def run_colour_scan(
    start_pixel,
    end_pixel,
    output_directory,
    scan_name
):

    final_directory = (
        output_directory
        /
        "final_states"
    )

    final_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    results = []
    summary_frames = []


    print("\n======================================")
    print(scan_name)
    print("======================================")


    for index, alpha in enumerate(
        ALPHAS
    ):

        pixel = (
            (1.0 - alpha)
            * start_pixel
            +
            alpha
            * end_pixel
        )


        (
            state,
            result,
            _
        ) = run_pixel(
            pixel
        )


        crab_percent = int(
            round(
                (1.0 - alpha)
                * 100
            )
        )


        microbe_percent = int(
            round(
                alpha * 100
            )
        )


        image = state_to_rgba(
            state
        )


        filename = (
            final_directory
            /
            f"alpha_{index:03d}_"
            f"{result['classification'].lower()}.png"
        )


        Image.fromarray(
            image,
            mode="RGBA"
        ).save(
            filename
        )


        # ----------------------------------------------------
        # GIF frame
        # ----------------------------------------------------

        frame = Image.fromarray(
            image,
            mode="RGBA"
        ).convert(
            "RGB"
        )


        draw = ImageDraw.Draw(
            frame
        )


        draw.text(
            (5, 5),
            (
                f"alpha={alpha:.2f}"
            )
        )


        draw.text(
            (5, 20),
            result[
                "classification"
            ]
        )


        draw.text(
            (5, 35),
            (
                f"{result['crab_fraction']*100:.1f}% Crab-like | "
                f"{result['microbe_fraction']*100:.1f}% Microbe-like"
            )
        )


        summary_frames.append(
            frame
        )


        pixel_np = np.asarray(
            pixel
        )


        row = {

            "alpha":
                float(alpha),

            "start_percent":
                crab_percent,

            "end_percent":
                microbe_percent,

            "R":
                float(
                    pixel_np[0]
                ),

            "G":
                float(
                    pixel_np[1]
                ),

            "B":
                float(
                    pixel_np[2]
                ),

            "A":
                float(
                    pixel_np[3]
                ),

            "R_255":
                float(
                    pixel_np[0]
                    * 255
                ),

            "G_255":
                float(
                    pixel_np[1]
                    * 255
                ),

            "B_255":
                float(
                    pixel_np[2]
                    * 255
                ),

            "A_255":
                float(
                    pixel_np[3]
                    * 255
                ),

            **result,
        }


        results.append(
            row
        )


        print(
            f"{index:3d}% | "
            f"RGB="
            f"[{pixel_np[0]*255:6.1f}, "
            f"{pixel_np[1]*255:6.1f}, "
            f"{pixel_np[2]*255:6.1f}] | "
            f"{result['crab_fraction']*100:5.1f}% Crab-like | "
            f"{result['microbe_fraction']*100:5.1f}% Microbe-like | "
            f"-> {result['classification']}"
        )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        output_directory
        /
        "results.csv"
    )


    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # --------------------------------------------------------
    # GIF
    # --------------------------------------------------------

    gif_path = (
        output_directory
        /
        "final_states.gif"
    )


    summary_frames[0].save(
        gif_path,
        save_all=True,
        append_images=
            summary_frames[1:],
        duration=150,
        loop=0
    )


    return results


# ============================================================
# 13. SPECIAL PIXEL TEST
# ============================================================

def run_special_test(
    name,
    pixel
):

    print("\n======================================")
    print(name.upper())
    print("======================================")


    (
        state,
        result,
        frames
    ) = run_pixel(
        pixel,
        collect_frames=True
    )


    # final image

    image = state_to_rgba(
        state
    )


    final_path = (
        SPECIAL_DIR
        /
        f"{name}_final.png"
    )


    Image.fromarray(
        image,
        mode="RGBA"
    ).save(
        final_path
    )


    # rollout GIF

    gif_path = (
        SPECIAL_DIR
        /
        f"{name}_rollout.gif"
    )


    frames[0].save(
        gif_path,
        save_all=True,
        append_images=
            frames[1:],
        duration=100,
        loop=0
    )


    print(
        "Seed RGBA:",
        np.asarray(
            pixel
        )
    )

    print(
        "Classification:",
        result[
            "classification"
        ]
    )

    print(
        "Crab-like:",
        f"{result['crab_fraction']*100:.1f}%"
    )

    print(
        "Microbe-like:",
        f"{result['microbe_fraction']*100:.1f}%"
    )

    print(
        "MSE Crab:",
        result[
            "mse_crab"
        ]
    )

    print(
        "MSE Microbe:",
        result[
            "mse_microbe"
        ]
    )


    return result


# ============================================================
# 14. TEST 1:
# NORMALE CRAB -> MICROBE FARBLINIE
# ============================================================

normal_results = run_colour_scan(
    crab_pixel,
    microbe_pixel,
    NORMAL_DIR,
    "NORMAL COLOUR SCAN"
)


# ============================================================
# 15. TEST 2:
# TRANSPARENTER / FARBLOSER PIXEL
# ============================================================

transparent_result = run_special_test(
    "transparent_pixel",
    transparent_pixel
)


# ============================================================
# 16. TEST 3:
# SCHWARZER PIXEL
# ============================================================

black_result = run_special_test(
    "black_pixel",
    black_pixel
)


# ============================================================
# 17. TEST 4:
# KOMPLEMENTÄRFARBEN IN 1%-SCHRITTEN
# ============================================================

complement_results = run_colour_scan(
    complement_crab,
    complement_microbe,
    COMPLEMENT_DIR,
    "COMPLEMENTARY COLOUR SCAN"
)


# ============================================================
# 18. SUMMARY
# ============================================================

def print_counts(
    name,
    results
):

    print(
        "\n",
        name
    )

    for category in [
        "CRAB",
        "INTERMEDIATE",
        "MICROBE",
        "NO_VISIBLE_PATTERN",
    ]:

        count = sum(
            result[
                "classification"
            ]
            ==
            category

            for result
            in results
        )

        print(
            category,
            "=",
            count
        )


print("\n======================================")
print("FINAL SUMMARY")
print("======================================")


print_counts(
    "NORMAL SCAN",
    normal_results
)


print_counts(
    "COMPLEMENTARY SCAN",
    complement_results
)


print(
    "\nTransparent pixel:",
    transparent_result[
        "classification"
    ]
)


print(
    "Black pixel:",
    black_result[
        "classification"
    ]
)


print("\n======================================")
print("OUTPUT")
print("======================================")


print(
    OUTPUT_ROOT.resolve()
)

print("\nDONE.")
