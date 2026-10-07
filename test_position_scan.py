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






BUNDLE = Path(
    "models/local/emoji_multi_attractor_example_v2/"
    "bundles/nca-emoji/emoji-multi-attractor-example-v2/"
    "20260830T174447Z-cfga63e288e1e9e-db1f45fc"
)

MODEL_PATH = BUNDLE / "model.eqx"
CONFIG_PATH = BUNDLE / "config.yaml"

OUTPUT_ROOT = Path(
    "inference_outputs/spatial_position_scan"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

STEPS = 128




SEED = 0

PATCH_SIZE = 1


ALPHAS = np.arange(
    0,
    101
) / 100.0










MIXED_MIN_FRACTION = 0.20

ACTIVE_ALPHA_THRESHOLD = 0.10


# RGB spectrum scan:
# A 64x64 matrix contains 4096 fields. Using 16 equally spaced values
# for each RGB channel gives exactly 16^3 = 4096 unique colours.
SPECTRUM_GRID_SIZE = 64
SPECTRUM_LEVELS = 16
SPECTRUM_TILE_SIZE = 16
SPECTRUM_UPSCALE = 8

CLASSIFICATION_COLOURS = {
    "CRAB": (255, 0, 0, 255),
    "MICROBE": (0, 255, 0, 255),
    "INTERMEDIATE": (255, 255, 0, 255),
    "NO_VISIBLE_PATTERN": (0, 0, 255, 255),
}






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

print("Family:", cfg.model.family)
print("Channels:", cfg.model.channels)
print("Fire rate:", cfg.model.fire_rate)
print("Downsample:", downsample)
print("Iterations:", iterations)
print("Training t:", run_t)






model_result = build_model(
    cfg.model,
    key=jr.PRNGKey(0)
)

if isinstance(model_result, tuple):

    model = model_result[0]

else:

    model = model_result


model = model.load(
    MODEL_PATH
)


print("\nMODEL LOADED:")
print(
    model.get_config()
)


N_CHANNELS = int(
    cfg.model.channels
)






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
        "Crab and microbe shapes differ."
    )


C, H, W = crab.shape






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
print("SEED PIXELS")
print("======================================")

print(
    "Crab:",
    np.round(
        np.asarray(crab_pixel) * 255
    ).astype(int)
)

print(
    "Microbe:",
    np.round(
        np.asarray(microbe_pixel) * 255
    ).astype(int)
)






def complementary_pixel(pixel):

    result = pixel

    result = result.at[
        :3
    ].set(
        1.0 - pixel[:3]
    )

    return result


complement_crab = complementary_pixel(
    crab_pixel
)

complement_microbe = complementary_pixel(
    microbe_pixel
)






transparent_pixel = jnp.array(
    [
        0.0,
        0.0,
        0.0,
        0.0
    ],
    dtype=jnp.float32
)


black_pixel = jnp.array(
    [
        0.0,
        0.0,
        0.0,
        1.0
    ],
    dtype=jnp.float32
)






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
    +
    pad_top
    +
    pad_bottom
)

canvas_W = (
    W
    +
    pad_left
    +
    pad_right
)






training_x = (
    patch_x
    +
    pad_left
)

training_y = (
    patch_y
    +
    pad_top
)


print("\n======================================")
print("CANVAS")
print("======================================")

print(
    "Full matrix:",
    canvas_H,
    "x",
    canvas_W
)

print(
    "Training seed position:",
    (training_x, training_y)
)















POSITIONS = [

    (
        "center",
        training_x,
        training_y
    ),

    (
        "top_left",
        0,
        0
    ),

    (
        "top_center",
        training_x,
        0
    ),

    (
        "top_right",
        canvas_W - 1,
        0
    ),

    (
        "center_left",
        0,
        training_y
    ),

    (
        "center_right",
        canvas_W - 1,
        training_y
    ),

    (
        "bottom_left",
        0,
        canvas_H - 1
    ),

    (
        "bottom_center",
        training_x,
        canvas_H - 1
    ),

    (
        "bottom_right",
        canvas_W - 1,
        canvas_H - 1
    ),
]


print("\nStart positions:")

for name, px, py in POSITIONS:

    print(
        name,
        "->",
        (px, py)
    )








full_padding = (
    (0, 0),
    (pad_top, pad_bottom),
    (pad_left, pad_right)
)


crab_target_full = jnp.pad(
    crab,
    full_padding,
    mode="constant"
)

microbe_target_full = jnp.pad(
    microbe,
    full_padding,
    mode="constant"
)






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











def shift_no_wrap(
    array,
    dy,
    dx
):

    arr = np.asarray(
        array
    )

    result = np.zeros_like(
        arr
    )

    height = arr.shape[-2]
    width = arr.shape[-1]


    dst_y0 = max(
        0,
        dy
    )

    dst_y1 = min(
        height,
        height + dy
    )


    src_y0 = max(
        0,
        -dy
    )

    src_y1 = (
        src_y0
        +
        (
            dst_y1
            -
            dst_y0
        )
    )


    dst_x0 = max(
        0,
        dx
    )

    dst_x1 = min(
        width,
        width + dx
    )


    src_x0 = max(
        0,
        -dx
    )

    src_x1 = (
        src_x0
        +
        (
            dst_x1
            -
            dst_x0
        )
    )


    if (
        dst_y1 > dst_y0
        and
        dst_x1 > dst_x0
    ):

        result[
            ...,
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = arr[
            ...,
            src_y0:src_y1,
            src_x0:src_x1
        ]


    return result









def align_for_analysis(
    state,
    seed_x,
    seed_y
):

    dx = (
        training_x
        -
        seed_x
    )

    dy = (
        training_y
        -
        seed_y
    )


    aligned = shift_no_wrap(
        jax.device_get(
            state
        ),
        dy,
        dx
    )


    return jnp.asarray(
        aligned
    )






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






def run_pixel(
    pixel,
    seed_x,
    seed_y
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
        seed_y,
        seed_x
    ].set(
        pixel
    )


    key = jr.PRNGKey(
        SEED
    )


    for _ in range(
        STEPS
    ):

        key, subkey = jr.split(
            key
        )

        state = nca_step(
            model,
            state,
            subkey
        )


    return state












def classify_full_state(
    raw_state,
    seed_x,
    seed_y
):

    aligned = align_for_analysis(
        raw_state,
        seed_x,
        seed_y
    )


    visible = aligned[
        :data_channels
    ]






    mse_crab = float(
        jnp.mean(
            (
                visible
                -
                crab_target_full[
                    :data_channels
                ]
            ) ** 2
        )
    )


    mse_microbe = float(
        jnp.mean(
            (
                visible
                -
                microbe_target_full[
                    :data_channels
                ]
            ) ** 2
        )
    )






    final_rgb = visible[
        :3
    ]

    crab_rgb = crab_target_full[
        :3
    ]

    microbe_rgb = microbe_target_full[
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


    if data_channels >= 4:

        active_mask = (
            visible[3]
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

            "aligned_state":
                aligned,
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


    crab_fraction = float(
        jnp.sum(
            crab_like
        )
        /
        active_pixels
    )


    microbe_fraction = float(
        jnp.sum(
            microbe_like
        )
        /
        active_pixels
    )


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

        "aligned_state":
            aligned,
    }




















def compare_to_center(
    raw_state,
    aligned_state,
    center_state
):

    raw_visible_mse = float(
        jnp.mean(
            (
                raw_state[
                    :data_channels
                ]
                -
                center_state[
                    :data_channels
                ]
            ) ** 2
        )
    )


    aligned_visible_mse = float(
        jnp.mean(
            (
                aligned_state[
                    :data_channels
                ]
                -
                center_state[
                    :data_channels
                ]
            ) ** 2
        )
    )


    raw_all_channels_mse = float(
        jnp.mean(
            (
                raw_state
                -
                center_state
            ) ** 2
        )
    )


    aligned_all_channels_mse = float(
        jnp.mean(
            (
                aligned_state
                -
                center_state
            ) ** 2
        )
    )


    return {
        "raw_visible_mse_vs_center":
            raw_visible_mse,

        "aligned_visible_mse_vs_center":
            aligned_visible_mse,

        "raw_all_channels_mse_vs_center":
            raw_all_channels_mse,

        "aligned_all_channels_mse_vs_center":
            aligned_all_channels_mse,
    }






reference_states = {

    "normal": {},

    "complementary": {},

    "black": None,

    "transparent": None,
}


reference_classes = {

    "normal": {},

    "complementary": {},

    "black": None,

    "transparent": None,
}



all_results = []






def run_colour_scan_at_position(
    scan_name,
    start_pixel,
    end_pixel,
    position_name,
    seed_x,
    seed_y
):

    output_dir = (
        OUTPUT_ROOT
        /
        position_name
        /
        scan_name
    )

    raw_dir = (
        output_dir
        /
        "raw_full_matrix"
    )

    aligned_dir = (
        output_dir
        /
        "aligned_analysis_only"
    )


    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    aligned_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    rows = []
    frames = []

    previous_class = None


    print(
        f"\n--- {position_name} / {scan_name} ---"
    )


    for index, alpha in enumerate(
        ALPHAS
    ):

        pixel = (
            (1.0 - alpha)
            *
            start_pixel
            +
            alpha
            *
            end_pixel
        )

        raw_state = run_pixel(
            pixel,
            seed_x,
            seed_y
        )

        analysis = classify_full_state(
            raw_state,
            seed_x,
            seed_y
        )

        aligned_state = analysis[
            "aligned_state"
        ]

        classification = analysis[
            "classification"
        ]

        if position_name == "center":

            reference_states[
                scan_name
            ][index] = raw_state

            reference_classes[
                scan_name
            ][index] = classification

            comparison = {
                "raw_visible_mse_vs_center":
                    0.0,

                "aligned_visible_mse_vs_center":
                    0.0,

                "raw_all_channels_mse_vs_center":
                    0.0,

                "aligned_all_channels_mse_vs_center":
                    0.0,
            }


            class_changed = False

        else:

            center_state = (
                reference_states[
                    scan_name
                ][index]
            )

            comparison = compare_to_center(
                raw_state,
                aligned_state,
                center_state
            )

            center_class = (
                reference_classes[
                    scan_name
                ][index]
            )

            class_changed = (
                classification
                !=
                center_class
            )

        raw_image = state_to_rgba(
            raw_state
        )

        raw_path = (
            raw_dir
            /
            f"alpha_{index:03d}_"
            f"{classification.lower()}.png"
        )

        Image.fromarray(
            raw_image,
            mode="RGBA"
        ).save(
            raw_path
        )

        aligned_image = state_to_rgba(
            aligned_state
        )

        aligned_path = (
            aligned_dir
            /
            f"alpha_{index:03d}_"
            f"{classification.lower()}.png"
        )

        Image.fromarray(
            aligned_image,
            mode="RGBA"
        ).save(
            aligned_path
        )

        frame = Image.fromarray(
            raw_image,
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
                f"{position_name} | "
                f"{scan_name} | "
                f"alpha={alpha:.2f}"
            )
        )

        draw.text(
            (5, 20),
            classification
        )

        frames.append(
            frame
        )

        pixel_np = np.asarray(
            pixel
        )

        row = {

            "scan":
                scan_name,

            "position":
                position_name,

            "seed_x":
                seed_x,

            "seed_y":
                seed_y,

            "alpha":
                float(alpha),

            "R":
                float(pixel_np[0]),

            "G":
                float(pixel_np[1]),

            "B":
                float(pixel_np[2]),

            "A":
                float(pixel_np[3]),

            "classification":
                classification,

            "class_changed_vs_center":
                class_changed,

            "mse_crab":
                analysis[
                    "mse_crab"
                ],

            "mse_microbe":
                analysis[
                    "mse_microbe"
                ],

            "crab_fraction":
                analysis[
                    "crab_fraction"
                ],

            "microbe_fraction":
                analysis[
                    "microbe_fraction"
                ],

            "active_pixels":
                analysis[
                    "active_pixels"
                ],

            **comparison,
        }

        rows.append(
            row
        )

        all_results.append(
            row
        )

        if (
            index % 10 == 0
            or
            classification
            !=
            previous_class
        ):

            print(
                f"alpha={alpha:4.2f} | "
                f"{classification:18s} | "
                f"aligned MSE="
                f"{comparison['aligned_visible_mse_vs_center']:.6f}"
            )


        previous_class = classification

    csv_path = (
        output_dir
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
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    gif_path = (
        output_dir
        /
        "raw_full_matrix_scan.gif"
    )


    frames[0].save(
        gif_path,
        save_all=True,
        append_images=
            frames[1:],
        duration=100,
        loop=0
    )

def run_special_at_position(
    scan_name,
    pixel,
    position_name,
    seed_x,
    seed_y
):

    output_dir = (
        OUTPUT_ROOT
        /
        position_name
        /
        scan_name
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    raw_state = run_pixel(
        pixel,
        seed_x,
        seed_y
    )


    analysis = classify_full_state(
        raw_state,
        seed_x,
        seed_y
    )


    aligned_state = analysis[
        "aligned_state"
    ]


    classification = analysis[
        "classification"
    ]


    if position_name == "center":

        reference_states[
            scan_name
        ] = raw_state


        reference_classes[
            scan_name
        ] = classification


        comparison = {
            "raw_visible_mse_vs_center":
                0.0,

            "aligned_visible_mse_vs_center":
                0.0,

            "raw_all_channels_mse_vs_center":
                0.0,

            "aligned_all_channels_mse_vs_center":
                0.0,
        }


        class_changed = False


    else:

        comparison = compare_to_center(
            raw_state,
            aligned_state,
            reference_states[
                scan_name
            ]
        )


        class_changed = (
            classification
            !=
            reference_classes[
                scan_name
            ]
        )



    Image.fromarray(
        state_to_rgba(
            raw_state
        ),
        mode="RGBA"
    ).save(
        output_dir
        /
        "raw_full_matrix.png"
    )



    Image.fromarray(
        state_to_rgba(
            aligned_state
        ),
        mode="RGBA"
    ).save(
        output_dir
        /
        "aligned_analysis_only.png"
    )


    pixel_np = np.asarray(
        pixel
    )


    row = {

        "scan":
            scan_name,

        "position":
            position_name,

        "seed_x":
            seed_x,

        "seed_y":
            seed_y,

        "alpha":
            "",

        "R":
            float(pixel_np[0]),

        "G":
            float(pixel_np[1]),

        "B":
            float(pixel_np[2]),

        "A":
            float(pixel_np[3]),

        "classification":
            classification,

        "class_changed_vs_center":
            class_changed,

        "mse_crab":
            analysis[
                "mse_crab"
            ],

        "mse_microbe":
            analysis[
                "mse_microbe"
            ],

        "crab_fraction":
            analysis[
                "crab_fraction"
            ],

        "microbe_fraction":
            analysis[
                "microbe_fraction"
            ],

        "active_pixels":
            analysis[
                "active_pixels"
            ],

        **comparison,
    }

    all_results.append(
        row
    )

    print(
        f"{position_name:15s} | "
        f"{scan_name:12s} | "
        f"{classification:18s} | "
        f"aligned MSE="
        f"{comparison['aligned_visible_mse_vs_center']:.6f}"
    )

def classification_to_rgba(classification):

    return CLASSIFICATION_COLOURS.get(
        classification,
        (255, 0, 255, 255)
    )


def make_rgb_seed_pixel(r, g, b):

    # Build a seed with exactly data_channels entries so this also stays
    # valid if the visible data representation contains more than RGBA.
    pixel = jnp.zeros(
        (data_channels,),
        dtype=jnp.float32
    )

    if data_channels >= 1:
        pixel = pixel.at[0].set(r)

    if data_channels >= 2:
        pixel = pixel.at[1].set(g)

    if data_channels >= 3:
        pixel = pixel.at[2].set(b)

    if data_channels >= 4:
        pixel = pixel.at[3].set(1.0)

    return pixel


def run_rgb_spectrum_scan(
    position_name="center",
    seed_x=training_x,
    seed_y=training_y
):

    if SPECTRUM_GRID_SIZE != 64:
        raise ValueError(
            "SPECTRUM_GRID_SIZE must be 64 for the requested 64x64 matrix."
        )

    if SPECTRUM_LEVELS != 16:
        raise ValueError(
            "SPECTRUM_LEVELS must be 16 so that 16^3 = 4096 colours."
        )

    if SPECTRUM_TILE_SIZE != SPECTRUM_LEVELS:
        raise ValueError(
            "SPECTRUM_TILE_SIZE must equal SPECTRUM_LEVELS."
        )

    if SPECTRUM_GRID_SIZE != 4 * SPECTRUM_TILE_SIZE:
        raise ValueError(
            "The 64x64 layout requires four 16x16 tiles per axis."
        )

    if data_channels < 3:
        raise RuntimeError(
            "RGB spectrum scan requires at least three data channels."
        )

    output_dir = (
        OUTPUT_ROOT
        /
        "rgb_spectrum_scan"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_matrix = np.zeros(
        (
            SPECTRUM_GRID_SIZE,
            SPECTRUM_GRID_SIZE,
            4
        ),
        dtype=np.uint8
    )

    classification_matrix = np.zeros(
        (
            SPECTRUM_GRID_SIZE,
            SPECTRUM_GRID_SIZE,
            4
        ),
        dtype=np.uint8
    )

    rows = []

    print("\n")
    print("======================================")
    print("RGB SPECTRUM SCAN")
    print("======================================")
    print(
        "Seed position:",
        position_name,
        (seed_x, seed_y)
    )
    print(
        "4096 unique colours = 16 red levels x "
        "16 green levels x 16 blue levels"
    )
    print(
        "Layout: 4x4 blue-level tiles; inside each tile "
        "x=red and y=green"
    )

    total_runs = SPECTRUM_GRID_SIZE * SPECTRUM_GRID_SIZE
    completed = 0

    for y in range(SPECTRUM_GRID_SIZE):

        for x in range(SPECTRUM_GRID_SIZE):

            tile_col = x // SPECTRUM_TILE_SIZE
            tile_row = y // SPECTRUM_TILE_SIZE

            blue_index = (
                tile_row * 4
                +
                tile_col
            )

            red_index = x % SPECTRUM_LEVELS
            green_index = y % SPECTRUM_LEVELS

            r = red_index / (SPECTRUM_LEVELS - 1)
            g = green_index / (SPECTRUM_LEVELS - 1)
            b = blue_index / (SPECTRUM_LEVELS - 1)

            pixel = make_rgb_seed_pixel(
                r,
                g,
                b
            )

            raw_state = run_pixel(
                pixel,
                seed_x,
                seed_y
            )

            analysis = classify_full_state(
                raw_state,
                seed_x,
                seed_y
            )

            classification = analysis[
                "classification"
            ]

            input_matrix[
                y,
                x
            ] = (
                int(round(r * 255)),
                int(round(g * 255)),
                int(round(b * 255)),
                255
            )

            classification_matrix[
                y,
                x
            ] = classification_to_rgba(
                classification
            )

            row = {
                "x": x,
                "y": y,
                "position": position_name,
                "seed_x": seed_x,
                "seed_y": seed_y,
                "tile_row": tile_row,
                "tile_col": tile_col,
                "red_index": red_index,
                "green_index": green_index,
                "blue_index": blue_index,
                "R": float(r),
                "G": float(g),
                "B": float(b),
                "A": 1.0 if data_channels >= 4 else "",
                "classification": classification,
                "mse_crab": analysis[
                    "mse_crab"
                ],
                "mse_microbe": analysis[
                    "mse_microbe"
                ],
                "crab_fraction": analysis[
                    "crab_fraction"
                ],
                "microbe_fraction": analysis[
                    "microbe_fraction"
                ],
                "active_pixels": analysis[
                    "active_pixels"
                ],
            }

            rows.append(
                row
            )

            completed += 1

            if (
                completed % 64 == 0
                or
                completed == total_runs
            ):

                print(
                    f"RGB scan progress: "
                    f"{completed}/{total_runs}"
                )

    csv_path = (
        output_dir
        /
        "rgb_spectrum_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    input_64_path = (
        output_dir
        /
        "rgb_spectrum_input_64x64.png"
    )

    class_64_path = (
        output_dir
        /
        "rgb_spectrum_classification_64x64.png"
    )

    Image.fromarray(
        input_matrix,
        mode="RGBA"
    ).save(
        input_64_path
    )

    Image.fromarray(
        classification_matrix,
        mode="RGBA"
    ).save(
        class_64_path
    )

    input_large_path = (
        output_dir
        /
        "rgb_spectrum_input_large.png"
    )

    class_large_path = (
        output_dir
        /
        "rgb_spectrum_classification_large.png"
    )

    Image.fromarray(
        input_matrix,
        mode="RGBA"
    ).resize(
        (
            SPECTRUM_GRID_SIZE * SPECTRUM_UPSCALE,
            SPECTRUM_GRID_SIZE * SPECTRUM_UPSCALE
        ),
        resample=Image.Resampling.NEAREST
    ).save(
        input_large_path
    )

    Image.fromarray(
        classification_matrix,
        mode="RGBA"
    ).resize(
        (
            SPECTRUM_GRID_SIZE * SPECTRUM_UPSCALE,
            SPECTRUM_GRID_SIZE * SPECTRUM_UPSCALE
        ),
        resample=Image.Resampling.NEAREST
    ).save(
        class_large_path
    )

    crab_count = sum(
        row["classification"] == "CRAB"
        for row in rows
    )

    microbe_count = sum(
        row["classification"] == "MICROBE"
        for row in rows
    )

    intermediate_count = sum(
        row["classification"] == "INTERMEDIATE"
        for row in rows
    )

    no_pattern_count = sum(
        row["classification"] == "NO_VISIBLE_PATTERN"
        for row in rows
    )

    summary = {
        "position": position_name,
        "seed_x": seed_x,
        "seed_y": seed_y,
        "total_runs": len(rows),
        "crab_count": crab_count,
        "microbe_count": microbe_count,
        "intermediate_count": intermediate_count,
        "no_visible_pattern_count": no_pattern_count,
        "csv_path": str(csv_path.resolve()),
        "input_64_path": str(input_64_path.resolve()),
        "class_64_path": str(class_64_path.resolve()),
        "input_large_path": str(input_large_path.resolve()),
        "class_large_path": str(class_large_path.resolve()),
    }

    summary_path = (
        output_dir
        /
        "rgb_spectrum_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        file.write("======================================\n")
        file.write("RGB SPECTRUM SCAN SUMMARY\n")
        file.write("======================================\n\n")

        file.write(f"Position: {position_name}\n")
        file.write(f"Seed position: ({seed_x}, {seed_y})\n")
        file.write(f"Total tested colours: {len(rows)}\n")
        file.write("RGB levels per channel: 16\n")
        file.write("Alpha: 1.0 for RGBA data\n\n")

        file.write("CLASSIFICATION COLOURS\n")
        file.write("--------------------------------------\n")
        file.write("CRAB: red (255, 0, 0)\n")
        file.write("MICROBE: green (0, 255, 0)\n")
        file.write("INTERMEDIATE: yellow (255, 255, 0)\n")
        file.write("NO_VISIBLE_PATTERN: blue (0, 0, 255)\n\n")

        file.write("CLASSIFICATION COUNTS\n")
        file.write("--------------------------------------\n")
        file.write(f"CRAB: {crab_count}\n")
        file.write(f"MICROBE: {microbe_count}\n")
        file.write(f"INTERMEDIATE: {intermediate_count}\n")
        file.write(f"NO_VISIBLE_PATTERN: {no_pattern_count}\n\n")

        file.write("MATRIX LAYOUT\n")
        file.write("--------------------------------------\n")
        file.write(
            "The 64x64 matrix is divided into a 4x4 grid of 16x16 tiles.\n"
        )
        file.write(
            "Each tile represents one of the 16 blue levels.\n"
        )
        file.write(
            "Inside every tile, x represents red and y represents green.\n"
        )
        file.write(
            "Therefore every one of the 16^3 = 4096 quantised RGB colours "
            "is tested exactly once.\n\n"
        )

        file.write("SAVED FILES\n")
        file.write("--------------------------------------\n")
        file.write(f"CSV: {csv_path.resolve()}\n")
        file.write(f"Input 64x64: {input_64_path.resolve()}\n")
        file.write(f"Classification 64x64: {class_64_path.resolve()}\n")
        file.write(f"Input large: {input_large_path.resolve()}\n")
        file.write(f"Classification large: {class_large_path.resolve()}\n")

    summary[
        "summary_path"
    ] = str(
        summary_path.resolve()
    )

    print("\nRGB spectrum scan completed.")
    print(
        "CSV:",
        csv_path.resolve()
    )
    print(
        "Input image:",
        input_large_path.resolve()
    )
    print(
        "Classification image:",
        class_large_path.resolve()
    )
    print(
        "Summary:",
        summary_path.resolve()
    )

    return rows, summary


for (
    position_name,
    seed_x,
    seed_y
) in POSITIONS:

    print("\n")
    print("======================================")
    print(
        "POSITION:",
        position_name,
        (seed_x, seed_y)
    )
    print("======================================")

    run_colour_scan_at_position(
        "normal",
        crab_pixel,
        microbe_pixel,
        position_name,
        seed_x,
        seed_y
    )

    run_colour_scan_at_position(
        "complementary",
        complement_crab,
        complement_microbe,
        position_name,
        seed_x,
        seed_y
    )

    run_special_at_position(
        "black",
        black_pixel,
        position_name,
        seed_x,
        seed_y
    )

    run_special_at_position(
        "transparent",
        transparent_pixel,
        position_name,
        seed_x,
        seed_y
    )

rgb_spectrum_rows, rgb_spectrum_summary = run_rgb_spectrum_scan(
    position_name="center",
    seed_x=training_x,
    seed_y=training_y
)




global_csv = (
    OUTPUT_ROOT
    /
    "all_position_results.csv"
)

with open(
    global_csv,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=
            all_results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )

summary_rows = []


for (
    position_name,
    seed_x,
    seed_y
) in POSITIONS:

    for scan_name in [
        "normal",
        "complementary",
        "black",
        "transparent",
    ]:

        selected = [

            row

            for row
            in all_results

            if (
                row["position"]
                ==
                position_name

                and

                row["scan"]
                ==
                scan_name
            )
        ]


        if not selected:
            continue


        aligned_values = [
            float(
                row[
                    "aligned_visible_mse_vs_center"
                ]
            )
            for row
            in selected
        ]


        raw_values = [
            float(
                row[
                    "raw_visible_mse_vs_center"
                ]
            )
            for row
            in selected
        ]


        class_changes = sum(
            bool(
                row[
                    "class_changed_vs_center"
                ]
            )
            for row
            in selected
        )


        crab_count = sum(
            row["classification"]
            ==
            "CRAB"

            for row
            in selected
        )


        microbe_count = sum(
            row["classification"]
            ==
            "MICROBE"

            for row
            in selected
        )


        intermediate_count = sum(
            row["classification"]
            ==
            "INTERMEDIATE"

            for row
            in selected
        )


        no_pattern_count = sum(
            row["classification"]
            ==
            "NO_VISIBLE_PATTERN"

            for row
            in selected
        )


        summary_rows.append(
            {

                "position":
                    position_name,

                "seed_x":
                    seed_x,

                "seed_y":
                    seed_y,

                "scan":
                    scan_name,

                "runs":
                    len(selected),

                "classification_changes_vs_center":
                    class_changes,

                "mean_raw_visible_mse_vs_center":
                    float(
                        np.mean(
                            raw_values
                        )
                    ),

                "mean_aligned_visible_mse_vs_center":
                    float(
                        np.mean(
                            aligned_values
                        )
                    ),

                "max_aligned_visible_mse_vs_center":
                    float(
                        np.max(
                            aligned_values
                        )
                    ),

                "crab_count":
                    crab_count,

                "intermediate_count":
                    intermediate_count,

                "microbe_count":
                    microbe_count,

                "no_visible_pattern_count":
                    no_pattern_count,
            }
        )


summary_csv = (
    OUTPUT_ROOT
    /
    "position_summary.csv"
)


with open(
    summary_csv,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=
            summary_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )

print("\n")
print("======================================")
print("POSITION EFFECT SUMMARY")
print("======================================")


for row in summary_rows:

    if row["position"] == "center":
        continue


    print(
        f"{row['position']:15s} | "
        f"{row['scan']:13s} | "
        f"class changes="
        f"{row['classification_changes_vs_center']:3d} | "
        f"mean raw MSE="
        f"{row['mean_raw_visible_mse_vs_center']:.6f} | "
        f"mean aligned MSE="
        f"{row['mean_aligned_visible_mse_vs_center']:.6f}"
    )


print("\n======================================")
print("OUTPUT")
print("======================================")

print(
    "Full result CSV:"
)

print(
    global_csv.resolve()
)

print(
    "\nPosition summary:"
)

print(
    summary_csv.resolve()
)

print(
    "\nAll full-size images:"
)

print(
    OUTPUT_ROOT.resolve()
)
report_path = OUTPUT_ROOT / "report.txt"

with open(report_path, "w") as f:

    f.write("======================================\n")
    f.write("SPATIAL POSITION SCAN REPORT\n")
    f.write("======================================\n\n")

    # Configuration
    f.write("CONFIGURATION\n")
    f.write("--------------------------------------\n")
    f.write(f"Model family: {cfg.model.family}\n")
    f.write(f"Channels: {cfg.model.channels}\n")
    f.write(f"Fire rate: {cfg.model.fire_rate}\n")
    f.write(f"Downsample: {downsample}\n")
    f.write(f"Training iterations: {iterations}\n")
    f.write(f"Training t: {run_t}\n")
    f.write(f"Inference steps: {STEPS}\n")
    f.write(f"Canvas size: {canvas_H} x {canvas_W}\n")
    f.write(f"Training seed position: ({training_x}, {training_y})\n\n")

    # Seed pixels
    f.write("SEED PIXELS\n")
    f.write("--------------------------------------\n")
    f.write(f"Crab pixel: {np.asarray(crab_pixel)}\n")
    f.write(f"Microbe pixel: {np.asarray(microbe_pixel)}\n")
    f.write(f"Complement crab: {np.asarray(complement_crab)}\n")
    f.write(f"Complement microbe: {np.asarray(complement_microbe)}\n")
    f.write(f"Black pixel: {np.asarray(black_pixel)}\n")
    f.write(f"Transparent pixel: {np.asarray(transparent_pixel)}\n\n")

    # Positions
    f.write("POSITIONS\n")
    f.write("--------------------------------------\n")
    for name, x, y in POSITIONS:
        f.write(f"{name}: ({x}, {y})\n")

    # All individual results
    f.write("\n\nALL RESULTS\n")
    f.write("======================================\n")

    for row in all_results:
        f.write("\n")
        for key, value in row.items():
            f.write(f"{key}: {value}\n")

    # Summary
    f.write("\n\nPOSITION SUMMARY\n")
    f.write("======================================\n")

    for row in summary_rows:
        f.write("\n")
        for key, value in row.items():
            f.write(f"{key}: {value}\n")

    # RGB spectrum scan summary
    f.write("\n\nRGB SPECTRUM SCAN SUMMARY\n")
    f.write("======================================\n")
    for key, value in rgb_spectrum_summary.items():
        f.write(f"{key}: {value}\n")

    # Every individual RGB spectrum result
    f.write("\n\nRGB SPECTRUM ALL RESULTS\n")
    f.write("======================================\n")
    for row in rgb_spectrum_rows:
        f.write("\n")
        for key, value in row.items():
            f.write(f"{key}: {value}\n")


print("Text report:")
print(report_path.resolve())
print("\nRGB spectrum CSV:")
print(rgb_spectrum_summary["csv_path"])

print("\nRGB spectrum input matrix:")
print(rgb_spectrum_summary["input_64_path"])

print("\nRGB spectrum classification matrix:")
print(rgb_spectrum_summary["class_64_path"])

print("\nRGB spectrum summary:")
print(rgb_spectrum_summary["summary_path"])

print("\nDONE.")
