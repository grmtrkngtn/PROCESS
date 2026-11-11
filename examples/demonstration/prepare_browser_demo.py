#!/usr/bin/env python3
"""Run locally, with PROCESS installed, to prepare browser-only tutorial assets.

Run from anywhere with: python examples/demonstration/prepare_browser_demo.py
This executes PROCESS / loads valid existing caches ONCE, then saves plots as PNG.
No GitHub API requests. Do not run on attendee machines or in WebAssembly.
"""
import itertools
import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE.parents[1]))  # use the checked-out PROCESS source
sys.path.insert(0, str(HERE))
import process_response_popcon as explorer
from process.main import SingleRun
from process.core.io.mfile import MFile
from process.core.io.plot.summary.api import create_thickness_builds
from process.core.io.plot.summary.geometry.poloidal import poloidal_cross_section
from process.core.io.plot.summary.geometry.toroidal import toroidal_cross_section
from process.core.io.plot.summary.plasma.overview import plot_main_plasma_information

public = HERE / "public"
(public / "plasma").mkdir(parents=True, exist_ok=True)
(public / "impurity").mkdir(parents=True, exist_ok=True)

PLASMA_INPUT_FILE = "data/spherical_tokamak_eval.IN.DAT"

DATA_DIR = Path("data")

HEATING_CONTROL = {

        "variable": "current_drive.p_hcd_primary_extra_heat_mw",

        "range": (0.0, 80.0),

        "n": 5,

        "short_label": "Extra heating",

        "display_scale": 1.0,

        "unit": "MW",

        "format": ".1f",

        "include_reference": True,

    }

FUEL_CONTROL = {

        "variable": "physics.molflow_plasma_fuelling",

        "range": (4.2e21, 5.2e21),

        "n": 5,

        "short_label": "Fuel supply",

        "display_scale": 1.0e21,

        "unit": "×10²¹ s⁻¹",

        "format": ".3f",

        "include_reference": True,

    }

PLASMA_CONTROLS = [HEATING_CONTROL, FUEL_CONTROL]

PLASMA_MAP_POINTS = 10

REBUILD_PLASMA_SOLUTIONS = False

REBUILD_PLASMA_MAPS = False

PRECOMPUTE_PLASMA_MAPS = False

EXPLORER_WORKERS = 1

plasma_response = explorer.build_response(

        input_file=PLASMA_INPUT_FILE,

        controls=PLASMA_CONTROLS,

        performance={

            "variable": "physics.p_fusion_total_mw",

            "label": "Fusion power [MW]",

        },

        equalities=[

            {

                "constraint": 2,

                "label": "C2 = 0  global power balance",

                "color": "crimson",

            },

            {

                "constraint": 93,

                "label": "C93 = 0  fuel-ion balance",

                "color": "#9c27b0",

            },

        ],

        limits=[

            {

                "constraint": 5,

                "label": "C5 density upper limit",

                "color": "#2ca02c",

            },

            {

                "constraint": 24,

                "label": "C24 beta upper limit",

                "color": "#ff7f0e",

            },

        ],

        title="PROCESS plasma solution space",

        map_points=PLASMA_MAP_POINTS,

        cache_dir=DATA_DIR,

        cache_name="demo_plasma_response",

        rebuild_solutions=REBUILD_PLASMA_SOLUTIONS,

        rebuild_maps=REBUILD_PLASMA_MAPS,

        precompute_maps=PRECOMPUTE_PLASMA_MAPS,

        max_workers=EXPLORER_WORKERS,

    )

IMPURITY_INPUT_FILE = PLASMA_INPUT_FILE

IMPURITY_CONTROLS = [

        {

            "variable": "current_drive.p_hcd_primary_extra_heat_mw",

            "range": (0.0, 50.0),

            "n": 5,

            "short_label": "Extra heating",

            "display_scale": 1.0,

            "unit": "MW",

            "format": ".1f",

            "include_reference": True,

        },

        {

            "variable": "impurity_radiation.f_nd_impurity_electrons(9)",

            "values": [0.0, 2.5e-4, 5.0e-4, 7.5e-4],

            "short_label": "Argon seeding",

            "display_scale": 1.0e-2,

            "unit": "% nₑ",

            "format": ".3f",

            "include_reference": True,

        },

        {

            "variable": "physics.molflow_plasma_fuelling",

            "range": (4.2e21, 5.2e21),

            "n": 5,

            "short_label": "Fuel supply",

            "display_scale": 1.0e21,

            "unit": "×10²¹ s⁻¹",

            "format": ".3f",

            "include_reference": True,

        },

    ]

REBUILD_IMPURITY_SOLUTIONS = False

REBUILD_IMPURITY_MAPS = False

PRECOMPUTE_IMPURITY_MAPS = False

IMPURITY_WORKERS = 1

impurity_response = explorer.build_response(

        input_file=IMPURITY_INPUT_FILE,

        controls=IMPURITY_CONTROLS,

        performance={

            "variable": "physics.p_fusion_total_mw",

            "label": "Fusion power [MW]",

        },

        equalities=[

            {

                "constraint": 2,

                "label": "C2 = 0  global power balance",

                "color": "crimson",

            },

            {

                "constraint": 93,

                "label": "C93 = 0  fuel-ion balance",

                "color": "#9c27b0",

            },

        ],

        limits=[

            {

                "constraint": 68,

                "label": "C68 divertor protection",

                "color": "#ff7f0e",

            },

            {

                "constraint": 5,

                "label": "C5 density limit",

                "color": "#00a6d6",

            },

            {

                "constraint": 24,

                "label": "C24 beta limit",

                "color": "#2ca02c",

            },

        ],

        title="Heating, fuelling and impurity-seeding trade-off",

        map_points=20,

        te_padding_min_kev=0.8,

        ne_padding_min=0.04e20,

        cache_dir=DATA_DIR,

        cache_name="demo_impurity_response",

        rebuild_solutions=REBUILD_IMPURITY_SOLUTIONS,

        rebuild_maps=REBUILD_IMPURITY_MAPS,

        precompute_maps=PRECOMPUTE_IMPURITY_MAPS,

        max_workers=IMPURITY_WORKERS,

    )

def store_response(tag, response, prefix, show_solution_cloud=True):
    axes = response["control_axes"]
    folder = public / tag
    total = 1
    for axis in axes: total *= len(axis)
    for counter, indices in enumerate(itertools.product(*(range(len(axis)) for axis in axes)), 1):
        fig, ax = explorer.plot_response(
            response, control_indices=indices,
            show_solution_cloud=show_solution_cloud,
        )
        name = prefix + "_".join(map(str,indices)) + ".png"
        fig.savefig(folder / name, dpi=115, bbox_inches="tight")
        plt.close(fig)
        if counter % 10 == 0 or counter == total:
            print(f"{tag}: {counter}/{total} plots")
    return {
        "control_axes": [[float(v) for v in axis] for axis in axes],
        "reference_indices": [int(x) for x in response["reference_indices"]],
    }

manifest = {
    "plasma": store_response("plasma", plasma_response, "p_"),
    "impurity": store_response("impurity", impurity_response, "i_", show_solution_cloud=False),
}
(public / "manifest.json").write_text(json.dumps(manifest, indent=2))

# Make overview, poloidal and toroidal figures from one evaluation.
single_run = SingleRun("data/large_tokamak_eval_IN.DAT", update_obsolete=True)
single_run.run()
mfile = MFile(single_run.mfile_path)
fig, ax = plt.subplots(figsize=(12,10))
plot_main_plasma_information(axis=ax, mfile=mfile, scan=-1, colour_scheme=1, fig=fig)
ax.set_aspect("equal", adjustable="box")
fig.savefig(public / "overview.png", dpi=135, bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(figsize=(4,6))
poloidal_cross_section(axis=ax,mfile=mfile,scan=-1,demo_ranges=False,
                       radial_build=create_thickness_builds(mfile,-1),colour_scheme=1)
ax.set_aspect("equal")
fig.savefig(public / "poloidal.png",dpi=135,bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(figsize=(4,4))
toroidal_cross_section(axis=ax,mfile=mfile,scan=-1,demo_ranges=False,colour_scheme=1)
ax.set_aspect("equal")
fig.savefig(public / "toroidal.png",dpi=135,bbox_inches="tight")
plt.close(fig)
print("All browser demo images and manifest saved to",public)
