import marimo

__generated_with = "0.25.2"

app = marimo.App(auto_download=["html"])

@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json
    from pathlib import Path

    def load_manifest(section):
        import sys
        path = mo.notebook_location() / "public" / "manifest.json"
        if sys.platform == "emscripten":
            from pyodide.http import open_url
            manifest = json.load(open_url(str(path)))
        else:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing generated assets: {path}. Run prepare_browser_demo.py."
                )
            manifest = json.loads(path.read_text())
        return manifest[section]
    def show_asset(relative_path, alt):
        import sys
        if sys.platform == "emscripten":
            path = mo.notebook_location() / "public" / relative_path
            return mo.md(f"![{alt}]({path})")
        import base64
        path = mo.notebook_location() / "public" / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated image: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return mo.md(f"![{alt}](data:image/png;base64,{encoded})")
    return mo, Path, load_manifest, show_asset


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

# PROCESS: From Integrated Reactor Design to Plasma Equilibrium Exploration

**PROCESS** is a systems code for the conceptual design and optimisation of fusion power plants.

It couples reduced-order plasma-physics and engineering models so that a reactor can be assessed as a

**single self-consistent system**, rather than as a collection of independent subsystems.

PROCESS is developed by the Power Plant Modelling and Integration (PPMI) group within Fusion Technology at UKAEA.

---

/// details | 🌍 Role in fusion design studies

- Explore broad reactor design spaces quickly.

- Test whether proposed designs satisfy coupled physics and engineering requirements.

- Identify trade-offs between plasma performance, magnets, geometry, power balance and plant performance.

- Provide an integrated starting point for more detailed design studies.

///

---

/// details | 🔧 Modelling approach

- **Reduced-order models** — mainly 0D–1D descriptions of the plasma and plant systems.

- **Coupled calculation** — changing one part of the reactor can alter many others.

- **Fast evaluation** — suitable for parameter scans, optimisation and uncertainty studies.

///

---

/// details | 🧭 Learning goals

By the end of the notebook you should understand:

1. How to represent a fusion power plant as a coupled system of models;

2. How a self-consistent fusion power plant plasma state is defined;

3. How changing design choices moves the solution through design space;

4. **How PROCESS helps us make answer the above points and make design decisions** by exposing the trade-offs between performance, feasibility and operating margin.

///

""")



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Interactive demonstration

This version runs **in your browser**, without a Python server or a PROCESS installation.
Use the heating, fuelling, and impurity-seeding sliders to explore the **precomputed PROCESS results**.
The plotted figures were generated from PROCESS in advance; changing a slider selects the
corresponding result rather than rerunning PROCESS.
""")


@app.cell(hide_code=True)
def _():
    # PROCESS was run when preparing the static figures.
    # No PROCESS computation or GitHub download occurs in attendees' browsers.
    pass


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

### PROCESS: Integrated Reactor Design

A single PROCESS solution calculates quantities across the whole plant, including:

- 🔥 **plasma state and fusion performance**

- 📐 **geometry and magnetic fields**

- ⚡ **heating and current drive**

- 🔌 **power flows and plant performance**

- 🧲 **pulse and magnet requirements**

- ⚠️ **physics and engineering limits**

The important point is not any one number: **all of these quantities belong to the same coupled reactor design.**

A change to one part of the reactor can therefore propagate through many other systems.

""")



@app.cell(hide_code=True)
def _(show_asset):
    show_asset("overview.png", "PROCESS integrated reactor overview")


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

### **Poloidal and Toroidal Cross-Sections**

We also use PROCESS to create initial **cross-section views of the tokamak reactor:**

1. **Poloidal section** (vertical cut through the plasma)

2. **Toroidal section** (horizontal slice around the machine’s ring).

- 🟡 **Plasma:** The hot ionised gas confined by strong magnetic fields at the centre.

- 🟣 **First wall:** The inner surface facing the plasma that acts as armour, absorbing heat and particle flux.

- 🟢 **Blanket:** Surrounds the first wall - captures neutrons to breed tritium and generate useful heat.

- ⚫ **Vacuum vessel & shield:** Provides mechanical support and radiation protection to components.

- 🔵 **Magnetic Coils:** Superconducting coils that shape and confine the plasma.

- 🟥 **Cryostat:** The outer shell keeping the superconducting magnets cold and isolated.

PROCESS is used to calculate these cross sections self-consistently ensuring all the components align and there are no overlaps.

""")



@app.cell(hide_code=True)
def _(show_asset):
    show_asset("poloidal.png", "PROCESS poloidal cross-section")


@app.cell(hide_code=True)
def _(show_asset):
    show_asset("toroidal.png", "PROCESS toroidal cross-section")


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

## 🧩 PROCESS: Problem Formulation

PROCESS can be thought of as a **complex function which maps inputs to outputs** describing a steady-state fusion power plant.

$$
\mathbf y = F(\mathbf x,\mathbf d,\mathbf u,\boldsymbol{\theta})
$$

/// admonition

<b>In simple terms:</b> we specify what we know or choose, PROCESS solves the coupled physics and engineering problem, and returns the resulting reactor state.

///

---

### 🎛️ Inputs

It is useful to separate the inputs conceptually into

$$
\mathbf d = \text{design variables},
\qquad
\mathbf u = \text{control variables},
\qquad
\boldsymbol{\theta} = \text{uncertain parameters}.
$$

- 🏗️ **Design variables** describe the system we choose to build — geometry, magnetic field, component dimensions and technology choices.

- 🎚️ **Control variables** describe how the system is operated — auxiliary heating, fuelling, current drive and other actuator-like quantities.

- 🎲 **Uncertain parameters** describe quantities that are not controlled precisely — confinement scalings, efficiencies, material properties and model assumptions.

---

This distinction is useful because different analyses choose different quantities to **prescribe, vary, solve for, or treat as uncertain**.

$$
\boxed{
\text{Inputs}
\;\longrightarrow\;
\text{PROCESS}
\;\longrightarrow\;
\text{Outputs}
}
$$

""")



@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

## Steady-State Plasma

The prescribed inputs alone do not completely define the reactor state. PROCESS must also determine a **self-consistent plasma operating state**.

For the response problem used in this notebook, we represent this state by

$$
\mathbf x =
\begin{bmatrix}
T_e \\
n_e
\end{bmatrix},
$$

where $T_e$ and $n_e$ are the volume-averaged electron temperature and density.

For a given set of design variables $\mathbf d$, control variables $\mathbf u$, and uncertain parameters $\boldsymbol{\theta}$, PROCESS determines $\mathbf x$ by satisfying two governing equations.

---

/// details | 🔥 Global plasma power balance

The power entering the plasma must balance the power leaving it,

$$P_{\mathrm{heating}} = P_{\mathrm{loss}}$$

Heating includes alpha-particle, ohmic and auxiliary heating, while losses include transport and plasma radiation.

In PROCESS this is represented by the **global power-balance equation**.

///

---

/// details | ⛽ Fuel-ion balance

The supplied fuel rate must be consistent with the rate at which fuel is consumed and lost from the plasma,

$$
\dot N_{\mathrm{fuel}}
=
\frac{R_{\mathrm{fusion}}}{f_{\mathrm{burnup}}}
$$

In PROCESS this is represented by the **fuel-ion balance equation**.

///

---

/// details | 📏 What is a residual?

PROCESS expresses each governing equation as a **residual**.

A residual measures how far the current state is from satisfying the equation:

$$
R = \text{left-hand side} - \text{right-hand side}.
$$

If

$$
R = 0,
$$

the equation is satisfied exactly.

If $R \neq 0$, the current plasma state is not self-consistent with that governing equation.

For the two plasma governing equations, the residuals are collected into

$$
\mathbf R(\mathbf x;\mathbf d,\mathbf u,\boldsymbol{\theta})
=
\begin{bmatrix}
R_{\mathrm{power}} \\
R_{\mathrm{fuel}}
\end{bmatrix}.
$$

A PROCESS plasma solution requires

$$
\boxed{
\mathbf R =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
}
$$

so both governing equations are satisfied simultaneously.

///

---

For this response problem, PROCESS varies $T_e$ and $n_e$ until both residuals are zero.

This gives a closed two-equation nonlinear problem for the two plasma-state variables.

Once this state has been found, PROCESS evaluates the coupled physics and engineering models to determine the remaining outputs,

$$
\mathbf y = F(\mathbf x,\mathbf d,\mathbf u,\boldsymbol{\theta}).
$$

""")



@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

## 🎛️ How Does the Plasma Respond to Heating and Fuelling Choices?

Now that we have defined the PROCESS problem, we can use it to ask a simple but important question:

/// admonition

**If we change the controls, how does the steady-state plasma operating point respond?**

///

For this example we vary only two control variables:

- 🔥 **Extra heating** changes the auxiliary power supplied to the plasma by the neutral beam system.

- ⛽ **Fuel supply** changes the fuel particle throughput and is coupled to the fusion reaction rate through C93.

For each combination of heating and fuelling, these controls are prescribed and PROCESS solves the two governing equations for the corresponding plasma state,

$$
(\text{heating},\ \text{fuelling})
\longrightarrow
(T_e,n_e).
$$

In this way, PROCESS can be used to **simulate the steady-state plasma response to changes in the control inputs**.

---

/// details | 🗺️ How to Read the Plasma Solution Space

For each combination of heating and fuelling, PROCESS varies $T_e$ and $n_e$ until the two governing equations are satisfied.

The resulting plasma state is therefore an **output of the PROCESS solution**.

### 🎯 PROCESS solutions

Each coloured marker is a **PROCESS solution** generated from one sampled combination of heating and fuelling.

Its position gives the resulting plasma state,

$$
(T_e,n_e),
$$

while its colour shows the **fusion power** of that solution.

The large outlined marker highlights the currently selected solution.

---

### 🔗 Governing-equation contours

For the currently selected heating and fuelling settings:

- <span style="color:#d62728;"><b>C2 = 0</b></span> shows where **global power balance** is satisfied.

- <span style="color:#9c27b0;"><b>C93 = 0</b></span> shows where **fuel-ion balance** is satisfied.

The solid curves show how the governing-equation residuals behave around the selected solution.

Their intersection identifies the plasma state where both equations are satisfied simultaneously:

$$
\boxed{
C_2 = 0
\qquad \text{and} \qquad
C_{93} = 0
}
$$

---

### 🚧 Physics limits

The dashed curves show where selected physics limits become active, including:

- **density limit**

- **beta limit**

These limits do not define the PROCESS solution — they determine whether that solution is **feasible**.

A solution can therefore satisfy both governing equations while still violating one or more operating limits.

///

---

## 🎛️ Exercise 1: Explore the Plasma Response

You now have two control levers: **heating** and **fuelling**.

Change them and watch how the resulting PROCESS solution moves through the plasma operating space.

/// details | 🔎 Questions to explore

- 🔥 At fixed fuelling, how does increasing extra heating move the plasma operating point?

- ⛽ At fixed heating, how does increasing the fuelling rate move it?

- 🔄 Do heating and fuelling affect $T_e$ and $n_e$ in the same way, or do they act differently?

- ⚛️ Where are the highest-fusion-power solutions?

- 🚧 How does moving through the solution space change the margin to the density and beta limits?

- 🤝 Do the two controls appear to work with or against each other in different parts of the operating space?

///

/// admonition

<b>Aim:</b> understand the link between the <b>controls you choose</b> and the <b>plasma state that results</b>, and how performance and constraint margin change as that solution moves.

///

""")



@app.cell
def _(Path):

    explorer = None

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

    REBUILD_PLASMA_SOLUTIONS = True

    REBUILD_PLASMA_MAPS = True

    PRECOMPUTE_PLASMA_MAPS = True

    EXPLORER_WORKERS = 8

    return (

        DATA_DIR,

        EXPLORER_WORKERS,

        PLASMA_CONTROLS,

        PLASMA_INPUT_FILE,

        PLASMA_MAP_POINTS,

        PRECOMPUTE_PLASMA_MAPS,

        REBUILD_PLASMA_MAPS,

        REBUILD_PLASMA_SOLUTIONS,

        explorer,

    )



@app.cell(hide_code=True)
def _(load_manifest):
    plasma_response = load_manifest("plasma")
    return (plasma_response,)


@app.cell(hide_code=True)
def _(mo, plasma_response):

    # Select only previously sampled heating and fuelling values.

    _axes = plasma_response["control_axes"]

    _reference = plasma_response["reference_indices"]

    plasma_heating_slider = mo.ui.slider(

        steps=[float(_v) for _v in _axes[0]],

        value=float(_axes[0][_reference[0]]),

        label="Heating (MW)",

        show_value=True,

    )

    plasma_fuel_slider = mo.ui.slider(

        steps=[float(_v / 1e21) for _v in _axes[1]],

        value=float(_axes[1][_reference[1]] / 1e21),

        label="Fuelling (×10²¹ s⁻¹)",

        show_value=True,

    )

    mo.vstack([plasma_heating_slider, plasma_fuel_slider])

    return plasma_fuel_slider, plasma_heating_slider



@app.cell(hide_code=True)
def _(plasma_fuel_slider, plasma_heating_slider, plasma_response, show_asset):
    _axes = plasma_response["control_axes"]
    _heat_idx = min(range(len(_axes[0])), key=lambda _i: abs(float(_axes[0][_i]) - plasma_heating_slider.value))
    _fuel_idx = min(range(len(_axes[1])), key=lambda _i: abs(float(_axes[1][_i] / 1e21) - plasma_fuel_slider.value))
    show_asset(f"plasma/p_{_heat_idx}_{_fuel_idx}.png", "Selected PROCESS plasma response")


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

---

""")



@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

## ⚖️ Exercise 2: Heating, Fuelling and Impurity Seeding

The previous example showed how the plasma responds when we change **heating** and **fuelling**.

We now introduce a third control: **argon impurity seeding**.

For each combination of the three controls, PROCESS solves the governing equations and returns the resulting steady-state plasma operating point.

---

### 🎛️ Three interacting controls

- 🔥 **Heating** supplies additional energy to the plasma.

- ⛽ **Fuelling** controls the fuel particle throughput and is coupled to the fusion reaction rate through the fuel-ion balance.

- 💨 **Argon seeding** introduces a radiating impurity and changes how power is exhausted from the plasma.

Argon can reduce the power reaching the exhaust by increasing plasma radiation. However, it also changes **fuel effective charge**, **$Z_{\mathrm{eff}}$**, and the overall plasma power balance.

The controls therefore interact:

$$
\boxed{
\text{change controls}
\;\longrightarrow\;
\text{new plasma state}
\;\longrightarrow\;
\text{new performance + constraint margins}
}
$$

In this example, the starting PROCESS solution is **self-consistent but not feasible**.

---

### 👥 Group Exercise: Find a Feasible Solution

Use **heating, fuelling and argon seeding** to move the PROCESS solution until all three selected limits are satisfied:

- 🚧 **density**

- 🚧 **beta**

- 🚧 **divertor protection**

#### 🔎 While you explore

Pay attention to:

- how each control moves the plasma operating point;

- which constraint margins improve or worsen;

- whether different controls work with or against each other;

- which constraint is hardest to satisfy.

#### ✅ When you find a feasible solution

Be ready to explain:

- Which control changes were most important?

- What performance was gained or lost?

- Which constraint became limiting?

- Could a different combination of controls give another feasible solution?

/// admonition

<b>Aim:</b> do not just find a feasible point — understand <b>why</b> it is feasible and what trade-offs were required to reach it.

///

""")



@app.cell
def _(PLASMA_INPUT_FILE):

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

    REBUILD_IMPURITY_SOLUTIONS = True

    REBUILD_IMPURITY_MAPS = True

    PRECOMPUTE_IMPURITY_MAPS = True

    IMPURITY_WORKERS = 8

    return (

        IMPURITY_CONTROLS,

        IMPURITY_INPUT_FILE,

        IMPURITY_WORKERS,

        PRECOMPUTE_IMPURITY_MAPS,

        REBUILD_IMPURITY_MAPS,

        REBUILD_IMPURITY_SOLUTIONS,

    )



@app.cell(hide_code=True)
def _(load_manifest):
    impurity_response = load_manifest("impurity")
    return (impurity_response,)


@app.cell(hide_code=True)
def _(impurity_response, mo):

    # Controls are ordered heating, argon, fuelling in IMPURITY_CONTROLS.

    _axes = impurity_response["control_axes"]

    _reference = impurity_response["reference_indices"]

    impurity_heating_slider = mo.ui.slider(

        steps=[float(_v) for _v in _axes[0]],

        value=float(_axes[0][_reference[0]]),

        label="Heating (MW)",

        show_value=True,

    )

    impurity_argon_slider = mo.ui.slider(

        steps=[float(_v * 100) for _v in _axes[1]],

        value=float(_axes[1][_reference[1]] * 100),

        label="Argon seeding (% nₑ)",

        show_value=True,

    )

    impurity_fuel_slider = mo.ui.slider(

        steps=[float(_v / 1e21) for _v in _axes[2]],

        value=float(_axes[2][_reference[2]] / 1e21),

        label="Fuelling (×10²¹ s⁻¹)",

        show_value=True,

    )

    mo.vstack([impurity_heating_slider, impurity_argon_slider, impurity_fuel_slider])

    return impurity_argon_slider, impurity_fuel_slider, impurity_heating_slider



@app.cell(hide_code=True)
def _(impurity_argon_slider, impurity_fuel_slider, impurity_heating_slider, impurity_response, show_asset):
    _axes = impurity_response["control_axes"]
    _heat_idx = min(range(len(_axes[0])), key=lambda _i: abs(float(_axes[0][_i]) - impurity_heating_slider.value))
    _argon_idx = min(range(len(_axes[1])), key=lambda _i: abs(float(_axes[1][_i] * 100) - impurity_argon_slider.value))
    _fuel_idx = min(range(len(_axes[2])), key=lambda _i: abs(float(_axes[2][_i] / 1e21) - impurity_fuel_slider.value))
    show_asset(f"impurity/i_{_heat_idx}_{_argon_idx}_{_fuel_idx}.png", "Selected PROCESS impurity response")


@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

---

""")



@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

## From Exploration to Optimisation

In this notebook we changed a small number of controls ourselves and watched how the PROCESS solution moved through the available design space.

We then had to decide:

- which solutions were **feasible**;

- how much **margin** they had to the limits;

- what **performance** they achieved;

- what trade-offs were required to get there.

---

### 🎯 Formalising the Design Choice

PROCESS formalises this process when optimising. Optimisation performs the same kind of search systematically.

We define:

- **iteration variables** - what PROCESS is allowed to change;

- **consistency equations** - what must be satisfied for the design to be self-consistent;

- **limits** - what defines the feasible design space;

- **an objective** - what we want the optimiser to improve.

The optimiser then searches the available solution space for the feasible design that best satisfies that objective.

---

$$
\boxed{
\text{Explore the solution space}
\qquad\longrightarrow\qquad
\text{understand the trade-offs}
\qquad\longrightarrow\qquad
\text{formalise the design objective}
}
$$

**We decide what matters, and encode those engineering priorities in the optimisation problem, and use the solver to find the optimimum solution**

""")



@app.cell
def _():

    return



@app.cell(hide_code=True)
def _(mo):

    mo.md(r"""

### PROCESS

🔗 **PROCESS source code:** [github.com/ukaea/PROCESS](https://github.com/ukaea/PROCESS)

### Bluemira

🔗 **Bluemira source code:** [github.com/Fusion-Power-Plant-Framework/bluemira](https://github.com/Fusion-Power-Plant-Framework/bluemira)

### Fusion Training Program

This material forms part of the **Fusion Training Program**: [Nucleus](https://nucleus.ukaea.uk/page/11260)

""")



if __name__ == "__main__":
    app.run()