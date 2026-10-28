"""Objective function routine."""

import numpy as np

from process.core.data_structure.base import DataStructure
from process.core.exceptions import ProcessValueError
from process.data_structure.numerics import FiguresOfMerit
from process.models.availability import AvailabilityModel

# Smooth-min sharpness for objective 21.
# Larger values approach the exact minimum constraint residual more closely.
SOFT_MIN_BETA = 50.0


def objective_function(i_figure_merit: int, data: DataStructure) -> float:
    """Calculate the specified objective function.

    A negative ``i_figure_merit`` indicates maximisation.
    A positive ``i_figure_merit`` indicates minimisation.

    Relevant UQ objectives
    -----------------------
    20 : L8 norm of actual inequality violations only.
         Intended for Stage-2 least-bad equilibrium recovery.

    21 : Smooth approximation to the signed minimum inequality margin.
         Positive means the limiting constraint is satisfied; negative means
         at least one inequality is violated.

         Use ``i_figure_merit = -21`` to search for extreme satisfaction.

    Raises
    ------
    ProcessValueError
        If ``i_figure_merit`` does not correspond to a valid figure of merit.
    """
    try:
        figure_of_merit = FiguresOfMerit(abs(i_figure_merit))
    except ValueError as err:
        raise ProcessValueError(
            f"Invalid i_figure_merit value: {i_figure_merit}"
        ) from err

    objective_sign = np.sign(i_figure_merit)

    match FiguresOfMerit(figure_of_merit):
        case FiguresOfMerit.MAJOR_RADIUS:
            objective_metric = 0.2 * data.physics.rmajor

        case FiguresOfMerit.NEUTRON_WALL_LOAD:
            objective_metric = data.physics.pflux_fw_neutron_mw

        case FiguresOfMerit.P_TF_PLUS_P_PF:
            objective_metric = (data.tfcoil.tfcmw + 1e-3 * data.pf_power.srcktpm) / 10.0

        case FiguresOfMerit.FUSION_GAIN_Q:
            objective_metric = data.current_drive.big_q_plasma

        case FiguresOfMerit.COST_OF_ELECTRICITY:
            objective_metric = data.costs.coe / 100.0

        case FiguresOfMerit.CAPITAL_COST:
            objective_metric = (
                data.costs.cdirt / 1.0e3
                if data.costs.ireactor == 0
                else data.costs.concost / 1.0e4
            )

        case FiguresOfMerit.ASPECT_RATIO:
            objective_metric = data.physics.aspect

        case FiguresOfMerit.DIVERTOR_HEAT_LOAD:
            objective_metric = data.divertor.pflux_div_heat_load_mw

        case FiguresOfMerit.TOROIDAL_FIELD:
            objective_metric = data.physics.b_plasma_toroidal_on_axis

        case FiguresOfMerit.TOTAL_INJECTED_POWER:
            objective_metric = data.current_drive.p_hcd_injected_total_mw

        case FiguresOfMerit.PULSE_LENGTH:
            objective_metric = data.times.t_plant_pulse_burn / 2.0e4

        case FiguresOfMerit.PLANT_AVAILABILITY_FACTOR:
            if (
                AvailabilityModel(data.costs.i_plant_availability)
                == AvailabilityModel.USER_INPUT
            ):
                raise ProcessValueError(
                    "i_figure_merit=15 requires `f_t_plant_available` to be calculated, "
                    "not user input"
                )
            objective_metric = data.costs.f_t_plant_available

        case FiguresOfMerit.MIN_R0_MAX_TAU_BURN:
            objective_metric = 0.95 * (data.physics.rmajor / 9.0) - 0.05 * (
                data.times.t_plant_pulse_burn / 7200.0
            )

        case FiguresOfMerit.NET_ELECTRICAL_OUTPUT:
            objective_metric = data.heat_transport.p_plant_electric_net_mw / 500.0

        case FiguresOfMerit.NULL_FIGURE_OF_MERIT:
            objective_metric = 1.0

        case FiguresOfMerit.MAX_Q_MAX_T_PLANT_PULSE_BURN:
            objective_metric = -0.5 * (data.current_drive.big_q_plasma / 20.0) - 0.5 * (
                data.times.t_plant_pulse_burn / 7200.0
            )

        case FiguresOfMerit.MIN_INEQ_CONSTRAINT_VIOLATION:
            start = data.numerics.n_equality_constraints
            end = start + data.numerics.n_inequality_constraints
            inequality_residuals = np.asarray(
                data.numerics.constraint_residuals_normalised[start:end],
                dtype=float,
            )

            # PROCESS convention:
            #   residual >= 0 : satisfied
            #   residual <  0 : violated
            #
            # Only actual violations contribute.
            violations = np.maximum(-inequality_residuals, 0.0)

            # L8 norm used in the earlier UQ study:
            #
            #     J8 = (sum(v_i**8))**(1/8)
            #
            # This strongly weights the largest violation while still retaining
            # information from all violated constraints.
            objective_metric = (
                float(np.linalg.norm(violations, ord=8)) if violations.size > 0 else 0.0
            )

        case FiguresOfMerit.SIGNED_MIN_INEQ_MARGIN:
            start = data.numerics.n_equality_constraints
            end = start + data.numerics.n_inequality_constraints
            inequality_residuals = np.asarray(
                data.numerics.constraint_residuals_normalised[start:end],
                dtype=float,
            )

            if inequality_residuals.size == 0:
                objective_metric = 0.0
            else:
                # Smooth approximation to min(r_i):
                #
                #   softmin(r) = -(1/beta) log(sum(exp(-beta*r_i)))
                #
                # This is differentiable when the limiting constraint changes,
                # unlike np.min(r).  It approaches the true minimum as beta
                # increases, and is always slightly more conservative than the
                # exact minimum.
                #
                # Stable evaluation: shift by r_min so the exponentials are <= 1.
                r_min = float(np.min(inequality_residuals))
                shifted = inequality_residuals - r_min
                objective_metric = float(
                    r_min
                    - np.log(np.sum(np.exp(-SOFT_MIN_BETA * shifted))) / SOFT_MIN_BETA
                )

        case _:
            raise ProcessValueError(f"Unknown figure_of_merit: {figure_of_merit}")

    return objective_sign * objective_metric
