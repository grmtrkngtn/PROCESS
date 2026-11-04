"""An adapter for different solvers."""

import importlib
import logging
import time
from abc import ABC, abstractmethod

import cvxpy
import numpy as np
from pyvmcon import (
    AbstractProblem,
    LineSearchConvergenceException,
    QSPSolverException,
    Result,
    VMCONConvergenceException,
    solve,
)
from scipy import optimize
from scipy.optimize import fsolve
from scipy.stats import qmc

from process.core import constants, process_output
from process.core.data_structure.base import DataStructure
from process.core.exceptions import ProcessValueError
from process.core.solver.evaluators import Evaluators
from process.data_structure.numerics import SolverOutputCondition

logger = logging.getLogger(__name__)


class _Solver(ABC):
    """Base class for different solver implementations."""

    def __init__(self, *, data: DataStructure):
        """Initialise a solver."""
        # Exit code for the solver
        self.ifail = 0
        self.data = data
        self.tolerance = self.data.numerics.epsvmc
        self.b: float | None = None
        self.maxcal = self.data.globals.maxcal

    def set_evaluators(self, evaluators: Evaluators):
        """Set objective and constraint functions and their gradient evaluators.

        Parameters
        ----------
        evaluators : Evaluators
            objective and constraint evaluators

        """
        self.evaluators = evaluators

    def set_opt_params(self, x_0: np.ndarray):
        """Define the initial optimisation parameters.

        Parameters
        ----------
        x_0 : np.ndarray
            optimisation parameters vector

        """
        self.x_0 = x_0

    def set_bounds(
        self,
        bndl: np.ndarray,
        bndu: np.ndarray,
        ilower: np.ndarray | None = None,
        iupper: np.ndarray | None = None,
    ):
        """Set the bounds on the optimisation parameters.

        Parameters
        ----------
        bndl : np.ndarray
            lower bounds for the optimisation parameters
        bndu : np.ndarray
            upper bounds for the optimisation parameters
        ilower : np.ndarray, optional
            array of 0s and 1s to activate lower bounds on
            optimsation parameters in x
        iupper : np.ndarray, optional
            array of 0s and 1s to activate upper bounds on
            optimsation parameters in x
        """
        self.bndl = bndl
        self.bndu = bndu

        # TODO Remove ilower/iupper and use finite vs. infinite values in bndl/bndu
        # instead to determine defined vs. undefined bounds
        # If lower and upper bounds switch arrays aren't specified, set all bounds
        # to defined
        if ilower is None and iupper is None:
            ilower = np.ones(len(bndl), dtype=int)
            iupper = np.ones(len(bndu), dtype=int)

        self.ilower = ilower
        self.iupper = iupper

    def set_constraints(self, m: int, meq: int):
        """Set the total number of constraints and equality constraints.

        Parameters
        ----------
        m : int
            number of constraint equations
        meq : int
            of the constraint equations, how many are equalities
        """
        self.m = m
        self.meq = meq

    def set_tolerance(self, tolerance: float):
        """Set tolerance for solver termination.

        Parameters
        ----------
        tolerance : float
            tolerance for solver termination
        """
        self.tolerance = tolerance

    def set_b(self, b: float):
        """Set the multiplier for the Hessian approximation.

        Parameters
        ----------
        b : float
            multiplier for an identity matrix as input for the Hessian b(n,n)
        """
        self.b = b

    @abstractmethod
    def solve(self) -> SolverOutputCondition:
        """Run the optimisation.

        Returns
        -------
        SolverOutputCondition
            solver error code
        """


class VmconProblem(AbstractProblem):
    """Set up the Vmcon problem"""

    def __init__(self, evaluator, nequality, ninequality):
        self._evaluator = evaluator
        self._nequality = nequality
        self._ninequality = ninequality

    def __call__(self, x: np.ndarray) -> Result:
        """Call a problem"""
        n = x.shape[0]
        objf, conf = self._evaluator.fcnvmc1(n, self.total_constraints, x, 0)
        fgrd, cnorm = self._evaluator.fcnvmc2(n, self.total_constraints, x, n)

        return Result(
            objf,
            fgrd,
            conf[: self.num_equality],
            cnorm[:, : self.num_equality].T,
            conf[self.num_equality :],
            cnorm[:, self.num_equality :].T,
        )

    @property
    def num_equality(self) -> int:
        """Number of equality constraints"""
        return self._nequality

    @property
    def num_inequality(self) -> int:
        """Number of inequality constraints"""
        return self._ninequality


class Vmcon(_Solver):
    """New VMCON implementation."""

    def solve(self) -> SolverOutputCondition:
        """Optimise using new VMCON.

        Returns
        -------
        SolverOutputCondition
            solver error code

        Raises
        ------
        ValueError
            If the initial point vector is the wrong shape or out of bounds
        """
        problem = VmconProblem(self.evaluators, self.meq, self.m - self.meq)

        bb = None
        if self.b is not None:
            bb = np.identity(self.data.numerics.n_iteration_variables) * self.b

        def _solver_callback(i: int, _result, _x, convergence_param: float):
            self.data.numerics.n_solver_iterations = i + 1
            self.data.globals.convergence_parameter = convergence_param
            print(
                f"{i + 1} | Convergence Parameter: {convergence_param:.3E}",
                end="\r",
                flush=True,
            )

        def _ineq_cons_satisfied(
            result: Result,
            _x: np.ndarray,
            _delta: np.ndarray,
            _lambda_eq: np.ndarray,
            _lambda_in: np.ndarray,
        ) -> bool:
            """Check that inequality constraints are satisfied.

            This additional convergence criterion ensures that solutions have
            satisfied inequality constraints.

            Parameters
            ----------
            result : Result
                evaluation of current optimisation parameter vector
            _x : np.ndarray
                current optimisation parameter vector
            _delta : np.ndarray
                search direction for line search
            _lambda_eq : np.ndarray
                equality Lagrange multipliers
            _lambda_in : np.ndarray
                inequality Lagrange multipliers

            Returns
            -------
            bool
                True if inequality constraints satisfied

            """
            # negative constraint value = violated
            # Check all ineqs are satisfied to within the tolerance
            # E.g. the relative violations are no more than v=0-tolerance
            return bool(
                np.all(result.ie >= -self.data.numerics.force_vmcon_inequality_tolerance)
            )

        try:
            x, _, _, res = solve(
                problem,
                np.array(self.x_0),
                np.array(self.bndl),
                np.array(self.bndu),
                max_iter=self.maxcal,
                epsilon=self.tolerance,
                qsp_options={"solver": cvxpy.CLARABEL},
                initial_B=bb,
                callback=_solver_callback,
                additional_convergence=_ineq_cons_satisfied
                if self.data.numerics.force_vmcon_inequality_satisfication
                else None,
            )
        except VMCONConvergenceException as e:
            if isinstance(e, LineSearchConvergenceException):
                self.info = SolverOutputCondition.MAX_LINE_SEARCHES
            elif isinstance(e, QSPSolverException):
                self.info = SolverOutputCondition.NO_SOLUTION
            else:
                self.info = SolverOutputCondition.MAX_ITERATIONS

            logger.critical(str(e))

            x = e.x
            res = e.result

        except ValueError:
            itervar_name_list = ""
            for count, iter_var in enumerate(
                self.data.numerics.ixc[: self.data.numerics.n_iteration_variables]
            ):
                itervar_name = self.data.numerics.lablxc[iter_var - 1]
                itervar_name_list += f"{count}: {itervar_name} \n"

            logger.warning(f"Active iteration variables are : \n{itervar_name_list}")
            raise

        else:
            self.info = SolverOutputCondition.CONVERGED

        # print a blank line because of the carridge return
        # in the callback
        print()

        self.x = x
        self.objf = res.f
        self.conf = np.hstack((res.eq, res.ie))

        return self.info

    def verror(self):
        """Routine to print out relevant messages in the case of an
        unfeasible result from a VMCON (optimisation) run

        This routine prints out relevant messages in the case of
        an unfeasible result from a VMCON (optimisation) run.
        """
        strings = "\n".join(
            {
                SolverOutputCondition.USER_TERMINATED: (
                    "User-terminated execution of VMCON.",
                ),
                SolverOutputCondition.IMPROPER_INPUT: (
                    "Improper input parameters to the VMCON routine.",
                    "PROCESS coding must be checked.",
                ),
                SolverOutputCondition.MAX_ITERATIONS: (
                    "The maximum number of calls has been reached without solution.",
                    (
                        "The code may be stuck in a minimum in the residual space that"
                        " is significantly above zero.\n"
                    ),
                    "There is either no solution possible, or the code",
                    "is failing to escape from a deep local minimum.",
                    "Try changing the variables in IXC, or modify their initial values.",
                ),
                SolverOutputCondition.MAX_LINE_SEARCHES: (
                    "The line search required the maximum of 10 calls.",
                    "A feasible solution may be difficult to achieve.",
                    "Try changing or adding variables to IXC.",
                ),
                SolverOutputCondition.NO_SOLUTION: (
                    "The quadratic programming technique was unable to",
                    "find a feasible point.\n",
                    "Try changing or adding variables to IXC, or modify",
                    "their initial values (especially if only 1 optimisation",
                    "iteration was performed).",
                ),
            }.get(self.info, "Unknown Error code")
        )

        process_output.ocmmnt(constants.NOUT, strings)
        print(strings)


class VmconBounded(Vmcon):
    """A solver that uses VMCON but checks x is in bounds before running"""

    def set_opt_params(self, x_0: np.ndarray):
        """Define the initial optimisation parameters.

        Parameters
        ----------
        x_0 : np.ndarray
            optimisation parameters vector

        """
        lower_violated = np.less(x_0, self.bndl)
        upper_violated = np.greater(x_0, self.bndu)

        for index, entry in enumerate(lower_violated):
            if entry:
                x_0[index] = self.bndl[index]
        for index, entry in enumerate(upper_violated):
            if entry:
                x_0[index] = self.bndu[index]
        self.x_0 = x_0


class FSolve(_Solver):
    """Solve equality constraints to ensure model consistency."""

    def evaluate_eq_cons(self, x: np.ndarray) -> np.ndarray:
        """Evaluate equality constraints.

        Parameters
        ----------
        x : np.ndarray
            parameter vector

        Returns
        -------
        np.ndarray
            equality constraint vector
        """
        # Evaluate equality constraints only
        _, conf = self.evaluators.fcnvmc1(x.shape[0], self.meq, x, 0)

        return conf

    def solve(self) -> int:
        """Solve equality constraints.

        Returns
        -------
        int
            solver error code
        """
        print("Solving equality constraints using fsolve")
        self.x, _info, err, msg = fsolve(
            self.evaluate_eq_cons, self.x_0, full_output=True
        )

        # Evaluate equality and inequality constraints at equality-satisfying solution
        # (or at last iteration of x if solution not found)
        _, self.conf = self.evaluators.fcnvmc1(self.x.shape[0], self.m, self.x, 0)

        # err == 1 for successful solve
        if err != 1:
            print(f"fsolve error code {err}: {msg}")
        self.info = err
        # No objective function
        self.objf = None
        return self.info


def _slsqp_residual(x, solver):
    """RMSE of PROCESS' already-normalised equality residuals.

    ``fcnvmc1`` returns the normalised PROCESS constraint vector.  Stage 3
    therefore works directly with the active equality residuals rather than
    reconstructing dimensional plasma derivatives and introducing a second
    normalisation scheme.
    """
    _, conf = solver._evaluate_state(
        np.asarray(x, dtype=float),
        m_eval=solver.meq,
    )
    eq = np.asarray(conf[: solver.meq], dtype=float)
    if eq.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(eq**2)))


class ScipySLSQP(_Solver):
    """Robust SciPy SLSQP solver with low-overhead diagnostics.

    This keeps the original three-stage recovery structure:

    1. Fully constrained solve: objective + equalities + inequalities.
    2. Objective-20 recovery: minimise PROCESS inequality-violation RMSE while
       enforcing the governing equalities, retaining constraint 30 as a hard
       domain guard.
    3. Equality-residual recovery: minimise the RMSE of PROCESS's already-
       normalised equality residuals, retaining constraint 30 as a domain guard.

    Stage 1 now tries the nominal control guess, optional fuelling-only guesses,
    then reproducible Sobol interior guesses before proceeding to Stage 2.
    Stage 2 and Stage 3 each run at most once; the former post-fallback adaptive
    multistart wrapper has been removed.

    Performance changes relative to the original experimental implementation:

    * only the primary finite-difference step is tried at each stage;
    * objective/constraint calls share a one-state PROCESS evaluation cache;
    * the progress callback evaluates PROCESS at most once per reported iterate;
    * the selected final point is always explicitly re-evaluated so PROCESS state
      matches the returned solution.
    """

    SOLVER_TOL = 1.0e-5
    EQ_CONSTRAINT_TOL = 1.0e-6
    INEQ_CONSTRAINT_TOL = 1.0e-5

    FULL_MAXITER = 50
    EQ_MAXITER = 50
    RESIDUAL_MAXITER = 100

    # Retry the FULLY constrained Stage 1 before Stage 2/3 fallback.
    # 1 nominal start + up to 2 fuelling-only starts + 2 Sobol starts.
    # Fuelling-only starts are used only if a fuel iteration variable exists.
    STAGE1_FUEL_FRACTIONS = (0.25, 0.75)
    STAGE1_N_SOBOL_STARTS = 2
    STAGE1_SOBOL_SEED = 1729
    STAGE1_SOBOL_INTERIOR_MARGIN = 0.10

    def __init__(self, *, data: DataStructure):
        super().__init__(data=data)
        self.iteration = 0
        self.n = 0
        self.bounds = None

        # One-state cache. This avoids repeated fcnvmc1 calls when SciPy asks
        # for objective and constraints at exactly the same x.
        self._cache_x: np.ndarray | None = None
        self._cache_objf: float | None = None
        self._cache_conf: np.ndarray | None = None
        self._cache_m: int | None = None

        # Best Stage-1 point for handing off to Stage 2.  A failed SLSQP solve
        # can terminate at a worse point than one visited earlier, so do not
        # blindly use result_full.x as the recovery start.
        self._stage1_best_x: np.ndarray | None = None
        self._stage1_best_score: tuple[float, ...] | None = None
        self._stage1_best_metrics: dict[str, float | bool] | None = None

        # Best Stage-2/3 points are tracked for the same reason: a failed local
        # optimiser can terminate at a worse point than one visited earlier.
        self._stage2_best_x: np.ndarray | None = None
        self._stage2_best_score: tuple[float, ...] | None = None
        self._stage2_best_metrics: dict[str, float | bool] | None = None
        self._stage3_best_x: np.ndarray | None = None
        self._stage3_best_eq_rmse: float | None = None

    def _stage1_start_points(
        self,
        nominal_x: np.ndarray,
        bndl: np.ndarray,
        bndu: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """Build deterministic Stage-1 starts in solver-scaled control space.

        The input physics, uncertainty realisation, active iteration variables,
        constraints and bounds are unchanged. Only the *initial guesses* for
        the active iteration variables differ.

        Start with nominal; if a fuel-named active variable exists, vary only
        its initial value to two fractions of its bounds. Finally add interior
        scrambled-Sobol starts spanning all finitely bounded controls.
        """
        nominal = np.asarray(nominal_x, dtype=float).copy()
        starts = [("nominal", nominal)]
        if nominal.size == 0:
            return starts

        finite_box = np.isfinite(bndl) & np.isfinite(bndu) & (bndu > bndl)

        def add_start(label: str, candidate: np.ndarray):
            candidate = np.clip(np.asarray(candidate, dtype=float), bndl, bndu)
            if not any(np.array_equal(candidate, previous) for _, previous in starts):
                starts.append((label, candidate.copy()))

        # Prioritise the control that exposed this issue. The diagnostic
        # name comes from the active iteration variable label, not from
        # a hard-coded variable index (which could change with IXC).
        try:
            active_codes = self.data.numerics.ixc[: nominal.size]
            names = [
                str(self.data.numerics.lablxc[int(code) - 1]).lower()
                for code in active_codes
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
            names = []

        fuel_indices = [
            i for i, name in enumerate(names)
            if "fuel" in name and finite_box[i]
        ]
        if fuel_indices:
            fuel_idx = fuel_indices[0]
            print(f"Stage 1 targeted multistart control: {names[fuel_idx]}")
            for fraction in self.STAGE1_FUEL_FRACTIONS:
                candidate = nominal.copy()
                candidate[fuel_idx] = (
                    bndl[fuel_idx]
                    + float(fraction) * (bndu[fuel_idx] - bndl[fuel_idx])
                )
                add_start(f"fuelling {fraction:.0%}", candidate)
        else:
            print("Stage 1: no active, finitely bounded fuelling control identified")

        n_sobol = int(self.STAGE1_N_SOBOL_STARTS)
        if n_sobol < 0:
            raise ProcessValueError("STAGE1_N_SOBOL_STARTS must be non-negative")
        if n_sobol:
            margin = float(self.STAGE1_SOBOL_INTERIOR_MARGIN)
            if not 0.0 <= margin < 0.5:
                raise ProcessValueError(
                    "STAGE1_SOBOL_INTERIOR_MARGIN must satisfy 0 <= margin < 0.5"
                )

            sampler = qmc.Sobol(
                d=nominal.size,
                scramble=True,
                seed=self.STAGE1_SOBOL_SEED,
            )
            n_power_two = 1 << (n_sobol - 1).bit_length()
            unit_points = sampler.random_base2(m=n_power_two.bit_length() - 1)
            unit_points = margin + (1.0 - 2.0 * margin) * unit_points

            for i, unit_point in enumerate(unit_points[:n_sobol], start=1):
                candidate = nominal.copy()
                candidate[finite_box] = (
                    bndl[finite_box]
                    + unit_point[finite_box]
                    * (bndu[finite_box] - bndl[finite_box])
                )
                add_start(f"Sobol {i}", candidate)

        return starts

    def _invalidate_cache(self):
        self._cache_x = None
        self._cache_objf = None
        self._cache_conf = None
        self._cache_m = None

    def _reset_stage1_recovery_candidate(self):
        """Reset the best Stage-1 point retained for Stage-2 recovery."""
        self._stage1_best_x = None
        self._stage1_best_score = None
        self._stage1_best_metrics = None

    def _consider_stage1_recovery_candidate(
        self,
        x: np.ndarray,
        equality_residuals: np.ndarray,
        inequality_residuals: np.ndarray,
    ):
        """Retain the most useful Stage-1 point for starting Stage 2.

        Preference is lexicographic:

        1. Any point that already satisfies the governing equalities is preferred
           over one that does not.
        2. Among equality-satisfying points, retain the one with the smallest
           inequality-violation RMSE, matching the Stage-2 objective.
        3. If no equality-satisfying point has been seen, retain the point with
           the smallest equality RMSE, then the smallest inequality-violation
           RMSE as a tie-breaker.

        This is deliberately a *recovery* score, not the user's PROCESS objective.
        """
        if self.data.numerics.solver_problem_type != 0:
            return

        eq = np.asarray(equality_residuals, dtype=float)
        ineq = np.asarray(inequality_residuals, dtype=float)

        if not np.all(np.isfinite(eq)) or not np.all(np.isfinite(ineq)):
            return

        eq_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        eq_max = float(np.max(np.abs(eq))) if eq.size else 0.0

        violations = np.maximum(-ineq, 0.0)
        ineq_rmse = float(np.sqrt(np.mean(violations**2))) if violations.size else 0.0
        ineq_max = float(np.max(violations)) if violations.size else 0.0

        equality_tol = max(self.EQ_CONSTRAINT_TOL, self.SOLVER_TOL)
        equalities_recovered = eq_max <= equality_tol

        if equalities_recovered:
            score = (0.0, ineq_rmse, eq_rmse)
        else:
            score = (1.0, eq_rmse, ineq_rmse)

        if self._stage1_best_score is None or score < self._stage1_best_score:
            self._stage1_best_x = np.asarray(x, dtype=float).copy()
            self._stage1_best_score = score
            self._stage1_best_metrics = {
                "equalities_recovered": equalities_recovered,
                "equality_rmse": eq_rmse,
                "equality_max": eq_max,
                "inequality_rmse": ineq_rmse,
                "inequality_max": ineq_max,
            }

    def _reset_stage2_recovery_candidate(self):
        """Reset the best Stage-2 point retained for Stage-3/handoff recovery."""
        self._stage2_best_x = None
        self._stage2_best_score = None
        self._stage2_best_metrics = None

    def _consider_stage2_recovery_candidate(
        self,
        x: np.ndarray,
        equality_residuals: np.ndarray,
        inequality_residuals: np.ndarray,
    ):
        """Retain the best Stage-2 point for either classification or Stage 3.

        If the governing equalities have been recovered, prefer the point with
        the smallest inequality-violation RMSE.  Otherwise prefer the point
        closest to governing equilibrium.
        """
        if self.data.numerics.solver_problem_type != 1:
            return

        eq = np.asarray(equality_residuals, dtype=float)
        ineq = np.asarray(inequality_residuals, dtype=float)

        if not np.all(np.isfinite(eq)) or not np.all(np.isfinite(ineq)):
            return

        eq_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        eq_max = float(np.max(np.abs(eq))) if eq.size else 0.0
        violations = np.maximum(-ineq, 0.0)
        ineq_rmse = float(np.sqrt(np.mean(violations**2))) if violations.size else 0.0
        ineq_max = float(np.max(violations)) if violations.size else 0.0

        equality_tol = max(self.EQ_CONSTRAINT_TOL, self.SOLVER_TOL)
        equalities_recovered = eq_max <= equality_tol

        if equalities_recovered:
            score = (0.0, ineq_rmse, eq_rmse)
        else:
            score = (1.0, eq_rmse, ineq_rmse)

        if self._stage2_best_score is None or score < self._stage2_best_score:
            self._stage2_best_x = np.asarray(x, dtype=float).copy()
            self._stage2_best_score = score
            self._stage2_best_metrics = {
                "equalities_recovered": equalities_recovered,
                "equality_rmse": eq_rmse,
                "equality_max": eq_max,
                "inequality_rmse": ineq_rmse,
                "inequality_max": ineq_max,
            }

    def _reset_stage3_recovery_candidate(self):
        """Reset the best governing-residual point visited during Stage 3."""
        self._stage3_best_x = None
        self._stage3_best_eq_rmse = None

    def _consider_stage3_recovery_candidate(
        self,
        x: np.ndarray,
        equality_residuals: np.ndarray,
    ):
        """Retain the Stage-3 point with the smallest governing-equation RMSE."""
        if self.data.numerics.solver_problem_type != 2:
            return

        eq = np.asarray(equality_residuals, dtype=float)
        if not np.all(np.isfinite(eq)):
            return

        eq_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        if self._stage3_best_eq_rmse is None or eq_rmse < self._stage3_best_eq_rmse:
            self._stage3_best_x = np.asarray(x, dtype=float).copy()
            self._stage3_best_eq_rmse = eq_rmse

    def _evaluate_state(
        self,
        x: np.ndarray,
        *,
        m_eval: int | None = None,
        force: bool = False,
    ):
        """Evaluate PROCESS once at ``x`` and cache objective/constraints.

        ``m_eval`` allows the Stage-3 governing recovery to evaluate only the
        leading active constraints it actually needs, rather than forcing all
        engineering constraints to be valid merely to measure constraints 2
        and 93.
        """
        x_arr = np.asarray(x, dtype=float)
        m_requested = self.m if m_eval is None else int(m_eval)

        if (
            not force
            and self._cache_x is not None
            and self._cache_m == m_requested
            and np.array_equal(x_arr, self._cache_x)
        ):
            return self._cache_objf, self._cache_conf

        objf, conf = self.evaluators.fcnvmc1(self.n, m_requested, x_arr, self.ifail)
        self._cache_x = x_arr.copy()
        self._cache_m = m_requested
        self._cache_objf = objf
        self._cache_conf = np.asarray(conf, dtype=float).copy()
        return self._cache_objf, self._cache_conf

    def obj_func(self, x):
        _, conf = self._evaluate_state(x)

        if abs(self.data.numerics.i_figure_merit) == 21:
            ineq = np.asarray(conf[self.meq : self.m], dtype=float)

            beta = 50.0
            rmin = np.min(ineq)
            softmin = rmin - np.log(np.sum(np.exp(-beta * (ineq - rmin)))) / beta

            return np.sign(self.data.numerics.i_figure_merit) * softmin

        objf, _ = self._evaluate_state(x)
        return objf

    def constraint_eq_vec(self, x):
        _, conf = self._evaluate_state(x)
        return conf[: self.meq]

    def constraint_ineq_vec(self, x):
        _, conf = self._evaluate_state(x)
        return conf[self.meq : self.m]

    def constraint_ineq_30(self, x):
        """Return constraint 30 (injected-power upper limit) as a domain guard."""
        con30_idx = np.where(self.data.numerics.icc[: self.m] == 30)[0]
        if con30_idx.size == 0:
            # No active constraint 30: return a permanently satisfied guard.
            return np.array([1.0])

        # fcnvmc1 evaluates the first m active constraints.  Evaluate only as
        # far as constraint 30 rather than all engineering constraints.  This
        # prevents an unrelated later constraint (e.g. 60) from killing the
        # Stage-3 governing-residual diagnostic.
        idx = int(con30_idx[0])
        _, conf = self._evaluate_state(x, m_eval=idx + 1)
        return np.asarray([conf[idx]], dtype=float)

    def convergence_progress(self, x_current):
        """Print the same useful progress without three extra PROCESS runs."""
        self.iteration += 1

        # Stages 1 and 2 need the complete constraint vector.  Stage 3 is
        # deliberately a governing-equation diagnostic, so do not let an
        # unrelated invalid engineering constraint kill its progress callback.
        if self.data.numerics.solver_problem_type == 2:
            _, conf = self._evaluate_state(x_current, m_eval=self.meq)
            eqs = np.asarray(conf[: self.meq], dtype=float)
            obj = float(np.sqrt(np.mean(eqs**2))) if eqs.size else 0.0
            ineqs = np.array([], dtype=float)
        else:
            obj, conf = self._evaluate_state(x_current)
            eqs = np.asarray(conf[: self.meq], dtype=float)
            ineqs = np.asarray(conf[self.meq : self.m], dtype=float)

        violations = np.maximum(-ineqs, 0.0)
        ineqs_rms = float(np.sqrt(np.mean(violations**2))) if violations.size else 0.0
        eqs_rms = float(np.sqrt(np.mean(eqs**2))) if eqs.size else 0.0

        if self.data.numerics.solver_problem_type == 0:
            self._consider_stage1_recovery_candidate(x_current, eqs, ineqs)
        elif self.data.numerics.solver_problem_type == 1:
            self._consider_stage2_recovery_candidate(x_current, eqs, ineqs)
        elif self.data.numerics.solver_problem_type == 2:
            self._consider_stage3_recovery_candidate(x_current, eqs)

        print(f"\nIteration {self.iteration}:")
        print(f"Obj func = {obj}")
        print(f"Equality RMSE = {eqs_rms:.3e}")
        if self.data.numerics.solver_problem_type == 2:
            print("Inequality diagnostics = skipped in governing-residual stage")
        else:
            print(f"Inequality violation RMSE = {ineqs_rms:.3e}")

        print("\nEquality constraints:")
        for i in np.argsort(np.abs(eqs))[::-1]:
            print(f"Constraint {self.data.numerics.icc[i]} = {eqs[i]:.3e}")

        print("\nViolated inequality constraints:")
        for i in np.argsort(ineqs):
            if ineqs[i] < 0.0:
                print(
                    f"Constraint {self.data.numerics.icc[len(eqs) + i]} = {ineqs[i]:.3e}"
                )

        print("\nRange-normalised opt params:")
        range_norm_opt_params = {}
        for i in range(self.data.numerics.n_iteration_variables):
            lower = self.data.numerics.itv_scaled_lower_bounds[i]
            upper = self.data.numerics.itv_scaled_upper_bounds[i]

            if upper == lower:
                xnorm = 1.0
            else:
                xnorm = min(max((x_current[i] - lower) / (upper - lower), 0.0), 1.0)

            number = self.data.numerics.ixc[i]
            range_norm_opt_params[number] = xnorm

        for number, xnorm in sorted(
            range_norm_opt_params.items(), key=lambda item: item[1]
        ):
            name = self.data.numerics.lablxc[number - 1]
            print(f"{xnorm:.3f}: {number}, {name}")

    def _run_full_solve(
        self, constraints, finite_diff_step, x_start, *, reset_recovery_candidate=True
    ):
        self.iteration = 0
        self._invalidate_cache()
        if reset_recovery_candidate:
            self._reset_stage1_recovery_candidate()

        # The starting point is also a legitimate recovery candidate.
        try:
            _, conf_start = self._evaluate_state(
                np.asarray(x_start, dtype=float),
                force=True,
            )
            self._consider_stage1_recovery_candidate(
                x_start,
                conf_start[: self.meq],
                conf_start[self.meq : self.m],
            )
        except (ValueError, ProcessValueError, RuntimeError):
            self._invalidate_cache()

        try:
            result = optimize.minimize(
                self.obj_func,
                np.asarray(x_start, dtype=float),
                method="SLSQP",
                jac="3-point",
                bounds=self.bounds,
                constraints=constraints,
                callback=self.convergence_progress,
                options={
                    "disp": True,
                    "maxiter": self.FULL_MAXITER,
                    "ftol": self.SOLVER_TOL,
                    "finite_diff_rel_step": finite_diff_step,
                },
            )
            # SciPy's terminal x is not guaranteed to have been the best recovery
            # point visited, but include it in the candidate set in case the
            # callback was not invoked for the final iterate.
            try:
                _, conf_final = self._evaluate_state(result.x, force=True)
                self._consider_stage1_recovery_candidate(
                    result.x,
                    conf_final[: self.meq],
                    conf_final[self.meq : self.m],
                )
            except (ValueError, ProcessValueError, RuntimeError):
                self._invalidate_cache()

            print(
                f"SLSQP full constrained solve: status {result.status}: {result.message}"
            )
            return result
        except (ValueError, ProcessValueError) as exc:
            print(f"SLSQP full constrained solve model error: {exc}")
            return None

    def _run_eq_solve(self, eq_constraints, finite_diff_step, x_start):
        self.iteration = 0
        self._invalidate_cache()
        self._reset_stage2_recovery_candidate()

        # Keep the incoming handoff point as a valid fallback.  This is
        # especially important for a Stage-3 -> Stage-2 rescue: if Stage 3 has
        # already recovered the governing equations, Stage 2 must never lose
        # that point merely because its local optimisation terminates poorly.
        try:
            _, conf_start = self._evaluate_state(
                np.asarray(x_start, dtype=float),
                force=True,
            )
            self._consider_stage2_recovery_candidate(
                x_start,
                conf_start[: self.meq],
                conf_start[self.meq : self.m],
            )
        except (ValueError, ProcessValueError, RuntimeError):
            self._invalidate_cache()

        try:
            result = optimize.minimize(
                self.obj_func,
                np.asarray(x_start, dtype=float),
                method="SLSQP",
                jac="3-point",
                bounds=self.bounds,
                constraints=eq_constraints,
                callback=self.convergence_progress,
                options={
                    "disp": True,
                    "maxiter": self.EQ_MAXITER,
                    "ftol": self.SOLVER_TOL,
                    "finite_diff_rel_step": finite_diff_step,
                },
            )
            try:
                _, conf_final = self._evaluate_state(result.x, force=True)
                self._consider_stage2_recovery_candidate(
                    result.x,
                    conf_final[: self.meq],
                    conf_final[self.meq : self.m],
                )
            except (ValueError, ProcessValueError, RuntimeError):
                self._invalidate_cache()

            print(f"SLSQP equality recovery: status {result.status}: {result.message}")
            return result
        except (ValueError, ProcessValueError) as exc:
            print(f"SLSQP equality recovery model error: {exc}")
            return None

    def _run_residual_solve(self, guard_constraint, finite_diff_step, x_start):
        self.iteration = 0
        self._invalidate_cache()
        self._reset_stage3_recovery_candidate()

        try:
            _, conf_start = self._evaluate_state(
                np.asarray(x_start, dtype=float),
                m_eval=self.meq,
                force=True,
            )
            self._consider_stage3_recovery_candidate(
                x_start,
                conf_start[: self.meq],
            )
        except (ValueError, ProcessValueError, RuntimeError):
            self._invalidate_cache()

        try:
            result = optimize.minimize(
                _slsqp_residual,
                np.asarray(x_start, dtype=float),
                args=(self,),
                method="SLSQP",
                jac="3-point",
                bounds=self.bounds,
                constraints=guard_constraint,
                callback=self.convergence_progress,
                options={
                    "disp": True,
                    "maxiter": self.RESIDUAL_MAXITER,
                    "ftol": self.SOLVER_TOL,
                    "finite_diff_rel_step": finite_diff_step,
                },
            )
            try:
                _, conf_final = self._evaluate_state(
                    result.x,
                    m_eval=self.meq,
                    force=True,
                )
                self._consider_stage3_recovery_candidate(
                    result.x,
                    conf_final[: self.meq],
                )
            except (ValueError, ProcessValueError, RuntimeError):
                self._invalidate_cache()

            print(
                "SLSQP normalised equality-residual recovery: "
                f"status {result.status}: {result.message}"
            )
            return result
        except (ValueError, ProcessValueError) as exc:
            print(f"SLSQP equality-residual recovery model error: {exc}")
            return None

    def _set_final_state(self, x, objf=None):
        """Re-evaluate the selected x so PROCESS output matches the returned point."""
        self._invalidate_cache()
        final_objf, final_conf = self._evaluate_state(np.asarray(x), force=True)
        self.x = np.asarray(x, dtype=float).copy()
        self.objf = final_objf if objf is None else objf
        self.conf = np.asarray(final_conf, dtype=float).copy()

    def _update_residual_diagnostics(self):
        """Populate the robust-solver residual fields written to OUT.DAT."""
        eq = np.asarray(self.conf[: self.meq], dtype=float)
        ineq = np.asarray(self.conf[self.meq : self.m], dtype=float)

        equality_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        equality_max = float(np.max(np.abs(eq))) if eq.size else 0.0

        # Same definition as objective 20: satisfied inequalities contribute 0,
        # and the RMSE denominator is the full active inequality set.
        violations = np.maximum(-ineq, 0.0)
        inequality_rmse = (
            float(np.sqrt(np.mean(violations**2))) if violations.size else 0.0
        )
        inequality_max = float(np.max(violations)) if violations.size else 0.0

        self.data.numerics.equality_residual_rmse = equality_rmse
        self.data.numerics.equality_residual_max = equality_max
        self.data.numerics.inequality_violation_rmse = inequality_rmse
        self.data.numerics.inequality_violation_max = inequality_max

        if ineq.size:
            limiting_index = int(np.argmin(ineq))
            minimum_inequality_margin = float(ineq[limiting_index])
            limiting_inequality_constraint = int(
                self.data.numerics.icc[self.meq + limiting_index]
            )
        else:
            minimum_inequality_margin = 0.0
            limiting_inequality_constraint = 0

        self.data.numerics.minimum_inequality_margin = minimum_inequality_margin
        self.data.numerics.limiting_inequality_constraint = (
            limiting_inequality_constraint
        )

        return equality_rmse, equality_max, inequality_rmse, inequality_max

    def _print_final_metrics(self, label: str):
        eq = np.asarray(self.conf[: self.meq], dtype=float)
        ineq = np.asarray(self.conf[self.meq : self.m], dtype=float)
        eq_rmse, eq_max, ineq_rmse, ineq_max = self._update_residual_diagnostics()
        violations = np.maximum(-ineq, 0.0)
        violated_count = int(np.count_nonzero(violations > 0.0))

        print(f"\n{label}")
        print(f"  equality RMSE            = {eq_rmse:.3e}")
        print(f"  max |equality residual|  = {eq_max:.3e}")
        print(f"  violated inequalities    = {violated_count}")
        print(f"  violation RMSE           = {ineq_rmse:.3e}")
        print(f"  max inequality violation = {ineq_max:.3e}")

    def _report_duration(self, start_time):
        duration = time.time() - start_time
        logger.info("Robust SciPy SLSQP duration: %.3f s", duration)
        print(f"\nTotal robust SLSQP duration: {duration:.3f} s")

    def solve(self) -> SolverOutputCondition:
        """Run Stage-1 multistart, then one Stage-2/3 fallback chain if needed."""
        self.x_0 = np.asarray(self.x_0, dtype=float).copy()
        self.n = self.x_0.shape[0]

        # Respect inactive bounds in the same way as the original implementation.
        bndl = np.asarray(self.bndl, dtype=float).copy()
        bndu = np.asarray(self.bndu, dtype=float).copy()

        if not (np.all(self.ilower) and np.all(self.iupper)):
            bndl[self.ilower == 0] = -np.inf
            bndu[self.iupper == 0] = np.inf

        # Keep the initial point in the active bounds.
        self.x_0 = np.clip(np.asarray(self.x_0, dtype=float), bndl, bndu)
        self.bounds = optimize.Bounds(lb=bndl, ub=bndu)

        eq_constraint = None
        constraints = []

        if self.meq > 0:
            eq_constraint = optimize.NonlinearConstraint(
                self.constraint_eq_vec,
                -self.EQ_CONSTRAINT_TOL,
                self.EQ_CONSTRAINT_TOL,
            )
            constraints.append(eq_constraint)

        if self.meq < self.m:
            ineq_constraint = optimize.NonlinearConstraint(
                self.constraint_ineq_vec,
                0.0,
                np.inf,
            )
            constraints.append(ineq_constraint)

        guard_constraint = optimize.NonlinearConstraint(
            self.constraint_ineq_30,
            0.0,
            np.inf,
        )

        finite_diff_step = self.data.numerics.epsfcn
        original_figure_merit = int(self.data.numerics.i_figure_merit)
        start_time = time.time()

        # ------------------------------------------------------------------
        # Stage 1: retry the original full constrained objective from several
        # initial control vectors before giving up and entering Stage 2.
        # The same physics inputs, active constraints, bounds and objective
        # are used in every attempt. Each restart has a fresh SLSQP history.
        # ------------------------------------------------------------------
        print("\n=== Stage 1: multistart full equality + inequality constrained SLSQP ===")
        print(f"finite diff step = {finite_diff_step}")
        self.data.numerics.solver_problem_type = 0

        stage1_starts = self._stage1_start_points(self.x_0, bndl, bndu)
        self._reset_stage1_recovery_candidate()
        last_result_full = None
        print(f"Stage 1 will attempt up to {len(stage1_starts)} initial guesses")

        for start_index, (label, x_start) in enumerate(stage1_starts, start=1):
            print(f"\n--- Stage 1 attempt {start_index}/{len(stage1_starts)} ({label}) ---")
            result_full = self._run_full_solve(
                constraints,
                finite_diff_step,
                x_start,
                reset_recovery_candidate=False,
            )
            last_result_full = result_full

            # A successful SciPy termination must still be checked against
            # PROCESS constraints. Nonconverged but feasible points remain
            # recovery candidates; do not label them a successful Stage 1
            # optimisation of the user's original objective.
            candidates = []
            if result_full is not None and hasattr(result_full, "x"):
                candidates.append(("terminal", np.asarray(result_full.x, dtype=float)))
            if self._stage1_best_x is not None:
                candidates.append(("best visited", self._stage1_best_x.copy()))

            for candidate_label, candidate_x in candidates:
                try:
                    self._set_final_state(candidate_x)
                    eq_rmse, eq_max, ineq_rmse, ineq_max = (
                        self._update_residual_diagnostics()
                    )
                except (ValueError, ProcessValueError, RuntimeError) as exc:
                    print(f"Stage 1 {candidate_label} validation failed: {exc}")
                    continue

                verified_feasible = (
                    np.isfinite(eq_max)
                    and np.isfinite(ineq_max)
                    and eq_max <= max(self.EQ_CONSTRAINT_TOL, self.SOLVER_TOL)
                    and ineq_max <= self.INEQ_CONSTRAINT_TOL
                )
                if verified_feasible:
                    if not (
                        candidate_label == "terminal"
                        and result_full is not None
                        and result_full.success
                    ):
                        # Useful for Stage-2 recovery, but do not mistake a
                        # merely feasible iterate for a converged Stage-1
                        # optimisation of the user's original objective.
                        print(
                            f"Stage 1 {candidate_label} point is feasible, "
                            "but the original objective solve did not converge"
                        )
                        continue

                    print(f"Stage 1 converged and verified from {label}")
                    self._print_final_metrics("Stage 1 verified converged solution")
                    self.info = SolverOutputCondition.CONVERGED
                    self.data.numerics.n_solver_iterations = int(result_full.nit)
                    self._report_duration(start_time)
                    return self.info

            if result_full is not None and result_full.success:
                print(
                    "Stage 1 reported SciPy success, but the returned point "
                    "failed direct constraint verification; trying next start"
                )

        # Only now enter Stage 2. Crucially, we retain the best recovery
        # candidate visited across *all* Stage-1 starts, not the final start.
        if self._stage1_best_x is not None:
            stage_2_start = self._stage1_best_x.copy()
            metrics = self._stage1_best_metrics or {}
            print(
                "Stage 2 start: best recovery candidate across all Stage-1 attempts "
                f"(equalities_recovered={metrics.get('equalities_recovered')}, "
                f"eq_rmse={metrics.get('equality_rmse', np.nan):.3e}, "
                f"ineq_rmse={metrics.get('inequality_rmse', np.nan):.3e})"
            )
        elif last_result_full is not None and hasattr(last_result_full, "x"):
            stage_2_start = np.asarray(last_result_full.x, dtype=float).copy()
            print("Stage 2 start: last Stage-1 terminal point (no valid tracked candidate)")
        else:
            stage_2_start = self.x_0.copy()
            print("Stage 2 start: original x0 (Stage 1 returned no usable point)")

        # ------------------------------------------------------------------
        # Stage 2: PROCESS objective 20.  Keep the governing equalities hard,
        # retain constraint 30 as a hard guard, and minimise the one-sided
        # normalised inequality-violation RMSE defined by objective_function().
        # ------------------------------------------------------------------
        print("\n=== Stage 2: objective-20 inequality recovery SLSQP ===")
        self.data.numerics.solver_problem_type = 1

        eq_constraints = [guard_constraint]
        if eq_constraint is not None:
            eq_constraints.insert(0, eq_constraint)

        self.data.numerics.i_figure_merit = 20
        self._invalidate_cache()
        try:
            result_eq = self._run_eq_solve(
                eq_constraints, finite_diff_step, stage_2_start
            )
        finally:
            # OUT.DAT should continue to describe the user's original PROCESS
            # objective, not the temporary recovery objective.
            self.data.numerics.i_figure_merit = original_figure_merit
            self._invalidate_cache()

        # Select the best Stage-2 point actually visited, rather than trusting
        # SciPy's terminal x.  Physical residuals determine whether equilibrium
        # was recovered; formal SLSQP success is secondary.
        if self._stage2_best_x is not None:
            stage_2_candidate = self._stage2_best_x.copy()
            stage_2_metrics = self._stage2_best_metrics or {}
            print(
                "Stage 2 candidate: best visited point "
                f"(equalities_recovered={stage_2_metrics.get('equalities_recovered')}, "
                f"eq_rmse={stage_2_metrics.get('equality_rmse', np.nan):.3e}, "
                f"ineq_rmse={stage_2_metrics.get('inequality_rmse', np.nan):.3e})"
            )
        elif result_eq is not None and hasattr(result_eq, "x"):
            stage_2_candidate = np.asarray(result_eq.x, dtype=float).copy()
            print("Stage 2 candidate: terminal point (no valid tracked candidate)")
        else:
            stage_2_candidate = stage_2_start.copy()
            print("Stage 2 candidate: incoming Stage-2 start")

        stage_2_candidate_valid = False
        try:
            self._set_final_state(stage_2_candidate)
            self._print_final_metrics("Stage 2 best recovered point")
            stage_2_candidate_valid = True
        except (ValueError, ProcessValueError) as exc:
            print(f"Stage 2 best point could not be fully evaluated: {exc}")

        if stage_2_candidate_valid:
            eq_max = self.data.numerics.equality_residual_max
            ineq_max = self.data.numerics.inequality_violation_max
            equality_tol = max(self.EQ_CONSTRAINT_TOL, self.SOLVER_TOL)

            if eq_max <= equality_tol:
                # We have physically recovered the governing equations.  Do not
                # send this case to Stage 3 just because SLSQP's status flag was
                # non-success.
                print(
                    "  Governing equalities recovered at the best Stage-2 point "
                    f"(formal SLSQP success={bool(result_eq and result_eq.success)})"
                )
                self.data.numerics.n_solver_iterations = (
                    int(result_eq.nit) if result_eq is not None else 0
                )

                if ineq_max <= self.INEQ_CONSTRAINT_TOL:
                    print("  Stage 2 recovered a fully feasible point")
                    self.data.numerics.solver_problem_type = 0
                    self.info = SolverOutputCondition.CONVERGED
                else:
                    print(
                        "  Governing equalities recovered, requirements remain violated"
                    )
                    self.data.numerics.solver_problem_type = 1
                    self.info = SolverOutputCondition.NO_SOLUTION

                self._report_duration(start_time)
                return self.info

        # Stage 2 genuinely did not recover the governing equations.  Hand Stage
        # 3 the best Stage-2 point visited, not the terminal SLSQP point.
        stage_3_start = stage_2_candidate
        print("Stage 3 start: best Stage-2 recovery candidate")

        # ------------------------------------------------------------------
        # Stage 3: minimise only the already-normalised governing residuals.
        # Constraint 30 remains a hard guard.  If Stage 3 recovers governing
        # equilibrium, hand that point back to Stage 2 so requirement severity
        # is minimised on the recovered equilibrium manifold.
        # ------------------------------------------------------------------
        print("\n=== Stage 3: minimum governing-residual SLSQP ===")
        self.data.numerics.solver_problem_type = 2

        result_residual = self._run_residual_solve(
            guard_constraint,
            finite_diff_step,
            stage_3_start,
        )

        if result_residual is None and self._stage3_best_x is None:
            raise ProcessValueError(
                "Residual recovery failed after full and objective-20 SLSQP "
                "stages: model evaluation failed"
            )

        if result_residual is not None and not result_residual.success:
            print(
                "Stage 3 did not report formal convergence; using the best "
                f"governing-residual point visited: {result_residual.message}"
            )

        if self._stage3_best_x is not None:
            stage_3_candidate = self._stage3_best_x.copy()
            print(
                "Stage 3 candidate: best visited governing-residual point "
                f"(eq_rmse={self._stage3_best_eq_rmse:.3e})"
            )
        elif result_residual is not None and hasattr(result_residual, "x"):
            stage_3_candidate = np.asarray(result_residual.x, dtype=float).copy()
            print("Stage 3 candidate: terminal point (no valid tracked candidate)")
        else:
            stage_3_candidate = stage_3_start.copy()

        # First evaluate only the governing equalities, which is always the
        # quantity Stage 3 is responsible for recovering.
        _, eq_conf = self._evaluate_state(
            stage_3_candidate,
            m_eval=self.meq,
            force=True,
        )
        eq = np.asarray(eq_conf[: self.meq], dtype=float)
        stage_3_eq_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        stage_3_eq_max = float(np.max(np.abs(eq))) if eq.size else 0.0
        equality_tol = max(self.EQ_CONSTRAINT_TOL, self.SOLVER_TOL)

        print(
            "Stage 3 governing residuals: "
            f"RMSE={stage_3_eq_rmse:.3e}, max={stage_3_eq_max:.3e}"
        )

        if stage_3_eq_max <= equality_tol:
            print(
                "Stage 3 recovered the governing equations; handing the best "
                "Stage-3 point back to Stage 2."
            )

            # Re-run objective 20 from the recovered equilibrium.  The Stage-2
            # tracker preserves the incoming Stage-3 point, so even if this
            # rescue optimisation terminates badly we cannot lose the recovered
            # equilibrium.
            self.data.numerics.solver_problem_type = 1
            self.data.numerics.i_figure_merit = 20
            self._invalidate_cache()
            try:
                result_eq_rescue = self._run_eq_solve(
                    eq_constraints,
                    finite_diff_step,
                    stage_3_candidate,
                )
            finally:
                self.data.numerics.i_figure_merit = original_figure_merit
                self._invalidate_cache()

            if self._stage2_best_x is not None:
                rescue_candidate = self._stage2_best_x.copy()
            elif result_eq_rescue is not None and hasattr(result_eq_rescue, "x"):
                rescue_candidate = np.asarray(result_eq_rescue.x, dtype=float).copy()
            else:
                rescue_candidate = stage_3_candidate.copy()

            self._set_final_state(rescue_candidate)
            self._print_final_metrics("Stage 2 rescue after Stage 3")

            eq_max = self.data.numerics.equality_residual_max
            ineq_max = self.data.numerics.inequality_violation_max
            self.data.numerics.n_solver_iterations = (
                int(result_eq_rescue.nit) if result_eq_rescue is not None else 0
            )

            # Because _run_eq_solve seeded its tracker with the Stage-3 point,
            # this should normally remain equality-recovered.  Keep the check
            # explicit rather than silently misclassifying a numerical failure.
            if eq_max <= equality_tol:
                if ineq_max <= self.INEQ_CONSTRAINT_TOL:
                    print("  Stage 3 -> Stage 2 rescue found a fully feasible point")
                    self.data.numerics.solver_problem_type = 0
                    self.info = SolverOutputCondition.CONVERGED
                else:
                    print(
                        "  Stage 3 -> Stage 2 rescue recovered equilibrium; "
                        "requirements remain violated"
                    )
                    self.data.numerics.solver_problem_type = 1
                    self.info = SolverOutputCondition.NO_SOLUTION

                self._report_duration(start_time)
                return self.info

            print(
                "  Stage 2 rescue did not retain the recovered governing point; "
                "falling back to the best Stage-3 diagnostic point."
            )

        # Genuine governing failure: Stage 3 could not recover the governing
        # equations within tolerance. Stage 3 is deliberately an equality-only
        # diagnostic: do not evaluate or validate any inequality constraint here.
        self.data.numerics.solver_problem_type = 2

        # Preserve the best Stage-3 point for SolverHandler output. The models
        # have already been evaluated at this point above with m_eval=self.meq.
        self.x = np.asarray(stage_3_candidate, dtype=float).copy()
        self.objf = stage_3_eq_rmse

        # SolverHandler expects a full-length constraint vector. Store only the
        # governing equality residuals and mark inequalities as unavailable.
        self.conf = np.full(self.m, np.nan, dtype=float)
        self.conf[: self.meq] = eq

        self.data.numerics.equality_residual_rmse = stage_3_eq_rmse
        self.data.numerics.equality_residual_max = stage_3_eq_max
        self.data.numerics.inequality_violation_rmse = np.nan
        self.data.numerics.inequality_violation_max = np.nan
        self.data.numerics.minimum_inequality_margin = np.nan
        self.data.numerics.limiting_inequality_constraint = 0

        print(
            "Stage 3 governing failure: inequality constraints were not "
            "evaluated at the selected governing-residual point"
        )

        self.info = SolverOutputCondition.NO_SOLUTION
        self.data.numerics.n_solver_iterations = (
            int(result_residual.nit) if result_residual is not None else 0
        )
        self._report_duration(start_time)
        return self.info


def get_solver(data: DataStructure, solver_name: str = "vmcon") -> _Solver:
    """Return a solver instance.

    Parameters
    ----------
    solver_name : str, optional
        solver to create, defaults to "vmcon"

    Returns
    -------
    _Solver
        solver to use for optimisation

    Raises
    ------
    ProcessValueError
        If solver name is not an inbuilt PROCESS solver or recognised package
    """
    solver: _Solver

    if solver_name == "vmcon":
        solver = Vmcon(data=data)
    elif solver_name == "vmcon_bounded":
        solver = VmconBounded(data=data)
    elif solver_name == "fsolve":
        solver = FSolve(data=data)
    elif solver_name == "scipy_slsqp":
        solver = ScipySLSQP(data=data)
    else:
        try:
            solver = load_external_solver(solver_name)
        except Exception as e:
            raise ProcessValueError(
                "Solver name is not an inbuilt PROCESS solver or recognised package "
                f'"{solver_name}"'
            ) from e

    return solver


def load_external_solver(package: str):
    """Attempts to load a package of name `package`.

    If a package of the name is available, return the `__process_solver__`
    attribute of that package or raise an `AttributeError`.

    Parameters
    ----------
    package: str :

    Raises
    ------
    AttributeError
        If module does not have a '__process_solver__' attribute

    """
    module = importlib.import_module(package)

    solver = getattr(module, "__process_solver__", None)

    if solver is None:
        raise AttributeError(
            f"Module {module.__name__} does not have a '__process_solver__' attribute."
        )

    return solver()