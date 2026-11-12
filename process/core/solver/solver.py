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


class ScipySLSQP(_Solver):
    """Jon's three-stage SciPy SLSQP solver, adapted to current PROCESS.

    Stage 1: optimise the original objective subject to all active constraints.
    Stage 2: minimise PROCESS objective 20 (L8 inequality violation) subject
             only to the governing equalities and variable bounds. Engineering
             inequalities, including C30, may be violated in this stage.
    Stage 3: minimise the RMSE of the current PROCESS equality residuals,
             subject to inequality 30 if active.

    Stage 1 makes up to three attempts: the original start and two small,
    reproducible, bound-respecting perturbations (5% of each variable range).
    All attempts use the same ``epsfcn`` finite-difference step.
    The best verified Stage-1 candidate warm-starts a single Stage-2 attempt;
    a single Stage-3 attempt follows only if equilibrium was not recovered.
    There is no finite-difference sweep or adaptive rescue chain.

    Compared with Jon's historical implementation, Stage 2 uses the current
    inequality-violation objective (FOM 20), while Stage 3 uses the current
    normalized governing-equation residuals instead of plasma derivatives.
    """

    SOLVER_TOL = 1.0e-5
    ACCEPT_EQ_TOL = 1.0e-5
    ACCEPT_INEQ_TOL = 1.0e-5
    FULL_MAXITER = 100
    EQ_MAXITER = 100
    RESIDUAL_MAXITER = 100

    # A positive raw separatrix power is a physical-domain requirement, not
    # an engineering inequality to relax in Stage 2 or add to FOM 20.
    PSEP_MIN_MW = 1.0e-3
    PSEP_SCALE_MW = 100.0

    def __init__(self, *, data: DataStructure):
        super().__init__(data=data)
        self.n = 0
        self.iteration = 0
        self._stage_name = ""
        self._solve_started_at = None

    def _evaluate(self, x: np.ndarray, *, equality_only: bool = False):
        """Call PROCESS; its evaluator maps normalized x into model state."""
        x = np.asarray(x, dtype=float)
        n_constraints = self.meq if equality_only else self.m
        return self.evaluators.fcnvmc1(self.n, n_constraints, x, self.ifail)

    def _raw_psep_from_state(self) -> float:
        """Raw power reaching the separatrix, before PROCESS regularisation.

        Both terms are updated by the existing PROCESS model evaluation. This
        avoids requiring an extra field or any modifications outside solver.py.

        Raises
        ------
        ProcessValueError
            If raw separatrix power is not finite.
        """
        physics = self.data.physics
        raw = float(physics.pscalingmw - physics.p_plasma_edge_rad_mw)
        if not np.isfinite(raw):
            raise ProcessValueError("Non-finite raw separatrix power")
        return raw

    def _psep_domain_margin(self, x: np.ndarray) -> float:
        """Hard physical-domain constraint (SLSQP convention: >= 0).

        Re-evaluate at *this* x: reading only global model state would give
        stale values when SciPy calls the constraint separately from the
        objective or requests finite-difference perturbations.
        """
        self._evaluate(x, equality_only=True)
        return (self._raw_psep_from_state() - self.PSEP_MIN_MW) / self.PSEP_SCALE_MW

    def _objective(self, x: np.ndarray) -> float:
        obj, _ = self._evaluate(x)
        return float(obj)

    def _equalities(self, x: np.ndarray) -> np.ndarray:
        _, conf = self._evaluate(x)
        return np.asarray(conf[: self.meq], dtype=float)

    def _inequalities(self, x: np.ndarray) -> np.ndarray:
        _, conf = self._evaluate(x)
        return np.asarray(conf[self.meq : self.m], dtype=float)

    def _constraint_30(self, x: np.ndarray) -> np.ndarray:
        """Evaluate only up to the active C30 guard, when present."""
        _, conf = self.evaluators.fcnvmc1(
            self.n,
            self._constraint_30_position + 1,
            np.asarray(x, dtype=float),
            self.ifail,
        )
        return np.asarray([conf[self._constraint_30_position]], dtype=float)

    def _residual(self, x: np.ndarray) -> float:
        """RMSE of the current PROCESS normalized governing equalities."""
        _, conf = self._evaluate(x, equality_only=True)
        eq = np.asarray(conf[: self.meq], dtype=float)
        return float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0

    @staticmethod
    def _metrics(conf: np.ndarray, meq: int) -> tuple[float, float, float, float]:
        eq = np.asarray(conf[:meq], dtype=float)
        ie = np.asarray(conf[meq:], dtype=float)
        violation = np.maximum(-ie, 0.0)
        return (
            float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0,
            float(np.max(np.abs(eq))) if eq.size else 0.0,
            float(np.sqrt(np.mean(violation**2))) if violation.size else 0.0,
            float(np.max(violation)) if violation.size else 0.0,
        )

    def _active_constraint_ids(self) -> np.ndarray:
        """PROCESS orders active equalities before active inequalities."""
        return np.asarray(self.data.numerics.icc[: self.m], dtype=int)

    def _print_range_normalized_variables(self, x: np.ndarray) -> None:
        """Show each active x relative to its active bounds, without model calls."""
        x = np.asarray(x, dtype=float)
        lower = np.asarray(self.bounds.lb, dtype=float)
        upper = np.asarray(self.bounds.ub, dtype=float)
        identifiers = np.asarray(self.data.numerics.ixc[: self.n], dtype=int)
        labels = self.data.numerics.lablxc

        entries = []
        for index, value in enumerate(x):
            lo, hi = lower[index], upper[index]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                fraction = (value - lo) / (hi - lo)
            elif np.isfinite(lo) and np.isfinite(hi) and hi == lo:
                fraction = 0.0
            else:
                fraction = float("nan")  # Unbounded variable has no [0, 1] position.

            number = int(identifiers[index])
            label = labels[number - 1]
            entries.append((fraction, number, label))

        print("\nRange-normalised opt params (0=lower bound, 1=upper bound):")
        for fraction, number, label in sorted(
            entries,
            key=lambda entry: (
                not np.isfinite(entry[0]),
                entry[0] if np.isfinite(entry[0]) else 0.0,
            ),
        ):
            print(f"  {fraction:.3f}: {number}, {label}")

    def _print_constraint_details(
        self, conf: np.ndarray, *, governing_only: bool = False
    ) -> None:
        """Use only the constraints already evaluated at this iterate."""
        conf = np.asarray(conf, dtype=float)
        eq = conf[: self.meq]
        ids = self._active_constraint_ids()

        print("\nEquality constraints (largest absolute residual first):")
        if eq.size:
            for index in np.argsort(-np.abs(eq)):
                print(f"  Constraint {ids[index]}: {eq[index]:+.3e}")
        else:
            print("  None active")

        if governing_only:
            print("\nInequality diagnostics: not evaluated in governing-residual stage")
            return

        inequalities = conf[self.meq : self.m]
        indices = [
            i
            for i in np.argsort(inequalities)
            if not np.isfinite(inequalities[i]) or inequalities[i] < 0.0
        ]
        print("\nViolated inequality constraints (most negative first):")
        if not indices:
            print("  None")
        for index in indices:
            print(f"  Constraint {ids[self.meq + index]}: {inequalities[index]:+.3e}")

    def _report_iteration(self, x: np.ndarray):
        """Detailed progress with one extra model evaluation per iterate.

        Stage 3 evaluates *only* governing equalities: computing engineering
        inequalities there can raise a model error at an otherwise useful
        governing-residual candidate.
        """
        self.iteration += 1
        stage = {
            "full constraints": "FULL",
            "equality-only": "EQUALITY_ONLY",
            "governing residual": "GOVERNING_RESIDUAL",
        }.get(self._stage_name, self._stage_name.upper())
        print(f"\nIteration {self.iteration} [{stage}]")

        try:
            self._print_iteration_diagnostics(x)
        except (ValueError, ProcessValueError, FloatingPointError, RuntimeError) as exc:
            print(f"  Constraint diagnostics unavailable: {exc}")
            logger.warning("SLSQP iteration diagnostic unavailable: %s", exc)

        # Does not call the physics model and does not modify the optimiser state.
        self._print_range_normalized_variables(x)

    def _print_iteration_diagnostics(self, x: np.ndarray) -> None:
        """Evaluate and display stage-specific constraint diagnostics."""
        if self._stage_name == "governing residual":
            _, conf = self._evaluate(x, equality_only=True)
            conf = np.asarray(conf, dtype=float)
            eq = conf[: self.meq]
            objective = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
            eq_max = float(np.max(np.abs(eq))) if eq.size else 0.0
            print(f"  Stage objective:          {objective:.6e}")
            print(f"  Equality RMSE:           {objective:.3e}")
            print(f"  Equality max:            {eq_max:.3e}")
            print("  Inequality viol. RMSE:   not evaluated")
            print("  Inequality viol. max:    not evaluated")
            self._print_constraint_details(conf, governing_only=True)
        else:
            objective, conf = self._evaluate(x)
            conf = np.asarray(conf, dtype=float)
            eq_rmse, eq_max, ineq_rmse, ineq_max = self._metrics(conf, self.meq)
            n_violated = int(np.count_nonzero(conf[self.meq :] < 0.0))
            print(f"  Stage objective:          {float(objective):.6e}")
            print(f"  Equality RMSE:           {eq_rmse:.3e}")
            print(f"  Equality max:            {eq_max:.3e}")
            print(f"  Inequality viol. RMSE:   {ineq_rmse:.3e}")
            print(f"  Inequality viol. max:    {ineq_max:.3e}")
            print(f"  Violated inequalities:   {n_violated}")
            self._print_constraint_details(conf)

    @staticmethod
    def _stage1_starts(x0: np.ndarray, bounds: optimize.Bounds) -> list[np.ndarray]:
        """Original start plus two reproducible bounded 5%-range perturbations.

        Perturb all non-fixed variables in normalised optimisation coordinates.
        A variable starting exactly on a bound is pushed inward. This is a
        starting-point perturbation, not a finite-difference-step change.
        """
        x0 = np.asarray(x0, dtype=float)
        lower = np.asarray(bounds.lb, dtype=float)
        upper = np.asarray(bounds.ub, dtype=float)
        finite_range = np.isfinite(lower) & np.isfinite(upper)
        span = np.zeros_like(x0)
        span[finite_range] = np.maximum(upper[finite_range] - lower[finite_range], 0.0)
        scale = np.where(finite_range, span, np.maximum(1.0, np.abs(x0)))
        fixed = finite_range & (upper <= lower)
        rng = np.random.default_rng(1729)
        starts = [x0.copy()]
        for _ in range(2):
            delta = 0.05 * scale * rng.uniform(-1.0, 1.0, size=x0.size)
            at_lower = np.isfinite(lower) & (x0 <= lower)
            at_upper = np.isfinite(upper) & (x0 >= upper)
            delta[at_lower] = np.abs(delta[at_lower])
            delta[at_upper & ~at_lower] = -np.abs(delta[at_upper & ~at_lower])
            delta[fixed] = 0.0
            starts.append(np.clip(x0 + delta, lower, upper))
        return starts

    def _candidate_rank(self, candidate) -> tuple:
        """Rank by feasibility, not by the original design objective alone.

        A recovered governing equilibrium beats an unrecovered one. Among
        equilibrium points favour smaller requirement violations. Otherwise
        minimise the largest tolerance-normalised violation. Objective breaks
        ties only after constraint quality.
        """
        _, obj, _, (eq_rmse, eq_max, ineq_rmse, ineq_max), raw_psep = candidate
        admissible = raw_psep >= self.PSEP_MIN_MW
        if eq_max <= self.ACCEPT_EQ_TOL and admissible:
            return (0, ineq_max, ineq_rmse, obj)
        return (
            1,
            max(
                eq_max / self.ACCEPT_EQ_TOL,
                ineq_max / self.ACCEPT_INEQ_TOL,
                max(self.PSEP_MIN_MW - raw_psep, 0.0) / self.PSEP_SCALE_MW,
            ),
            eq_max / self.ACCEPT_EQ_TOL,
            ineq_max / self.ACCEPT_INEQ_TOL,
            eq_rmse,
            obj,
        )

    def _run_stage(self, name, objective, constraints, step, maxiter, x0):
        """One SLSQP attempt; preserve Jon's 3-point objective Jacobian."""
        self._stage_name = name
        self.iteration = 0
        print(f"\n=== SLSQP {name}: finite-difference step {step:.3e} ===")
        attempt_start = time.perf_counter()
        try:
            result = optimize.minimize(
                objective,
                np.asarray(x0, dtype=float).copy(),
                method="SLSQP",
                jac="3-point",
                bounds=self.bounds,
                constraints=constraints,
                callback=self._report_iteration,
                options={
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": self.SOLVER_TOL,
                    "finite_diff_rel_step": step,
                },
            )
        except (ValueError, ProcessValueError, FloatingPointError, RuntimeError) as exc:
            logger.warning("SLSQP %s failed during model evaluation: %s", name, exc)
            print(f"SLSQP {name}: model evaluation error: {exc}")
            print(f"SLSQP {name}: duration={time.perf_counter() - attempt_start:.1f} s")
            return None
        print(
            f"SLSQP {name}: status={result.status}, success={result.success}, "
            f"iterations={result.nit}, nfev={result.nfev}, "
            f"duration={time.perf_counter() - attempt_start:.1f} s, "
            f"message={result.message}"
        )
        return result

    def _full_result(self, result):
        """Re-evaluate a candidate and return verified constraint diagnostics."""
        if result is None:
            return None
        try:
            obj, conf = self._evaluate(result.x)
            conf = np.asarray(conf, dtype=float)
            if (
                conf.size != self.m
                or not np.isfinite(obj)
                or not np.all(np.isfinite(conf))
            ):
                return None
            metrics = self._metrics(conf, self.meq)
            raw_psep = self._raw_psep_from_state()
            return (
                np.asarray(result.x, dtype=float).copy(),
                float(obj),
                conf,
                metrics,
                raw_psep,
            )
        except (ValueError, ProcessValueError, FloatingPointError, RuntimeError) as exc:
            logger.warning("SLSQP candidate failed re-evaluation: %s", exc)
            return None

    def _set_problem_type(self, problem_type: int):
        """Set the optional three-stage output classifier when supported."""
        nums = self.data.numerics
        if hasattr(nums, "solver_problem_type"):
            nums.solver_problem_type = problem_type

    def _write_diagnostics(self, conf: np.ndarray, problem_type: int):
        """Populate optional newer PROCESS numerical diagnostic fields."""
        nums = self.data.numerics
        self._set_problem_type(problem_type)
        eq_rmse, eq_max, ineq_rmse, ineq_max = self._metrics(conf, self.meq)
        fields = {
            "equality_residual_rmse": eq_rmse,
            "equality_residual_max": eq_max,
            "inequality_violation_rmse": ineq_rmse,
            "inequality_violation_max": ineq_max,
        }
        inequalities = np.asarray(conf[self.meq :], dtype=float)
        if inequalities.size and np.any(np.isfinite(inequalities)):
            idx = int(np.nanargmin(inequalities))
            fields["minimum_inequality_margin"] = float(inequalities[idx])
            fields["limiting_inequality_constraint"] = int(nums.icc[self.meq + idx])
        else:
            fields["minimum_inequality_margin"] = float("nan")
            fields["limiting_inequality_constraint"] = 0
        for name, value in fields.items():
            if hasattr(nums, name):
                setattr(nums, name, value)

    def _accept(self, x, obj, conf, *, problem_type, iterations, converged=False):
        self.x = np.asarray(x, dtype=float).copy()
        self.conf = np.asarray(conf, dtype=float).copy()
        self.objf = float(obj)
        self.info = (
            SolverOutputCondition.CONVERGED
            if problem_type == 0 and converged
            else SolverOutputCondition.NO_SOLUTION
        )
        self.data.numerics.n_solver_iterations = int(iterations)
        self._write_diagnostics(self.conf, problem_type)
        eq_rmse, eq_max, ineq_rmse, ineq_max = self._metrics(self.conf, self.meq)
        print(
            f"\nSelected SLSQP result: problem_type={problem_type}, "
            f"converged={converged}, "
            f"eq_rmse={eq_rmse:.3e}, eq_max={eq_max:.3e}, "
            f"ineq_rmse={ineq_rmse:.3e}, ineq_max={ineq_max:.3e}"
        )
        if problem_type != 2:
            self._print_constraint_details(self.conf)
        else:
            self._print_constraint_details(self.conf[: self.meq], governing_only=True)
        if self._solve_started_at is not None:
            print(
                f"Total SciPy SLSQP duration: "
                f"{time.perf_counter() - self._solve_started_at:.3f} s"
            )
        return self.info

    def solve(self) -> SolverOutputCondition:
        """Run the three-stage SLSQP solve with bounded Stage-1 starts.

        Raises
        ------
        ProcessValueError
            If the finite-difference step is invalid or the final
            governing-residual stage fails to return valid residuals.
        """
        self._solve_started_at = time.perf_counter()
        self.n = len(self.x_0)
        lower = np.asarray(self.bndl, dtype=float).copy()
        upper = np.asarray(self.bndu, dtype=float).copy()
        lower[np.asarray(self.ilower) == 0] = -np.inf
        upper[np.asarray(self.iupper) == 0] = np.inf
        self.bounds = optimize.Bounds(lower, upper)
        x0 = np.clip(np.asarray(self.x_0, dtype=float), lower, upper)

        epsfcn = float(self.data.numerics.epsfcn)
        if not np.isfinite(epsfcn) or epsfcn <= 0.0:
            raise ProcessValueError("SLSQP requires positive finite numerics.epsfcn")

        # PROCESS returns equalities first, with inequalities >= 0 on feasibility.
        eq_constraint = optimize.NonlinearConstraint(self._equalities, 0.0, 0.0)
        ineq_constraint = (
            optimize.NonlinearConstraint(self._inequalities, 0.0, np.inf)
            if self.meq < self.m
            else None
        )
        # The physical domain is hard in ALL stages. Unlike the engineering
        # inequalities it must NOT enter FOM 20 or the violation diagnostics.
        psep_domain = optimize.NonlinearConstraint(self._psep_domain_margin, 0.0, np.inf)
        full_constraints = [
            c for c in (eq_constraint, ineq_constraint, psep_domain) if c is not None
        ]

        active = np.asarray(self.data.numerics.icc[: self.m], dtype=int)
        con30_pos = np.flatnonzero(active == 30)
        self._constraint_30_position = int(con30_pos[0]) if con30_pos.size else None
        guard = (
            optimize.NonlinearConstraint(self._constraint_30, 0.0, np.inf)
            if self._constraint_30_position is not None
            else None
        )
        # Stage 2 must allow violations of *all* engineering inequalities.
        # The C30 guard remains only for Stage 3, where it limits excursions
        # during unconstrained governing-residual minimisation.
        equality_constraints = [eq_constraint, psep_domain]
        guard_constraints = ([guard] if guard is not None else []) + [psep_domain]

        starts = self._stage1_starts(x0, self.bounds)
        print(
            "\n=== Jon's SciPy SLSQP (three starting points) ===\n"
            f"{self.n} iteration variables, {self.meq} equalities, "
            f"{self.m - self.meq} inequalities; "
            f"Stage-1 starts={len(starts)} (original + 2 bounded 5% wiggles); "
            f"finite_diff_rel_step={epsfcn:.3e} (fixed); "
            "Stage 2=FOM 20 / equalities only (1 attempt); "
            "Stage 3=governing residual (1 attempt); "
            f"Stage-3 C30 guard={'active' if guard is not None else 'inactive'}; "
            f"raw Psep >= {self.PSEP_MIN_MW:g} MW in all stages"
        )

        # Each stage's candidate: (fully evaluated candidate, iteration count).
        # Retain the best *constraint-quality* candidate from all Stage-1 tries.
        best_stage1 = None
        self._set_problem_type(0)
        for attempt, start in enumerate(starts, start=1):
            print(f"\n--- Stage 1 attempt {attempt}/{len(starts)} ---")
            if attempt > 1:
                print(f"  max |start - original| = {np.max(np.abs(start - x0)):.3e}")
            result = self._run_stage(
                "full constraints",
                self._objective,
                full_constraints,
                epsfcn,
                self.FULL_MAXITER,
                start,
            )
            candidate = self._full_result(result)
            if candidate is None:
                continue
            _, eq_max, _, ineq_max = candidate[3]
            print(
                f"  Stage 1 candidate: eq_max={eq_max:.3e}, "
                f"ineq_max={ineq_max:.3e}, "
                f"Psep-domain={'OK' if candidate[4] >= self.PSEP_MIN_MW else 'FAILED'}, "
                f"SLSQP success={result.success}"
            )
            if best_stage1 is None or self._candidate_rank(
                candidate
            ) < self._candidate_rank(best_stage1[0]):
                best_stage1 = (candidate, int(result.nit))
                print("  Best Stage 1 candidate updated")

            # True optimisation convergence requires SLSQP success AND
            # independently rechecked governing/engineering feasibility.
            if (
                result.success
                and eq_max <= self.ACCEPT_EQ_TOL
                and ineq_max <= self.ACCEPT_INEQ_TOL
                and candidate[4] >= self.PSEP_MIN_MW
            ):
                print(f"Stage 1 attempt {attempt} converged and is verified feasible")
                return self._accept(
                    *candidate[:3],
                    problem_type=0,
                    iterations=result.nit,
                    converged=True,
                )

        # Stage 2: use the closest-to-feasible Stage-1 endpoint. If all three
        # calls threw model errors, retain original x0 as a safe fallback start.
        stage2_start = best_stage1[0][0].copy() if best_stage1 is not None else x0.copy()
        if best_stage1 is not None:
            _, eq_max, _, ineq_max = best_stage1[0][3]
            print(
                "\nStage 2 warm start: best Stage 1 candidate "
                f"(eq_max={eq_max:.3e}, ineq_max={ineq_max:.3e})"
            )
        else:
            print("\nStage 2 warm start: original x0 (no valid Stage 1 endpoints)")

        # Stage 2: solve the governing equalities but minimise the L8 norm of
        # inequality *violations*. Do not enforce any engineering inequality as
        # a hard constraint: a point with valid governing physics but violated
        # requirements is precisely a legitimate Stage-2 result.
        #
        # Restore the user's original FOM *before* re-evaluating the candidate,
        # so the stored objective and downstream PROCESS output are consistent.
        self._set_problem_type(1)
        original_fom = self.data.numerics.i_figure_merit
        self.data.numerics.i_figure_merit = 20  # MIN_INEQ_CONSTRAINT_VIOLATION
        print(
            "\nStage 2: minimise inequality violations (FOM 20, L8); "
            "hard constraints = governing equalities and variable bounds"
        )
        try:
            result2 = self._run_stage(
                "equality-only",
                self._objective,
                equality_constraints,
                epsfcn,
                self.EQ_MAXITER,
                stage2_start,
            )
            if result2 is not None and np.isfinite(result2.fun):
                print(f"  Stage 2 final FOM 20: {float(result2.fun):.6e}")
        finally:
            self.data.numerics.i_figure_merit = original_fom

        # Both the original objective and constraint values are now evaluated
        # with the original FOM restored. No engineering requirement is used as
        # a Stage-2 acceptance condition; only governing equalities are checked.
        candidate2 = self._full_result(result2)

        # Keep any equilibrium found during Stage 1 even if Stage 2 fails.
        equilibrium_candidates = []
        if (
            best_stage1 is not None
            and best_stage1[0][3][1] <= self.ACCEPT_EQ_TOL
            and best_stage1[0][4] >= self.PSEP_MIN_MW
        ):
            equilibrium_candidates.append(best_stage1)
        if (
            candidate2 is not None
            and candidate2[3][1] <= self.ACCEPT_EQ_TOL
            and candidate2[4] >= self.PSEP_MIN_MW
        ):
            equilibrium_candidates.append((candidate2, int(result2.nit)))

        if equilibrium_candidates:
            selected, iterations = min(
                equilibrium_candidates,
                key=lambda item: self._candidate_rank(item[0]),
            )
            # Re-evaluate because subsequent PROCESS calls mutate model state.
            restored = self._full_result(optimize.OptimizeResult(x=selected[0]))
            if (
                restored is not None
                and restored[3][1] <= self.ACCEPT_EQ_TOL
                and restored[4] >= self.PSEP_MIN_MW
            ):
                ineq_max = restored[3][3]
                problem_type = 0 if ineq_max <= self.ACCEPT_INEQ_TOL else 1
                if self._constraint_30_position is not None:
                    c30 = restored[2][self._constraint_30_position]
                    if c30 < -self.ACCEPT_INEQ_TOL:
                        print(
                            f"  Note: C30 is violated ({c30:.3e}); "
                            "engineering inequalities are allowed in Stage 2"
                        )
                print(
                    "Recovered governing equilibrium; returning best requirement state"
                )
                # A recovered feasible point is not proof of Stage-1 optimality.
                return self._accept(
                    *restored[:3],
                    problem_type=problem_type,
                    iterations=iterations,
                )

        # Stage 3: one minimisation of governing residuals from the most
        # promising available Stage 1/2 point. No finite-difference sweep.
        self._set_problem_type(2)
        stage3_start = stage2_start
        if candidate2 is not None and (
            best_stage1 is None
            or self._candidate_rank(candidate2) < self._candidate_rank(best_stage1[0])
        ):
            stage3_start = candidate2[0].copy()
        print("\nStage 3: one governing-residual minimisation")
        result3 = self._run_stage(
            "governing residual",
            self._residual,
            guard_constraints,
            epsfcn,
            self.RESIDUAL_MAXITER,
            stage3_start,
        )
        if result3 is None:
            raise ProcessValueError(
                "SLSQP governing-residual stage returned no valid point"
            )
        try:
            _, eq_conf = self._evaluate(result3.x, equality_only=True)
            eq = np.asarray(eq_conf[: self.meq], dtype=float)
            raw_psep = self._raw_psep_from_state()
        except (ValueError, ProcessValueError, FloatingPointError, RuntimeError) as exc:
            raise ProcessValueError(
                f"SLSQP Stage 3 final evaluation failed: {exc}"
            ) from exc
        if eq.size != self.meq or not np.all(np.isfinite(eq)):
            raise ProcessValueError("Stage 3 produced invalid governing residuals")

        x = np.asarray(result3.x, dtype=float).copy()
        if (
            float(np.max(np.abs(eq))) if eq.size else 0.0
        ) <= self.ACCEPT_EQ_TOL and raw_psep >= self.PSEP_MIN_MW:
            restored = self._full_result(result3)
            if restored is not None and restored[4] >= self.PSEP_MIN_MW:
                ineq_max = restored[3][3]
                problem_type = 0 if ineq_max <= self.ACCEPT_INEQ_TOL else 1
                print("Stage 3 recovered equilibrium; classifying requirements")
                return self._accept(
                    *restored[:3],
                    problem_type=problem_type,
                    iterations=result3.nit,
                )

        # Engineering inequalities may be undefined outside governing equilibrium.
        conf = np.full(self.m, np.nan, dtype=float)
        conf[: self.meq] = eq
        eq_rmse = float(np.sqrt(np.mean(eq**2))) if eq.size else 0.0
        print(
            "Stage 3 returned a governing-equation residual "
            "(physically admissible equilibrium not recovered)"
        )
        return self._accept(
            x,
            eq_rmse,
            conf,
            problem_type=2,
            iterations=result3.nit,
        )


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
