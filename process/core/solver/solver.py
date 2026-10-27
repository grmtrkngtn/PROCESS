"""An adapter for different solvers."""

import importlib
import logging
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
from process.core.solver.objectives import objective_function
from process.data_structure.numerics import (
    FiguresOfMerit,
    SolverOutputCondition,
    SolverProblemType,
)

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


class ScipySLSQP(_Solver):
    """Optimise using SciPy's SLSQP solver."""

    SOLVER_TOL = 1.0e-5
    EQ_CONSTRAINT_TOL = 1.0e-6
    MODEL_FAILURE_PENALTY = 1.0e6

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the PROCESS objective function.

        Parameters
        ----------
        x : np.ndarray
            Normalised optimisation parameter vector.

        Returns
        -------
        float
            Objective function value.
        """
        try:
            objf, _ = self.evaluators.fcnvmc1(
                len(x),
                self.m,
                x,
                self.ifail,
            )
        except (ProcessValueError, RuntimeError):
            return self.MODEL_FAILURE_PENALTY

        if not np.isfinite(objf):
            return self.MODEL_FAILURE_PENALTY

        return objf

    def equality_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the equality constraint residuals.

        Parameters
        ----------
        x : np.ndarray
            Normalised optimisation parameter vector.

        Returns
        -------
        np.ndarray
            Equality constraint residuals.
        """
        try:
            _, conf = self.evaluators.fcnvmc1(
                len(x),
                self.m,
                x,
                self.ifail,
            )
        except (ProcessValueError, RuntimeError):
            return np.full(self.meq, self.MODEL_FAILURE_PENALTY)

        equality_residuals = conf[: self.meq]

        if not np.all(np.isfinite(equality_residuals)):
            return np.full(self.meq, self.MODEL_FAILURE_PENALTY)

        return equality_residuals

    def equality_constraints_satisfied(self, x: np.ndarray) -> bool:
        """Check whether all equality constraints satisfy the PROCESS tolerance."""
        equality_residuals = self.equality_constraints(x)

        return bool(np.all(np.abs(equality_residuals) <= self.EQ_CONSTRAINT_TOL))

    def inequality_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the inequality constraint residuals.

        Parameters
        ----------
        x : np.ndarray
            Normalised optimisation parameter vector.

        Returns
        -------
        np.ndarray
            Inequality constraint residuals.
        """
        try:
            _, conf = self.evaluators.fcnvmc1(
                len(x),
                self.m,
                x,
                self.ifail,
            )
        except (ProcessValueError, RuntimeError):
            return np.full(
                self.m - self.meq,
                -self.MODEL_FAILURE_PENALTY,
            )

        inequality_residuals = conf[self.meq : self.m]

        if not np.all(np.isfinite(inequality_residuals)):
            return np.full(
                self.m - self.meq,
                -self.MODEL_FAILURE_PENALTY,
            )

        return inequality_residuals

    def inequality_violation_objective(self, x: np.ndarray) -> float:
        """Evaluate the inequality constraint violation objective.

        Parameters
        ----------
        x : np.ndarray
            Normalised optimisation parameter vector.

        Returns
        -------
        float
            Root-mean-square normalised inequality constraint violation.
        """
        try:
            # Evaluate the models and constraints at the current point.
            self.evaluators.fcnvmc1(
                len(x),
                self.m,
                x,
                self.ifail,
            )
        except (ProcessValueError, RuntimeError):
            return self.MODEL_FAILURE_PENALTY

        objective = objective_function(
            FiguresOfMerit.MIN_INEQ_CONSTRAINT_VIOLATION,
            self.data,
        )

        if not np.isfinite(objective):
            return self.MODEL_FAILURE_PENALTY

        return objective

    def equality_residual_objective(self, x: np.ndarray) -> float:
        """Evaluate the equality constraint residual objective.

        Parameters
        ----------
        x : np.ndarray
            Normalised optimisation parameter vector.

        Returns
        -------
        float
            Root-mean-square normalised equality constraint residual.
        """
        equality_residuals = self.equality_constraints(x)

        return (
            np.sqrt(np.mean(equality_residuals**2))
            if equality_residuals.size > 0
            else 0.0
        )

    def convergence_progress(self, x_current: np.ndarray) -> None:
        """Print convergence diagnostics after each SLSQP iteration."""
        self.iteration += 1

        problem_type = SolverProblemType(self.data.numerics.solver_problem_type)
        print(f"\nIteration {self.iteration} [{problem_type.name}]")

        try:
            objf, conf = self.evaluators.fcnvmc1(
                len(x_current),
                self.m,
                x_current,
                self.ifail,
            )
        except (ProcessValueError, RuntimeError) as error:
            print(f"  Invalid PROCESS trial point: {error}")
            return

        equality_residuals = conf[: self.meq]
        inequality_residuals = conf[self.meq : self.m]
        inequality_violations = np.maximum(-inequality_residuals, 0.0)

        equality_rmse = (
            np.sqrt(np.mean(equality_residuals**2))
            if equality_residuals.size > 0
            else 0.0
        )
        equality_max = (
            np.max(np.abs(equality_residuals)) if equality_residuals.size > 0 else 0.0
        )
        inequality_rmse = (
            np.sqrt(np.mean(inequality_violations**2))
            if inequality_violations.size > 0
            else 0.0
        )
        inequality_max = (
            np.max(inequality_violations) if inequality_violations.size > 0 else 0.0
        )

        if problem_type == SolverProblemType.FULLY_CONSTRAINED:
            print(f"  PROCESS objective:        {objf:.6e}")
        elif problem_type == SolverProblemType.EQUALITY_ONLY:
            print(f"  Stage objective:          {inequality_rmse:.6e}")
        else:
            print(f"  Stage objective:          {equality_rmse:.6e}")

        print(f"  Equality RMSE:           {equality_rmse:.3e}")
        print(f"  Equality max:            {equality_max:.3e}")
        print(f"  Inequality viol. RMSE:   {inequality_rmse:.3e}")
        print(f"  Inequality viol. max:    {inequality_max:.3e}")

        print("  Equality constraints:")
        if equality_residuals.size == 0:
            print("    None")
        else:
            for i in np.argsort(-np.abs(equality_residuals)):
                constraint_id = self.data.numerics.icc[i]
                print(f"    Constraint {constraint_id}: {equality_residuals[i]:+.3e}")

        violated = np.where(inequality_residuals < 0.0)[0]

        print("  Violated inequality constraints:")
        if violated.size == 0:
            print("    None")
        else:
            violated = violated[np.argsort(inequality_residuals[violated])]
            for i in violated:
                constraint_id = self.data.numerics.icc[self.meq + i]
                print(f"    Constraint {constraint_id}: {inequality_residuals[i]:+.3e}")

    def solve(self) -> SolverOutputCondition:
        """Optimise using the three-stage robust solve."""
        lower_bounds = np.where(self.ilower, self.bndl, -np.inf)
        upper_bounds = np.where(self.iupper, self.bndu, np.inf)
        bounds = optimize.Bounds(lower_bounds, upper_bounds)
        x_0 = np.clip(self.x_0, lower_bounds, upper_bounds)

        eq_constraints = optimize.NonlinearConstraint(
            self.equality_constraints,
            -self.EQ_CONSTRAINT_TOL,
            self.EQ_CONSTRAINT_TOL,
        )

        constraints = [eq_constraints]

        if self.meq < self.m:
            constraints.append(
                optimize.NonlinearConstraint(
                    self.inequality_constraints,
                    0.0,
                    np.inf,
                )
            )

        finite_diff_step = self.data.numerics.epsfcn
        finite_diff_steps = (
            finite_diff_step,
            finite_diff_step * 10.0,
            finite_diff_step / 10.0,
        )

        # Stage 1: solve the fully constrained PROCESS optimisation problem.
        self.data.numerics.solver_problem_type = SolverProblemType.FULLY_CONSTRAINED
        result = None
        stage_1_success = False

        for step in finite_diff_steps:
            self.iteration = 0
            print(
                "\nStage 1: fully constrained solve "
                f"(finite difference step = {step:.3e})"
            )

            result = optimize.minimize(
                self.objective,
                x_0,
                method="SLSQP",
                jac="3-point",
                bounds=bounds,
                constraints=constraints,
                callback=self.convergence_progress,
                options={
                    "maxiter": self.maxcal,
                    "ftol": self.SOLVER_TOL,
                    "finite_diff_rel_step": step,
                },
            )

            if result.success and self.equality_constraints_satisfied(result.x):
                stage_1_success = True
                break

        # Stage 2: if no feasible solution is found, satisfy the governing
        # equalities while minimising inequality constraint violation.
        stage_2_success = False

        if not stage_1_success:
            self.data.numerics.solver_problem_type = SolverProblemType.EQUALITY_ONLY

            for step in finite_diff_steps:
                self.iteration = 0
                print(
                    "\nStage 2: equality-only solve "
                    f"(finite difference step = {step:.3e})"
                )

                result = optimize.minimize(
                    self.inequality_violation_objective,
                    x_0,
                    method="SLSQP",
                    jac="3-point",
                    bounds=bounds,
                    constraints=eq_constraints,
                    callback=self.convergence_progress,
                    options={
                        "maxiter": self.maxcal,
                        "ftol": self.SOLVER_TOL,
                        "finite_diff_rel_step": step,
                    },
                )

                if result.success and self.equality_constraints_satisfied(result.x):
                    stage_2_success = True
                    break

        # Stage 3: if the governing equalities cannot be satisfied, minimise
        # their normalised residuals and return the closest point found.
        if not stage_1_success and not stage_2_success:
            self.data.numerics.solver_problem_type = (
                SolverProblemType.MINIMUM_EQUALITY_RESIDUAL
            )

            for step in finite_diff_steps:
                self.iteration = 0
                print(
                    "\nStage 3: minimum equality residual "
                    f"(finite difference step = {step:.3e})"
                )

                result = optimize.minimize(
                    self.equality_residual_objective,
                    x_0,
                    method="SLSQP",
                    jac="3-point",
                    bounds=bounds,
                    callback=self.convergence_progress,
                    options={
                        "maxiter": self.maxcal,
                        "ftol": self.SOLVER_TOL,
                        "finite_diff_rel_step": step,
                    },
                )

                if result.success:
                    break

        objf, conf = self.evaluators.fcnvmc1(
            len(result.x),
            self.m,
            result.x,
            self.ifail,
        )

        equality_residuals = conf[: self.meq]
        self.data.numerics.equality_residual_rmse = (
            np.sqrt(np.mean(equality_residuals**2))
            if equality_residuals.size > 0
            else 0.0
        )
        self.data.numerics.equality_residual_max = (
            np.max(np.abs(equality_residuals)) if equality_residuals.size > 0 else 0.0
        )

        inequality_residuals = conf[self.meq : self.m]
        inequality_violations = np.maximum(-inequality_residuals, 0.0)

        self.data.numerics.inequality_violation_rmse = (
            np.sqrt(np.mean(inequality_violations**2))
            if inequality_violations.size > 0
            else 0.0
        )

        self.data.numerics.inequality_violation_max = (
            np.max(inequality_violations) if inequality_violations.size > 0 else 0.0
        )

        self.x = result.x
        self.objf = objf
        self.conf = conf
        self.data.numerics.n_solver_iterations = result.nit
        if (
            self.data.numerics.solver_problem_type
            == SolverProblemType.MINIMUM_EQUALITY_RESIDUAL
        ):
            self.info = SolverOutputCondition.NO_SOLUTION
        else:
            self.info = (
                SolverOutputCondition.CONVERGED
                if result.success
                else SolverOutputCondition.NO_SOLUTION
            )
        return self.info


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
