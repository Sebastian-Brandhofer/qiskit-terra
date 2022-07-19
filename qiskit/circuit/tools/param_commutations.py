import inspect
import multiprocessing
import pickle
from multiprocessing import Process
from typing import List

import sympy
from sympy import solve, zoo, lambdify, solveset, pprint, nonlinsolve, trigsimp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, ControlledGate, ParameterVector, Parameter
from qiskit.circuit.library import C3SXGate, C4XGate
from qiskit.circuit.tools.symbolic_unitary_simulator import SymbolicUnitarySimulatorPy
import numpy as np

from qiskit.dagcircuit import DAGOpNode


def _handle_corner_case(entry):
    if isinstance(entry, str):
        return True, None

    if isinstance(entry, bool):
        return True, entry

    if isinstance(entry, list) and len(entry) == 0:
        return True, True

    # this is sympy's way of telling us the solution is the empty set
    if len(entry) == 2 and len(entry[0]) == 0 and len(entry[1]) == 0:
        return True, False

    if len(entry) == 2 and zoo in next(iter(entry[1])):
        return True, False

    return False, None


def _sympy_expressions_string(sympy_expr, precision=6):
    simplified_expr = set()
    for expr_tuple in sympy_expr:
        curated_expr = []
        for expr in expr_tuple:
            expr_str = str(expr)
            try:
                param_val = float(expr_str)
                param_val = round(param_val % (4 * np.pi), precision)
                if np.abs(param_val) < 10**-precision:
                    param_val = 0
                param_val = str(param_val)
            except:
                param_val = expr_str
            curated_expr.append(param_val.replace("I", "1j"))
        simplified_expr.add(tuple(curated_expr))
    return simplified_expr


def _construct_lambda_function_string(params, sympy_expr):
    sympy_expressions = _sympy_expressions_string(sympy_expr)
    lambda_definition = (
        "lambda "
        + ", ".join(map(str, params))
        + ": ["
        + ", ".join(map(str, sympy_expressions))
        + "]"
    )
    return lambda_definition.replace("'", "")


def _postprocess_sympy_param_equations(entry):
    if isinstance(entry, bool):
        return entry

    is_corner_case, result = _handle_corner_case(entry)
    #is_corner_case, result = False, None
    if is_corner_case:
        return result
    else:
        # This must be a regular lambda function
        try:
            # sympy reports params with assignment in entry[0] and expressions in entry[1], expressions may contain
            # params that are not assigned
            assert isinstance(entry, tuple) and len(entry) == 2
            params = list(
                set(entry[0]).union(set([s for e in entry[1] for el in e for s in el.free_symbols]))
            )
            # make sure params are in order
            params.sort(key=lambda x: int(str(x).split("_")[1]))
            lambda_str = _construct_lambda_function_string(params, entry[1])
            assigned_parms = tuple([int(str(e).split("_")[1]) for e in entry[0]])
            return assigned_parms, lambda_str
            # print(f"g0={g0}, g1={g1} @{placement} from {inspect.getsource(f)} to {to_lambda_function_str(f)}")
        except Exception as e:
            print(f"err1={e}")
            # return str(e).replace("lambda", "")
            return None


def _get_symbolic_commutator(d0: DAGOpNode, d1: DAGOpNode):
    """Compute the symbolic commutator of two operations using the SymbolicUnitarySimulator
    Args:
        d0 (DAGOpNode): first operation in the considered pairs of operations
        d1 (DAGOpNode): second operation in the considered pairs of operations
    Return:
        The symbolic commutator of the input operations
    """
    qargs = set(d0.qargs + d1.qargs)
    param_offset = 0
    d0_params = [Parameter("p_{}".format(param_offset+i)) for i in range(len(d0.op.params))]
    param_offset += len(d0_params)
    d1_params = [Parameter("p_{}".format(param_offset+i)) for i in range(len(d1.op.params))]

    d0.op.params = d0_params
    d1.op.params = d1_params
    sus = SymbolicUnitarySimulatorPy()
    g0g1 = QuantumCircuit(len(qargs))
    g0g1.append(d0.op, d0.qargs)
    g0g1.append(d1.op, d1.qargs)
    g0g1_u = sus.run_experiment(g0g1)

    sus = SymbolicUnitarySimulatorPy()
    g1g0 = QuantumCircuit(len(qargs))
    g1g0.append(d1.op, d1.qargs)
    g1g0.append(d0.op, d0.qargs)
    g1g0_u = sus.run_experiment(g1g0)
    return g0g1_u - g1g0_u


def _do_parameterized_operations_commute(d0: DAGOpNode, d1: DAGOpNode):
    """Determines the set of parameter constraints that must hold for two operations to commute.
    Args:
        d0 (DAGOpNode): first operation in the pair of operations
        d1 (DAGOpNode): second operation in the pair of operations
    """

    timeout_in_min = 10
    # get the commutator as a symbolic expression
    symbolic_commutator = _get_symbolic_commutator(d0, d1)
    commutation_equations = symbolic_commutator.reshape(
        symbolic_commutator.nrows() * symbolic_commutator.ncols(), 1
    )
    symbols = list(commutation_equations.free_symbols)
    # print("Solving for [{}, {}]".format(d0.op.name, d1.op.name))

    # if there are no free symbols in the symbolic commutator, the commutator does not depend on any parameter, i.e.
    # the gates are either commuting or not without the parameters having an impact
    if len(symbols) == 0:
        return all([r == 0 for r in commutation_equations])

    simplified_commutation_equations = None
    try:
        manager = multiprocessing.Manager()
        sympy_results = manager.list()

        def simplify_commutation_equations(simplified_commutation_conditions):
            try:
                simplified_commutation_conditions.append([trigsimp(s) for s in commutation_equations])
            except Exception as e:
                print("sympy trigsimp error", e)
                pass

        proc = Process(target=simplify_commutation_equations, args=(sympy_results,))
        proc.start()
        proc.join(timeout=60 * timeout_in_min)
        proc.terminate()

        if len(sympy_results) > 0:
            simplified_commutation_equations = sympy_results[0]

    except:
        print("Simplification of trigonometric functions failed!")
        simplified_commutation_equations = None

    if simplified_commutation_equations is not None:
        # symbolic expression evaluates to 0, params do not matter, operation do always commute
        if all([e == 0 for e in simplified_commutation_equations]):
            return True

        # The simplification may remove some free symbols
        symbols = list(set([s for eq in simplified_commutation_equations for s in eq.free_symbols]))

        if len(symbols) == 0:
            return all([r == 0 for r in simplified_commutation_equations])
    else:
        simplified_commutation_equations = commutation_equations


    try:
        manager = multiprocessing.Manager()
        sympy_results = manager.list()

        def sympy_solve_commutation_equations(commutation_conditions):
            try:
                commutation_conditions.append(solve(simplified_commutation_equations, symbols, set=True))
            except Exception as e:
                commutation_conditions.append("unknown: {}".format(e))

        proc = Process(target=sympy_solve_commutation_equations, args=(sympy_results,))
        proc.start()
        proc.join(timeout=60 * timeout_in_min)
        proc.terminate()

        if len(sympy_results) > 0:
            # print(sympy_results)
            return sympy_results[0]
        else:
            return "timeout"
    except:
        return "unknown"


def _get_param_gates(max_params: int, exclude_gates: List[Gate]) -> List[Gate]:
    """Returns a list of Gates that have at most max_params many input parameters
    Args:
        max_params (int): maximum number of considered parameters
        exclude_gates (List[Gates]): do not return these gates, even if they are parameterized
    """
    # do not consider gates in blocked gate types due to prohibitive runtime overhead
    blocked_gate_types = [C3SXGate, C4XGate]

    gate_params = [g for g in Gate.__subclasses__() if "standard_gates" in g.__module__] + [
        g for g in ControlledGate.__subclasses__() if g not in blocked_gate_types
    ]

    parameterized_gates = []
    gate_idx = 0
    for gate_type in gate_params:
        if gate_type in exclude_gates:
            continue

        # get minimum number of params for this gate
        for i in range(1, max_params):
            try:
                gate = gate_type(*ParameterVector("test_{}".format(gate_idx), length=i))
                parameterized_gates.append(gate)
                break
            except:
                pass
        gate_idx += 1

    return parameterized_gates
