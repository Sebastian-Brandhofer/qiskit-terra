import copy
import inspect
import itertools
import pickle
from functools import lru_cache
from typing import List

from qiskit.circuit import Gate, ControlledGate, Parameter
from qiskit.circuit.commutation import _order_operations
from qiskit.circuit.commutation_library import SessionCommutationLibrary
from qiskit.circuit.library import C3SXGate, C4XGate, TGate, RGate
from qiskit.circuit.tools.param_commutations import (
    _do_parameterized_operations_commute,
    _get_param_gates,
    _postprocess_sympy_param_equations, _construct_lambda_function_string,
)
from qiskit.dagcircuit import DAGOpNode


@lru_cache(None)
def _get_unparameterizable_gates() -> List[Gate]:
    """Retrieve a list of non-parmaterized gates with up to 3 qubits, using the python inspection module

    Return:
        A list of non-parameterized gates to be considered in the commutation library
    """

    # These two gates take too long in later processing steps
    blocked_types = [C3SXGate, C4XGate]
    gates_params = [g for g in Gate.__subclasses__() if "standard_gates" in g.__module__] + [
        g for g in ControlledGate.__subclasses__() if g not in blocked_types
    ]
    gates = []
    for g in gates_params:
        try:
            # just try to instantiate gate, if this is possible, the gate is in the set of simple gates we consider,
            # i.e. it does not have parameters
            g()
            gates.append(g)
        except:
            # gate may have parameters
            continue
    return gates

#TODO generate commutation_condition class, add here to docstring
def _generate_commutation_dict(considered_gates: List[Gate] = None) -> dict:
    """Compute the commutation relation of considered gates

    Args:
        considered_gates List[Gate]: a list of gates between which the commutation should be determined

    Return:
        A dictionary that includes the commutation relation for each considered pair of operations and each relative
        placement

    """
    commutations = {}
    if considered_gates is None:
        considered_gates = _get_unparameterizable_gates() + \
                           _get_param_gates(4, exclude_gates=_get_unparameterizable_gates())

    for gate0_type in considered_gates:
        if gate0_type in _get_unparameterizable_gates():
            gate0 = gate0_type()
        else:
            gate0 = gate0_type
        node0 = DAGOpNode(op=gate0, qargs=list(range(gate0.num_qubits)), cargs=[])
        for g1_type in considered_gates:
            if g1_type in _get_unparameterizable_gates():
                gate1 = g1_type()
            else:
                gate1 = copy.deepcopy(g1_type)

            # only consider canonical entries
            if _order_operations(gate0, gate1) != (gate0, gate1) and gate0.name != gate1.name:
                continue

            #TODO remove, just for debugging purposes!
            #if not (gate0.name == "p" and gate1.name == "u2"):
            #    continue

            # enumerate all relative gate placements with overlap between gate qubits
            gate_placements = itertools.permutations(range(gate0.num_qubits + gate1.num_qubits - 1), gate0.num_qubits)
            gate_pair_commutation = {}
            for permutation in gate_placements:
                permutation_list = list(permutation)
                gate1_qargs = []

                # use idx_non_overlapping qubits to represent qubits on g1 that are not connected to g0
                next_non_overlapping_qubit_idx = gate0.num_qubits
                for i in range(gate1.num_qubits):
                    if i in permutation_list:
                        gate1_qargs.append(permutation_list.index(i))
                    else:
                        gate1_qargs.append(next_non_overlapping_qubit_idx)
                        next_non_overlapping_qubit_idx += 1

                node1 = DAGOpNode(op=gate1, qargs=gate1_qargs, cargs=[])

                # replace non-overlapping qubits with None to act as a key in the commutation library
                relative_placement = tuple([i if i < gate1.num_qubits else None for i in permutation])

                if not gate0.is_parameterized() and not gate1.is_parameterized():
                    # if no gate includes parameters, compute commutation relation using matrix multiplication
                    commutation_relation = SessionCommutationLibrary.do_operations_commute(node0, node1)
                else:
                    # if one+ gate has parameters, use sympy to determine param values for which the gates commute
                    commuting_param_eq = _do_parameterized_operations_commute(node0, node1)
                    print(
                        "[{}, {}] @{} = {}".format(
                            node0.op.name, node1.op.name, relative_placement, commuting_param_eq
                        )
                    )
                    # transform sympy equations to numpy lambda functions
                    commutation_relation = _postprocess_sympy_param_equations(commuting_param_eq)

                assert relative_placement not in gate_pair_commutation
                gate_pair_commutation[relative_placement] = commutation_relation

            commutations[gate0.name, gate1.name] = gate_pair_commutation
    return commutations


def _simplify_commuting_dict(commuting_dict):
    """Compress some of the commutation library entries

    Args:
        commuting_dict (dict): A simplified commutation dictionary

    """
    # Remove relative placement key if commutation is independent of relative placement
    for ops in commuting_dict.keys():
        gates_commutations = set(commuting_dict[ops].values())
        if len(gates_commutations) == 1:
            commuting_dict[ops] = next(iter(gates_commutations))

    return commuting_dict


def _dump_commuting_dict_as_python(commutations: dict, fn: str = "../_standard_gates_commutations.py"):
    """Write commutation dictionary as python object to ./qiskit/circuit/_standard_gates_commutations.py.

    Args:
        commutations (dict): a dictionary that includes the commutation relation for each considered pair of operations

    """
    with open(fn, "w") as fp:
        dir_str = "from numpy import *\n\n"
        dir_str += "standard_gates_commutations = {\n"
        for k, v in commutations.items():
            if not isinstance(v, dict):
                dir_str += '    ("{}", "{}"): {},\n'.format(*k, v)
            else:
                dir_str += '    ("{}", "{}"): {{\n'.format(*k)

                for entry_key, entry_val in v.items():
                    # for redumping the commutations dict
                    if isinstance(entry_val, tuple) and callable(entry_val[1]) and entry_val[1].__name__ == "<lambda>":
                        dir_str += inspect.getsource(entry_val[1])
                    else:
                        dir_str += "        {}: {},\n".format(entry_key, entry_val)

                dir_str += "    },\n"
        dir_str += "}\n"
        fp.write(dir_str.replace("'", ""))


if __name__ == "__main__":
    dirpath = "/Users/sebastianbrandhofer/gh/qiskit-terra/qiskit/circuit/tools/param_commutation_dict_mod4pi.p"
    cons_gates = [TGate, RGate(Parameter("p_0"), Parameter("p_1"))]
    commutation_dict = _generate_commutation_dict(considered_gates=cons_gates)
    #pickle.dump(commutation_dict, open(dirpath, "wb"))
    #_dump_commuting_dict_as_python(commutation_dict)
    # You may want to run _validated_commutation_library to make sure commutation_dict only contains correct entries
    # sympy does not always report all solutions
