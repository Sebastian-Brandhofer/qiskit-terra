import copy
import itertools
import pickle
from typing import List

from qiskit.circuit import Gate, ControlledGate, ParameterVector
from qiskit.circuit.commutation import _order_operations, _commute, StandardGateCommutations
from qiskit.circuit.library import C3SXGate, C4XGate
from qiskit.circuit.tools.build_standard_commutations import _get_unparameterizable_gates, \
    _dump_commuting_dict_as_python
from qiskit.dagcircuit import DAGOpNode
from qiskit.dagcircuit.dagdependency import _does_commute
import numpy as np


def _get_param_gates(max_params: int, exclude_gates: List[Gate]) -> dict:
    """Get parameterized gates as a dictionary that indicates the number of parameters for each gate as a key
    Args:
        max_params (int): maximum number of parameters
        exclude_gates (List[Gate]): excluded quantum gates from consideration
    Return:
        A dictionary include the number of parameters as a key and a list of Quantum Gates with the defined number of
        parameters as value.
    """
    blocked_types = [C3SXGate, C4XGate]

    gates_params = [g for g in Gate.__subclasses__() if "standard_gates" in g.__module__] + [
        g for g in ControlledGate.__subclasses__() if g not in blocked_types
    ]

    param_gates = {}
    gidx = 0
    for g_t in gates_params:
        if g_t in exclude_gates:
            continue
        for i in reversed(range(1, max_params)):
            # get minimum number of params for this gate
            try:
                g = g_t(*ParameterVector("test_{}".format(gidx), length=i))
                param_gates.setdefault(i, []).append(g)
                break
            except:
                pass
        gidx += 1

    return param_gates


def get_num_params(g, param_dict):
    """Get the number of parameters of a Gate g by looking through the parameterized gate dictionary
    Args:
        g (Gate): gate whose number of parameters should be looked up
        param_dict (dict): num_parameters:Gates dict
    Return:
        The number of parameters of a quantum gate
    """
    for num_params, gs in param_dict.items():
        if g.name in gs:
            return num_params
    return 0


def get_param_space(num_params, data_points=5):
    """Compute a set of parameters in [0, 2*np.pi] that should be considered for num_params many parameters
    Args:
        num_params: Number of parameters for which values should be generated
        data_points: number of data points in [0, 2*pi] for each parameter
    Return:
        A set of parameters that should be evaluated for a pair of gates
    """
    pispace = np.linspace(0, 4*np.pi, data_points)
    return itertools.product(*[pispace] * num_params)


def _prune_incorrect_commutations(commutations, valid):
    for k, v in commutations.items():
        for k0, v0 in v.items():

            if k in valid and k0 in valid[k]:
            #print(f"k={k} v={v} k0={k0} v0={v0}")
                if valid[k][k0] == "incorrect":
                    v[k0] = None
    return commutations


def _validated_commutations():
    params = {i+1: list(get_param_space(i+1, data_points=13)) for i in range(6)}
    params[0] = []
    param_dict = _get_param_gates(4, exclude_gates=_get_unparameterizable_gates())
    considered_gates = _get_unparameterizable_gates() + [g for gs in param_dict.values() for g in gs]
    validated_gates_lib = {}
    param_dict = {k: [g.name for g in gs] for k, gs in param_dict.items()}
    print("Going through library")
    for g0_t in considered_gates:
        if g0_t in _get_unparameterizable_gates():
            g0 = g0_t()
        else:
            g0 = g0_t
        d0 = DAGOpNode(op=g0, qargs=list(range(g0.num_qubits)), cargs=[])
        for g1_t in considered_gates:
            if g1_t in _get_unparameterizable_gates():
                g1 = g1_t()
            else:
                g1 = copy.deepcopy(g1_t)

            # only consider canonical entries
            if _order_operations(g0, g1) != (g0, g1) and g0.name != g1.name:
                continue

            # all possible combinations of overlap between g1 and g0 qubits (-1 otherwise we needlessly consider completely disjunct cases)
            combinations = itertools.permutations(
                range(g0.num_qubits + g1.num_qubits - 1), g0.num_qubits
            )
            commute_qubit_dic = {}
            for permutation in combinations:
                permutation_list = list(permutation)
                g1_qargs = []

                # use idx_non_overlapping qubits to represent qubits on g1 that are not connected to g0
                idx_non_overlapping_qubits = g0.num_qubits
                for i in range(g1.num_qubits):
                    if i in permutation_list:
                        g1_qargs.append(permutation_list.index(i))
                    else:
                        g1_qargs.append(idx_non_overlapping_qubits)
                        idx_non_overlapping_qubits += 1

                d1 = DAGOpNode(op=g1, qargs=g1_qargs, cargs=[])

                relative_placement = tuple([i if i < g1.num_qubits else None for i in permutation])

                if "u3" in g0.name or "u3" in g1.name:
                    validated_gates_lib.setdefault((d0.op.name, d1.op.name), {})[relative_placement] = "incorrect"
                    break

                g0_num_params = get_num_params(d0.op, param_dict)
                g1_num_params = get_num_params(d1.op, param_dict)
                #print(g0.name)
                #print(g1.name)
                #print()
                #print(g0_num_params)
                #print(g1_num_params)
                #print()
                correct = True
                for param in params[g0_num_params + g1_num_params]:

                    pr = list(param)
                    #print(pr)
                    d0.op.params = pr[:g0_num_params]
                    d1.op.params = pr[g0_num_params: g0_num_params + g1_num_params]
                    commutes_matmul = _commute(d0, d1)
                    commutes_lib = _does_commute(d0, d1)
                    if commutes_lib != commutes_matmul:
                        validated_gates_lib.setdefault((d0.op.name, d1.op.name), {})[relative_placement] = "incorrect"
                        print(
                            f"bug @[{d0.op.name},{d1.op.name}][{relative_placement}] mmul={commutes_matmul} lib={commutes_lib} with params: {pr}")
                        correct = False
                        break
                    #else:
                if correct:
                    #print(f"correct @[{d0.op.name},{d1.op.name}][{relative_placement}]")
                    validated_gates_lib.setdefault((d0.op.name, d1.op.name), {})[
                        relative_placement] = "correct"
    # Stats
    cnt = 0
    crct = 0
    for v in validated_gates_lib.values():
        for v0 in v.values():
            cnt += 1
            if v0 == "correct":
                crct += 1

    print(f"Went through {cnt} commutation entries out of which {crct} where correct")
    return validated_gates_lib


if __name__ == "__main__":
    valid_dic = _validated_commutations()
    #pickle.dump(valid_dic, open("valid2.p", "wb"))
    valid_dic = pickle.load(open("valid2.p", "rb"))
    sgc = _prune_incorrect_commutations(StandardGateCommutations, valid=valid_dic)
    _dump_commuting_dict_as_python(sgc, "../_standard_gates_commutations_pruned.py")

