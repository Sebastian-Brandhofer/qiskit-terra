import itertools
from functools import lru_cache
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, ControlledGate
from qiskit.circuit.commutation import _get_ops_in_order
from qiskit.circuit.commutation_library import SessionCommutationLibrary
from qiskit.circuit.library import C3SXGate, C4XGate
from qiskit.circuit.tools.symbolic_unitary_simulator import SymbolicUnitarySimulatorPy
from qiskit.dagcircuit import DAGOpNode


@lru_cache(None)
def _get_simple_gates():
    """Using module inspection, retrieve a list of non-parmaterized gates with up to 3 qubits

    Return:
        A list of simple gates to be considered in the standard gates commutation library
    """
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
            # print("Gate may have parameters", g)
            continue
    return gates


def _get_symbolic_commutator(d0, d1):
    sus = SymbolicUnitarySimulatorPy()
    qargs = set(d0.qargs + d1.qargs)

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


def _get_commutation_dict():
    """Compute the commutation relation of considered gates

    Return:
        A dictionary that includes the commutation relation for each considered pair of operations
    """
    commuting_dict = {}
    considered_gates = _get_simple_gates()
    for g0_t in considered_gates:
        if g0_t in _get_simple_gates():
            g0 = g0_t()
        else:
            g0 = g0_t
        d0 = DAGOpNode(op=g0, qargs=list(range(g0.num_qubits)), cargs=[])
        for g1_t in considered_gates:
            if g1_t in _get_simple_gates():
                g1 = g1_t()
            else:
                g1 = g1_t

            # only consider canonical entries
            if _get_ops_in_order(g0, g1) != (g0, g1):
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

                if not g0.is_parameterized() and not g1.is_parameterized():
                    is_commuting = SessionCommutationLibrary.do_operations_commute(d0, d1)
                else:
                    is_commuting = None

                if relative_placement in commute_qubit_dic:
                    assert (
                        commute_qubit_dic[relative_placement] == is_commuting
                    ), "If there is already an entry, it must be equal"
                else:
                    commute_qubit_dic[relative_placement] = is_commuting

            commuting_dict[g0.name, g1.name] = commute_qubit_dic
    return commuting_dict


def _simplify_commuting_dict(commuting_dict):
    """Write commutation dictionary as python file.

    Args:
        commuting_dict (dict): A simplified commutation dictionary

    """
    # Set bool if commutation is independent of relative placement
    for ops in commuting_dict.keys():
        gates_commutations = set(commuting_dict[ops].values())
        if len(gates_commutations) == 1:
            commuting_dict[ops] = next(iter(gates_commutations))

    return commuting_dict


def _dump_commuting_dict_as_python(commutations):
    """Write commutation dictionary as python file.

    Args:
        commutations (dict): a dictionary that includes the commutation relation for each considered pair of operations

    """
    with open("../_standard_gates_commutations.py", "w") as fp:
        dir_str = "standard_gates_commutations = {\n"
        for k, v in commutations.items():
            if isinstance(v, bool):
                dir_str += '    ("{}", "{}"): {},\n'.format(*k, v)
            else:
                dir_str += '    ("{}", "{}"): {{\n'.format(*k)

                for entry_key, entry_val in v.items():
                    dir_str += "        {}: {},\n".format(entry_key, entry_val)

                dir_str += "    },\n"
        dir_str += "}\n"
        fp.write(dir_str)


if __name__ == "__main__":
    commutation_dict = _get_commutation_dict()
    simplified_commuting_dict = _simplify_commuting_dict(commutation_dict)
    _dump_commuting_dict_as_python(simplified_commuting_dict)
