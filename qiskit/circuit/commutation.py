"""Gate Commutation Library."""
import os
import pickle
from typing import Union
import numpy as np

from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator
from qiskit.dagcircuit import DAGOpNode, DAGDepNode

try:
    from qiskit.circuit.standard_gates_commutations import standard_gates_commutations

    StandardGateCommutations = standard_gates_commutations
except:
    print("Did not find StandardGateCommutations library!")
    StandardGateCommutations = {}


def _get_relative_placement(gate0: DAGOpNode, gate1: DAGOpNode) -> tuple:
    """Determines the relative placement of two gates. Note: this is NOT symmetric.

    Args:
        gate0 (DAGOpNode): first gate
        gate1 (DAGOpNode): second gate

    Return:
        A list that describes the placement of gate1 with respect to gate0.
        E.g. _get_relative_placement(CX(0, 1), CX(1, 2)) would return [None, 0]
    """
    qubits_g1 = {q: i for i, q in enumerate(gate1.qargs)}
    return tuple(qubits_g1.get(q, None) for i, q in enumerate(gate0.qargs))


def _get_ops_in_order(op0: Union[DAGOpNode, str], op1: [DAGOpNode, str]):
    if not isinstance(op0, str) and not isinstance(op1, str):
        return (op0, op1) if hash(op0.op.name) < hash(op1.op.name) else (op1, op0)

    if isinstance(op0, str) and isinstance(op1, str):
        return (op0, op1) if hash(op0) < hash(op1) else (op1, op0)

    op0str = op0 if isinstance(op0, str) else op0.op.name
    op1str = op1 if isinstance(op1, str) else op1.op.name
    return (op0str, op1str) if hash(op0str) < hash(op1str) else (op1str, op0str)


def _look_up_commutation_dict(
    op0: DAGOpNode, op1: DAGOpNode, _commutation_dict: dict
) -> Union[bool, None]:
    # TODO if op0 or op1 has params, evaluate stored_commutation
    first_op, second_op = _get_ops_in_order(op0, op1)
    commutation = _commutation_dict.get((first_op.op.name, second_op.op.name), None)
    if commutation is None or isinstance(commutation, bool):
        return commutation

    if isinstance(commutation, dict):
        return commutation.get(_get_relative_placement(first_op, second_op), None)
    else:
        raise ValueError("Expected commutation to be None, bool or a dict")


class CommutationLibrary:
    """A library containing commutation relationships of non-parameterized standard gates."""

    def __init__(self, cache_max_entries: int = 5e6):
        self._standard_commutations = StandardGateCommutations
        self._cached_commutations = {}
        # 5e6 entries should require less than 1GB memory
        self._cache_max_entries = cache_max_entries
        self._current_cache_entries = 0

    def is_op_in_library(self, gate0: DAGOpNode, gate1: DAGOpNode) -> bool:
        """Checks whether a gate is part of the commutation library.

        Args:
            gate0 (DAGOpNode): Gate to be checked.
            gate1 (DAGOpNode): Gate to be checked.

        Return:
            bool: True if gate is in commutation library, false otherwise
        """
        return (gate0.op.name, gate1.op.name) in self._standard_commutations

    def look_up_commutation_relation(self, op0: DAGOpNode, op1: DAGOpNode) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            op0 (DAGOpNode): a gate whose commutation should be checked
            op1 (DAGOpNode): a gate whose commutation should be checked

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        commutation = _look_up_commutation_dict(op0, op1, self._standard_commutations)
        if commutation is not None:
            return commutation

        return _look_up_commutation_dict(op0, op1, self._cached_commutations)

    def do_operations_commute(self, op0: DAGOpNode, op1: DAGOpNode, cache: bool = True) -> bool:
        """Determines the commutation relation between op0 and op1 by trying to loop up their relation in the
        commutation library or computing the relation explicitly using matrix multiplication

        Args:
            op0 (DAGOpNode): a gate whose commutation should be checked
            op1 (DAGOpNode): a gate whose commutation should be checked
            cache (bool): whether to store new commutation relations in the commutation library

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        if set(op0.qargs).isdisjoint(op1.qargs):
            return True

        if op0.qargs == op1.qargs and op0.op.name == op1.op.name:
            return True

        commutation_lookup = self.look_up_commutation_relation(op0, op1)

        if commutation_lookup is not None:
            return commutation_lookup

        # Compute commutation via matrix multiplication
        is_commuting = _commute(op0, op1)

        if cache and self._cache_max_entries > 0:
            if self._current_cache_entries >= self._cache_max_entries:
                self._cached_commutations.popitem()
                self._current_cache_entries -= 1

            # Store result in this session's commutation_library
            first_op, second_op = _get_ops_in_order(op0, op1)
            self._cached_commutations.setdefault((first_op.op.name, second_op.op.name), {})[
                _get_relative_placement(first_op, second_op)
            ] = is_commuting
            self._current_cache_entries += 1

        return is_commuting


def _commute(node1: DAGOpNode, node2: DAGOpNode) -> bool:
    """Function to verify commutation relation between two nodes in the DAG.

    Args:
        node1 (DAGnode): first node operation
        node2 (DAGnode): second node operation

    Return:
        bool: True if the nodes commute and false if it is not the case.
    """

    # Create set of qubits on which the operation acts
    qarg1 = [node1.qargs[i] for i in range(0, len(node1.qargs))]
    qarg2 = [node2.qargs[i] for i in range(0, len(node2.qargs))]

    # Create set of cbits on which the operation acts
    carg1 = [node1.cargs[i] for i in range(0, len(node1.cargs))]
    carg2 = [node2.cargs[i] for i in range(0, len(node2.cargs))]

    # Commutation for classical conditional gates
    # if and only if the qubits are different.
    # TODO: qubits can be the same if conditions are identical and
    # the non-conditional gates commute.
    if node1.op.condition or node2.op.condition:
        intersection = set(qarg1).intersection(set(qarg2))
        return not intersection

    # Commutation for non-unitary or parameterized or opaque ops
    # (e.g. measure, reset, directives or pulse gates)
    # if and only if the qubits and clbits are different.
    non_unitaries = ["measure", "reset", "initialize", "delay"]

    def _unknown_commutator(n):
        return n.op._directive or n.name in non_unitaries or n.op.is_parameterized()

    if _unknown_commutator(node1) or _unknown_commutator(node2):
        intersection_q = set(qarg1).intersection(set(qarg2))
        intersection_c = set(carg1).intersection(set(carg2))
        return not (intersection_q or intersection_c)

    # Known non-commuting gates (TODO: add more).
    non_commute_gates = [{"x", "y"}, {"x", "z"}]
    if qarg1 == qarg2 and ({node1.name, node2.name} in non_commute_gates):
        return False

    # Create matrices to check commutation relation if no other criteria are matched
    qarg = list(set(node1.qargs + node2.qargs))
    qbit_num = len(qarg)

    qarg1 = [qarg.index(q) for q in node1.qargs]
    qarg2 = [qarg.index(q) for q in node2.qargs]

    dim = 2**qbit_num
    id_op = np.reshape(np.eye(dim), (2, 2) * qbit_num)
    try:
        op1 = np.reshape(node1.op.to_matrix(), (2, 2) * len(qarg1))
    except (CircuitError, AttributeError):
        print("Op: {} has no to_matrix() method".format(node1.op))
        return False

    try:
        op2 = np.reshape(node2.op.to_matrix(), (2, 2) * len(qarg2))
    except (CircuitError, AttributeError):
        print("Op: {} has no to_matrix() method".format(node2.op))
        return False

    op = Operator._einsum_matmul(id_op, op1, qarg1)
    op12 = Operator._einsum_matmul(op, op2, qarg2, right_mul=False)
    op21 = Operator._einsum_matmul(op, op2, qarg2, shift=qbit_num, right_mul=True)

    return np.allclose(op12, op21)
