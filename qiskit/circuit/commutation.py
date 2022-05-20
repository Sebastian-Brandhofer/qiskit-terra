"""Gate Commutation Library."""
import os
import pickle
from typing import Union
import numpy as np

from qiskit.quantum_info import Operator
from qiskit.dagcircuit import DAGOpNode

dirname = os.path.dirname(__file__)

StandardGateCommutations = pickle.load(open(dirname + "/standard_gates_commutations.p", "rb"))


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


class CommutationLibrary:
    """A library containing commutation relationships of non-parameterized standard gates."""

    def __init__(self):
        self._standard_commutations = StandardGateCommutations

    def is_op_in_library(self, gate0: DAGOpNode, gate1: DAGOpNode) -> bool:
        """Checks whether a gate is part of the commutation library.

        Args:
            gate0 (DAGOpNode): Gate to be checked.
            gate1 (DAGOpNode): Gate to be checked.

        Return:
            bool: True if gate is in commutation library, false otherwise
        """
        return (type(gate0.op), type(gate1.op)) in self._standard_commutations[type(gate0.op)]

    def get_stored_commutation_relation(self, op0: DAGOpNode, op1: DAGOpNode) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            op0 (DAGOpNode): a gate whose commutation should be checked
            op1 (DAGOpNode): a gate whose commutation should be checked

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        relative_placement = _get_relative_placement(op0, op1)
        op0op1 = self._standard_commutations.get((type(op0.op), type(op1.op)), None)
        if op0op1 is None:
            return None

        if isinstance(op0op1, bool):
            return op0op1

        return op0op1.get(relative_placement, None)

    def do_gates_commute(self, op0: DAGOpNode, op1: DAGOpNode, cache: bool = True) -> bool:
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

        # pylint: disable=unidiomatic-typecheck
        if op0.qargs == op1.qargs and type(op0.op) == type(op1.op):
            return True

        stored_commutation = self.get_stored_commutation_relation(op0, op1)

        if stored_commutation is not None:
            # TODO if op0 or op1 has params, evaluate stored_commutation
            return stored_commutation

        # Compute commutation via matrix multiplication
        is_commuting = _commute(op0, op1)

        # TODO add a LRU cache
        if cache:
            # Store result in this session's commutation_library
            self._standard_commutations.setdefault((type(op0.op), type(op1.op)), {})[
                _get_relative_placement(op0, op1)
            ] = is_commuting
            self._standard_commutations.setdefault((type(op1.op), type(op0.op)), {})[
                _get_relative_placement(op1, op0)
            ] = is_commuting


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

    op1 = np.reshape(node1.op.to_matrix(), (2, 2) * len(qarg1))
    op2 = np.reshape(node2.op.to_matrix(), (2, 2) * len(qarg2))

    op = Operator._einsum_matmul(id_op, op1, qarg1)
    op12 = Operator._einsum_matmul(op, op2, qarg2, right_mul=False)
    op21 = Operator._einsum_matmul(op, op2, qarg2, shift=qbit_num, right_mul=True)

    return np.allclose(op12, op21)
