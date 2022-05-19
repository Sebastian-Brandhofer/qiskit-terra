"""Gate Commutation Library."""

import pickle

from qiskit.quantum_info import Operator

from qiskit.circuit import Instruction
from typing import Union
import numpy as np
from qiskit.dagcircuit import DAGOpNode

StandardGateCommutations = pickle.load(open("standard_gates_commutations.p", "rb"))


def get_relative_placement(gate0, gate1):
    #for i, q in sorted(gate0.qubits):
    #   dic_g0[i] = q
    # for i, q in sorted(gate1.qubits):
    #   dic_g1[q] = i
    return [dic_g1.get(dic_g0[i], None) for i in dic_g0.keys()]


class CommutationLibrary:
    """A library containing commutation relationships of non-parameterized standard gates."""

    def __init__(self):
        self._standard_commutations = StandardGateCommutations

    def is_gate_in_library(self, gate: Union[DAGOpNode, Instruction]):
        """ Checks whether a gate is part of the commutation library.

        Args:
            gate (DAGOpNode or Gate): Gate to be checked.

        Return:
            true if gate is in commutation library, false otherwise

        Raises:
            TypeError if gate parameter is not a DAGOpNode or Gate

        """
        if isinstance(gate, DAGOpNode):
            return type(gate.op) in self._standard_commutations
        elif isinstance(gate, Instruction):
            return type(gate) in self._standard_commutations
        else:
            raise TypeError("Expected a DAGOpNode or Gate object")

    def get_stored_commutation_relation(self, gate0: Instruction, gate1: Instruction):
        if not self.is_gate_in_library(gate0):
            return None

        if not self.is_gate_in_library(gate1):
            return None

        relative_placement = get_relative_placement(gate0, gate1)
        return self._standard_commutations[gate0][gate1][relative_placement]

    def do_gates_commute(self, gate0: Union[DAGOpNode, Instruction], gate1: Union[DAGOpNode, Instruction]):
        g0 = DAGOpNode.op if isinstance(gate0, DAGOpNode) else gate0
        g1 = DAGOpNode.op if isinstance(gate1, DAGOpNode) else gate1

        if set(g0.qargs).isdisjoint(g1.qargs):
            return True

        #TODO check for conditional/classical operations?
        stored_commutation = self.get_stored_commutation_relation(g0, g1)

        if stored_commutation is not None:
            return stored_commutation

        # Compute commutation via matrix multiplication
        is_commuting = _commute(g0, g1)
        # Store result in this session's commutation_library
        self._standard_commutations[type(g0)][type(g1)][get_relative_placement(g0, g1)] = is_commuting
        self._standard_commutations[type(g1)][type(g0)][get_relative_placement(g1, g0)] = is_commuting

def _commute(node1, node2):
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
    if node1.type == "op" and node2.type == "op":
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

