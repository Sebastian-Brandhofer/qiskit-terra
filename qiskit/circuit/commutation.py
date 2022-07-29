"""Gate Commutation Library."""
import copy
import os
import pickle
from functools import lru_cache
from typing import Union
import numpy as np

from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator
from qiskit.dagcircuit import DAGOpNode

# Make the commutation library available as a singleton object
try:
    from qiskit.circuit._standard_gates_commutations import standard_gates_commutations
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
        E.g. _get_relative_placement(CX(0, 1), CX(1, 2)) would return [None, 0] as there is no overlap
        on the first qubit of the first gate but there is an overlap on the second qubit of the first gate, i.e. qubit 0
        of the second gate. _get_relative_placement(CX(1, 2), CX(0, 1)) would return [1, None]
    """
    qubits_g1 = {q_g1: i_g1 for i_g1, q_g1 in enumerate(gate1.qargs)}
    return tuple(qubits_g1.get(q_g0, None) for q_g0 in gate0.qargs)


@lru_cache(maxsize=10**3)
def _persistent_id(op_name: str) -> int:
    """Returns an integer id of a string that is persistent over different python executions (not that
        hash() can not be used, i.e. its value can change over two python executions)
    Args:
        op_name (str): The string whose integer id should be determined.
    Return:
        The integer id of the input string.
    """
    return int.from_bytes(bytes(op_name, encoding="ascii"), byteorder="big", signed=True)


def _order_operations(op0: Union[DAGOpNode, Instruction], op1: [DAGOpNode, Instruction]):
    """Orders two operations in a canonical way that is persistent over different python versions and executions
    Args:
        op0 (Union[DAGOpNode, Instruction]): one of the two operations to be ordered
        op1 (Union[DAGOpNode, Instruction]): one of the two operations to be ordered
    Return:
        The input operations in a persistent, canonical order.
    """
    if not isinstance(op0, Instruction) and not isinstance(op1, Instruction):
        least_qubits_op, most_qubits_op = (
            (op0, op1) if len(op0.qargs) < len(op1.qargs) else (op1, op0)
        )
        # prefer operation with least number of qubits as first key
        if len(op0.qargs) != len(op1.qargs):
            return least_qubits_op, most_qubits_op
        else:
            return (op0, op1) if _persistent_id(op0.op.name) < _persistent_id(op1.op.name) else (op1, op0)
    else:
        least_qubits_op, most_qubits_op = (
            (op0, op1) if op0.num_qubits < op1.num_qubits else (op1, op0)
        )
        # prefer operation with least number of qubits as first key
        if op0.num_qubits != op1.num_qubits:
            return least_qubits_op, most_qubits_op
        else:
            return (op0, op1) if _persistent_id(op0.name) < _persistent_id(op1.name) else (op1, op0)


def mod4pi(x):
    """Gets the real part of x, truncates it to zero if its magnitude is less than 1e-6 and then returns it modulo 4Pi
    """
    x = np.real(x)
    x = x if np.abs(x) > 1e-6 else 0
    return x % (4 * np.pi)


def _evaluate_parameterized_commutation(first_op, second_op, commutation):
    """Evaluates for a pair of operations whether their assigned parameters allow the operations to commute
    Args:
        first_op (DAGOpNode): first operation of the considered pairs of operations
        second_op (DAGOpNode): second operation of the considered pairs of operations
        commutation: True or False, describing the operation's commutation or a tuple of (indices, equations) where
        indices describe which operation parameters are used in equations and where equations is a lambda function
        that returns a list of commuting parameter assignments.
    Return:
        True if operations are commuting with their provided parameters, False otherwise
    """
    if not isinstance(commutation, tuple):
        return commutation

    lambda_function = commutation[1]
    bound_gate_parameters = list(map(mod4pi, [(first_op.op.params + second_op.op.params)[p] for p in commutation[0]]))
    determined_param_assignments = lambda_function(*bound_gate_parameters)
    determined_param_assignments = [[mod4pi(a) for a in assignments] for assignments in determined_param_assignments]

    # if one of the sympy equations yield the input parameters, the input parameters yield commuting gates
    any_allclose = any(
        [np.allclose(bound_gate_parameters, param_assignment) for param_assignment in determined_param_assignments]
    )
    return any_allclose


def _look_up_commutation(
    op0: DAGOpNode, op1: DAGOpNode, _commutation_lib: dict, only_std_gates=False
) -> Union[bool, None]:
    """Looks up and returns the commutation of a pair of operations from a provided commutation library
    Args:
        op0 (DAGOpNode): an operation in the pair of considered operations
        op1 (DAGOpNode): an operation in the pair of considered operations
        _commutation_lib (dict): dictionary of commutation relations
        only_std_gates (bool): only look up commutation relation for standard gates (for performance measurements)
    Return:
        True if op0 and op1 commute, False if they do not commute and None if the commutation is not in the library
    """
    first_op, second_op = _order_operations(op0, op1)
    commutation = _commutation_lib.get((first_op.op.name, second_op.op.name), None)

    if commutation is None or isinstance(commutation, bool):
        return commutation

    if _is_any_operation_parameterizable(first_op, second_op):
        if only_std_gates:
            return None
        # Commutation may be the same for every relative placement. Otherwise, commutation would include a dictionary
        # with the relative placement as a key.
        if isinstance(commutation, dict):
            commutation = commutation.get(_get_relative_placement(first_op, second_op), None)

        if first_op.op.name == second_op.op.name:
            # sympy does not consider this symmetry and just ignores one half of the solutions...
            return _evaluate_parameterized_commutation(first_op, second_op, commutation) or \
                   _evaluate_parameterized_commutation(second_op, first_op, commutation)
        else:
            return _evaluate_parameterized_commutation(first_op, second_op, commutation)
    else:
        # Commutation may be the same for every relative placement. Otherwise, commutation would include a dictionary
        # with the relative placement as a key.
        if commutation is None or isinstance(commutation, bool):
            return commutation

        if isinstance(commutation, dict):
            return commutation.get(_get_relative_placement(first_op, second_op), None)
        else:
            raise ValueError("Expected commutation to be None, bool or a dict")


def _is_any_operation_parameterizable(first_op, second_op):
    """Determines whether a pair of operations must be evaluated through parameter evaluation
    Args:
        first_op (DAGOpNode): first operation of the evaluated pair of operations
        second_op (DAGOpNode): second operation of the evaluated pair of operations

    Return:
        True if an operation contains parameters that are not a ParameterExpression, else False
    """
    if first_op.op.is_parameterized():
        return False
    if second_op.op.is_parameterized():
        return False
    return (len(first_op.op.params) > 0) or (len(second_op.op.params) > 0)


class CommutationLibrary:
    """A library containing commutation relationships of non-parameterized standard gates."""

    def __init__(self, cache_max_entries: int = 5*10**6):
        self._standard_commutations = StandardGateCommutations
        self._cached_commutations = {}
        # 5e6 entries should require less than 1GB memory
        self._cache_max_entries = cache_max_entries
        self._current_cache_entries = 0

        self._lookups = 0
        self._param_lookup_hits = 0
        self._param_lookup_misses = 0
        self._standard_gates_lookup_hits = 0
        self._standard_gates_lookup_misses = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def is_op_in_library(self, gate0: DAGOpNode, gate1: DAGOpNode) -> bool:
        """Checks whether a gate is part of the commutation library.

        Args:
            gate0 (DAGOpNode): Gate to be checked.
            gate1 (DAGOpNode): Gate to be checked.

        Return:
            bool: True if gate is in commutation library, false otherwise
        """
        return (gate0.op.name, gate1.op.name) in self._standard_commutations

    def look_up_commutation_relation(self, op0: DAGOpNode, op1: DAGOpNode, only_cache: bool,
                                     only_std_gates: bool) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            op0 (DAGOpNode): a gate whose commutation should be checked
            op1 (DAGOpNode): a gate whose commutation should be checked
            only_cache (bool): flag that is true if only a cache should be used for commutation resolution
            only_std_gates (bool): flag that is true if only standard gates should be resolved by the library

        Return:
            bool: True if the gates commute and false if it is not the case.
        """

        self._lookups += 1
        if not only_cache:
            commutation = _look_up_commutation(op0, op1, self._standard_commutations, only_std_gates)
            if commutation is not None:
                if _is_any_operation_parameterizable(op0, op1):
                    self._param_lookup_hits += 1
                else:
                    self._standard_gates_lookup_hits += 1
                return commutation
            else:
                if _is_any_operation_parameterizable(op0, op1):
                    self._param_lookup_misses += 1
                else:
                    self._standard_gates_lookup_misses += 1

        commutation = _look_up_commutation(op0, op1, self._cached_commutations)
        if commutation is None:
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        return commutation

    def do_operations_commute(self, op0: DAGOpNode, op1: DAGOpNode, cache: bool = True, only_cache=False,
                              only_std_gates=False, only_matmul=False) -> bool:
        """Determines the commutation relation between op0 and op1 by trying to loop up their relation in the
        commutation library or computing the relation explicitly using matrix multiplication

        Args:
            op0 (DAGOpNode): a gate whose commutation should be checked
            op1 (DAGOpNode): a gate whose commutation should be checked
            cache (bool): whether to store new commutation relations in the commutation library
            only_cache (bool): True if the gate commutations should only be cached
            only_std_gates (bool): flag that is true if only standard gates should be resolved by the library
            only_matmul (bool): flag that is true if only matrix multiplication should be used to determine commutation
        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        if not isinstance(op0, DAGOpNode) or not isinstance(op1, DAGOpNode):
            return False

        for nd in [op0, op1]:
            if nd.op._directive or nd.name in {"measure", "reset", "delay"}:
                return False

        if op0.op.condition or op1.op.condition:
            return False

        if not only_matmul:
            if set(op0.qargs).isdisjoint(op1.qargs):
                return True

            # identical operations
            if op0.qargs == op1.qargs and op0.op.name == op1.op.name and op0.op.params == op1.op.params:
                return True

            commutation_lookup = self.look_up_commutation_relation(op0, op1, only_cache, only_std_gates)

            if commutation_lookup is not None:
                return commutation_lookup

        # Compute commutation via matrix multiplication
        is_commuting = _commute(op0, op1)

        # TODO use cache only for non-parameterizable operations?
        use_cache = cache and not _is_any_operation_parameterizable(op0, op1) and not only_matmul
        if use_cache and self._cache_max_entries > 0:
            # Store result in this session's commutation_library
            if self._current_cache_entries >= self._cache_max_entries:
                self._cached_commutations.popitem()
                self._current_cache_entries -= 1

            first_op, second_op = _order_operations(op0, op1)
            """
            if _is_any_operation_parameterizable(first_op, second_op):
                param_entry = self._cached_commutations.setdefault((first_op.op.name, second_op.op.name),
                                                                   {}).setdefault(
                    _get_relative_placement(first_op, second_op), [])
                params_list = first_op.op.params + second_op.op.params
                param_entry.append((tuple(i for i in range(len(params_list))), copy.copy(params_list)))
            else:
            """
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
