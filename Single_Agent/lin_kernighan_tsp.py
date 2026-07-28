"""
This code is from the python-tsp package at the link
https://github.com/fillipe-gsm/python-tsp/tree/master

"""
from random import sample
from itertools import permutations
from typing import Any, List, Optional, TextIO, Tuple

import numpy as np


def setup_initial_solution(
    distance_matrix: np.ndarray, x0: Optional[List] = None
) -> Tuple[List[int], float]:
    """Return initial solution and its objective value

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Permutation of nodes from 0 to n - 1 indicating the starting solution.
        If not provided, a random list is created.

    Returns
    -------
    x0
        Permutation with initial solution. If ``x0`` was provided, it is the
        same list

    fx0
        Objective value of x0
    """

    if not x0:
        n = distance_matrix.shape[0]  # number of nodes
        x0 = [0] + sample(range(1, n), n - 1)  # ensure 0 is the first node

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0


def compute_permutation_distance(
    distance_matrix: np.ndarray, permutation: List[int]
) -> float:
    """Compute the total route distance of a given permutation

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric

    permutation
        A list with nodes from 0 to n - 1 in any order

    Returns
    -------
    Total distance of the path given in ``permutation`` for the provided
    ``distance_matrix``

    Notes
    -----
    Suppose the permutation [0, 1, 2, 3], with four nodes. The total distance
    of this path will be from 0 to 1, 1 to 2, 2 to 3, and 3 back to 0. This
    can be fetched from a distance matrix using:

        distance_matrix[ind1, ind2], where
        ind1 = [0, 1, 2, 3]  # the FROM nodes
        ind2 = [1, 2, 3, 0]  # the TO nodes

    This can easily be generalized to any permutation by using ind1 as the
    given permutation, and moving the first node to the end to generate ind2.
    """
    ind1 = permutation
    ind2 = permutation[1:] + permutation[:1]
    return distance_matrix[ind1, ind2].sum()

def solve_tsp_brute_force(
    distance_matrix: np.ndarray,
) -> Tuple[Optional[List], Any]:
    """Solve TSP to optimality with a brute force approach

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric

    Returns
    -------
    A permutation of nodes from 0 to n that produces the least total
    distance

    The total distance the optimal permutation produces

    Notes
    ----
    The algorithm checks all permutations and returns the one with smallest
    distance. In principle, the total number of possibilities would be n! for
    n nodes. However, we can fix node 0 and permutate only the remaining,
    reducing the possibilities to (n - 1)!.
    """

    # Exclude 0 from the range since it is fixed as starting point
    points = range(1, distance_matrix.shape[0])
    best_distance = np.inf
    best_permutation = None

    for partial_permutation in permutations(points):
        # Remember to add the starting node before evaluating it
        permutation = [0] + list(partial_permutation)
        distance = compute_permutation_distance(distance_matrix, permutation)

        if distance < best_distance:
            best_distance = distance
            best_permutation = permutation

    return best_permutation, best_distance

def _cycle_to_successors(cycle: List[int]) -> List[int]:
    """
    Convert a cycle representation to successors representation.

    Parameters
    ----------
    cycle
        A list representing a cycle.

    Returns
    -------
    List
        A list representing successors.
    """
    successors = cycle[:]
    n = len(cycle)
    for i, _ in enumerate(cycle):
        successors[cycle[i]] = cycle[(i + 1) % n]
    return successors


def _successors_to_cycle(successors: List[int]) -> List[int]:
    """
    Convert a successors representation to a cycle representation.

    Parameters
    ----------
    successors
        A list representing successors.

    Returns
    -------
    List
        A list representing a cycle.
    """
    cycle = successors[:]
    j = 0
    for i, _ in enumerate(successors):
        cycle[i] = j
        j = successors[j]
    return cycle


def _minimizes_hamiltonian_path_distance(
    tabu: np.ndarray,
    iteration: int,
    successors: List[int],
    ejected_edge: Tuple[int, int],
    distance_matrix: np.ndarray,
    hamiltonian_path_distance: float,
    hamiltonian_cycle_distance: float,
) -> Tuple[int, int, float]:
    """
    Minimize the Hamiltonian path distance after ejecting an edge.

    Parameters
    ----------
    tabu
        A NumPy array for tabu management.

    iteration
        The current iteration.

    successors
        A list representing successors.

    ejected_edge
        The edge that was ejected.

    distance_matrix
        A NumPy array representing the distance matrix.

    hamiltonian_path_distance
        The Hamiltonian path distance.

    hamiltonian_cycle_distance
        The Hamiltonian cycle distance.

    Returns
    -------
    Tuple
        The best c, d, and the new Hamiltonian path distance found.
    """
    a, b = ejected_edge
    best_c = c = last_c = successors[b]
    path_cb_distance = distance_matrix[c, b]
    path_bc_distance = distance_matrix[b, c]
    hamiltonian_path_distance_found = hamiltonian_cycle_distance

    while successors[c] != a:
        d = successors[c]
        path_cb_distance += distance_matrix[c, last_c]
        path_bc_distance += distance_matrix[last_c, c]
        new_hamiltonian_path_distance_found = (
            hamiltonian_path_distance
            + distance_matrix[b, d]
            - distance_matrix[c, d]
            + path_cb_distance
            - path_bc_distance
        )

        if (
            new_hamiltonian_path_distance_found + distance_matrix[a, c]
            < hamiltonian_cycle_distance
        ):
            return c, d, new_hamiltonian_path_distance_found

        if (
            tabu[c, d] != iteration
            and new_hamiltonian_path_distance_found
            < hamiltonian_path_distance_found
        ):
            hamiltonian_path_distance_found = (
                new_hamiltonian_path_distance_found
            )
            best_c = c

        last_c = c
        c = d

    return best_c, successors[best_c], hamiltonian_path_distance_found


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)


def _solve_tsp_brute_force(
    distance_matrix: np.ndarray,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    x, fx = solve_tsp_brute_force(distance_matrix)
    x = x or []

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )
    msg = (
        "Few nodes to use Lin-Kernighan heuristics, "
        "using Brute Force instead. "
    )
    if not x:
        msg += "No solution found."
    else:
        msg += f"Found value: {fx}"
    _print_message(msg, verbose, log_file_handler)

    if log_file_handler:
        log_file_handler.close()

    return x, fx


def solve_tsp_lin_kernighan(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesperson Problem using the Lin-Kernighan algorithm.

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path.

    log_file
        If not `None`, creates a log file with details about the whole
        execution.

    verbose
        If true, prints algorithm status every iteration.

    Returns
    -------
    Tuple
        A tuple containing the Hamiltonian cycle and its distance.

    References
    ----------
    Éric D. Taillard, "Design of Heuristic Algorithms for Hard Optimization,"
    Chapter 5, Section 5.3.2.1: Lin-Kernighan Neighborhood, Springer, 2023.
    """
    num_vertices = distance_matrix.shape[0]
    if num_vertices < 5:
        return _solve_tsp_brute_force(distance_matrix, log_file, verbose)

    hamiltonian_cycle, hamiltonian_cycle_distance = setup_initial_solution(
        distance_matrix=distance_matrix, x0=x0
    )
    vertices = list(range(num_vertices))
    iteration = 0
    improvement = True
    tabu = np.zeros(shape=(num_vertices, num_vertices), dtype=int)

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    while improvement:
        iteration += 1
        improvement = False
        successors = _cycle_to_successors(hamiltonian_cycle)

        # Eject edge [a, b] to start the chain and compute the Hamiltonian
        # path distance obtained by ejecting edge [a, b] from the cycle
        # as reference.
        a = int(distance_matrix[vertices, successors].argmax())
        b = successors[a]
        hamiltonian_path_distance = (
            hamiltonian_cycle_distance - distance_matrix[a, b]
        )

        while True:
            ejected_edge = a, b

            # Find the edge [c, d] that minimizes the Hamiltonian path obtained
            # by removing edge [c, d] and adding edge [b, d], with [c, d] not
            # removed in the current ejection chain.
            (
                c,
                d,
                hamiltonian_path_distance_found,
            ) = _minimizes_hamiltonian_path_distance(
                tabu,
                iteration,
                successors,
                ejected_edge,
                distance_matrix,
                hamiltonian_path_distance,
                hamiltonian_cycle_distance,
            )

            # If the Hamiltonian cycle cannot be improved, return
            # to the solution and try another ejection.
            if hamiltonian_path_distance_found >= hamiltonian_cycle_distance:
                break

            # Update Hamiltonian path distance reference
            hamiltonian_path_distance = hamiltonian_path_distance_found

            # Reverse the direction of the path from b to c
            i, si, successors[b] = b, successors[b], d
            while i != c:
                successors[si], i, si = i, si, successors[si]

            # Don't remove again the minimal edge found
            tabu[c, d] = tabu[d, c] = iteration

            # c plays the role of b in the next iteration
            b = c

            msg = (
                f"Current value: {hamiltonian_cycle_distance}; "
                f"Ejection chain: {iteration}"
            )
            _print_message(msg, verbose, log_file_handler)

            # If the Hamiltonian cycle improves, update the solution
            if (
                hamiltonian_path_distance + distance_matrix[a, b]
                < hamiltonian_cycle_distance
            ):
                improvement = True
                successors[a] = b
                hamiltonian_cycle = _successors_to_cycle(successors)
                hamiltonian_cycle_distance = (
                    hamiltonian_path_distance + distance_matrix[a, b]
                )

    if log_file_handler:
        log_file_handler.close()

    return hamiltonian_cycle, hamiltonian_cycle_distance