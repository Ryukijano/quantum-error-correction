"""Concrete quasi-cyclic lifted-product clustered-cyclic code construction.

This module implements a simplified clustered-cyclic qLDPC code by building two
sparse quasi-cyclic binary matrices ``A`` and ``B`` and then forming the CSS
hypergraph-product checks

    H_X = [I ⊗ A | B^T ⊗ I]
    H_Z = [B ⊗ I | I ⊗ A^T].

The resulting stabilizers are converted into a static Stim circuit using
ancilla-based syndrome extraction. The ``parallel_product_surgery`` interface is
kept via placeholder logical-Z supports that mark the cyclic cluster
boundaries where future surgery hooks can be inserted.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Coord = Tuple[float, ...]
SparseSupport = tuple[int, ...]


@dataclass(frozen=True)
class SparseCirculantMatrix:
    """Sparse quasi-cyclic binary matrix made from circulant permutation blocks."""

    block_count: int
    lift_size: int
    generators: tuple[tuple[int, int], ...]
    row_supports: tuple[SparseSupport, ...]
    column_supports: tuple[SparseSupport, ...]

    @property
    def size(self) -> int:
        return self.block_count * self.lift_size

    def dense(self) -> List[List[int]]:
        """Materialize the binary matrix for debugging or export."""

        dense_rows: List[List[int]] = []
        for support in self.row_supports:
            row = [0] * self.size
            for column in support:
                row[column] = 1
            dense_rows.append(row)
        return dense_rows


class _RecordBuilder:
    """Small helper for assembling Stim strings with measurement bookkeeping."""

    _MAX_SINGLE_QUBIT_ARGS = 256
    _MAX_TWO_QUBIT_PAIRS = 128

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.measurement_count = 0

    def add(self, line: str) -> None:
        if line.strip():
            self.lines.append(line.rstrip())

    def qubit_coords(self, coord_lookup: Mapping[int, Coord]) -> None:
        for qubit in sorted(coord_lookup):
            coords = ",".join(f"{value:.6g}" for value in coord_lookup[qubit])
            self.add(f"QUBIT_COORDS({coords}) {qubit}")

    def gate(self, name: str, qubits: Sequence[int]) -> None:
        for chunk in self._chunk(qubits, self._MAX_SINGLE_QUBIT_ARGS):
            self.add(f"{name} {' '.join(map(str, chunk))}")

    def pair_gate(self, name: str, pairs: Sequence[tuple[int, int]]) -> None:
        for chunk in self._chunk(pairs, self._MAX_TWO_QUBIT_PAIRS):
            flattened = [str(qubit) for pair in chunk for qubit in pair]
            self.add(f"{name} {' '.join(flattened)}")

    def depolarize(self, p: float, qubits: Sequence[int]) -> None:
        if p <= 0.0:
            return
        for chunk in self._chunk(qubits, self._MAX_SINGLE_QUBIT_ARGS):
            self.add(f"DEPOLARIZE1({p}) {' '.join(map(str, chunk))}")

    def measure(self, qubits: Sequence[int]) -> List[int]:
        indices: List[int] = []
        for chunk in self._chunk(qubits, self._MAX_SINGLE_QUBIT_ARGS):
            self.add(f"M {' '.join(map(str, chunk))}")
            chunk_indices = list(range(self.measurement_count, self.measurement_count + len(chunk)))
            indices.extend(chunk_indices)
            self.measurement_count += len(chunk)
        return indices

    def detector(self, coord: Coord, rec_indices: Sequence[int]) -> None:
        if not rec_indices:
            return
        offsets = [self.measurement_count - index for index in rec_indices]
        rec_terms = " ".join(f"rec[-{offset}]" for offset in offsets)
        coords = ",".join(f"{value:.6g}" for value in coord)
        self.add(f"DETECTOR({coords}) {rec_terms}")

    def observable(self, rec_indices: Sequence[int], observable_index: int = 0) -> None:
        if not rec_indices:
            return
        offsets = [self.measurement_count - index for index in rec_indices]
        rec_terms = " ".join(f"rec[-{offset}]" for offset in offsets)
        self.add(f"OBSERVABLE_INCLUDE({observable_index}) {rec_terms}")

    def build(self) -> str:
        return "\n".join(self.lines) + "\n"

    @staticmethod
    def _chunk(items: Sequence[int] | Sequence[tuple[int, int]], size: int) -> Iterable[Sequence[int] | Sequence[tuple[int, int]]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]


class ClusteredCyclicCode:
    """Concrete clustered-cyclic qLDPC construction based on a lifted product.

    Args:
        distance: Coarse size parameter controlling default cluster count/size.
        rounds: Number of repeated syndrome rounds.
        physical_error_rate: Per-round depolarizing noise rate applied to data.
        p: Alias for ``physical_error_rate``.
        num_clusters: Number of block-circulant clusters in the QC seed.
        cluster_size: Size of each circulant lift inside a cluster.
        check_weight: Number of non-zero circulant blocks per block row.
        seed: Random seed used to generate reproducible QC block shifts.
        parallel_product_surgery: Keep placeholder logical-boundary supports
            marking where future surgery gadgets would attach.
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        physical_error_rate: float | None = None,
        *,
        p: float | None = None,
        num_clusters: int | None = None,
        cluster_size: int | None = None,
        check_weight: int = 3,
        seed: int | None = None,
        parallel_product_surgery: bool = True,
    ) -> None:
        if physical_error_rate is None:
            physical_error_rate = p
        if physical_error_rate is None:
            raise ValueError("physical_error_rate (or alias p) must be provided.")
        if distance < 2:
            raise ValueError("distance must be at least 2.")
        if rounds <= 0:
            raise ValueError("rounds must be positive.")
        if not 0.0 <= physical_error_rate <= 1.0:
            raise ValueError("physical_error_rate must be between 0 and 1.")

        self.distance = distance
        self.rounds = rounds
        self.physical_error_rate = physical_error_rate
        self.num_clusters = num_clusters or max(2, distance)
        self.cluster_size = cluster_size or max(2, distance)
        self.check_weight = max(1, min(check_weight, self.num_clusters))
        self.seed = seed if seed is not None else (distance * 1009 + rounds * 97)
        self.parallel_product_surgery = parallel_product_surgery

        self._dimension = self.num_clusters * self.cluster_size
        self.a_matrix = self._build_qc_matrix(seed_offset=0xA5A5A5A5, reverse_bias=False)
        self.b_matrix = self._build_qc_matrix(seed_offset=0x5A5A5A5A, reverse_bias=True)
        self.hx_row_supports = self._build_hx_row_supports()
        self.hz_row_supports = self._build_hz_row_supports()

        # These supports indicate the cyclic cluster boundaries where an actual
        # parallel product surgery layer would splice logical operators.
        self.logical_z_placeholders = self._build_surgery_placeholders()

    @property
    def matrix_size(self) -> int:
        return self._dimension

    @property
    def num_data_qubits(self) -> int:
        return 2 * self.matrix_size * self.matrix_size

    @property
    def num_x_checks(self) -> int:
        return len(self.hx_row_supports)

    @property
    def num_z_checks(self) -> int:
        return len(self.hz_row_supports)

    def hx_matrix(self) -> List[List[int]]:
        """Materialize H_X as a dense binary matrix."""

        return self._materialize_dense_matrix(self.hx_row_supports, self.num_data_qubits)

    def hz_matrix(self) -> List[List[int]]:
        """Materialize H_Z as a dense binary matrix."""

        return self._materialize_dense_matrix(self.hz_row_supports, self.num_data_qubits)

    def build_circuit_string(self) -> str:
        """Return the clustered-cyclic lifted-product code as Stim text."""

        builder = _RecordBuilder()
        builder.qubit_coords(self._coord_lookup())

        data_qubits = list(range(self.num_data_qubits))
        x_ancillas = self._x_ancillas()
        z_ancillas = self._z_ancillas()
        x_layers = self._schedule_interactions(self.hx_row_supports, x_ancillas)
        z_layers = self._schedule_interactions(self.hz_row_supports, z_ancillas)
        x_coords = self._check_coords(layer=2.0)
        z_coords = self._check_coords(layer=3.0)

        builder.gate("R", data_qubits)
        builder.add("TICK")

        previous_x: List[int] | None = None
        previous_z: List[int] | None = None

        for round_index in range(self.rounds):
            builder.gate("R", z_ancillas)
            builder.add("TICK")
            for layer in z_layers:
                builder.pair_gate("CX", [(data_qubit, ancilla_qubit) for ancilla_qubit, data_qubit in layer])
                builder.add("TICK")
            current_z = builder.measure(z_ancillas)
            self._attach_round_detectors(builder, z_coords, current_z, previous_z)
            previous_z = current_z
            builder.depolarize(self.physical_error_rate, data_qubits)
            builder.add("TICK")

            builder.gate("R", x_ancillas)
            builder.gate("H", x_ancillas)
            builder.add("TICK")
            for layer in x_layers:
                builder.pair_gate("CX", layer)
                builder.add("TICK")
            builder.gate("H", x_ancillas)
            current_x = builder.measure(x_ancillas)
            self._attach_round_detectors(builder, x_coords, current_x, previous_x)
            previous_x = current_x
            builder.depolarize(self.physical_error_rate, data_qubits)
            if round_index != self.rounds - 1:
                builder.add("TICK")

        final_measurements = builder.measure(data_qubits)
        if previous_z is not None:
            self._attach_final_z_detectors(builder, z_coords, previous_z, final_measurements)

        observable_support = self._observable_support()
        builder.observable([final_measurements[qubit] for qubit in observable_support])
        return builder.build()

    def build_circuit(self) -> "stim.Circuit":
        """Return the code as a materialized ``stim.Circuit``."""

        try:
            import stim
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Stim is required to materialize ClusteredCyclicCode circuits.") from exc
        return stim.Circuit(self.build_circuit_string())

    def build(self) -> "stim.Circuit":
        """Alias for ``build_circuit``."""

        return self.build_circuit()

    def to_stim_circuit(self) -> "stim.Circuit":
        """Alias for ``build_circuit``."""

        return self.build_circuit()

    def _build_qc_matrix(self, *, seed_offset: int, reverse_bias: bool) -> SparseCirculantMatrix:
        rng = random.Random(self.seed ^ seed_offset)
        generators = self._choose_generators(rng, reverse_bias=reverse_bias)
        size = self.matrix_size
        row_supports: List[SparseSupport] = []
        column_sets = [set() for _ in range(size)]

        for block_row in range(self.num_clusters):
            for local_row in range(self.cluster_size):
                support: List[int] = []
                row_index = block_row * self.cluster_size + local_row
                for cluster_offset, shift in generators:
                    block_column = (block_row + cluster_offset) % self.num_clusters
                    local_column = (local_row + shift) % self.cluster_size
                    column = block_column * self.cluster_size + local_column
                    support.append(column)
                    column_sets[column].add(row_index)
                row_supports.append(tuple(sorted(support)))

        column_supports = tuple(tuple(sorted(rows)) for rows in column_sets)
        return SparseCirculantMatrix(
            block_count=self.num_clusters,
            lift_size=self.cluster_size,
            generators=generators,
            row_supports=tuple(row_supports),
            column_supports=column_supports,
        )

    def _build_hx_row_supports(self) -> tuple[SparseSupport, ...]:
        supports: List[SparseSupport] = []
        for outer in range(self.matrix_size):
            b_transpose_support = self.b_matrix.column_supports[outer]
            for inner in range(self.matrix_size):
                first_sector = [self._vv_qubit(outer, column) for column in self.a_matrix.row_supports[inner]]
                second_sector = [self._cc_qubit(column, inner) for column in b_transpose_support]
                supports.append(tuple(sorted(first_sector + second_sector)))
        return tuple(supports)

    def _build_hz_row_supports(self) -> tuple[SparseSupport, ...]:
        supports: List[SparseSupport] = []
        for outer in range(self.matrix_size):
            b_support = self.b_matrix.row_supports[outer]
            for inner in range(self.matrix_size):
                first_sector = [self._vv_qubit(column, inner) for column in b_support]
                second_sector = [self._cc_qubit(outer, column) for column in self.a_matrix.column_supports[inner]]
                supports.append(tuple(sorted(first_sector + second_sector)))
        return tuple(supports)

    def _build_surgery_placeholders(self) -> tuple[SparseSupport, ...]:
        if not self.parallel_product_surgery:
            return ()

        placeholders: List[SparseSupport] = []
        for cluster in range(self.num_clusters):
            support: List[int] = []
            next_cluster = (cluster + 1) % self.num_clusters
            for local in range(self.cluster_size):
                left_index = cluster * self.cluster_size + local
                right_index = next_cluster * self.cluster_size + local
                support.append(self._vv_qubit(left_index, right_index))
            placeholders.append(tuple(sorted(support)))
        return tuple(placeholders)

    def _coord_lookup(self) -> Dict[int, Coord]:
        coords: Dict[int, Coord] = {}
        for outer in range(self.matrix_size):
            for inner in range(self.matrix_size):
                row = self._row_index(outer, inner)
                coords[self._vv_qubit(outer, inner)] = self._pair_coord(outer, inner, layer=0.0)
                coords[self._cc_qubit(outer, inner)] = self._pair_coord(outer, inner, layer=1.0)
                coords[self._x_ancilla_qubit(row)] = self._pair_coord(outer, inner, layer=2.0)
                coords[self._z_ancilla_qubit(row)] = self._pair_coord(outer, inner, layer=3.0)
        return coords

    def _check_coords(self, *, layer: float) -> tuple[Coord, ...]:
        return tuple(
            self._pair_coord(outer, inner, layer=layer)
            for outer in range(self.matrix_size)
            for inner in range(self.matrix_size)
        )

    def _pair_coord(self, outer: int, inner: int, *, layer: float) -> Coord:
        outer_cluster, outer_local = divmod(outer, self.cluster_size)
        inner_cluster, inner_local = divmod(inner, self.cluster_size)
        cluster_pitch = float(self.cluster_size + 1)
        x_coord = inner_cluster * cluster_pitch + inner_local
        y_coord = outer_cluster * cluster_pitch + outer_local
        return (float(x_coord), float(y_coord), layer)

    def _observable_support(self) -> SparseSupport:
        if self.logical_z_placeholders:
            return self.hz_row_supports[0]
        return self.hz_row_supports[0]

    @staticmethod
    def _attach_round_detectors(
        builder: _RecordBuilder,
        coords: Sequence[Coord],
        current: Sequence[int],
        previous: Sequence[int] | None,
    ) -> None:
        if previous is None:
            return
        if len(previous) != len(current):
            raise ValueError("Detector bookkeeping requires a fixed check ordering across rounds.")

        for index, coord in enumerate(coords):
            builder.detector(coord, [previous[index], current[index]])

    def _attach_final_z_detectors(
        self,
        builder: _RecordBuilder,
        coords: Sequence[Coord],
        previous_z: Sequence[int],
        final_measurements: Sequence[int],
    ) -> None:
        for index, coord in enumerate(coords):
            parity_records = [final_measurements[data_qubit] for data_qubit in self.hz_row_supports[index]]
            builder.detector(coord, [previous_z[index], *parity_records])

    def _schedule_interactions(
        self,
        row_supports: Sequence[Sequence[int]],
        ancilla_qubits: Sequence[int],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        layers: List[List[tuple[int, int]]] = []
        busy_ancillas: List[set[int]] = []
        busy_data: List[set[int]] = []

        for check_index, support in enumerate(row_supports):
            ancilla = ancilla_qubits[check_index]
            for data_qubit in support:
                for layer_index, layer in enumerate(layers):
                    if ancilla in busy_ancillas[layer_index] or data_qubit in busy_data[layer_index]:
                        continue
                    layer.append((ancilla, data_qubit))
                    busy_ancillas[layer_index].add(ancilla)
                    busy_data[layer_index].add(data_qubit)
                    break
                else:
                    layers.append([(ancilla, data_qubit)])
                    busy_ancillas.append({ancilla})
                    busy_data.append({data_qubit})

        return tuple(tuple(layer) for layer in layers)

    def _x_ancillas(self) -> List[int]:
        start = self.num_data_qubits
        return list(range(start, start + self.num_x_checks))

    def _z_ancillas(self) -> List[int]:
        start = self.num_data_qubits + self.num_x_checks
        return list(range(start, start + self.num_z_checks))

    def _x_ancilla_qubit(self, row: int) -> int:
        return self.num_data_qubits + row

    def _z_ancilla_qubit(self, row: int) -> int:
        return self.num_data_qubits + self.num_x_checks + row

    def _row_index(self, outer: int, inner: int) -> int:
        return outer * self.matrix_size + inner

    def _vv_qubit(self, left: int, right: int) -> int:
        return left * self.matrix_size + right

    def _cc_qubit(self, left: int, right: int) -> int:
        return self.matrix_size * self.matrix_size + left * self.matrix_size + right

    def _choose_generators(self, rng: random.Random, *, reverse_bias: bool) -> tuple[tuple[int, int], ...]:
        offsets = self._ordered_local_offsets(reverse_bias=reverse_bias)
        locality_window = min(len(offsets), max(self.check_weight + 1, 3))
        local_pool = offsets[:locality_window]
        chosen_offsets = [0]

        if self.check_weight > 1:
            extra_offsets = [offset for offset in local_pool if offset != 0]
            rng.shuffle(extra_offsets)
            chosen_offsets.extend(extra_offsets[: self.check_weight - 1])

        generators: List[tuple[int, int]] = []
        for offset in chosen_offsets:
            generators.append((offset, rng.randrange(self.cluster_size)))
        return tuple(generators)

    def _ordered_local_offsets(self, *, reverse_bias: bool) -> List[int]:
        offsets: List[int] = [0]
        radius = 1
        while len(offsets) < self.num_clusters:
            if reverse_bias:
                candidates = [(-radius) % self.num_clusters, radius % self.num_clusters]
            else:
                candidates = [radius % self.num_clusters, (-radius) % self.num_clusters]
            for candidate in candidates:
                if candidate not in offsets:
                    offsets.append(candidate)
                    if len(offsets) == self.num_clusters:
                        break
            radius += 1
        return offsets

    @staticmethod
    def _materialize_dense_matrix(row_supports: Sequence[Sequence[int]], width: int) -> List[List[int]]:
        dense_rows: List[List[int]] = []
        for support in row_supports:
            row = [0] * width
            for column in support:
                row[column] = 1
            dense_rows.append(row)
        return dense_rows
