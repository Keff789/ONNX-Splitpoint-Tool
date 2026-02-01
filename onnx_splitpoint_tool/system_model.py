"""Simple system / link models for latency and energy estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .units import bandwidth_to_bytes_per_s


@dataclass
class LinkConstraints:
    """Optional feasibility constraints for a link (per inference).

    All fields are optional; if a field is None it is not enforced.
    """

    max_latency_ms: Optional[float] = None
    max_energy_mJ: Optional[float] = None
    max_bytes: Optional[int] = None


@dataclass
class LinkModelSpec:
    """Lightweight, extensible link model specification.

    This is intentionally simple (no external deps) and designed to be serialisable
    to/from JSON for dissertation-friendly reproducibility.

    type:
      - 'ideal':      time = overhead + bytes / bandwidth
      - 'packetized': time = overhead + n_packets*per_packet_overhead + (bytes+overhead_bytes*n_packets)/bandwidth
    """

    type: str = 'ideal'
    name: str = 'manual'

    bandwidth_value: Optional[float] = None
    bandwidth_unit: str = 'MB/s'

    overhead_ms: float = 0.0

    # Energy per transferred byte (payload only) in pJ / byte
    energy_pj_per_byte: Optional[float] = None

    # Packetisation extras (only used for type='packetized')
    mtu_payload_bytes: Optional[int] = None
    per_packet_overhead_ms: float = 0.0
    per_packet_overhead_bytes: int = 0

    constraints: LinkConstraints = field(default_factory=LinkConstraints)

    def bandwidth_Bps(self) -> Optional[float]:
        return bandwidth_to_bytes_per_s(self.bandwidth_value, self.bandwidth_unit)

    def estimate(self, payload_bytes: float) -> Dict[str, Optional[float]]:
        """Estimate link time/energy for sending payload_bytes once."""
        b = float(max(0.0, payload_bytes))
        bw = self.bandwidth_Bps()
        if bw is None or bw <= 0:
            t_ms = None
        else:
            if str(self.type).lower() == 'packetized':
                mtu = int(self.mtu_payload_bytes or 0)
                if mtu <= 0:
                    # Fallback to 'ideal' semantics when MTU is not provided
                    n_packets = 1
                    overhead_bytes = 0
                else:
                    n_packets = int(math.ceil(b / float(mtu))) if b > 0 else 0
                    overhead_bytes = int(self.per_packet_overhead_bytes or 0)
                extra_ms = float(self.per_packet_overhead_ms or 0.0) * float(n_packets)
                total_bytes = b + float(overhead_bytes) * float(n_packets)
                t_ms = float(self.overhead_ms) + float(extra_ms) + (1000.0 * float(total_bytes) / float(bw))
            else:
                # ideal / default
                t_ms = float(self.overhead_ms) + (1000.0 * b / float(bw))

        if self.energy_pj_per_byte is None:
            e_mJ = None
        else:
            # pJ -> mJ: 1 pJ = 1e-12 J = 1e-9 mJ
            e_mJ = float(self.energy_pj_per_byte) * b * 1e-9

        return {
            'time_ms': t_ms,
            'energy_mJ': e_mJ,
            'bandwidth_Bps': (float(bw) if bw is not None else None),
        }

    def is_feasible(self, payload_bytes: float) -> bool:
        """Check link constraints against the estimated metrics."""
        est = self.estimate(payload_bytes)
        t_ms = est.get('time_ms')
        e_mJ = est.get('energy_mJ')

        c = self.constraints
        if c.max_bytes is not None and float(payload_bytes) > float(c.max_bytes):
            return False
        if c.max_latency_ms is not None and t_ms is not None and float(t_ms) > float(c.max_latency_ms):
            return False
        if c.max_energy_mJ is not None and e_mJ is not None and float(e_mJ) > float(c.max_energy_mJ):
            return False
        return True


@dataclass
class ComputeSpec:
    """Compute-side approximation for latency/energy modelling."""

    gops: Optional[float] = None
    # Energy per FLOP in pJ / FLOP (optional)
    energy_pj_per_flop: Optional[float] = None


@dataclass
class MemoryConstraints:
    """Optional activation-memory feasibility constraints (peak during execution).

    Values are expressed in *bytes* and refer to the peak activation memory
    required by each partition while executing sequentially.

    Note: This is an approximation based on tensor lifetime spans (producer -> last consumer).
    It does not include weight/parameter memory.
    """

    max_peak_act_left_bytes: Optional[int] = None
    max_peak_act_right_bytes: Optional[int] = None


@dataclass
class SystemSpec:
    """System model = (left compute) + (link) + (right compute) + constant overhead."""

    left: ComputeSpec = field(default_factory=ComputeSpec)
    right: ComputeSpec = field(default_factory=ComputeSpec)
    link: LinkModelSpec = field(default_factory=LinkModelSpec)

    memory: MemoryConstraints = field(default_factory=MemoryConstraints)

    overhead_ms: float = 0.0

    def is_memory_feasible(self, *, peak_left_bytes: float, peak_right_bytes: float) -> bool:
        """Check activation-memory constraints for a boundary (if configured).

        This is separate from link feasibility because it depends on *peak* memory
        during execution of each partition, not just the cut payload size.
        """
        c = getattr(self, 'memory', None)
        if c is None:
            return True
        try:
            if c.max_peak_act_left_bytes is not None and float(peak_left_bytes) > float(c.max_peak_act_left_bytes):
                return False
            if c.max_peak_act_right_bytes is not None and float(peak_right_bytes) > float(c.max_peak_act_right_bytes):
                return False
        except Exception:
            # Never fail hard on bad config; treat as feasible.
            return True
        return True

    def estimate_boundary(self, *, comm_bytes: float, flops_left: float, flops_total: float) -> Dict[str, Optional[float]]:
        """Estimate end-to-end latency/energy for a candidate boundary."""
        fl_l = float(max(0.0, flops_left))
        fl_t = float(max(0.0, flops_total))
        fl_r = float(max(0.0, fl_t - fl_l))

        # Latency components
        if self.left.gops is None or self.left.gops <= 0:
            t_left_ms = None
        else:
            t_left_ms = 1000.0 * fl_l / (float(self.left.gops) * 1e9)

        if self.right.gops is None or self.right.gops <= 0:
            t_right_ms = None
        else:
            t_right_ms = 1000.0 * fl_r / (float(self.right.gops) * 1e9)

        link_est = self.link.estimate(comm_bytes)
        t_link_ms = link_est.get('time_ms')

        if t_left_ms is None or t_right_ms is None or t_link_ms is None:
            t_total_ms = None
        else:
            t_total_ms = float(t_left_ms) + float(t_link_ms) + float(t_right_ms) + float(self.overhead_ms)

        # Energy components
        if self.left.energy_pj_per_flop is None:
            e_left_mJ = None
        else:
            e_left_mJ = float(self.left.energy_pj_per_flop) * fl_l * 1e-9

        if self.right.energy_pj_per_flop is None:
            e_right_mJ = None
        else:
            e_right_mJ = float(self.right.energy_pj_per_flop) * fl_r * 1e-9

        e_link_mJ = link_est.get('energy_mJ')

        if e_left_mJ is None or e_right_mJ is None or e_link_mJ is None:
            e_total_mJ = None
        else:
            e_total_mJ = float(e_left_mJ) + float(e_link_mJ) + float(e_right_mJ)

        return {
            'latency_total_ms': t_total_ms,
            'latency_left_ms': t_left_ms,
            'latency_link_ms': t_link_ms,
            'latency_right_ms': t_right_ms,
            'energy_total_mJ': e_total_mJ,
            'energy_left_mJ': e_left_mJ,
            'energy_link_mJ': e_link_mJ,
            'energy_right_mJ': e_right_mJ,
        }

