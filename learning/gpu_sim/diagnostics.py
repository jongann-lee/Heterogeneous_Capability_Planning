"""Opt-in synchronized timings and CUDA allocator diagnostics."""

from contextlib import contextmanager
import json
from time import perf_counter

import torch


def prepare_cuda_allocator(device):
    """Reuse RAPIDS scratch blocks without taking over a caller's allocator.

    Only replace the default direct CUDA allocator, before constructing any
    terrain. The pool starts at 64 MiB and is bounded by 1 GiB and one eighth
    of currently free device memory. PyTorch retains its separate allocator.
    """
    if torch.device(device).type != "cuda":
        return
    import rmm

    with torch.cuda.device(device):
        current = rmm.mr.get_current_device_resource()
        if type(current) is not rmm.mr.CudaMemoryResource:
            return
        free, _total = torch.cuda.mem_get_info(device)
        maximum = min(1024 ** 3, free // 8) // 256 * 256
        if maximum < 64 * 1024 ** 2:
            return
        rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(
            current, initial_pool_size=64 * 1024 ** 2,
            maximum_pool_size=maximum))


class TrainingDiagnostics:
    """Keep only scalar metrics; never retain observations or GPU tensors.

    Timings include GPU completion and therefore add synchronization overhead.
    Nested rollout stages are subsets of the total rollout time. Device usage
    includes all processes; PyTorch counters only describe its own allocator.
    """

    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.metrics = {}
        if enabled and self.device.type == "cuda":
            # Enable before terrain construction so retained RAPIDS allocations
            # are visible too. This wraps the existing resource, not a new pool.
            from rmm.statistics import enable_statistics
            enable_statistics()

    def reset(self):
        self.metrics = {}
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.snapshot("start")

    def snapshot(self, label):
        if not self.enabled or self.device.type != "cuda":
            return
        free, total = torch.cuda.mem_get_info(self.device)
        mib = 1024 ** 2
        for name, value in {
            "torch_allocated": torch.cuda.memory_allocated(self.device),
            "torch_reserved": torch.cuda.memory_reserved(self.device),
            "torch_peak_allocated": torch.cuda.max_memory_allocated(self.device),
            "device_used": total - free,
            "device_free": free,
        }.items():
            self.metrics[f"memory/{label}/{name}_mib"] = value / mib
        from rmm.statistics import get_statistics
        statistics = get_statistics()
        if statistics is not None:
            self.metrics[f"memory/{label}/rmm_allocated_mib"] = (
                statistics.current_bytes / mib)
            # RMM's peak is cumulative since statistics were enabled, whereas
            # the PyTorch peak is reset for each optimizer update.
            self.metrics[f"memory/{label}/rmm_peak_since_start_mib"] = (
                statistics.peak_bytes / mib)
        import rmm
        resource = rmm.mr.get_current_device_resource()
        while resource is not None:
            if isinstance(resource, rmm.mr.PoolMemoryResource):
                self.metrics[f"memory/{label}/rmm_pool_reserved_mib"] = (
                    resource.pool_size() / mib)
                break
            resource = getattr(resource, "upstream_mr", None)

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            key = f"time/{name}_seconds"
            self.metrics[key] = self.metrics.get(key, 0.0) + perf_counter() - start

    def report(self, episodes_seen):
        if self.enabled:
            self.snapshot("end")
            print("training_profile=" + json.dumps({
                "episodes_seen": episodes_seen, **self.metrics,
            }, sort_keys=True), flush=True)
        return dict(self.metrics)
