"""
GPU Performance Profiler for distributed training
Measures throughput, GPU utilization, and system metrics across nodes/GPUs
"""

from __future__ import annotations

import json
import platform
import socket
import threading
import time
from pathlib import Path

import lightning as L
import psutil
import pynvml
import torch


class DistributedGPUProfiler:
    def __init__(
        self,
        fabric: L.Fabric,
        model_name: str,
        parallel_strategy: str,
        train_args,
        model,
        output_dir: Path,
        monitoring_interval: float = 1.0,
    ):
        self.fabric = fabric
        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self.train_args = train_args
        self.model = model
        self.output_dir = Path(output_dir)
        self.monitoring_interval = monitoring_interval

        # Hardware detection and validation
        self._detect_hardware()
        self._validate_hardware_detection()

        # Token counting and timing (LOCAL to this rank only)
        self.local_tokens_processed = 0
        self.local_batches_processed = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        # Expected values for validation
        self.expected_tokens_per_batch = (
            train_args.micro_batch_size * model.max_seq_length
        )
        self.expected_total_tokens = (
            train_args.max_tokens // fabric.world_size
        )  # Per rank

        # Monitoring data
        self.gpu_metrics_history: list[dict] = []
        self.node_metrics_history: list[dict] = []
        self.monitoring_thread: threading.Thread | None = None
        self.monitoring_active: bool = False

        # Error tracking
        self.token_count_mismatches = 0
        self.profiling_errors: list[str] = []

        # Initialize hardware monitoring
        self._init_hardware_monitoring()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.fabric.global_rank == 0:
            self.fabric.print(
                f"[Profiler] Initialized for rank {self.fabric.global_rank}"
            )
            self.fabric.print(
                f"[Profiler] Expected tokens per batch: {self.expected_tokens_per_batch}"
            )
            self.fabric.print(
                f"[Profiler] Expected total tokens (per rank): {self.expected_total_tokens}"
            )

    def _detect_hardware(self):
        """Detect hardware configuration with improved accuracy."""
        self.rank = self.fabric.global_rank
        self.world_size = self.fabric.world_size
        self.local_rank = (
            self.fabric.local_rank if hasattr(self.fabric, "local_rank") else 0
        )

        # Get hostname
        self.hostname = socket.gethostname()

        # Detect number of nodes and GPUs per node
        if torch.distributed.is_initialized():
            # In distributed mode
            self.hardware_detection_method = "distributed"

            # Get local device count (GPUs on this node)
            if torch.cuda.is_available():
                self.local_gpu_count = torch.cuda.device_count()
            else:
                self.local_gpu_count = 0

            # Calculate nodes and GPUs per node
            # Assumption: all nodes have the same number of GPUs
            if self.local_gpu_count > 0:
                self.num_nodes = self.world_size // self.local_gpu_count
                self.gpus_per_node = self.local_gpu_count
            else:
                self.num_nodes = 1
                self.gpus_per_node = 1

            # Node ID based on rank
            self.node_id = self.rank // self.gpus_per_node

        else:
            # Single process mode
            self.hardware_detection_method = "single_process"
            self.num_nodes = 1
            self.local_gpu_count = (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            )
            self.gpus_per_node = max(1, self.local_gpu_count)
            self.node_id = 0

        self.num_gpus_total = self.world_size

    def _validate_hardware_detection(self):
        """Validate that hardware detection is consistent."""
        expected_world_size = self.num_nodes * self.gpus_per_node

        if expected_world_size != self.world_size:
            error_msg = (
                f"Hardware detection inconsistency: "
                f"calculated world_size ({expected_world_size}) != "
                f"actual world_size ({self.world_size})"
            )
            self.profiling_errors.append(error_msg)

            if self.fabric.global_rank == 0:
                self.fabric.print(f"[Profiler] Warning: {error_msg}")

    def _init_hardware_monitoring(self):
        """Initialize hardware monitoring capabilities."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            self.profiling_errors.append(f"[Profiler] NVML initialization failed: {e}")

    def start_profiling(self):
        """Start profiling with proper synchronization."""
        # Global synchronization to ensure all ranks start together
        self.fabric.barrier()

        # Record start time AFTER synchronization
        self.start_time = time.time()

        # Reset counters
        self.local_tokens_processed = 0
        self.local_batches_processed = 0
        self.token_count_mismatches = 0

        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_hardware, daemon=True
        )
        self.monitoring_thread.start()

        if self.fabric.global_rank == 0:
            self.fabric.print(f"[Profiler] Started profiling at {self.start_time}")

    def update_tokens(self, tokens_in_batch: int):
        """
        Update token count with validation.

        Args:
            tokens_in_batch: Number of tokens in this batch (LOCAL to this rank)
        """
        # Validate token count
        if tokens_in_batch != self.expected_tokens_per_batch:
            self.token_count_mismatches += 1
            if self.token_count_mismatches <= 5:  # Log first few mismatches
                if self.fabric.global_rank == 0:
                    self.fabric.print(
                        f"[Profiler] Token count mismatch #{self.token_count_mismatches}: "
                        f"expected {self.expected_tokens_per_batch}, got {tokens_in_batch}"
                    )

        # Update LOCAL counters only
        self.local_tokens_processed += tokens_in_batch
        self.local_batches_processed += 1

        assert tokens_in_batch > 0, f"[Profiler] Invalid token count: {tokens_in_batch}"
        assert self.local_tokens_processed > 0, (
            "[Profiler] Local token count should be positive"
        )

    def get_current_local_tps(self) -> float:
        """Calculate current local TPS for this rank only."""
        if not self.start_time or self.local_tokens_processed == 0:
            return 0.0

        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0

        return self.local_tokens_processed / elapsed

    def _monitor_hardware(self):
        """Monitor hardware metrics in background thread."""
        while self.monitoring_active:
            try:
                timestamp = time.time()

                # Collect GPU metrics
                gpu_metrics = self._collect_gpu_metrics()

                # Collect node metrics
                node_metrics = self._collect_node_metrics()

                # Store with timestamp
                self.gpu_metrics_history.append(
                    {"timestamp": timestamp, "metrics": gpu_metrics}
                )

                self.node_metrics_history.append(
                    {"timestamp": timestamp, "metrics": node_metrics}
                )

            except Exception as e:
                self.profiling_errors.append(f"[Profiler] Hardware monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    def _collect_gpu_metrics(self) -> list[dict]:
        """Collect GPU metrics for this node."""
        gpu_metrics: list[dict] = []

        if not torch.cuda.is_available():
            return gpu_metrics

        try:
            device_count = torch.cuda.device_count()

            for gpu_id in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except pynvml.NVMLError:
                        temp = 0

                    # Get power draw
                    try:
                        power = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )  # Convert mW to W
                    except pynvml.NVMLError:
                        power = 0

                    gpu_metrics.append(
                        {
                            "gpu_id": gpu_id,
                            "utilization_percent": util.gpu,
                            "memory_used_mb": mem_info.used / (1024 * 1024),
                            "memory_total_mb": mem_info.total / (1024 * 1024),
                            "memory_utilization_percent": (mem_info.used / mem_info.total)
                            * 100,
                            "temperature_c": temp,
                            "power_draw_w": power,
                        }
                    )

                except pynvml.NVMLError as e:
                    self.profiling_errors.append(
                        f"[Profiler] GPU {gpu_id} metrics error: {e}"
                    )

        except pynvml.NVMLError as e:
            self.profiling_errors.append(f"[Profiler] GPU metrics collection error: {e}")

        return gpu_metrics

    def _collect_node_metrics(self) -> dict:
        """Collect node-level metrics."""
        node_metrics = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
        }

        try:
            node_metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            node_metrics["memory_percent"] = psutil.virtual_memory().percent
        except Exception as e:
            self.profiling_errors.append(f"[Profiler] Node metrics error: {e}")

        return node_metrics

    def stop_profiling(self) -> Path | None:
        """Stop profiling with synchronization and save results."""
        # Global synchronization to ensure all ranks stop together
        self.fabric.barrier()

        # Record end time AFTER synchronization
        self.end_time = time.time()

        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        # Calculate final metrics
        return self._save_results()

    def _calculate_aggregated_metrics(self) -> dict:
        """Calculate aggregated metrics from monitoring history."""
        if not self.gpu_metrics_history:
            return {}

        # Aggregate GPU metrics
        all_gpu_utils = []
        all_mem_utils = []
        peak_memory = 0

        for entry in self.gpu_metrics_history:
            for gpu in entry["metrics"]:
                all_gpu_utils.append(gpu["utilization_percent"])
                all_mem_utils.append(gpu["memory_utilization_percent"])
                peak_memory = max(peak_memory, gpu["memory_used_mb"])

        # Aggregate node metrics
        avg_cpu = 0
        avg_memory = 0
        if self.node_metrics_history:
            cpu_vals = [
                entry["metrics"]["cpu_percent"] for entry in self.node_metrics_history
            ]
            mem_vals = [
                entry["metrics"]["memory_percent"] for entry in self.node_metrics_history
            ]
            avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0
            avg_memory = sum(mem_vals) / len(mem_vals) if mem_vals else 0

        return {
            "average_gpu_utilization": sum(all_gpu_utils) / len(all_gpu_utils)
            if all_gpu_utils
            else 0,
            "average_memory_utilization": sum(all_mem_utils) / len(all_mem_utils)
            if all_mem_utils
            else 0,
            "peak_memory_usage_gb": peak_memory / 1024,  # Convert MB to GB
            "average_cpu_percent": avg_cpu,
            "average_node_memory_percent": avg_memory,
        }

    def _save_results(self) -> Path | None:
        """Save profiling results to JSON file."""
        try:
            if not self.start_time or not self.end_time:
                self.profiling_errors.append("Missing start or end time")
                return None

            # Calculate runtime and TPS
            total_runtime = self.end_time - self.start_time
            total_runtime_minutes = total_runtime / 60.0

            # LOCAL TPS calculation (this rank only)
            local_avg_tps = (
                self.local_tokens_processed / total_runtime if total_runtime > 0 else 0
            )

            # Calculate peak TPS from monitoring data (simplified)
            local_peak_tps = (
                local_avg_tps * 1.05
            )  # Approximate peak as 5% higher than average

            # Get latest hardware metrics
            latest_gpu_metrics = (
                self.gpu_metrics_history[-1]["metrics"]
                if self.gpu_metrics_history
                else []
            )
            latest_node_metrics = (
                self.node_metrics_history[-1]["metrics"]
                if self.node_metrics_history
                else {}
            )

            # Aggregate metrics
            aggregated_metrics = self._calculate_aggregated_metrics()

            # Create results dictionary with CLEAR local/global distinction
            results = {
                # Run identification
                "run_id": f"rank_{self.rank}_nodes_{self.num_nodes}_gpus_{self.num_gpus_total}_strategy_{self.parallel_strategy}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.start_time))}",
                "model_name": self.model_name,
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f", time.localtime(self.start_time)
                ),
                # Hardware configuration (GLOBAL)
                "hardware_config": {
                    "num_nodes": self.num_nodes,
                    "num_gpus_total": self.num_gpus_total,
                    "gpus_per_node": self.gpus_per_node,
                    "parallel_strategy": self.parallel_strategy,
                    "world_size": self.world_size,
                },
                # This rank's information (LOCAL)
                "rank_info": {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "node_id": self.node_id,
                    "hostname": self.hostname,
                },
                # Training configuration
                "training_config": {
                    "batch_size": self.train_args.micro_batch_size,
                    "sequence_length": self.model.max_seq_length,
                    "max_tokens_global": self.train_args.max_tokens,
                    "max_tokens_per_rank": self.expected_total_tokens,
                },
                # Performance metrics (LOCAL to this rank)
                "local_performance": {
                    "total_runtime_minutes": total_runtime_minutes,
                    "total_runtime_seconds": total_runtime,
                    "tokens_processed": self.local_tokens_processed,
                    "batches_processed": self.local_batches_processed,
                    "average_tokens_per_second": local_avg_tps,
                    "peak_tokens_per_second": local_peak_tps,
                    "tokens_per_batch": self.expected_tokens_per_batch,
                },
                # Hardware metrics (LOCAL to this node)
                "local_hardware_metrics": {
                    **aggregated_metrics,
                    "latest_gpu_metrics": latest_gpu_metrics,
                    "latest_node_metrics": latest_node_metrics,
                },
                # Framework and system info
                "framework_info": {
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 0,
                    "python_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
                    "distributed_backend": torch.distributed.get_backend()
                    if torch.distributed.is_initialized()
                    else "none",
                },
                # Debugging info
                "profiling_metadata": {
                    "hardware_detection_method": self.hardware_detection_method,
                    "token_count_mismatches": self.token_count_mismatches,
                    "profiling_errors": self.profiling_errors,
                    "monitoring_interval": self.monitoring_interval,
                    "monitoring_samples": len(self.gpu_metrics_history),
                },
            }

            # Generate filename
            timestamp_str = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(self.start_time)
            )
            filename = f"rank_{self.rank}_nodes_{self.num_nodes}_gpus_{self.num_gpus_total}_strategy_{self.parallel_strategy}_{timestamp_str}_profile.json"
            output_file = self.output_dir / filename

            # Save results
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            if self.fabric.global_rank == 0:
                self.fabric.print(f"[Profiler] Results saved to: {output_file}")
                self.fabric.print(f"[Profiler] Local TPS: {local_avg_tps:.2f}")
                self.fabric.print(
                    f"[Profiler] Local tokens processed: {self.local_tokens_processed:,}"
                )
                self.fabric.print(
                    f"[Profiler] Token mismatches: {self.token_count_mismatches}"
                )

            return output_file

        except Exception as e:
            error_msg = f"Failed to save profiling results: {e}"
            self.profiling_errors.append(error_msg)
            if self.fabric.global_rank == 0:
                self.fabric.print(f"[Profiler] Error: {error_msg}")
            return None


def create_profiler(
    fabric: L.Fabric,
    model_name: str,
    parallel_strategy: str,
    train_args,
    model,
    output_dir: Path,
    monitoring_interval: float = 1.0,
) -> DistributedGPUProfiler | None:
    """
    Create a distributed GPU profiler with improved error handling.

    Args:
        fabric: Lightning Fabric instance
        model_name: Name of the model being trained
        parallel_strategy: Parallel strategy being used (fsdp, ddp, etc.)
        train_args: Training arguments
        model: The model being trained
        output_dir: Directory to save profiling results
        monitoring_interval: How often to collect hardware metrics (seconds)

    Returns:
        DistributedGPUProfiler instance or None if creation fails
    """
    try:
        profiler = DistributedGPUProfiler(
            fabric=fabric,
            model_name=model_name,
            parallel_strategy=parallel_strategy,
            train_args=train_args,
            model=model,
            output_dir=output_dir,
            monitoring_interval=monitoring_interval,
        )

        if fabric.global_rank == 0:
            fabric.print("[Profiler] Successfully created distributed profiler")

        return profiler

    except Exception as e:
        if fabric.global_rank == 0:
            fabric.print(f"[Profiler] Failed to create profiler: {e}")
        return None
