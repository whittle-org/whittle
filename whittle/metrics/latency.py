import torch
import torch.profiler
from torch.profiler import record_function


# Define a sample language model (e.g., a simple RNN or transformer-based model)
def get_total_cpu_gpu_runtime(prof: torch.profiler.profile) -> tuple[int, int]:
    """
    Calculates the total runtime for CPU and GPU (CUDA) from the profiler events.

    This function extracts and sums the self CPU and CUDA times from the PyTorch profiler events.
    It handles both legacy and Kineto profiler events for accurate CPU and CUDA profiling.

    Args:
        prof (torch.profiler.profile): A PyTorch profiler object containing profiling events.

    Returns:
        tuple[int, int]: A tuple where the first value is the total CPU time (in microseconds),
                         and the second value is the total CUDA time (in microseconds).
    """
    events = prof.events()
    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])

    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == torch.device("cpu").type:
            # In legacy profiler, kernel info is stored in CPU events
            if evt.is_legacy:
                sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == torch.device("cuda").type:
            # In Kineto profiler, there are events with the correct device type (e.g., CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total

    return sum_self_cpu_time_total, sum_self_cuda_time_total


def compute_latency(
    model: torch.nn.Module,
    use_cuda: bool = False,
    batch_size: int = 8,
    n_samples: int = 10,
) -> float:
    """
    Profiles the latency of a PyTorch model for inference.

    This function measures the average latency of the model's forward pass over a specified number of samples
    using PyTorch's profiler. It supports both CPU and CUDA profiling.

    Args:
        model (torch.nn.Module): the LitGPT profiled.
        use_cuda (bool, optional): If True and CUDA is available, the model will be moved to the GPU for profiling. Defaults to False.
        batch_size (int, optional): The batch size for the input tensor. Defaults to 8.
        n_samples (int, optional): The number of samples to profile after the warm-up phase. Defaults to 10.

    Returns:
        float: The average inference time per sample in milliseconds.
    """
    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, model.max_seq_length)
    )
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Use PyTorch profiler to record compute_latency
    model.eval()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=10, active=n_samples),
    ) as profiler:
        # Actual forward pass inside the profiler
        with record_function("model_inference"):
            for i in range(11 + n_samples):
                _ = model(input_tensor)
                profiler.step()

    # Summing up CPU and CUDA times from the profiler using the correct methods
    cuda_time_us, cpu_time_us = get_total_cpu_gpu_runtime(profiler)

    # Convert time to milliseconds
    total_time_ms = (cpu_time_us + cuda_time_us) / 1000
    model = model.cpu()
    return total_time_ms / n_samples
