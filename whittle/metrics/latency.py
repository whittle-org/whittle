import torch
import torch.profiler
from torch.profiler import record_function


# Define a sample language model (e.g., a simple RNN or transformer-based model)
def get_total_cpu_gpu_runtime(prof):
    # from torch.autograd.profiler_utils.py (_build_table)
    # results is in micro-seconds
    from torch.autograd import DeviceType

    events = prof.events()
    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            # in legacy profiler, kernel info is stored in cpu events
            if evt.is_legacy:
                sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == DeviceType.CUDA:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total
    return sum_self_cpu_time_total, sum_self_cuda_time_total


def profile_model_latency(model, use_cuda=False, batch_size=8, n_samples=10):
    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, model.max_seq_length)
    )
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Use PyTorch profiler to record latency
    model.eval()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],  # CPU and CUDA
        schedule=torch.profiler.schedule(
            wait=1, warmup=10, active=n_samples
        ),  # wait for 4 batches, warmup for 5 profile one
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


def update_config(
    config,
    sub_network_n_embd: int,
    sub_network_intermediate_size: int,
    sub_network_num_heads: int,
    sub_network_n_layers: int,
    sub_network_query_groups=None,
    sub_network_head_size=None,
):
    config.n_embd = sub_network_n_embd
    config.intermediate_size = sub_network_intermediate_size
    config.n_head = sub_network_num_heads
    if sub_network_query_groups is not None:
        config.n_query_groups = sub_network_query_groups
    if sub_network_head_size is not None:
        config.head_size = sub_network_head_size
    return config


if __name__ == "__main__":
    # profile directly litgpt model to avoid indexing, and other inefficiencies
    from litgpt.model import GPT as LitGPT
    from litgpt import Config

    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 128
    config.intermediate_size = 128 * 4
    config.n_layer = 4
    model = LitGPT(config)
    print(f"Full model cuda {profile_model_latency(model,use_cuda=True)} ms")
    config = update_config(config, 64, 64 * 4, 4, 2, 4)
    model = LitGPT(config)
    print(f"Mini model cuda {profile_model_latency(model,use_cuda=True)} ms")
