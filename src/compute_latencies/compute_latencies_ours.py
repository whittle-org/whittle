import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from whittle.models.gpt import GPT

# from generate.base import generate
import pickle
from litgpt import Config
from llm_compression.search_spaces import search_spaces


def get_latency_from_string(s: str, sub_str: str = "CPU time total: "):
    index = s.find(sub_str)  # find the index of the substring
    if index != -1:  # check if substring is found
        # print(s[-2])
        # print(s[-3:-1])
        flag_units = (s[-3:-1] == "ms") or s[-3:-1] == "us" or s[-3:-1] == "ns"
        if s[-2] == "s" and not flag_units:
            # print(s)
            unit = "s"
            content = s[
                index + len(sub_str) : -3
            ]  # extract content following substring
        else:
            # print(s)
            unit = s[-3:-1]
            content = s[
                index + len(sub_str) : -3
            ]  # extract content following substring
        return content, unit
    else:
        print("Substring not found")


def torch_profiler_llm(
    model: torch.nn.Module,
    model_inputs_x: torch.Tensor,
    model_inputs_y: torch.Tensor,
    n: int = 10,
    use_gpu: bool = True,
    use_cpu: bool = False,
    gpu_dtype: torch.dtype = torch.bfloat16,
):
    times_profiler_cpu = []
    times_profiler_gpu = []
    mean_latency_gpu = None
    std_latency_gpu = None
    mean_latency_cpu = None
    std_latency_cpu = None
    unit_cpu = []
    unit_gpu = []
    if use_gpu:
        model = model.cuda()
        model_inputs_y = model_inputs_y.cuda()
        model_inputs_x = model_inputs_x.cuda()
        for i in range(n):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                with record_function("model_inference"):
                    with torch.amp.autocast(device_type="cuda", dtype=gpu_dtype):
                        model(
                            model_inputs_x
                        )  # 120, temperature=1.0, top_k=50, eos_id=50256)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
            time_gpu, unit_gpu_curr = get_latency_from_string(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=1),
                "CUDA time total: ",
            )
            # time_cpu, unit_cpu = get_latency_from_string(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
            # times_profiler_cpu.append(float(time_cpu))
            times_profiler_gpu.append(float(time_gpu))
            unit_gpu.append(unit_gpu_curr)
        mean_latency_gpu = np.mean(times_profiler_gpu)
        std_latency_gpu = np.std(times_profiler_gpu)
    if use_cpu:
        model = model.cpu()
        model_inputs_y = model_inputs_y.cpu()
        model_inputs_x = model_inputs_x.cpu()
        # inputs = inputs.to("cpu")
        for i in range(n):
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(model_inputs_x)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
            time_cpu, unit_cpu_curr = get_latency_from_string(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=1)
            )
            times_profiler_cpu.append(float(time_cpu))
            unit_cpu.append(unit_cpu_curr)
        mean_latency_cpu = np.mean(times_profiler_cpu)
        std_latency_cpu = np.mean(times_profiler_cpu)
    return (
        mean_latency_cpu,
        std_latency_cpu,
        mean_latency_gpu,
        std_latency_gpu,
        unit_cpu,
        unit_gpu,
        times_profiler_gpu,
        times_profiler_cpu,
    )


config = Config.from_file(
    str(
        f"//p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/checkpoints/meta-llama/Meta-Llama-3.1-8B/"
        + "model_config.yaml"
    )
)
config.fix_head_size = True
import pickle

with open(
    "//p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/checkpoints/meta-llama/Meta-Llama-3.1-8B/grid_llama2_mag_mean.pkl",
    "rb",
) as f:
    a = pickle.load(f)
latencies = []
model = GPT(config)
model.to(torch.bfloat16)
search_space = search_spaces["llama2"](config)
for arch in a[::-1]:
    # sub_network_config = search_space.cast(arch)
    # del arch["random_layers"]
    model.set_sub_network(**arch)
    inputs = torch.randint(0, config.vocab_size, (2, 512))  # .cuda()
    inputs_y = torch.randint(0, config.vocab_size, (2, 512))  # .cuda()
    out = torch_profiler_llm(model, inputs, inputs_y)
    print(out)
    latency = np.mean(out[-2])
    latencies.append(latency)
    print(latencies)
import pickle

with open("latencies_llama-3.1-8B.pkl", "wb") as f:
    pickle.dump(latencies, f)
