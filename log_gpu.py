import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

def log_gpu_usage(interval=1, log_file="gpu_usage_log.txt"):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # Use GPU 0; adjust for multiple GPUs

    with open(log_file, "w") as f:
        while True:
            utilization = nvmlDeviceGetUtilizationRates(handle)
            log_entry = f"Time: {time.ctime()}, GPU Usage: {utilization.gpu}%, Memory Usage: {utilization.memory}%\n"
            f.write(log_entry)
            f.flush()
            time.sleep(interval)

log_gpu_usage()

