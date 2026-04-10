import psutil
import os
import subprocess 
import time

def track_subprocess(command):
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps_proc = psutil.Process(proc.pid)

    cpu_time_start = ps_proc.cpu_times()
    mem_start = ps_proc.memory_info().rss

    # Wait for process to finish, but keep checking if it's still alive
    while proc.poll() is None:
        time.sleep(0.05)  # short sleep to avoid busy waiting

    try:
        cpu_time_end = ps_proc.cpu_times()
        mem_end = ps_proc.memory_info().rss
    except psutil.NoSuchProcess:
        # Fallback if process exited too fast
        cpu_time_end = cpu_time_start
        mem_end = mem_start

    cpu_time_used = (cpu_time_end.user + cpu_time_end.system) - (cpu_time_start.user + cpu_time_start.system)
    memory_used = (mem_end - mem_start) / (1024 * 1024)  # Convert to MB

    return round(cpu_time_used, 2), round(memory_used, 2)

def get_resource_usage():
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 ** 2)  # MB
        cpu_percent = process.cpu_percent(interval=1.0)  # %
        cpu_times = process.cpu_times()
        total_cpu_time = cpu_times.user + cpu_times.system  # seconds
        return rss, cpu_percent, total_cpu_time