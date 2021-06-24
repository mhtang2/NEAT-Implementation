import functools
import time


times = {}


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3

        name = func.__name__
        if name not in times:
            times[name] = 0
        times[name] += run_time
        return value
    return wrapper_timer


def resetTimer():
    global times
    times = {}


def printTimer(scale=1.0):
    x = {k: v/scale for k, v in times.items()}
    print({k: v for k, v in sorted(x.items(), key=lambda item: item[1])})


def totalTime():
    return sum(times.values())

def getTimes():
    return times
