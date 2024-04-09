"""
Collection of utility functions.
"""
import functools

import parselmouth
from frozendict import frozendict


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl


def freezeargs(func):
    """Transform mutable dictionary into immutable dictionary. Useful to be compatible with caching functions.

    Parameters
    ----------
    func : function
        Function to be wrapped.

    Returns
    -------
    function
        Wrapped function with immutable dictionary.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapped


def measure_praat_stats(sound: parselmouth.Sound, fmin: float, fmax: float):
    duration = parselmouth.praat.call(sound, "Get total duration")  # duration

    pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, fmin, fmax)  # create a praat pitch object
    meanF = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
    stdevF = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")  # get standard deviation

    harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", fmin, fmax)
    localJitter = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localAbsoluteJitter = parselmouth.praat.call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = parselmouth.praat.call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = parselmouth.praat.call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = parselmouth.praat.call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3,
                                            1.6)
    apq3Shimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        "duration": duration,
        "meanF": meanF,
        "stdevF": stdevF,
        "hnr": hnr,
        "localJitter": localJitter,
        "localAbsoluteJitter": localAbsoluteJitter,
        "rapJitter": rapJitter,
        "ppq5Jitter": ppq5Jitter,
        "ddpJitter": ddpJitter,
        "localShimmer": localShimmer,
        "localdbShimmer": localdbShimmer,
        "apq3Shimmer": apq3Shimmer,
        "aqpq5Shimmer": aqpq5Shimmer,
        "apq11Shimmer": apq11Shimmer,
        "ddaShimmer": ddaShimmer
    }
