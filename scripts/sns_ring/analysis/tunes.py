import numpy as np
import scipy.fft


def compute_tunes_phase_diff(coords):
    """Compute eigentunes by averaging turn-by-turn phase differences 
    in normalized x-x', y-y' planes.
    
    We assume `coords` is a list of shape (T, N, 6), where T is the number 
    of turns and N is the number of particles, and that the coordinates are
    normalized so that particles move in circles in the x-x' and y-y' planes.
    
    To do:
        - Handle varying number of particles per turn.
    """
    def get_phases_2d(x, xp):
        phases = np.arctan2(xp, x)
        phases[phases < 0.0] += (2.0 * np.pi)
        return phases
    
    def get_phases(X):
        phases_x = get_phases_2d(X[:, 0], X[:, 1])
        phases_y = get_phases_2d(X[:, 2], X[:, 3])
        phases = np.vstack([phases_x, phases_y]).T
        return phases
    
    phases_old_list = [get_phases(X) for X in coords[:-1]]
    phases_new_list = [get_phases(X) for X in coords[1:]]
    
    tunes_list = []
    for phases_old, phases_new in zip(phases_old_list, phases_new_list):
        tunes = (phases_old - phases_new) / (2.0 * np.pi)
        tunes[np.where(tunes < 0.0)] += 1.0
        tunes_list.append(tunes)
    tunes = np.mean(tunes_list, axis=0)
    return tunes


def fft(signal):
    """Return FFT without 0 component."""
    N = len(signal)
    M = N // 2
    freq = (1.0 / N) * np.arange(M)
    amp = (1.0 / M) * np.abs(scipy.fft.fft(signal)[:M])
    return freq[1:], amp[1:]


def compute_tunes_fft(coords):
    def compute_tune(signal):
        freq, amp = fft(signal)
        return freq[np.argmax(amp)]

    tunes = np.zeros((coords.shape[1], 2))
    for i in range(tunes.shape[0]):
        tune_x = compute_tune(coords[:, i, 0])
        tune_y = compute_tune(coords[:, i, 2])
        tunes[i, :] = [tune_x, tune_y]
    return tunes