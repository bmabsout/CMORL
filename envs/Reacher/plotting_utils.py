import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import fft
from scipy.fftpack import fft
from scipy import stats

theme = {'blue': '#265285', 'purple': '#8e0178', 'green': '#38aa38', 'orange': '#ee6e00', 'red': '#e41a1c', 'pink': '#984ea3'
         }

colors = list(theme.values())
line_styles = ['-', '--', ':', '-.']


def fourier_transform(actions, T):
    N = len(actions)
    yf = fft(actions)
    freq = np.linspace(0.0, 1.0/(2*T), N//2)
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    return freq, amplitudes


def smoothness(amplitudes):
    normalized_freqs = np.linspace(0, 1, amplitudes.shape[0])
    return np.mean(amplitudes * normalized_freqs)


def motors_smoothness(motors):
    smoothnesses = []
    for i in range(motors.shape[1]):
        freqs, amplitudes = fourier_transform(motors[:, i]*2-1, 1)
        smoothnesses.append(smoothness(amplitudes))
    return np.mean(smoothnesses)


def center_of_mass(freqs, amplitudes):
    return np.sum(freqs * amplitudes) / sum(amplitudes)


def cut_data(actionss, ep_lens):
    median = int(np.median(ep_lens))
    print("median:", median)
    same_len = map(lambda x: x[:median], filter(
        lambda x: len(x) >= median, actionss))
    return same_len


def to_array_truncate(l):
    min_len = min(map(len, l))
    return np.array(list(map(lambda x: x[:min_len], l)))


def combine(fouriers):
    freqs = fouriers[0][0]
    amplitudess = np.array(list(map(lambda x: x[1], fouriers)))

    amplitudes = np.mean(amplitudess, axis=0)
    return freqs, amplitudes


def from_actions(actionss, ep_lens):
    fouriers = list(map(fourier_transform, cut_data(actionss, ep_lens)))
    return combine(fouriers)


def plot_fourier(ax_m, freqs, amplitudes, amplitudes_std=None, main_color=theme['orange'], std_color=theme['blue']):
    if not (amplitudes_std is None):
        y = amplitudes + amplitudes_std
        ax_m.fill_between(freqs, 0, y, where=y >= 0,
                          facecolor=std_color, alpha=1, label="Mean$+\\sigma$")
    ax_m.fill_between(freqs, 0, amplitudes, where=amplitudes >=
                      0, facecolor=main_color, label="Mean")


def dict_elems_apply(fn, d):
    return {k: fn(d[k]) for k in d.keys()}


def dicts_list_to_list_dicts(l):
    return {k: [dic[k] for dic in l] for k in l[0]}


def plot_with_std(ax, x, y, y_std, **kwargs):
    line, = ax.plot(x, y, **kwargs)
    alpha = 1
    ax.fill_between(x, y-y_std, y+y_std,
                    color=line.get_color(), alpha=alpha*0.5)
    between, = ax.fill(np.NaN, np.NaN, color=line.get_color(), alpha=alpha*0.5)
    return (between, line)
