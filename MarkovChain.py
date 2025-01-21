import numpy as np
import tqdm
import tqdm.notebook as tnotebook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import itertools
from dataclasses import dataclass
from collections import defaultdict

max_up5_count = 6
max_up6_count = 6

@dataclass(unsafe_hash=True)
class GachaInfo:
    watermark_5: int
    watermark_6: int
    up5_count: int
    up6_count: int

    def __init__(self, watermark_5=0, watermark_6=0, up5_count=0, up6_count=0):
        self.watermark_5 = watermark_5
        self.watermark_6 = watermark_6
        self.up5_count = min(up5_count, max_up5_count)
        self.up6_count = min(up6_count, max_up6_count)

prob_6_base = np.zeros(80)
prob_6_base[0: 64] = 0.008
prob_6_base[65: 80] = 0.008 + np.arange(1, 16) * 0.05
prob_6_base[79] = 1.

prob_5_base = np.zeros(10)
prob_5_base[0: 9] = 0.08
prob_5_base[9] = 1.

prob_6_is_up_6 = 0.5
prob_5_is_up_5 = 0.5

def transition(
        state: GachaInfo,
        force_up6: bool = False,
    ) -> list[tuple[float, GachaInfo]]:
    ret_prob = []
    ret_state = []

    water_5 = state.watermark_5
    water_6 = state.watermark_6
    up5 = state.up5_count
    up6 = state.up6_count

    if force_up6 and up6 == 0:
        return [(1., GachaInfo(0, 0, up5, 1))]

    prob_6 = prob_6_base[water_6]
    prob_5 = np.clip(prob_5_base[water_5], 0., 1. - prob_6)
    prob_4 = 1 - prob_5 - prob_6

    # got a 6, is up
    ret_prob.append(prob_6 * prob_6_is_up_6)
    ret_state.append(GachaInfo(0, 0, up5, up6 + 1))
    # got a 6, is not up
    ret_prob.append(prob_6 * (1 - prob_6_is_up_6))
    ret_state.append(GachaInfo(0, 0, up5, up6))

    if water_6 < 79: # has chance to get a 5
        # got a 5, is up
        ret_prob.append(prob_5 * prob_5_is_up_5)
        ret_state.append(GachaInfo(0, water_6 + 1, up5 + 1, up6))
        # got a 5, is not up
        ret_prob.append(prob_5 * (1 - prob_5_is_up_5))
        ret_state.append(GachaInfo(0, water_6 + 1, up5, up6))
    
        if water_5 < 9:
            # got a 4
            ret_prob.append(prob_4)
            ret_state.append(GachaInfo(water_5 + 1, water_6 + 1, up5, up6))
    
    assert np.isclose(np.sum(ret_prob), 1., atol=1e-6)
    return list(zip(ret_prob, ret_state))

# save transition matrix
state_list = [
    GachaInfo(watermark_5=watermark_5, watermark_6=watermark_6, up5_count=up5_count, up6_count=up6_count)
    for watermark_5 in range(10)
    for watermark_6 in range(80)
    for up5_count in range(max_up5_count + 1)
    for up6_count in range(max_up6_count + 1)
]
state_idx_mapping = {state: idx for idx, state in enumerate(state_list)}
transition_matrix = scipy.sparse.dok_matrix((len(state_list), len(state_list)))
for state in state_list:
    for prob, next_state in transition(state):
        transition_matrix[state_idx_mapping[state], state_idx_mapping[next_state]] += prob

transition_force_up6_matrix = scipy.sparse.dok_matrix((len(state_list), len(state_list)))
for state in state_list:
    for prob, next_state in transition(state, force_up6=True):
        transition_force_up6_matrix[state_idx_mapping[state], state_idx_mapping[next_state]] += prob

transition_matrix = transition_matrix.tocsc()
transition_force_up6_matrix = transition_force_up6_matrix.tocsc()

get_stat_idx = lambda state: state.up5_count * (max_up6_count + 1) + state.up6_count
get_stat_idx_direct = lambda count5, count6: count5 * (max_up6_count + 1) + count6
get_stat = lambda idx: (idx // (max_up6_count + 1), idx % (max_up6_count + 1))

stat_state_count = (max_up5_count + 1) * (max_up6_count + 1)
stat_lookup = scipy.sparse.dok_matrix((len(state_list), stat_state_count))

for state in state_list:
    stat_lookup[state_idx_mapping[state], get_stat_idx(state)] = 1.
stat_lookup = stat_lookup.tocsc()

stat_lookup_accumulate_over_5 = scipy.sparse.dok_matrix((len(state_list), stat_state_count))
for state in state_list:
    c5, c6 = state.up5_count, state.up6_count
    for count5 in range(1, c5 + 1):
        stat_lookup_accumulate_over_5[state_idx_mapping[state], get_stat_idx_direct(count5, c6)] = 1.
stat_lookup_accumulate_over_5 = stat_lookup_accumulate_over_5.tocsc()

stat_lookup_accumulate_over_6 = scipy.sparse.dok_matrix((len(state_list), stat_state_count))
for state in state_list:
    c5, c6 = state.up5_count, state.up6_count
    for count6 in range(1, c6 + 1):
        stat_lookup_accumulate_over_6[state_idx_mapping[state], get_stat_idx_direct(c5, count6)] = 1.
stat_lookup_accumulate_over_6 = stat_lookup_accumulate_over_6.tocsc()

assert np.allclose(transition_matrix.sum(axis=1), 1., atol=1e-3)
assert np.allclose(transition_force_up6_matrix.sum(axis=1), 1., atol=1e-3)

def simulate_matrix(n: int) -> pd.DataFrame:
    mc_distr = np.zeros(len(state_list))
    mc_distr[state_idx_mapping[GachaInfo()]] = 1.

    trace = []
    trace.append(mc_distr)
    for it in tnotebook.tqdm(range(n)):
        tr_matrix = transition_matrix if it != 120 else transition_force_up6_matrix

        mc_distr = mc_distr @ tr_matrix
        trace.append(mc_distr)

    return np.vstack(trace)

def trace_to_stats(trace: np.ndarray) -> pd.DataFrame:
    assert trace.ndim == 2
    num_iter, num_states = trace.shape

    stats = []
    for it in range(num_iter):
        for idx in range(num_states):
            data = {
                "iteration": it,
                "up5_count": state_list[idx].up5_count,
                "up6_count": state_list[idx].up6_count,
                "prob": trace[it, idx],
            }
            stats.append(data)
    
    return pd.DataFrame(stats)

def trace_to_stats_matrix(
        trace: np.ndarray,
        lookup_matrix=None,
    ) -> pd.DataFrame:
    assert trace.ndim == 2
    num_iter, num_states = trace.shape
    num_stats = stat_state_count

    if lookup_matrix is None:
        lookup_matrix = stat_lookup

    stat_probs = trace @ lookup_matrix

    stats = []
    for it in range(num_iter):
        for idx in range(num_stats):
            count_5, count_6 = get_stat(idx)
            data = {
                "iteration": it,
                "up5_count": count_5,
                "up6_count": count_6,
                "prob": stat_probs[it, idx],
            }
            stats.append(data)
    
    return pd.DataFrame(stats)
    

res = simulate_matrix(1000)

def plot_stat(stat_df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    stat_5 = stat_df.drop(columns=["up6_count"])
    stat_5 = stat_5.groupby(["iteration", "up5_count"]).sum().reset_index()
    sns.lineplot(data=stat_5, x="iteration", y="prob", hue="up5_count", ax=ax[0])

    stat_6 = stat_df.drop(columns=["up5_count"])
    stat_6 = stat_6.groupby(["iteration", "up6_count"]).sum().reset_index()
    sns.lineplot(data=stat_6, x="iteration", y="prob", hue="up6_count", ax=ax[1])

    fig.tight_layout()

    return fig, ax

plot_stat(trace_to_stats_matrix(res))

plt.show()

def plot_stat_5(stat_df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    stat_5 = stat_df.drop(columns=["up6_count"])
    stat_5 = stat_5.groupby(["iteration", "up5_count"]).sum().reset_index()
    sns.lineplot(data=stat_5, x="iteration", y="prob", hue="up5_count", ax=ax)

    fig.tight_layout()

    return fig, ax

def plot_stat_6(stat_df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    stat_6 = stat_df.drop(columns=["up5_count"])
    stat_6 = stat_6.groupby(["iteration", "up6_count"]).sum().reset_index()
    sns.lineplot(data=stat_6, x="iteration", y="prob", hue="up6_count", ax=ax)

    fig.tight_layout()

    return fig, ax

plot_stat_5(trace_to_stats_matrix(res, lookup_matrix=stat_lookup_accumulate_over_5))

plt.show()

plot_stat_6(trace_to_stats_matrix(res, lookup_matrix=stat_lookup_accumulate_over_6))

plt.show()

