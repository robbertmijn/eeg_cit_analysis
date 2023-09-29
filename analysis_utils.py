import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import _eeg_preprocessing as preprocessing
import numpy as np
from datamatrix import convert as cnv, DataMatrix, operations as ops, functional as fnc, series as srs
import mne
from matplotlib import pyplot as plt
from eyelinkparser import parse, defaulttraceprocessor
from mne.time_frequency import tfr_morlet
import time_series_test as tst
import warnings
from itertools import product

COLDICT = {'target': '#c94733', 'probe': '#0d5b26', 'control': '#2e5fa1', 'distractor': 'gray'}
CHANNELS = ['Fz', 'Pz', 'Cz']
FULL_FREQS = np.arange(4, 30, 2)
T1_TRIGGER = 2
RSVP_TRIGGER = 1
PERMUTATIONS = 1000
DV = ['pz', 'theta', 'fz', 'cz', 'pupil']
LEVEL = ['probe', 'control', 'target']
CONDITION = ['familiar', 'unfamiliar']

@fnc.memoize(persistent=True, folder='_loaded')
def proc_subject(subject, stats=True):

    print(f'read subject {subject}')
    raw, events, metadata = eet.read_subject(subject, eeg_preprocessing = True)

    print(f'extracting dm {subject}')
    sdm = extract_dm(raw, events, metadata)

    if stats:
        print(f'calculate all stats {subject}')
        stats = do_all_cpt(sdm.T1_correct == 1, DV, LEVEL, CONDITION)
    else:
        stats = None

    return sdm, stats


@fnc.memoize(persistent=True, folder='_extracted')
def extract_dm(raw, events, metadata):
    # Select T1 triggers from all triggers
    t1_triggers = eet.epoch_trigger(events, T1_TRIGGER)

    # Create dm from metadata
    sdm = cnv.from_pandas(metadata)

    extract_pupil(sdm, raw, t1_triggers, metadata)
    extract_ERP(sdm, raw, t1_triggers, metadata)
    extract_TFR(sdm, raw, t1_triggers, metadata)

    # Remove practice trials
    sdm = sdm.practice == 'no'

    return sdm


def extract_pupil(sdm, raw, t1_triggers, metadata):
    # Edit object in-place
    pupil = eet.PupilEpochs(raw, t1_triggers,
                                    tmin=-.1, tmax=1, 
                                    metadata=metadata,
                                    baseline=(-.05, 0))
    sdm.pupil = cnv.from_mne_epochs(pupil)
    sdm.pupil = sdm.pupil[:, ...]
    

def extract_ERP(sdm, raw, t1_triggers, metadata):
    # Edit object in-place
    # eet.autoreject_epochs.clear()
    # Extract ERP data
    epochs = eet.autoreject_epochs(raw, t1_triggers,
                                    tmin=-.1, tmax=1, metadata=metadata,
                                    picks=CHANNELS)
    sdm.erp_T1 = cnv.from_mne_epochs(epochs)
    sdm.pz = sdm.erp_T1[:, 'Pz']
    sdm.fz = sdm.erp_T1[:, 'Fz']
    sdm.cz = sdm.erp_T1[:, 'Cz']
    del sdm.erp_T1


def extract_TFR(sdm, raw, t1_triggers, metadata):
    # Edit object in-place
    # Extract time-frequency data
    tfr_epochs = eet.autoreject_epochs(raw, t1_triggers,
                            tmin=-.5, tmax=2, metadata=metadata,
                            picks=CHANNELS, baseline=(-.1, 0))
    morlet = tfr_morlet(tfr_epochs, FULL_FREQS, decim=4, n_cycles=2,
                        average=False, return_itc=False)
    morlet.crop(-.1, 1)
    sdm.tfr_T1 = cnv.from_mne_tfr(morlet)
    sdm.tfr_T1_z = z_by_freq(sdm.tfr_T1)
    del sdm.tfr_T1  # save memory
    sdm.theta = sdm.tfr_T1_z[:, ..., :2][:, ...]
    sdm.alpha = sdm.tfr_T1_z[:, ..., 2:5][:, ...]
    sdm.beta = sdm.tfr_T1_z[:, ..., 5:][:, ...]


def get_merged_data(SUBJECTS, stats=True):
    dm = DataMatrix()
    all_stats = []
    for subject in SUBJECTS:
        sdm, stats = proc_subject(subject, stats)

        dm <<= sdm
        all_stats.append(stats)
        print(f'finished {subject}')
    return dm, all_stats


def z_by_freq(col):
    """Performs z-scoring across trials, channels, and time points but 
    separately for each frequency.
    
    Parameters
    ----------
    col: MultiDimensionalColumn
    
    Returns
    -------
    MultiDimensionalColumn
    """
    zcol = col[:]
    for i in range(zcol.shape[2]):
        zcol._seq[:, :, i] = (
            (zcol._seq[:, :, i] - np.nanmean(zcol._seq[:, :, i]))
            / np.nanstd(zcol._seq[:, :, i])
        )
    return zcol


@fnc.memoize(persistent=True, folder='_stats')
def do_cpt(dm, dv, level, condition):
    result = DataMatrix(1)
    tdm = dm.T1 == {'distractor', level}
    tdm = tdm.condition == condition
    groups = None if len(dm.subject_nr.unique) > 2 else "subject_nr"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        temp = tst.lmer_permutation_test(tdm, formula=f'{dv} ~ T1', groups=groups, iterations=PERMUTATIONS, cluster_p_threshold=.05, suppress_convergence_warnings=True)
    result.dv = dv
    result.level = level
    result.condition = condition
    ref = level if level != 'control' else 'distractor'
    if not temp[f'T1[T.{ref}]']:
        pvalue = 1
        result.start = 0
        result.end = 0
        result.zsum = 0
        result.p = pvalue
    else:
        pvalue = 1 - temp[f'T1[T.{ref}]'][0][3]
        result.start = temp[f'T1[T.{ref}]'][0][0]
        result.end = temp[f'T1[T.{ref}]'][0][1]
        result.zsum = temp[f'T1[T.{ref}]'][0][2]
        result.p = pvalue
    return result


@fnc.memoize(persistent=True, folder='_stats')
def do_all_cpt(dm, dv, level, condition):
    stats = DataMatrix()

    tests = list(product(dv, level, condition))

    for d, l, c in tests:
        print(f'pp {dm.subject_nr.unique[0]}, dv: {d}, level: {l}, condition: {c}')
        # do_cpt.clear()
        stats <<= do_cpt(dm, d, l, c)
    return stats


def set_xticks(ax, depth):
    ax.set_xticks(ticks=np.arange(4, depth, 100), labels=np.arange(0,  1000, 400))


def equal_axes(axes, depth):

    for ax in axes.flat:
        set_xticks(ax, depth)

    for ax in axes:
        min = np.min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
        max = np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
        ax[0].set_ylim(min, max)
        ax[1].set_ylim(min, max)


def ERP_plots(dm, title, stats=None):
    fcol = 0
    fig, axes = plt.subplots(3, 2, constrained_layout=True, sharex = True)
    for condition, cdm in ops.split(dm.condition):
        for t1, tdm in ops.split(cdm.T1):
            plot_dv(axes[0, fcol], tdm, t1, 'pupil', condition, stats)
            plot_dv(axes[1, fcol], tdm, t1, 'pz', condition, stats)
            plot_dv(axes[2, fcol], tdm, t1, 'theta', condition, stats)

        fcol += 1

    equal_axes(axes, tdm.pupil.depth)
    fig.supxlabel('Time since T1 onset (ms)')
    axes[0,0].legend(frameon=False)
    fig.suptitle(title)


def plot_dv(ax, dm, t1, dv, condition, stats=None):
    x = np.arange(0, dm[dv].depth)
    n = (~np.isnan(dm[dv])).sum(axis=0)
    y = dm[dv].mean
    std = y  / np.sqrt(n)
    ax.plot(y, label=t1)
    ax.fill_between(x, y - std, y + std, alpha=.2, color=COLDICT[t1])
    ax.set_title(f'{dv} {condition}')
    
    if stats and t1 in ['probe', 'control', 'target']:
        stat = (stats.condition == condition) & (stats.dv == dv) & (stats.level == t1)
        if stat.p < .05:
            start = stat[0].start
            end = stat[0].end
            ax.plot(x[start:end], y[start:end], linewidth=3, color=COLDICT[t1])