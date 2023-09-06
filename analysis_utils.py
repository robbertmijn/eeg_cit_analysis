import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import _eeg_preprocessing as preprocessing
import numpy as np
from datamatrix import convert as cnv, DataMatrix, operations as ops, functional as fnc, series as srs
import mne
from matplotlib import pyplot as plt
from eyelinkparser import parse, defaulttraceprocessor
from mne.time_frequency import tfr_morlet
import time_series_test as tst


CHANNELS = ['Fz', 'Pz', 'Cz']
FULL_FREQS = np.arange(4, 30, 2)
T1_TRIGGER = 2
RSVP_TRIGGER = 1
PERMUTATIONS = 1000

@fnc.memoize(persistent=True, folder='_loaded')
def proc_subject(subject):

    print(f'read subject {subject}')
    raw, events, metadata = eet.read_subject(subject, eeg_preprocessing = True)

    print(f'extracting dm {subject}')
    sdm = extract_dm(raw, events, metadata)

    print(f'calculate all stats {subject}')
    stats = do_all_cpt(sdm.T1_correct == 1, ['pz', 'theta', 'pupil'], ['probe', 'control'], ['familiar', 'unfamiliar'])
    
    return sdm, stats


@fnc.memoize(persistent=True, folder='_extracted')
def extract_dm(raw, events, metadata):
    # Select T1 triggers from all triggers
    t1_triggers = eet.epoch_trigger(events, T1_TRIGGER)

    # Create dm from metadata
    sdm = cnv.from_pandas(metadata)

    pupil = eet.PupilEpochs(raw, t1_triggers,
                                    tmin=-.1, tmax=1, 
                                    metadata=metadata,
                                    baseline=(-.05, 0))
    sdm.pupil = cnv.from_mne_epochs(pupil)
    sdm.pupil = sdm.pupil[:, ...]
    
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

    # Extract time-frequency data
    tfr_epochs = eet.autoreject_epochs(raw, t1_triggers,
                            tmin=-.5, tmax=2, metadata=metadata,
                            picks=CHANNELS, baseline=(-.1, 0))
    morlet = tfr_morlet(tfr_epochs, FULL_FREQS, decim=1, n_cycles=2,
                        average=False, return_itc=False)
    morlet.crop(-.1, 1)
    sdm.tfr_T1 = cnv.from_mne_tfr(morlet)
    sdm.tfr_T1_z = z_by_freq(sdm.tfr_T1)
    del sdm.tfr_T1  # save memory
    sdm.theta = sdm.tfr_T1_z[:, ..., :2][:, ...]
    sdm.alpha = sdm.tfr_T1_z[:, ..., 2:5][:, ...]
    sdm.beta = sdm.tfr_T1_z[:, ..., 5:][:, ...]

    sdm = sdm.practice == 'no'

    return sdm


def get_merged_data(SUBJECTS):
    dm = DataMatrix()
    all_stats = []
    for subject in SUBJECTS:
        sdm, stats = proc_subject(subject)

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
    temp = tst.lmer_permutation_test(tdm, formula=f'{dv} ~ T1', groups=None, iterations=PERMUTATIONS, cluster_p_threshold=.05, suppress_convergence_warnings=True)
    result.dv = dv
    result.level = level
    result.condition = condition
    ref = 'probe' if level == 'probe' else 'distractor'
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


def do_all_cpt(dm, dv, level, condition):
    stats = DataMatrix()
    for d in dv:
        for l in level:
            for c in condition:
                print(f'dv: {d}, level: {l}, condition: {c}')
                # do_cpt.clear()
                stats <<= do_cpt(dm, d, l, c)
    return stats


def set_xticks(ax, dv, step):
    ax.set_xticks(ticks = np.arange(4, dv.depth, 100), labels=np.arange(0,  1000, 400))