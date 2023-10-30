#%%
from datamatrix import convert as cnv, operations as ops, functional as fnc, series as srs, DataMatrix
import eeg_eyetracking_parser as eet
from matplotlib import pyplot as plt
import numpy as np
import analysis_utils as au
COLDICT = {'target': '#c94733', 'probe': '#0d5b26', 'control': '#2e5fa1', 'distractor': 'gray'}
# SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
SUBJECTS = list(range(1, 24))

#%%
## Memoized functions; uncomment to rerun and cache
# au.eet.read_subject.clear()
# au.eet.autoreject_epochs.clear()
# au.proc_subject.clear()
# au.get_merged_data.clear()
dm, stats = au.get_merged_data(SUBJECTS, stats=True)
GA_stats = au.do_all_cpt(dm.T1_correct == 1, ['pz', 'theta', 'fz', 'cz', 'pupil'], ['probe', 'control', 'target'], ['familiar', 'unfamiliar'])

dm.fznans = srs.nancount(dm.fz)
dm.fznans[dm.fznans > 0] = 1

dm.pupilnans = srs.nancount(dm.pupil)
dm.pupilnans[dm.pupilnans > 0] = 1

dm.thetanans = srs.nancount(dm.theta)
dm.thetanans[dm.thetanans > 0] = 1

# print accuracy
for s, sdm in ops.split(dm.subject_nr):
    print(f"pp {s} accuracy:")
    for t1, tdm in ops.split(sdm.T1):
        print(f"{t1}: {tdm.T1_correct.mean*100:.1f}% correct, pupil: {tdm.pupilnans.mean*100:.1f}, fz: {tdm.fznans.mean*100:.1f}, theta: {tdm.thetanans.mean*100:.1f}")

plt.figure()
[plt.plot(sdm.sum_points, label=s) for s, sdm in ops.split(dm.subject_nr)]
plt.legend()
plt.show()
    
dm = dm.T1_correct == 1

#%% ERP GA plots
'''
"Familiar": the probe in these blocks are the pp parent
"Unfamiliar": the probe in these blocks are another random control
Blocks are presented in random order


Plot traces, mark clusters "bold" that show a difference between that trace and the distractor. 

In the unfamiliar blocks, we expect only "target" to show a difference.
'''
au.ERP_plots(dm, f'Grand Average {len(dm.subject_nr.unique)} pps', GA_stats)

#%% ERP per participant
for s in SUBJECTS:
    sdm, stats = au.proc_subject(s, stats=False)
    au.ERP_plots(sdm, f'pp {s}', stats)

#%% Time Frequency Plots
x = np.arange(0, dm.pupil.depth)

fig, axes = plt.subplots(2, 2, constrained_layout=True, sharey=True, sharex=True)
frow = 0
for condition, cdm in ops.split(dm.condition):
    # Plot overall Z scores and the difference between probe and control
    dm_control = cdm.T1 == 'control'
    dm_probe = cdm.T1 == 'probe'
    dm_distractor = cdm.T1 == 'distractor'
    probe = dm_probe.tfr_T1_z[..., ...] - dm_distractor.tfr_T1_z[..., ...]
    control = dm_control.tfr_T1_z[..., ...] - dm_distractor.tfr_T1_z[..., ...]
    m = axes[frow, 0].imshow(control, aspect='auto', cmap='jet',vmax = 0.3, vmin = -0.1)
    axes[frow, 0].set_yticks(np.arange(0, 13, step=2), np.arange(4, 30, step=4))
    axes[frow, 0].set_title(f'Control - Distractor ({condition})')
    n = axes[frow, 1].imshow(probe, aspect='auto', cmap='jet', vmin = -0.1, vmax = 0.3)
    axes[frow, 1].set_title(f'Probe - Distractor ({condition})')
    frow += 1

plt.colorbar(n, ticks=np.arange(-0.1, 0.35, step=0.1), ax=axes[:, 1])
for ax in axes.flat:
    au.set_xticks(ax, dm.pupil, 400)
fig.supxlabel('Time since T1 onset (ms)')
fig.supxlabel('Frequency (Hz)')
fig.supylabel('Z difference')
fig.suptitle(f'Grand Average {len(dm.subject_nr.unique)} pps')
plt.show()
