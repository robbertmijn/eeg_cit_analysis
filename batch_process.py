"""
Memoization process for preprocessing and merging data from the face study by van der Mijn et al.

Runs for a couple hours on a CPU system with 15 cores
"""
import argparse
import multiprocessing as mp
from itertools import product

import analysis_utils as au

# List of pp numbers to process
SUBJECTS = list(range(1, 23+1))

# Calculate individual stats (takes long)
INDIV_STATS = False

# Optionally, uncomment these next lines to clear cache
# au.eet.read_subject.clear()
# au.eet.autoreject_epochs.clear()
# au.proc_subject.clear()
# au.extract_dm.clear()
# au.do_cpt.clear()
# au.do_all_cpt.clear()


def process_subject(subject_nr):
    """
    process subject

    Parameters
    ----------
    subject_nr: int
        subject number

    Returns
    -------
    sdm: DataMatrix
        preprocessed data of the subject
    """
     
    print("START preprocessing pp {}".format(subject_nr))
    au.proc_subject(subject_nr, stats=INDIV_STATS)
    print("DONE preprocessing pp {}".format(subject_nr))


if __name__ == '__main__':
    """
    This function is run when this whole python script is run from the terminal
    """
    
    # specify which parameters can be passed in the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-process', action='store', default=2)
    parser.add_argument('--subjects', action='store', default='all')
    parser.add_argument('--GA_stats', action='store', default=True)
    args = parser.parse_args()
    print(f'using {args.n_process} processes')

    # process all pps as a batch, or just a subset specified in the arguments
    if args.subjects == 'all':
        subjects = SUBJECTS
    else:
        subjects = [int(s) for s in args.subjects.split(',')]
    print(f'processing subjects: {subjects}')

    # if more than 1 pp is processed, we use the pool to do these in parallel
    if len(subjects) == 1:
        process_subject(subjects[0])
    else:
        with mp.Pool(int(args.n_process)) as pool:
            pool.map(process_subject, subjects)
    
    # Merge data of all pps. This will call process_subject again. But because the results have been memoized by the previous operation, it should just load the resulst instead of actually redoing it
    print("START merge")
    dm, stats = au.get_merged_data(subjects, stats=INDIV_STATS)
    if args.GA_stats:
        print("GA stats...")
        au.do_all_cpt(dm.T1_correct == 1, ['pz', 'theta', 'fz', 'cz', 'pupil'], ['probe', 'control', 'target'], ['familiar', 'unfamiliar'])
    print("DONE merge")