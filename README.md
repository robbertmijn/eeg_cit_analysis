# EEG-pupil CIT Analysis

This repository hold the preprocessing and analysis script for the EEG-pupil CIT study conducted by Robbert van der Mijn and Elkan Akyurek in 2023.

## Getting started

### Package managers

To run these analyses, clone this repository to your own system

```
git clone https://github.com/robbertmijn/eeg_cit_analysis
```

Install a package manager such as [Anaconda](https://www.anaconda.com/). The GUI of Anaconda allows you to find IDE's in which you can write, review and run python scripts (such as Spyder or VSCode). From the terminal, you can ask Anaconda to create a new environment which can help keeping packages and their versions separated between projects: 

```
conda create --name eeg
```

Each time you use python somewhere, make sure to activate that environment:

```
conda activate eeg
```

Most GUI's allow you to select which environment is used by default (which saves you from having to activate it each time you start the GUI.

### Dependencies

Install [eeg_eyetracking_parser](https://github.com/smathot/eeg_eyetracking_parser). This will also install a number of other required packages. If installation is not succesfull, try to manually install the packages listed [here](https://github.com/smathot/eeg_eyetracking_parser?tab=readme-ov-file#dependencies). Make sure you do this step while your conda env is activated:

```
pip install eeg_eyetracking_parser
```

### Processing data

Copy participant data to your cloned directory. The data should be in [BIDS](https://github.com/smathot/eeg_eyetracking_parser?tab=readme-ov-file#file-and-folder-structure) format. Start with 1 pp to test the pipeline.

Process data using the batch processor. We can pass some arguments to the scripts by using this syntax:

```
python batch_process.py --n-process 4 --subjects 3 --GA_stats False
```



