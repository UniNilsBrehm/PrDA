# A dopaminergic visuomotor pathway to the zebrafish optic tectum

**Nils Brehm, Shagnik Chakraborty, Johann H. Bollmann** — *iScience*, Cell Press (2026)

This repository contains the analysis code for Ca²⁺ imaging, ventral root recordings and visual stimulus data, as well as the scripts used to create all figures of the manuscript.

> [!NOTE]
> The required input data files and the generated output files are listed in the header of each script.

---

## Table of contents

- [System requirements](#system-requirements)
- [Installation](#installation)
- [Analysis pipeline](#analysis-pipeline)
  - [Preprocessing](#preprocessing)
  - [Ventral root recordings](#ventral-root-recordings)
  - [Response analysis](#response-analysis)
  - [Population analysis](#population-analysis)
  - [Shared analysis modules](#shared-analysis-modules)
- [Figures](#figures)
  - [Main figures](#main-figures)
  - [Supplementary figures](#supplementary-figures)
  - [Shared plotting modules](#shared-plotting-modules)

---

## System requirements

All scripts were written and tested with **Python 3.13.7** and depend on the following packages:

| Package | Purpose |
|---|---|
| [`numpy`](https://numpy.org) | Numerical computing |
| [`pandas`](https://pandas.pydata.org) | Data handling |
| [`matplotlib`](https://matplotlib.org) | Plotting |
| [`scipy`](https://scipy.org) | Filtering, peak detection, curve fitting, statistics |
| [`statsmodels`](https://www.statsmodels.org) | Linear regression (OLS), ECDF |
| [`scikit-learn`](https://scikit-learn.org) | ROC AUC / effect sizes |
| [`seaborn`](https://seaborn.pydata.org) | Statistical plotting |
| [`joblib`](https://joblib.readthedocs.io) | Parallel processing |
| [`tqdm`](https://tqdm.github.io) | Progress bars |
| [`tifffile`](https://github.com/cgohlke/tifffile) | Reading TIFF recordings |
| [`tables`](https://www.pytables.org) | HDF5 file support for pandas |
| [`ipython`](https://ipython.org) | Interactive debugging |
| [`numba`](https://numba.pydata.org) | JIT compilation |

Statistical modelling of the synergy effect (GLMM) is done in **R**.

The figures use the **Arial** font, which must be installed on your system for the PDFs to render as intended.

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install numpy pandas matplotlib scipy statsmodels scikit-learn seaborn joblib tqdm tifffile tables ipython numba
```

Before running any analysis script, set `BASE_DIR` in [`analysis/config.py`](analysis/config.py) to the directory that contains your data.

---

## Analysis pipeline

All analysis scripts are located in [`analysis/`](analysis). Run them in the order listed below. Each step uses the outputs of the steps before it.

### Preprocessing

#### 1. Metadata extraction
| | |
|---|---|
| **Script** | [`get_meta_data_from_sweeps.py`](analysis/get_meta_data_from_sweeps.py) |
| **Description** | Collects metadata from Ca²⁺ imaging recordings, ventral root data and stimulus logs. |
| **Dependencies** | `utils.py`, `config.py` |

#### 2. Match sampling rates
| | |
|---|---|
| **Script** | [`match_sampling_rates.py`](analysis/match_sampling_rates.py) |
| **Description** | Resamples recordings so that all of them share the same sampling rate. |
| **Dependencies** | `utils.py`, `config.py` |

#### 3. Collect sweeps
| | |
|---|---|
| **Script** | [`collect_sweeps.py`](analysis/collect_sweeps.py) |
| **Description** | Selects good recordings for further analysis and converts them to ΔF/F. |
| **Dependencies** | `utils.py`, `config.py` |

#### 4. Stimulus preprocessing
| | |
|---|---|
| **Script** | [`stimulus_pre_processing.py`](analysis/stimulus_pre_processing.py) |
| **Description** | Builds stimulus protocols, onsets/offsets and binary traces for the analysis pipeline. |
| **Dependencies** | `utils.py`, `config.py` |

### Ventral root recordings

#### 5. Preprocessing
| | |
|---|---|
| **Script** | [`ventral_root_pre_processing.py`](analysis/ventral_root_pre_processing.py) |
| **Description** | Converts the chunked Olympus `.txt` files into one recording per sweep. |
| **Dependencies** | `utils.py` |

#### 6. Time alignment
| | |
|---|---|
| **Script** | [`ventral_root_align_recordings_with_ca_imaging.py`](analysis/ventral_root_align_recordings_with_ca_imaging.py) |
| **Description** | Aligns ventral root recordings in time with the Ca²⁺ imaging data. |
| **Dependencies** | `utils.py` |

#### 7. Event detection
| | |
|---|---|
| **Script** | [`ventral_root_event_detection.py`](analysis/ventral_root_event_detection.py) |
| **Description** | Detects swim bouts and other motor events in ventral root traces. |
| **Dependencies** | `utils.py` |

### Response analysis

#### 8. Linear scoring of responses
| | |
|---|---|
| **Script** | [`linear_scoring_analysis.py`](analysis/linear_scoring_analysis.py) |
| **Description** | Scores ROI responses to visual stimuli and motor events with a linear regression model (see below). |
| **Dependencies** | `utils.py`, `config.py` |

$$\text{Score} = R^2 \times \text{slope}$$

#### 9. Prepare data for synergy effect statistics
| | |
|---|---|
| **Script** | [`synergy_effect.py`](analysis/synergy_effect.py) |
| **Description** | Prepares linear scoring data for a generalized linear mixed model (GLMM) and computes selectivity indices per neuron based on the visual, motor and combined scores. Further statistical analysis is done in R. |
| **Dependencies** | `config.py` |

#### 10. Remove motor events from Ca²⁺ responses
| | |
|---|---|
| **Script** | [`remove_motor_events_from_ca_responses.py`](analysis/remove_motor_events_from_ca_responses.py) |
| **Description** | Removes motor-related activity from ΔF/F traces. |
| **Dependencies** | `utils.py`, `config.py` |

#### 11. Fit Ca²⁺ kernel time constants
| | |
|---|---|
| **Script** | [`cirf_find_parameters_parallel.py`](analysis/cirf_find_parameters_parallel.py) |
| **Description** | Finds the optimal rise and decay time constants of the calcium impulse response function by fitting the data. |
| **Dependencies** | `utils.py`, `config.py`, `plotting_utils.py` |

#### 12. Direction selectivity indices
| | |
|---|---|
| **Script** | [`compute_ds_index.py`](analysis/compute_ds_index.py) |
| **Description** | Computes direction selectivity indices from forward and backward grating scores and runs the statistical tests. |
| **Dependencies** | `utils.py`, `figures/plotting_style.py` |

### Population analysis

#### 13. Z-score-thresholded population activity
| | |
|---|---|
| **Script** | [`zscore_population_activity.py`](analysis/zscore_population_activity.py) |
| **Description** | Computes event-aligned population activity: the fraction of active neurons and the fraction of jointly active neuron pairs. |
| **Dependencies** | `utils.py`, `config.py` |

#### 14. Pairwise lagged cross-correlation
| | |
|---|---|
| **Script** | [`pairwise_cross_correlations.py`](analysis/pairwise_cross_correlations.py) |
| **Description** | Computes lag-resolved pairwise cross-correlations between neurons and compares them against a circular-shift null. |
| **Dependencies** | `utils.py`, `config.py` |

### Shared analysis modules

| Module | Description |
|---|---|
| [`config.py`](analysis/config.py) | Base directory, file paths and analysis settings |
| [`utils.py`](analysis/utils.py) | Shared helper functions: data loading, filtering, ΔF/F, regressors and more |

---

## Figures

All figure scripts are located in [`figures/`](figures). Each script reads its input data from `figures/data/<figure>/` and exports the finished figure as an A4 PDF to `figures/figs/`.

> [!IMPORTANT]
> The scripts use relative paths, so run them from inside the `figures/` directory:
> ```bash
> cd figures
> python figure_1.py
> ```

### Main figures

| Figure | Script | Title | Input data | Output |
|:---:|---|---|---|---|
| **1** | [`figure_1.py`](figures/figure_1.py) | Anatomical organization and in situ HCR of marker genes in PrDA neurons | `data/fig1/` | `01_figure_1.pdf` |
| **2** | [`figure_2.py`](figures/figure_2.py) | Functional imaging of PrDA neurons | `data/fig2/` | `02_figure_2.pdf` |
| **3** | [`figure_3.py`](figures/figure_3.py) | Spontaneous motor events are associated with phasic PrDA neuron activity | `data/fig3/` | `03_figure_3.pdf` |
| **4** | [`figure_4.py`](figures/figure_4.py) | PrDA neurons are driven by visual stimulation and respond bilaterally | `data/fig4/` | `04_figure_4.pdf` |
| **5** | [`figure_5.py`](figures/figure_5.py) | PrDA neurons combine motor and visual inputs | `data/fig4/`, `data/fig5/` | `05_figure_5.pdf` |
| **6** | [`figure_6.py`](figures/figure_6.py) | PrDA neurons exhibit strong synchronized activity | `data/fig6/` | `06_figure_6.pdf` |
| **7** | [`figure_7.py`](figures/figure_7.py) | Correcting direction selectivity of PrDA neurons by including motor activity | `data/fig7/` | `07_figure_7.pdf` |

> [!NOTE]
> Some figure scripts run a short data preparation step before plotting and write intermediate files back into `data/`:
> - **Figure 3:** computes response probabilities and anatomical data from the linear scoring results.
> - **Figure 5:** classifies neurons (M+/V+, M+/V−, M?/V+, M−/V+).
> - **Figure 7:** computes the Kolmogorov–Smirnov tests on the direction selectivity indices.

### Supplementary figures

| Figure | Script | Title | Input data | Output |
|:---:|---|---|---|---|
| **S3** | [`supp_figure_3.py`](figures/supp_figure_3.py) | Swim bout detection, Ca²⁺ kernel fitting and linear scoring workflow | `data/supp_fig3/` | `10_supp_figure_3.pdf` |
| **S4** | [`supp_figure_4.py`](figures/supp_figure_4.py) | Number of triggered swim bouts per stimulus type | `data/supp_fig4/` | `11_supp_figure_4.pdf` |

### Shared plotting modules

| Module | Description | Used by |
|---|---|---|
| [`plotting_style.py`](figures/plotting_style.py) | Global matplotlib defaults and the `PlotStyle` class (colors, line, marker and text styles) | All figures |
| [`plotting_utils.py`](figures/plotting_utils.py) | Helpers for scale bars, pictograms, labels, axis spines and colorbars | Figures 1–5, 7, S3 |
| [`utils.py`](figures/utils.py) | Copy of the analysis helper functions (regressors, Ca²⁺ kernel, HDF5 loading) | Figures 2–5, S3 |

---

<sub>**Author:** Nils Brehm · 2026</sub>