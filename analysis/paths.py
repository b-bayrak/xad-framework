"""Locations used by the analysis scripts, relative to the repository root.

Inputs that are not part of the repository (raw and prepared data, run outputs)
live in data/ and runs/, which are created by the download, preparation and
job scripts.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(REPO, 'data')
RAW = os.path.join(DATA, 'raw')                      # downloaded files of the ten added datasets
PREPARED = os.path.join(DATA, 'prepared')            # prepared versions (datasets/prepare.py)

RUNS_SKAB_SWAT = os.path.join(REPO, 'runs', 'skab_swat')   # jobs/skab_main, swat_main, skab_swat_supp
RUNS_BENCHMARK = os.path.join(REPO, 'runs', 'benchmark')   # jobs/benchmark_*
RUNS_POSTHOC = os.path.join(REPO, 'runs', 'posthoc')       # jobs/posthoc_*

RESULTS = os.path.join(REPO, 'results')
PREP_LOGS = os.path.join(REPO, 'datasets', 'logs')
REGISTER = os.path.join(REPO, 'datasets', 'register')
RUNLOG = os.path.join(REPO, 'docs', 'RUNLOG.md')
