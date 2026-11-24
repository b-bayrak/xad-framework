# Auto-imported by Python at startup when this directory is on PYTHONPATH.
# Active ONLY when XAD_EXTRA=1 (set in jobs/benchmark_*.sbatch), so it
# reaches the exp_*.py subprocesses that run_all_experiments.py spawns while
# leaving every other Python process untouched. See xad_extra/__init__.py.
import os

if os.environ.get('XAD_EXTRA') == '1':
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    for _p in (_repo, _here):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import xad_extra
    xad_extra.install(verbose=os.environ.get('XAD_EXTRA_VERBOSE') == '1')
