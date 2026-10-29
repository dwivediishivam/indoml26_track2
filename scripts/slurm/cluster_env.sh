# Sourced by OMNI jobs: existing venv + our extra packages; offline HF.
export PATH=$HOME/.venvs/indoml/bin:$PATH
export PYTHONPATH=$HOME/indoml-t2/pylib${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$HOME/indoml-t2/hf_home
export ROOT=$HOME/indoml-t2
if [ -r /etc/profile.d/modules.sh ]; then set +u; source /etc/profile.d/modules.sh; module add cuda12.8/toolkit/12.8.1 2>/dev/null || true; fi
# Nodes expose 224 cores but the cgroup grants few: cap per-process threads to avoid thrashing.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
