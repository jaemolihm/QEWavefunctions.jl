#!/bin/bash
# Example: rigidly-shifted Wannier functions for Si.
#
# Runs the reference (ref/) and perturbed (pert/) systems through QE + Wannier90,
# then rebuilds the perturbed amn from rigidly-displaced reference Wannier
# functions and re-wannierizes. Run from this directory:
#
#   ./run.sh
#
# Adjust the QE / Wannier90 / QEWavefunctions.jl paths below for your system.

QE=$HOME/program/qe_epw/bin
W90=$HOME/program/wannier90/wannier90.x
QEWAN=$HOME/forces_DMFT/QEWavefunctions.jl

# Bundled helper scripts (my_qe_bands.py, plotband.py), found relative to run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/../bin"

# SCF + bands + nscf + Wannier90 .nnkp/.amn/.mmn/.eig for the current directory
run_qe_w90() {
    mpirun -np 12 $QE/pw.x -nk 4 -ndiag 1 -in scf.in   > scf.out
    mpirun -np 12 $QE/pw.x -nk 4 -ndiag 1 -in bands.in > bands.out
    $BIN/my_qe_bands.py si temp
    mpirun -np 12 $QE/pw.x -nk 4 -ndiag 1 -in nscf.in  > nscf.out
    mpirun -np 1  $W90 -pp si
    mpirun -np 12 $QE/pw2wannier90.x -nk 4 -ndiag 1 -in pw2wan.in > pw2wan.out
}

# === Reference system ===
cd ref
run_qe_w90
mpirun -np 12 $W90 si
$BIN/plotband.py si --png
cd ..

# === Perturbed system ===
cd pert
run_qe_w90

# Baseline: Wannierize from the atomic-projection amn produced by pw2wannier90.
mpirun -np 12 $W90 si
cp si.wout si.atomic.wout

# Rigidly-shifted Wannier functions: rebuild si.amn from the reference Wannier
# functions displaced by the difference of projection centers in ref/si.nnkp and
# pert/si.nnkp, then re-wannierize.
julia --project=$QEWAN $QEWAN/scripts/generate_amn.jl \
  si \
  ../ref \
  ../pert \
  ../ref/temp/si.save \
  ../pert/temp/si.save \
  si.amn
mpirun -np 12 $W90 si
cp si.wout si.rigidWF.wout
$BIN/plotband.py si --png
cd ..
