# QEWavefunctions

[![Build Status](https://github.com/jaemolihm/QEWavefunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/QEWavefunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package for reading Quantum ESPRESSO wavefunction files (`wfcN.hdf5` and
`wfcN.dat`) and working with the wavefunctions in reciprocal and real space.

## Features

- Read QE wavefunctions in HDF5 (`read_qe_wfc_hdf5`) or Fortran binary
  (`read_qe_wfc_dat`) format, with `read_qe_wfc` auto-dispatching on the file
  extension.
- Write wavefunctions back to QE HDF5 format (`write_qe_wfc_hdf5`).
- Optional band selection and metadata-only reads.
- Transform G-space coefficients to real space, by explicit phase summation
  (`compute_real_space_wfc`) or FFT (`compute_real_space_wfc_fft`).
- Compute the z-projected density (`compute_z_density`).
- Compute overlaps between two wavefunctions on shared Miller indices, with an
  optional Wannier-center phase shift (`match_miller_indices`,
  `compute_wfc_overlap`).

## Installation

This package is not registered, so install it from the Git URL (or a local
path). In the Julia REPL, press `]` to enter the Pkg mode:

```julia
pkg> add https://github.com/jaemolihm/QEWavefunctions.jl
```

Or, equivalently, from regular Julia code:

```julia
using Pkg
Pkg.add(url="https://github.com/jaemolihm/QEWavefunctions.jl")
```

### Developing locally

If you have a local clone and want to hack on it:

```julia
using Pkg
Pkg.develop(path="/path/to/QEWavefunctions.jl")
# install dependencies and run the tests
Pkg.instantiate()
Pkg.test("QEWavefunctions")
```

## Usage

```julia
using QEWavefunctions

# Read a wavefunction (format inferred from extension)
wfc = read_qe_wfc("wfc1.hdf5")

# Only the first 20 bands
wfc = read_qe_wfc("wfc1.dat"; bands = 1:20)

# Metadata only (skips the large mill / evc arrays)
meta = read_qe_wfc("wfc1.hdf5"; metadata_only = true)

# Real-space periodic part u_nk(r) on a (nx, ny, nz) grid -> (nx, ny, nz, nbnd)
ngrid = (36, 36, 36)
u = compute_real_space_wfc_fft(wfc, ngrid)

# z-projected density -> (nz, nbnd)
rho = compute_z_density(wfc, ngrid[3])

# Overlap between two wavefunctions on their shared G-vectors
wfc2 = read_qe_wfc("wfc1_other_run.hdf5")
S = compute_wfc_overlap(wfc, wfc2)

# Write back to QE HDF5 format
write_qe_wfc_hdf5("wfc1_out.hdf5", wfc)
```

The fields of `QEWavefunction` (k-point `xk`, reciprocal lattice
`recip_lattice`, Miller indices `mill`, coefficients `evc`, etc.) are documented
in the docstrings; access them in the REPL with `?QEWavefunction`.

## Dependencies

- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) — reading/writing QE HDF5 files
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) — fixed-size
  lattice vectors and Miller indices
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) — real-space transforms
- LinearAlgebra (stdlib)

Julia 1.6 or later.
