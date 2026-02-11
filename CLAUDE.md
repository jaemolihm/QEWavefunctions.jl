# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QEWavefunctions.jl is a Julia package for reading wavefunction data from Quantum ESPRESSO HDF5 files (wfcN.hdf5 format). The package provides a single main function `read_qe_wfc_hdf5` that parses HDF5 wavefunction files and returns a `QEWavefunction` struct containing k-point data, Miller indices, reciprocal lattice vectors, and wavefunction coefficients.

## Development Commands

### Testing
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Build/Install Package
```bash
julia --project -e 'using Pkg; Pkg.build()'
```

### Instantiate Dependencies
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Start Julia REPL with Package
```bash
julia --project
```
Then in REPL:
```julia
using QEWavefunctions
```

## Architecture

### Core Data Structure
- **QEWavefunction struct** (src/QEWavefunctions.jl:13-26): Contains all wavefunction data from a QE HDF5 file
  - Metadata: gamma_only, ik, ispin, xk, scalef, ngw, igwx, npol, nbnd
  - Reciprocal lattice: 3×3 static matrix
  - Miller indices: Vector of 3D static vectors
  - Wavefunction coefficients: Complex matrix (npol*igwx, nbnd)

### Key Implementation Details
- Uses StaticArrays (SVector, SMatrix) for fixed-size arrays (k-points, lattice vectors, Miller indices)
- HDF5 reading handled with automatic file closure via `do` block pattern
- Complex numbers in QE HDF5 files are stored as interleaved real arrays [re, im, re, im, ...] and reinterpreted using `reinterpret(ComplexF64, ...)`
- Fortran logical values are stored as strings (".true." or ".false.") and parsed with string matching
- Supports `metadata_only=true` flag to skip reading large datasets (mill, evc)

### QE wfc HDF5 File Format

Structure of `wfcN.hdf5` files produced by Quantum ESPRESSO (verified with `h5ls -rv`):

```
/                           (root group)
  Attributes:
    gamma_only              7-byte space-padded ASCII string (".TRUE." or ".FALSE.")
    ik                      Int32, k-point index
    ispin                   Int32, spin index
    xk                      Float64[3], k-point in Cartesian (2pi/a) units
    nbnd                    Int32, number of bands
    ngw                     Int32, number of G-vectors for this k-point
    igwx                    Int32, max number of G-vectors across k-points
    npol                    Int32, number of spin polarizations (1 or 2)
    scale_factor            Float64

/MillerIndices              Dataset {igwx, 3} Int32   (HDF5 C-order; Julia reads as (3, igwx))
  Attributes:
    bg1                     Float64[3], reciprocal lattice vector b1 (Cartesian, 2pi/a)
    bg2                     Float64[3], reciprocal lattice vector b2
    bg3                     Float64[3], reciprocal lattice vector b3
    doc                     ASCII string (description, optional)

/evc                        Dataset {nbnd, 2*npol*igwx} Float64  (HDF5 C-order; Julia reads as (2*npol*igwx, nbnd))
  Attributes:
    doc:                    ASCII string (description, optional)
```

Complex wavefunction coefficients are stored as interleaved real/imaginary pairs:
`[Re(c1), Im(c1), Re(c2), Im(c2), ...]`, reinterpreted in Julia via `reinterpret(ComplexF64, ...)`.

The `MillerIndices` dataset bg attributes are accessed in HDF5.jl as `file["MillerIndices"]["bg1"]`
(HDF5.jl maps string indexing on a Dataset to attribute access).

Trimmed wfc files written by `write_qe_wfc_hdf5` add an optional root attribute:
- `band_range`: Int32 vector of original band indices stored in `evc`

### Reference Implementation
The read_qe_wfc_hdf5 function follows the algorithm in Quantum ESPRESSO's Fortran subroutine `read_wfc` in `Modules/io_base.f90`.

## Dependencies
- HDF5.jl (v0.17.2): Reading HDF5 files from Quantum ESPRESSO
- StaticArrays.jl (v1.9.15): Efficient fixed-size arrays for lattice vectors and k-points
- Julia compatibility: 1.6+
