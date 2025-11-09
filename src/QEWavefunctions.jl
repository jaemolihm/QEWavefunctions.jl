module QEWavefunctions

using StaticArrays
using HDF5

export read_qe_wfc_hdf5
export compute_real_space_wfc

"""
    QEWavefunction

A struct to hold the data from a Quantum ESPRESSO wfcN.hdf5 file.
"""
struct QEWavefunction
    gamma_only :: Bool
    ik :: Int
    ispin :: Int
    xk  :: SVector{3, Float64}
    scalef :: Float64
    ngw :: Int
    igwx :: Int
    npol :: Int
    nbnd :: Int
    recip_lattice :: SMatrix{3, 3, Float64, 9}
    mill :: Vector{SVector{3, Int}}
    evc :: Matrix{ComplexF64}
end

"""
    read_wfc_hdf5(filename; metadata_only=false) -> QEWavefunction

Reads wavefunction data from a Quantum ESPRESSO HDF5 file.

The function extracts attributes and datasets corresponding to a specific
k-point and reconstructs the complex wavefunction coefficients.
See subroutine read_wfc in Modules/io_base.f90 of Quantum ESPRESSO for reference.

# Arguments
- `filename`: The path to the `.hdf5` file (e.g., "wfc1.hdf5").
- `metadata_only`: If `true`, only reads metadata (attributes) and skips reading
  large datasets (`mill` and `evc`), returning empty arrays for those fields.
  Default is `false`.

# Returns
- A `QEWavefunction` struct containing the loaded data.
"""
function read_qe_wfc_hdf5(filename; metadata_only::Bool = false)
    # Open the HDF5 file in read-only mode
    # The 'do' block ensures the file is closed automatically
    h5open(filename, "r") do file
        # HDF5 attributes are attached to the root group
        root_attrs = attrs(file)

        # --- Read all metadata from attributes ---
        ik = root_attrs["ik"]
        xk = SVector(root_attrs["xk"]...)
        ispin = root_attrs["ispin"]
        ngw = root_attrs["ngw"]
        nbnd = root_attrs["nbnd"]
        npol = root_attrs["npol"]
        igwx = root_attrs["igwx"]
        scalef = root_attrs["scale_factor"]

        # Fortran logicals are stored as strings (e.g., ".true.")
        gamma_only = occursin("true", lowercase(root_attrs["gamma_only"]))

        b1 = read(file["MillerIndices"]["bg1"])
        b2 = read(file["MillerIndices"]["bg2"])
        b3 = read(file["MillerIndices"]["bg3"])
        recip_lattice = SMatrix{3, 3}(hcat(b1, b2, b3))

        if metadata_only
            # Read onlty metadata, return empty arrays for large datasets
            mill = SVector{3, Int}[]
            evc = Matrix{ComplexF64}(undef, 0, 0)

        else
            # --- Read the large arrays from datasets ---
            # Miller indices are stored directly as list of SVector{3, Int}
            mill = Vector(vec(reinterpret(SVector{3, Int32}, read(file["MillerIndices"]))))
            @assert length(mill) == igwx

            # QE stores complex numbers as a real array of twice the size [re, im, re, im, ...]
            # Read the entire real-valued dataset for the wavefunctions and reinterpret
            # the real array as a complex array.
            evc_complex_flat = reinterpret(ComplexF64, read(file["evc"]))

            # Reshape the flat complex vector into the correct matrix dimensions
            evc = reshape(evc_complex_flat, (npol * igwx, nbnd))
        end

        # --- Construct and return the struct ---
        QEWavefunction(gamma_only, ik, ispin, xk, scalef, ngw, igwx, npol, nbnd, recip_lattice, mill, evc)
    end
end

include("real_space.jl")


end
