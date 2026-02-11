module QEWavefunctions

using StaticArrays
using HDF5

export read_qe_wfc_hdf5
export write_qe_wfc_hdf5
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
    read_wfc_hdf5(filename; metadata_only=false, bands=nothing) -> QEWavefunction

Reads wavefunction data from a Quantum ESPRESSO HDF5 file.

The function extracts attributes and datasets corresponding to a specific
k-point and reconstructs the complex wavefunction coefficients.
See subroutine read_wfc in Modules/io_base.f90 of Quantum ESPRESSO for reference.

# Arguments
- `filename`: The path to the `.hdf5` file (e.g., "wfc1.hdf5").
- `metadata_only`: If `true`, only reads metadata (attributes) and skips reading
  large datasets (`mill` and `evc`), returning empty arrays for those fields.
  Default is `false`.
- `bands`: If specified, only reads the selected bands (e.g., `1:20` or `[1, 3, 5]`).
  Default is `nothing`, which reads all bands.

# Returns
- A `QEWavefunction` struct containing the loaded data.
"""
function read_qe_wfc_hdf5(filename; metadata_only::Bool = false, bands = nothing)
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
            # Read the real-valued dataset for the wavefunctions and reinterpret as complex.
            if bands === nothing
                evc_complex_flat = reinterpret(ComplexF64, read(file["evc"]))
                evc = reshape(evc_complex_flat, (npol * igwx, nbnd))
            elseif bands isa AbstractRange
                evc_complex_flat = reinterpret(ComplexF64, file["evc"][:, bands])
                evc = reshape(evc_complex_flat, (npol * igwx, length(bands)))
            else
                # bands is a Vector: convert to range if contiguous, otherwise read all and select
                bands_vec = collect(bands)
                if length(bands_vec) > 0 && bands_vec == first(bands_vec):last(bands_vec)
                    bands_range = first(bands_vec):last(bands_vec)
                    evc_complex_flat = reinterpret(ComplexF64, file["evc"][:, bands_range])
                    evc = reshape(evc_complex_flat, (npol * igwx, length(bands_range)))
                else
                    # Non-contiguous: read all, then select
                    evc_complex_flat = reinterpret(ComplexF64, read(file["evc"]))
                    evc = reshape(evc_complex_flat, (npol * igwx, nbnd))[:, bands_vec]
                end
            end
        end

        # --- Construct and return the struct ---
        QEWavefunction(gamma_only, ik, ispin, xk, scalef, ngw, igwx, npol, nbnd, recip_lattice, mill, evc)
    end
end

"""
    write_qe_wfc_hdf5(filename, wfc::QEWavefunction; band_range=nothing)

Writes wavefunction data to an HDF5 file in Quantum ESPRESSO format.
The output file can be read back using `read_qe_wfc_hdf5`.

# Arguments
- `filename`: Output HDF5 file path.
- `wfc`: QEWavefunction struct to write.
- `band_range`: Optional vector of original band indices stored in `evc`.
  Stored as an additional `band_range` attribute in the output file.
"""
function write_qe_wfc_hdf5(filename, wfc::QEWavefunction; band_range=nothing)
    h5open(filename, "w") do file
        # Write root attributes matching QE format
        attrs(file)["gamma_only"] = wfc.gamma_only ? ".TRUE." : ".FALSE."
        attrs(file)["ik"] = Int32(wfc.ik)
        attrs(file)["xk"] = Vector{Float64}(wfc.xk)
        attrs(file)["ispin"] = Int32(wfc.ispin)
        attrs(file)["ngw"] = Int32(wfc.ngw)
        attrs(file)["igwx"] = Int32(wfc.igwx)
        attrs(file)["npol"] = Int32(wfc.npol)
        attrs(file)["nbnd"] = Int32(size(wfc.evc, 2))
        attrs(file)["scale_factor"] = Float64(wfc.scalef)

        # Write Miller indices dataset with reciprocal lattice vectors as attributes
        file["MillerIndices"] = Int32.(reduce(hcat, wfc.mill))  # (3, igwx)
        attrs(file["MillerIndices"])["bg1"] = Vector{Float64}(wfc.recip_lattice[:, 1])
        attrs(file["MillerIndices"])["bg2"] = Vector{Float64}(wfc.recip_lattice[:, 2])
        attrs(file["MillerIndices"])["bg3"] = Vector{Float64}(wfc.recip_lattice[:, 3])

        # Write wavefunction coefficients (complex -> real interleaved format)
        file["evc"] = collect(reinterpret(Float64, wfc.evc))  # (2*npol*igwx, nbnd)

        # Write optional band range metadata
        if band_range !== nothing
            attrs(file)["band_range"] = Int32.(collect(band_range))
        end
    end
end

include("real_space.jl")


end
