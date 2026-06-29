module QEWavefunctions

using StaticArrays
using HDF5
using LinearAlgebra

export QEWavefunction
export read_qe_wfc
export read_qe_wfc_hdf5
export read_qe_wfc_dat
export write_qe_wfc_hdf5
export match_miller_indices
export compute_wfc_overlap
export compute_real_space_wfc
export compute_real_space_wfc_fft
export compute_z_density

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

"""
    read_qe_wfc(filename; metadata_only=false, bands=nothing) -> QEWavefunction

Dispatches to `read_qe_wfc_hdf5` or `read_qe_wfc_dat` based on the file extension
(`.hdf5` or `.dat`).
"""
function read_qe_wfc(filename; kwargs...)
    if endswith(filename, ".hdf5")
        read_qe_wfc_hdf5(filename; kwargs...)
    elseif endswith(filename, ".dat")
        read_qe_wfc_dat(filename; kwargs...)
    else
        error("Unknown extension for QE wfc file: $filename (expected .hdf5 or .dat)")
    end
end

# Read one Fortran sequential unformatted record. Records are framed by 4-byte
# little-endian length markers at both ends. Returns the payload as a byte vector.
function _read_fortran_record(io::IO)
    n = read(io, Int32)
    bytes = read(io, Int(n))
    length(bytes) == n || error("Truncated Fortran record (expected $n bytes, got $(length(bytes)))")
    n_end = read(io, Int32)
    n_end == n || error("Fortran record marker mismatch: head=$n tail=$n_end")
    bytes
end

function _skip_fortran_record(io::IO)
    n = read(io, Int32)
    skip(io, Int(n))
    n_end = read(io, Int32)
    n_end == n || error("Fortran record marker mismatch while skipping: head=$n tail=$n_end")
    nothing
end

"""
    read_qe_wfc_dat(filename; metadata_only=false, bands=nothing) -> QEWavefunction

Read a Quantum ESPRESSO wavefunction stored in Fortran sequential unformatted
binary (`wfcN.dat`). Mirrors `read_qe_wfc_hdf5` for HDF5 inputs.

Record layout (see `write_wfc` in QE Modules/io_base.f90):
1. `ik (i4), xk (r8 × 3), ispin (i4), gamma_only (l4), scalef (r8)` — 44 bytes
2. `ngw (i4), igwx (i4), npol (i4), nbnd (i4)` — 16 bytes
3. `b1, b2, b3 (r8 × 9)` — 72 bytes
4. `mill (i4 × 3 × igwx)`
5..`nbnd+4`. one per band: `evc(1:npol*igwx)` (c16)
"""
function read_qe_wfc_dat(filename; metadata_only::Bool = false, bands = nothing)
    open(filename, "r") do io
        # Record 1
        rec = _read_fortran_record(io)
        length(rec) == 44 || error("Unexpected size for record 1: $(length(rec)) (expected 44)")
        ik = Int(reinterpret(Int32, @view rec[1:4])[1])
        xk = SVector{3, Float64}(reinterpret(Float64, @view rec[5:28]))
        ispin = Int(reinterpret(Int32, @view rec[29:32])[1])
        gamma_only = reinterpret(Int32, @view rec[33:36])[1] != 0
        scalef = reinterpret(Float64, @view rec[37:44])[1]

        # Record 2
        rec = _read_fortran_record(io)
        length(rec) == 16 || error("Unexpected size for record 2: $(length(rec)) (expected 16)")
        h = reinterpret(Int32, rec)
        ngw, igwx, npol, nbnd = Int(h[1]), Int(h[2]), Int(h[3]), Int(h[4])

        # Record 3: reciprocal lattice vectors
        rec = _read_fortran_record(io)
        length(rec) == 72 || error("Unexpected size for record 3: $(length(rec)) (expected 72)")
        bg = reshape(collect(reinterpret(Float64, rec)), (3, 3))
        recip_lattice = SMatrix{3, 3, Float64}(bg)

        if metadata_only
            return QEWavefunction(gamma_only, ik, ispin, xk, scalef, ngw, igwx, npol, nbnd,
                                  recip_lattice, SVector{3, Int}[], Matrix{ComplexF64}(undef, 0, 0))
        end

        # Record 4: Miller indices
        rec = _read_fortran_record(io)
        length(rec) == 12 * igwx || error("Unexpected size for Miller indices record: $(length(rec)) (expected $(12*igwx))")
        mill_i32 = collect(reinterpret(SVector{3, Int32}, rec))
        mill = SVector{3, Int}.(mill_i32)

        # Records 5..nbnd+4: evc, one record per band, npol*igwx ComplexF64 each
        if bands === nothing
            evc = Matrix{ComplexF64}(undef, npol * igwx, nbnd)
            for j in 1:nbnd
                rec = _read_fortran_record(io)
                length(rec) == 16 * npol * igwx || error(
                    "Unexpected size for evc record $j: $(length(rec)) (expected $(16*npol*igwx))")
                evc[:, j] .= reinterpret(ComplexF64, rec)
            end
        else
            sel = collect(bands)
            keep_idx = Dict(b => i for (i, b) in enumerate(sel))
            evc = Matrix{ComplexF64}(undef, npol * igwx, length(sel))
            for j in 1:nbnd
                if haskey(keep_idx, j)
                    rec = _read_fortran_record(io)
                    length(rec) == 16 * npol * igwx || error(
                        "Unexpected size for evc record $j: $(length(rec)) (expected $(16*npol*igwx))")
                    evc[:, keep_idx[j]] .= reinterpret(ComplexF64, rec)
                else
                    _skip_fortran_record(io)
                end
            end
        end

        QEWavefunction(gamma_only, ik, ispin, xk, scalef, ngw, igwx, npol, nbnd,
                       recip_lattice, mill, evc)
    end
end

"""
    match_miller_indices(wfc1::QEWavefunction, wfc2::QEWavefunction) -> (idx1, idx2)

Find Miller indices shared by `wfc1` and `wfc2`, returning index vectors such that
`wfc1.mill[idx1] == wfc2.mill[idx2]` elementwise.

The returned indices preserve the order of `wfc1.mill`, so downstream code that
pre-computes phases from `wfc1.mill[idx1]` indexes consistently into the matched
rows of `wfc1.evc` and `wfc2.evc`.

Useful when comparing wavefunctions from two QE runs whose G-spheres differ
(e.g. different geometries with slightly different reciprocal lattices).
"""
function match_miller_indices(wfc1::QEWavefunction, wfc2::QEWavefunction)
    mill2_to_idx = Dict{SVector{3, Int}, Int}()
    sizehint!(mill2_to_idx, length(wfc2.mill))
    for (i, G) in enumerate(wfc2.mill)
        mill2_to_idx[G] = i
    end
    n_max = min(length(wfc1.mill), length(wfc2.mill))
    idx1 = Int[]
    idx2 = Int[]
    sizehint!(idx1, n_max)
    sizehint!(idx2, n_max)
    for (i, G) in enumerate(wfc1.mill)
        j = get(mill2_to_idx, G, 0)
        if j != 0
            push!(idx1, i)
            push!(idx2, j)
        end
    end
    return idx1, idx2
end

"""
    compute_wfc_overlap(wfc1::QEWavefunction, wfc2::QEWavefunction;
                        dR=SVector(0., 0., 0.)) -> Matrix{ComplexF64}

Compute the overlap matrix `S[m, n] = <ψ1_m | exp(-i (k+G) · dR) | ψ2_n>` between
two wavefunctions, restricted to Miller indices present in both.

`dR` is an optional Wannier-center shift (Cartesian, Bohr); when zero the phase
factor is skipped.

`wfc1.recip_lattice` and `wfc1.xk` are used to build the k+G vectors that enter
the phase — same-Miller-index is treated as same-plane-wave, the appropriate
convention when the two structures share a nearly identical lattice.

For spinor wavefunctions (`npol == 2`) the overlap is summed over both spin
components: `S[m, n] = Σ_σ Σ_G conj(c1_m[G, σ]) e^{-i(k+G)·dR} c2_n[G, σ]`. The
translation phase is spin-diagonal, so the same phase factor is applied to both
components. In QE's `evc` layout, rows `1:igwx` hold spin component 1 and rows
`igwx+1:2*igwx` hold spin component 2.
"""
function compute_wfc_overlap(wfc1::QEWavefunction, wfc2::QEWavefunction;
                             dR::SVector{3, Float64} = SVector(0., 0., 0.))
    @assert wfc1.npol == wfc2.npol "npol mismatch: $(wfc1.npol) vs $(wfc2.npol)"
    @assert wfc1.ispin == wfc2.ispin "ispin mismatch: $(wfc1.ispin) vs $(wfc2.ispin)"
    idx1, idx2 = match_miller_indices(wfc1, wfc2)
    phase = iszero(dR) ? nothing : begin
        kGs = Ref(wfc1.recip_lattice) .* wfc1.mill[idx1] .+ Ref(wfc1.xk)
        cis.(.-dot.(kGs, Ref(dR)))
    end
    S = zeros(ComplexF64, size(wfc1.evc, 2), size(wfc2.evc, 2))
    for σ in 1:wfc1.npol
        rows1 = idx1 .+ (σ - 1) * wfc1.igwx
        rows2 = idx2 .+ (σ - 1) * wfc2.igwx
        if phase === nothing
            S .+= wfc1.evc[rows1, :]' * wfc2.evc[rows2, :]
        else
            S .+= wfc1.evc[rows1, :]' * Diagonal(phase) * wfc2.evc[rows2, :]
        end
    end
    return S
end

include("real_space.jl")


end
