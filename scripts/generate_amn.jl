#!/usr/bin/env julia
# Generate amn file using rigidly displaced Wannier functions as initial guesses.
#
# The per-Wannier-function rigid shift dR_n is read from the projection centers
# of two prefix.nnkp files (reference and perturbed): dR_n = center_pert_n -
# center_orig_n (in crystal/fractional coordinates). Each reference Wannier
# function n is translated by dR_n before overlapping with the perturbed bands.
#
# Usage:
#   ./scripts/generate_amn.jl prefix folder_orig folder_pert wfcdir_orig wfcdir_pert output.amn
#
# or, if julia is not on PATH / the environment is installed elsewhere:
#   julia --project=/path/to/QEWavefunctions.jl scripts/generate_amn.jl ARGS...
#
# Arguments:
#   prefix        Wannier90 prefix (reads <prefix>.chk and <prefix>.nnkp)
#   folder_orig   Folder with <prefix>.nnkp and <prefix>.chk of the reference system
#   folder_pert   Folder with <prefix>.nnkp of the perturbed system
#   wfcdir_orig   QE save folder with reference wfcN.hdf5 or wfcN.dat
#   wfcdir_pert   QE save folder with perturbed wfcN.hdf5 or wfcN.dat
#   output.amn    Output amn file path
#
# wfc files in either HDF5 (wfcN.hdf5) or Fortran binary (wfcN.dat) format are
# auto-detected per folder.

# Activate the package environment (repo root) regardless of how the script is
# launched, so QEWavefunctions and WannierIO resolve.
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using QEWavefunctions
using WannierIO
using StaticArrays
using LinearAlgebra
using HDF5

"""Locate the wfc file for k-point `ik` in `folder`, preferring HDF5 over .dat."""
function _find_wfc(folder, ik)
    h5 = joinpath(folder, "wfc$ik.hdf5")
    dat = joinpath(folder, "wfc$ik.dat")
    isfile(h5) && return h5
    isfile(dat) && return dat
    error("No wfc$ik.hdf5 or wfc$ik.dat found in $folder")
end

"""Check if a wfc file is a trimmed HDF5 file (has band_range attribute). .dat files are never trimmed."""
function _has_band_range(filename)
    endswith(filename, ".hdf5") || return false
    h5open(filename, "r") do file
        haskey(attrs(file), "band_range")
    end
end

"""Read the band_range attribute from a trimmed wfc HDF5 file."""
function _read_band_range(filename)
    h5open(filename, "r") do file
        Int.(attrs(file)["band_range"])
    end
end

"""
Parse the centers of the `spinor_projections` block of an nnkp file (crystal
coordinates). WannierIO's `read_nnkp` only parses the scalar `projections` block,
so spinor projections are read here directly.

Each spinor projection occupies three lines; the center is the first three
numbers of the first line:
    cx cy cz   l mr rad
    zx zy zz   xx xy xz   zona
    spin   sqx sqy sqz
"""
function _read_spinor_projection_centers(nnkp_file)
    lines = readlines(nnkp_file)
    i = findfirst(l -> occursin("begin spinor_projections", l), lines)
    i === nothing && return nothing
    n_projs = parse(Int, strip(lines[i + 1]))
    centers = SVector{3, Float64}[]
    for j in 1:n_projs
        toks = split(strip(lines[i + 1 + 3 * (j - 1) + 1]))
        push!(centers, SVector{3, Float64}(parse.(Float64, toks[1:3])))
    end
    return centers
end

"""
Read projection centers (crystal coordinates) from an nnkp file, supporting both
the scalar `projections` block (collinear) and the `spinor_projections` block
(spinor / noncollinear).
"""
function _read_projection_centers(nnkp_file)
    proj = get(read_nnkp(nnkp_file), :projections, nothing)
    if proj !== nothing
        return [SVector{3, Float64}(p.center) for p in proj]
    end
    centers = _read_spinor_projection_centers(nnkp_file)
    centers === nothing && error("No `projections` or `spinor_projections` block in $nnkp_file")
    return centers
end

"""
Read per-Wannier-function rigid shifts from the projection centers of the two
nnkp files: dR_n = center_pert_n - center_orig_n, in crystal coordinates.
"""
function _read_projection_shifts(nnkp_orig, nnkp_pert)
    c_orig = _read_projection_centers(nnkp_orig)
    c_pert = _read_projection_centers(nnkp_pert)
    @assert length(c_orig) == length(c_pert) (
        "Number of projections differ: orig=$(length(c_orig)), pert=$(length(c_pert))")
    return [SVector{3, Float64}(p - o) for (o, p) in zip(c_orig, c_pert)]
end

function generate_amn_using_rigid_WF(prefix, folder_orig, folder_pert,
                                     wfcdir_orig, wfcdir_pert, filename_output)
    # Reference Wannier functions
    chk = read_chk(joinpath(folder_orig, prefix * ".chk"))

    # Per-WF rigid shift from projection centers (crystal coordinates)
    dR_crys = _read_projection_shifts(joinpath(folder_orig, prefix * ".nnkp"),
                                      joinpath(folder_pert, prefix * ".nnkp"))
    println("# Per-WF rigid shift dR_n = center_pert_n - center_orig_n (crystal coordinates):")
    for (n, dR) in enumerate(dR_crys)
        println("#   WF $n : [$(dR[1]), $(dR[2]), $(dR[3])]")
    end
    flush(stdout)

    wfc1_orig = _find_wfc(wfcdir_orig, 1)
    wfc1_pert = _find_wfc(wfcdir_pert, 1)

    wfc_metadata = read_qe_wfc(wfc1_orig; metadata_only = true)
    flush(stderr)

    # Determine if wfc files are already trimmed to selected bands
    orig_trimmed = _has_band_range(wfc1_orig)
    pert_trimmed = _has_band_range(wfc1_pert)

    if orig_trimmed
        band_rng = _read_band_range(wfc1_orig)
        @assert wfc_metadata.nbnd == length(band_rng) (
            "Trimmed orig wfc nbnd ($(wfc_metadata.nbnd)) != length(band_range) ($(length(band_rng)))")
        println("# Orig wfc: trimmed ($(length(band_rng)) bands, band_range=$band_rng)")
    else
        # Indices of selected bands
        band_rng = setdiff(1:wfc_metadata.nbnd, chk.exclude_bands)
        println("# Orig wfc: full ($(wfc_metadata.nbnd) bands, selecting $(length(band_rng)))")
    end

    if pert_trimmed
        pert_band_rng = _read_band_range(wfc1_pert)
        @assert pert_band_rng == band_rng (
            "band_range mismatch: orig=$band_rng, pert=$pert_band_rng")
        println("# Pert wfc: trimmed ($(length(pert_band_rng)) bands, band_range=$pert_band_rng)")
    else
        println("# Pert wfc: full")
    end

    # Verify band / projection counts are consistent with Wannier90 rotation matrices
    num_bands_expected = length(chk.Udis) > 0 ? size(chk.Udis[1], 1) : size(chk.Uml[1], 1)
    @assert length(band_rng) == num_bands_expected (
        "Number of selected bands ($(length(band_rng))) != expected from chk ($num_bands_expected)")
    n_wann = size(get_U(chk)[1], 2)
    @assert length(dR_crys) == n_wann (
        "Number of projections ($(length(dR_crys))) != number of Wannier functions ($n_wann)")

    flush(stdout)

    A = @views map(1:chk.n_kpts) do ik
        if mod(ik, 100) == 0
            println("# generate_amn.jl : processing k-point $ik / $(chk.n_kpts)")
            flush(stdout)
        end

        wfc_orig = if orig_trimmed
            read_qe_wfc(_find_wfc(wfcdir_orig, ik))
        else
            read_qe_wfc(_find_wfc(wfcdir_orig, ik); bands=band_rng)
        end

        wfc_pert = if pert_trimmed
            read_qe_wfc(_find_wfc(wfcdir_pert, ik))
        else
            read_qe_wfc(_find_wfc(wfcdir_pert, ik); bands=band_rng)
        end

        xk_crys_orig = wfc_orig.recip_lattice \ wfc_orig.xk
        xk_crys_pert = wfc_pert.recip_lattice \ wfc_pert.xk
        @assert xk_crys_orig ≈ xk_crys_pert "k-points do not match between wfc files"
        @assert xk_crys_orig ≈ chk.kpoints[ik] "k-points do not match with chk file"
        @assert wfc_orig.npol == wfc_pert.npol "npol mismatch: orig=$(wfc_orig.npol), pert=$(wfc_pert.npol)"

        U = get_U(chk)[ik]

        # Per-WF shifted overlap, all in crystal coordinates.
        #   A[m, n] = < wfc_pert_m | exp(-i (k+G)·dR_n) | wan_orig_n >
        # with wan_orig_n = sum_m' wfc_orig_m' U[m', n] and, since b_i·a_j = 2π δ_ij,
        #   (k+G)·dR_n = 2π (κ + mill_G)·dR_crys_n  (κ = k in crystal coords).
        # For spinors (npol=2) the overlap is summed over both spin components; the
        # translation phase is spin-diagonal and identical for both. In QE's evc
        # layout, rows 1:igwx hold spin component 1 and igwx+1:2igwx component 2.
        idx_pert, idx_orig = match_miller_indices(wfc_pert, wfc_orig)
        κ = SVector{3, Float64}(chk.kpoints[ik])
        mill_m = wfc_orig.mill[idx_orig]
        phase = [cispi(-2 * dot(κ + mill_m[g], dR_crys[n]))
                 for g in eachindex(mill_m), n in 1:n_wann]
        A_ik = zeros(ComplexF64, size(wfc_pert.evc, 2), n_wann)
        for σ in 1:wfc_orig.npol
            rows_orig = idx_orig .+ (σ - 1) * wfc_orig.igwx
            rows_pert = idx_pert .+ (σ - 1) * wfc_pert.igwx
            D = wfc_orig.evc[rows_orig, :] * U                 # orig WFs in G-space (matched, this spin)
            A_ik .+= wfc_pert.evc[rows_pert, :]' * (D .* phase)
        end

        A_ik
    end

    header = "# Created using rigidly-shifted Wannier functions (shift from nnkp projection centers)"
    write_amn(filename_output, A; header)
end;


function main(args)
    if length(args) != 6
        println(stderr, """
        Usage: generate_amn.jl prefix folder_orig folder_pert wfcdir_orig wfcdir_pert output.amn

          prefix        Wannier90 prefix (reads <prefix>.chk and <prefix>.nnkp)
          folder_orig   Folder with <prefix>.nnkp and <prefix>.chk of the reference system
          folder_pert   Folder with <prefix>.nnkp of the perturbed system
          wfcdir_orig   QE save folder with reference wfcN.hdf5 or wfcN.dat
          wfcdir_pert   QE save folder with perturbed wfcN.hdf5 or wfcN.dat
          output.amn    Output amn file path
        """)
        exit(1)
    end

    prefix = args[1]
    folder_orig = args[2]
    folder_pert = args[3]
    wfcdir_orig = args[4]
    wfcdir_pert = args[5]
    filename_output = args[6]

    println("# Generating amn file using rigidly displaced Wannier functions")
    println("# Wannier90 prefix                      : $prefix")
    println("# nnkp/chk folder of reference system   : $folder_orig")
    println("# nnkp folder of perturbed system       : $folder_pert")
    println("# QE save folder of reference system    : $wfcdir_orig")
    println("# QE save folder of perturbed system    : $wfcdir_pert")
    println("# Output amn file                       : $filename_output")
    flush(stdout)

    generate_amn_using_rigid_WF(prefix, folder_orig, folder_pert,
                                wfcdir_orig, wfcdir_pert, filename_output)

    println("# amn file written")
    flush(stdout)
end

main(ARGS)
