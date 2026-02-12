using StaticArrays
using LinearAlgebra
using FFTW

"""
    compute_real_space_wfc(wfc::QEWavefunction, ngrid::NTuple{3, Int}) -> Array{ComplexF64, 4}

Compute real-space wavefunctions from reciprocal-space (G-space) representation.

Uses a hybrid approach with pre-computed phase factors for each spatial direction
to reduce redundant trigonometric calculations and improve cache locality.

The output is the periodic part of the wavefunction u_nk(r). To compute the full Bloch
wavefunction, use ψ_nk(r) = e^(ik·r) * u_nk(r).
```
rgrid = QEWavefunctions._real_space_grid(ngrid)
phase = map(r -> cispi(2 * dot(k, r)), rgrid)
```

`u_nk(r)` is normalized as ``∑ᵣ |u_nk(r)|² = 1`` over the unit cell.

# Arguments
- `wfc::QEWavefunction`: Wavefunction data in reciprocal space
- `ngrid::NTuple{3, Int}`: Real-space grid dimensions (nx, ny, nz)

# Returns
- 4D array of size (nx, ny, nz, nbnd) containing real-space wavefunctions

# Algorithm
The transformation is: ψ(r) = (1/√(nx*ny*nz)) Σ_G c_G e^(iG·r)
where the phase factors are pre-computed separately for each direction.
"""
function compute_real_space_wfc(wfc::QEWavefunction, ngrid::NTuple{3, Int})
    nx, ny, nz = ngrid
    wfc_r = zeros(ComplexF64, nx, ny, nz, wfc.nbnd)

    # Pre-compute phase factors for each spatial direction
    # This avoids recomputing exp(i*G·r) for each (r, G) pair
    # Instead we use: exp(i*G·r) = exp(i*Gx*x) * exp(i*Gy*y) * exp(i*Gz*z)
    nG = wfc.igwx
    phases_x = Matrix{ComplexF64}(undef, nx, nG)
    phases_y = Matrix{ComplexF64}(undef, ny, nG)
    phases_z = Matrix{ComplexF64}(undef, nz, nG)

    for (iG, G) in enumerate(wfc.mill)
        for ix in 1:nx
            phases_x[ix, iG] = cispi(2 * G[1] * (ix - 1) / nx)
        end
        for iy in 1:ny
            phases_y[iy, iG] = cispi(2 * G[2] * (iy - 1) / ny)
        end
        for iz in 1:nz
            phases_z[iz, iG] = cispi(2 * G[3] * (iz - 1) / nz)
        end
    end

    # Loop over real-space grid points with optimized memory access pattern
    # Combine pre-computed phase factors
    phase_matrix = zeros(ComplexF64, prod(ngrid), nG)
    @views for ir in 1:prod(ngrid)
        ix, iy, iz = CartesianIndices(ngrid)[ir].I
        @. phase_matrix[ir, :] = phases_x[ix, :] * phases_y[iy, :] * phases_z[iz, :]
    end

    mul!(Base.ReshapedArray(wfc_r, (prod(ngrid), wfc.nbnd), ()), phase_matrix, wfc.evc)

    # Normalization of Fourier transform
    wfc_r ./= sqrt(prod(ngrid))

    return wfc_r
end

function _real_space_grid(ngrid :: NTuple{3, Int})
    rgrid_arr = zeros(Float64, 3, ngrid...)
    rgrid_arr[1, :, :, :] .= reshape(collect(0:ngrid[1]-1) ./ ngrid[1], (ngrid[1], 1, 1))
    rgrid_arr[2, :, :, :] .= reshape(collect(0:ngrid[2]-1) ./ ngrid[2], (1, ngrid[2], 1))
    rgrid_arr[3, :, :, :] .= reshape(collect(0:ngrid[3]-1) ./ ngrid[3], (1, 1, ngrid[3]))

    rgrid_vec = collect(reshape(reinterpret(SVector{3, Float64}, rgrid_arr), prod(ngrid)))
    rgrid_vec
end

"""
    compute_real_space_wfc_fft(wfc::QEWavefunction, ngrid::NTuple{3, Int}) -> Array{ComplexF64, 4}

Compute real-space wavefunctions from G-space using FFT.

Equivalent to `compute_real_space_wfc` but uses FFTW's IFFT instead of explicit
matrix multiplication, which is much faster for large grids.

The output `u_nk(r)` is normalized as ``∑ᵣ |u_nk(r)|² = 1`` over the unit cell.

# Arguments
- `wfc::QEWavefunction`: Wavefunction data in reciprocal space
- `ngrid::NTuple{3, Int}`: Real-space grid dimensions (nx, ny, nz)

# Returns
- 4D array of size (nx, ny, nz, nbnd) containing real-space wavefunctions
"""
function compute_real_space_wfc_fft(wfc::QEWavefunction, ngrid::NTuple{3, Int})
    nx, ny, nz = ngrid
    nbnd = size(wfc.evc, 2)
    coeff = zeros(ComplexF64, nx, ny, nz, nbnd)
    for (iG, G) in enumerate(wfc.mill)
        ix = mod(G[1], nx) + 1
        iy = mod(G[2], ny) + 1
        iz = mod(G[3], nz) + 1
        for n in 1:nbnd
            coeff[ix, iy, iz, n] = wfc.evc[iG, n]
        end
    end
    # IFFT along spatial dims; FFTW IFFT divides by N, we want 1/√N normalization
    wfc_r = ifft(coeff, (1, 2, 3))
    wfc_r .*= sqrt(prod(ngrid))
    return wfc_r
end

"""
    compute_z_density(wfc::QEWavefunction, nz::Int) -> Matrix{Float64}

Compute z-projected density for each band using FFT along z.

Returns `ρ[iz, n] = (1/nz) Σ_{Gx,Gy} |Σ_{Gz} c(Gx,Gy,Gz,n) e^{2πi Gz (iz-1)/nz}|²`

This is memory-efficient: only allocates O(n_gxy_groups × nz) instead of the full 3D grid.

# Arguments
- `wfc::QEWavefunction`: Wavefunction data in reciprocal space
- `nz::Int`: Number of grid points along z

# Returns
- Matrix of size (nz, nbnd) containing z-projected density
"""
function compute_z_density(wfc::QEWavefunction, nz::Int)
    nbnd = size(wfc.evc, 2)

    # Group G-vectors by (Gx, Gy)
    gxy_keys = [(G[1], G[2]) for G in wfc.mill]
    unique_gxy = unique(gxy_keys)
    gxy_to_idx = Dict(k => i for (i, k) in enumerate(unique_gxy))
    n_groups = length(unique_gxy)

    coeff = zeros(ComplexF64, n_groups, nz)
    rho = zeros(Float64, nz, nbnd)

    for n in 1:nbnd
        fill!(coeff, 0)
        for (iG, G) in enumerate(wfc.mill)
            g = gxy_to_idx[(G[1], G[2])]
            gz_idx = mod(G[3], nz) + 1
            coeff[g, gz_idx] += wfc.evc[iG, n]
        end
        f = ifft(coeff, 2)  # IFFT along z dimension: (n_groups, nz)
        for iz in 1:nz
            for g in 1:n_groups
                rho[iz, n] += abs2(f[g, iz])
            end
        end
    end
    # ifft divides by nz; we want ρ = (1/nz)|Σ c e^{...}|² = nz * Σ|ifft|²
    rho .*= nz
    return rho
end

