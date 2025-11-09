using StaticArrays
using LinearAlgebra

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

