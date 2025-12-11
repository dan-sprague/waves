using GLMakie
using LinearAlgebra

# --- Constants ---
const N = 34          
const ITER = 2        
const DT = 0.2        
const VISC = 0.0001   
const STOCHASTIC_STR = 60.0 # Strength of random turbulence
const FLOW_SPEED = 2.0      # Strength of the continuous wind

mutable struct Fluid3D
    u::Array{Float64, 3}; v::Array{Float64, 3}; w::Array{Float64, 3}
    u0::Array{Float64, 3}; v0::Array{Float64, 3}; w0::Array{Float64, 3}
    div::Array{Float64, 3}; p::Array{Float64, 3}
end

function Fluid3D(n)
    z() = zeros(n, n, n)
    Fluid3D(z(), z(), z(), z(), z(), z(), z(), z())
end

# --- PERIODIC BOUNDARIES (The Secret to Flow) ---
# Instead of walls, we copy values from the opposite side.
function set_periodic!(x, N)
    # Wrap X
    @inbounds for k in 1:N, j in 1:N
        x[1, j, k] = x[N-1, j, k]
        x[N, j, k] = x[2, j, k]
    end
    # Wrap Y
    @inbounds for k in 1:N, i in 1:N
        x[i, 1, k] = x[i, N-1, k]
        x[i, N, k] = x[i, 2, k]
    end
    # Wrap Z
    @inbounds for j in 1:N, i in 1:N
        x[i, j, 1] = x[i, j, N-1]
        x[i, j, N] = x[i, j, 2]
    end
end

function lin_solve3d!(x, x0, a, c, N)
    cRecip = 1.0 / c
    @inbounds for n in 1:ITER
        for k in 2:N-1, j in 2:N-1, i in 2:N-1
            x[i, j, k] = (x0[i, j, k] + a * (
                x[i-1, j, k] + x[i+1, j, k] +
                x[i, j-1, k] + x[i, j+1, k] +
                x[i, j, k-1] + x[i, j, k+1]
            )) * cRecip
        end
        set_periodic!(x, N)
    end
end

function project3d!(u, v, w, p, div, N)
    h = 1.0/N
    @inbounds for k in 2:N-1, j in 2:N-1, i in 2:N-1
        div[i, j, k] = -1.0/3.0 * h * (
            u[i+1, j, k] - u[i-1, j, k] +
            v[i, j+1, k] - v[i, j-1, k] +
            w[i, j, k+1] - w[i, j, k-1]
        )
        p[i, j, k] = 0
    end
    set_periodic!(div, N); set_periodic!(p, N)
    
    lin_solve3d!(p, div, 1, 6, N)
    
    @inbounds for k in 2:N-1, j in 2:N-1, i in 2:N-1
        u[i, j, k] -= 0.5 * (p[i+1, j, k] - p[i-1, j, k]) / h
        v[i, j, k] -= 0.5 * (p[i, j+1, k] - p[i, j-1, k]) / h
        w[i, j, k] -= 0.5 * (p[i, j, k+1] - p[i, j, k-1]) / h
    end
    set_periodic!(u, N); set_periodic!(v, N); set_periodic!(w, N)
end

function advect3d!(d, d0, u, v, w, dt, N)
    dt0 = dt * N
    @inbounds for k in 2:N-1, j in 2:N-1, i in 2:N-1
        x = i - dt0 * u[i, j, k]
        y = j - dt0 * v[i, j, k]
        z = k - dt0 * w[i, j, k]
        
        # Periodic Wrap for the Backtrace
        # If we look back too far left, wrap to the right side
        if x < 1.5 x += (N - 2) end; if x > N - 0.5 x -= (N - 2) end
        if y < 1.5 y += (N - 2) end; if y > N - 0.5 y -= (N - 2) end
        if z < 1.5 z += (N - 2) end; if z > N - 0.5 z -= (N - 2) end

        i0, j0, k0 = floor(Int, x), floor(Int, y), floor(Int, z)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        s1, t1, r1 = x - i0, y - j0, z - k0
        s0, t0, r0 = 1 - s1, 1 - t1, 1 - r1
        
        d[i, j, k] = 
            r0 * (t0 * (s0 * d0[i0, j0, k0] + s1 * d0[i1, j0, k0]) +
                  t1 * (s0 * d0[i0, j1, k0] + s1 * d0[i1, j1, k0])) +
            r1 * (t0 * (s0 * d0[i0, j0, k1] + s1 * d0[i1, j0, k1]) +
                  t1 * (s0 * d0[i0, j1, k1] + s1 * d0[i1, j1, k1]))
    end
    set_periodic!(d, N)
end

function step_fluid3d!(f::Fluid3D)
    lin_solve3d!(f.u0, f.u, VISC, 1 + 6*VISC, N); lin_solve3d!(f.v0, f.v, VISC, 1 + 6*VISC, N); lin_solve3d!(f.w0, f.w, VISC, 1 + 6*VISC, N)
    project3d!(f.u0, f.v0, f.w0, f.p, f.div, N)
    advect3d!(f.u, f.u0, f.u0, f.v0, f.w0, DT, N); advect3d!(f.v, f.v0, f.u0, f.v0, f.w0, DT, N); advect3d!(f.w, f.w0, f.u0, f.v0, f.w0, DT, N)
    project3d!(f.u, f.v, f.w, f.p, f.div, N)
end

function apply_forces!(f::Fluid3D)
    # 1. THE FLOW: Constant push to the Right (+X)
    # We add this every frame.
    @inbounds for k in 2:N-1, j in 2:N-1, i in 2:N-1
        f.u[i, j, k] += 0.05 * FLOW_SPEED
    end

    # 2. THE CHAOS: Stochastic Kicks
    for _ in 1:6
        i, j, k = rand(2:N-1), rand(2:N-1), rand(2:N-1)
        # Random kicks in all directions
        f.u[i, j, k] += randn() * STOCHASTIC_STR
        f.v[i, j, k] += randn() * STOCHASTIC_STR
        f.w[i, j, k] += randn() * STOCHASTIC_STR
    end
end

# --- PARTICLES ---
mutable struct Particles3D
    pos::Vector{Point3f}
    vel::Vector{Point3f}
end

function update_particles!(parts, f::Fluid3D)
    @inbounds for idx in 1:length(parts.pos)
        p = parts.pos[idx]
        # Map to grid
        gx, gy, gz = p[1]*N, p[2]*N, p[3]*N
        i, j, k = clamp(floor(Int, gx), 1, N), clamp(floor(Int, gy), 1, N), clamp(floor(Int, gz), 1, N)
        
        # Get velocity
        u, v, w = f.u[i, j, k], f.v[i, j, k], f.w[i, j, k]
        
        # SDE Update
        nx = p[1] + (u * 0.005) + (randn() * 0.0005)
        ny = p[2] + (v * 0.005) + (randn() * 0.0005)
        nz = p[3] + (w * 0.005) + (randn() * 0.0005)
        
        # Periodic Wrap (The Portal Effect)
        if nx > 1.0 nx -= 1.0 end; if nx < 0.0 nx += 1.0 end
        if ny > 1.0 ny -= 1.0 end; if ny < 0.0 ny += 1.0 end
        if nz > 1.0 nz -= 1.0 end; if nz < 0.0 nz += 1.0 end
        
        parts.pos[idx] = Point3f(nx, ny, nz)
        parts.vel[idx] = Point3f(u, v, w)
    end
end

# --- RUN ---
fluid = Fluid3D(N)
# Start with some random chaos so it doesn't look like a rigid block at frame 1
for _ in 1:10 apply_forces!(fluid); step_fluid3d!(fluid) end 

parts = Particles3D([Point3f(rand(), rand(), rand()) for _ in 1:10000], 
                    [Point3f(0,0,0) for _ in 1:10000])

pts_obs = Observable(parts.pos)
col_obs = Observable(zeros(10000))

fig = Figure(size = (1000, 1000), backgroundcolor = :black)
ax = Axis3(fig[1, 1], aspect = :data, backgroundcolor = :black, azimuth = 0.6, elevation = 0.4)
hidedecorations!(ax); hidespines!(ax)

meshscatter!(ax, pts_obs, color = col_obs, colormap = :plasma, markersize = 0.012, shading = NoShading, transparency=true, alpha=0.5)

display(fig)

@async begin
    while isopen(fig.scene)
        apply_forces!(fluid)
        step_fluid3d!(fluid)
        update_particles!(parts, fluid)
        pts_obs[] = parts.pos
        # Color by velocity magnitude to highlight fast currents
        col_obs[] = [norm(v) for v in parts.vel]
        sleep(1/60)
    end
end