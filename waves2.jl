using GLMakie

# --- Constants ---
const N = 100          # Grid Size
const ITER = 4         # Solver iterations
const DT = 0.1         # Time step
const VISC = 0.0001    # Fluid Viscosity
const FORCE_STR = 50.0 # Strength of random stochastic kicks
const PARTICLE_COUNT = 50

# --- Fluid Struct (The Grid) ---
mutable struct Fluid
    u::Matrix{Float64}   # Velocity X
    v::Matrix{Float64}   # Velocity Y
    u0::Matrix{Float64}  # Previous Vel X
    v0::Matrix{Float64}  # Previous Vel Y
    div::Matrix{Float64} # Divergence (scratch space)
    p::Matrix{Float64}   # Pressure (scratch space)
end

function Fluid(n)
    zeros_mat() = zeros(n, n)
    Fluid(zeros_mat(), zeros_mat(), zeros_mat(), zeros_mat(), zeros_mat(), zeros_mat())
end

# --- Navier-Stokes Solvers (Stable Fluids) ---
# (Simplified specific solver for velocity only - no density dye needed)

function set_bnd!(b, x, N)
    # Simple reflective boundaries
    @inbounds for i in 2:N-1
        x[1, i] = b == 1 ? -x[2, i] : x[2, i]
        x[N, i] = b == 1 ? -x[N-1, i] : x[N-1, i]
        x[i, 1] = b == 2 ? -x[i, 2] : x[i, 2]
        x[i, N] = b == 2 ? -x[i, N-1] : x[i, N-1]
    end
end

function lin_solve!(b, x, x0, a, c, N)
    cRecip = 1.0 / c
    @inbounds for k in 1:ITER
        for j in 2:N-1
            for i in 2:N-1
                x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) * cRecip
            end
        end
        set_bnd!(b, x, N)
    end
end

function project!(u, v, p, div, N)
    h = 1.0/N
    @inbounds for j in 2:N-1
        for i in 2:N-1
            div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
            p[i, j] = 0
        end
    end
    set_bnd!(0, div, N)
    set_bnd!(0, p, N)
    lin_solve!(0, p, div, 1, 4, N)
    
    @inbounds for j in 2:N-1
        for i in 2:N-1
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h
        end
    end
    set_bnd!(1, u, N)
    set_bnd!(2, v, N)
end

function advect!(b, d, d0, u, v, dt, N)
    dt0 = dt * N
    @inbounds for j in 2:N-1
        for i in 2:N-1
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            if x < 1.5 x = 1.5 end
            if x > N - 0.5 x = N - 0.5 end
            if y < 1.5 y = 1.5 end
            if y > N - 0.5 y = N - 0.5 end
            i0, j0 = floor(Int, x), floor(Int, y)
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1.0 - s1, 1.0 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                      s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
        end
    end
    set_bnd!(b, d, N)
end

function step_fluid!(f::Fluid)
    # 1. Diffusion
    lin_solve!(1, f.u0, f.u, VISC * DT * N * N, 1 + 4 * VISC * DT * N * N, N)
    lin_solve!(2, f.v0, f.v, VISC * DT * N * N, 1 + 4 * VISC * DT * N * N, N)
    
    # 2. Projection
    project!(f.u0, f.v0, f.p, f.div, N)
    
    # 3. Advection
    advect!(1, f.u, f.u0, f.u0, f.v0, DT, N)
    advect!(2, f.v, f.v0, f.u0, f.v0, DT, N)
    
    # 4. Final Project
    project!(f.u, f.v, f.p, f.div, N)
end

# --- STOCHASTIC FORCING (The "S" in SPDE) ---
function add_stochastic_force!(f::Fluid)
    # Inject random momentum at random spots
    # This simulates turbulent energy injection
    for _ in 1:3
        i = rand(3:N-2)
        j = rand(3:N-2)
        # Random direction and strength
        f.u[i, j] += randn() * FORCE_STR
        f.v[i, j] += randn() * FORCE_STR
    end
end

# --- SDE PARTICLE SYSTEM ---
mutable struct Particles
    pos::Vector{Point2f}
    vel::Vector{Point2f}
end

function update_particles!(parts, f::Fluid)
    # Move particles according to the fluid field + Brownian noise
    for k in 1:length(parts.pos)
        px, py = parts.pos[k]
        
        # 1. Sample Fluid Velocity at particle position (Linear Interpolation)
        # Map 0..1 coordinates to 1..N grid
        gx = px * N
        gy = py * N
        
        # Clamp to grid
        gx = clamp(gx, 1.1, N-0.1)
        gy = clamp(gy, 1.1, N-0.1)
        
        i, j = floor(Int, gx), floor(Int, gy)
        
        # Simple nearest neighbor for speed (or bilinear for smoothness)
        u_curr = f.u[i, j]
        v_curr = f.v[i, j]
        
        # 2. SDE Update: dX = u*dt + sigma*dW
        # Drift (Fluid Flow)
        dx = u_curr * 0.005 
        dy = v_curr * 0.005
        
        # Diffusion (Brownian Jitter) - The "Stochastic" part
        noise_x = randn() * 0.001
        noise_y = randn() * 0.001
        
        new_x = px + dx + noise_x
        new_y = py + dy + noise_y
        
        # 3. Boundary Wrap/Reset
        if new_x < 0 || new_x > 1 || new_y < 0 || new_y > 1
            # Respawn randomly
            new_x, new_y = rand(), rand()
        end
        
        parts.pos[k] = Point2f(new_x, new_y)
        
        # Color based on velocity
        vel_mag = sqrt(u_curr^2 + v_curr^2)
        parts.vel[k] = Point2f(vel_mag, 0) # Store magnitude in X for coloring
    end
end

# --- MAIN VISUALIZATION ---
fluid = Fluid(N)
parts = Particles([Point2f(rand(), rand()) for _ in 1:PARTICLE_COUNT], 
                  [Point2f(0,0) for _ in 1:PARTICLE_COUNT])

# Observables
pts_obs = Observable(parts.pos)
color_obs = Observable(zeros(PARTICLE_COUNT))

fig = Figure(size = (900, 900), backgroundcolor = :black)
ax = Axis(fig[1, 1], backgroundcolor = :black, aspect = 1)
hidedecorations!(ax); hidespines!(ax)

# Plot particles
scatter!(ax, pts_obs, 
    color = color_obs, 
    colormap = :plasma, 
    markersize = 4,
    transparency = true,
    alpha = 0.6
)

display(fig)

# --- RUN LOOP ---
@async begin
    while isopen(fig.scene)
        # 1. Physics Step
        add_stochastic_force!(fluid) # The SPDE noise
        step_fluid!(fluid)           # The Navier-Stokes Solver
        
        # 2. Particle Step (Lagrangian SDE)
        update_particles!(parts, fluid)
        
        # 3. Update Graphics
        pts_obs[] = parts.pos
        # Update colors based on velocity magnitude (stored in parts.vel.x)
        color_obs[] = [p[1] * 0.5 for p in parts.vel] 
        
        sleep(1/60)
    end
end