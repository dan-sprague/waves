using GLMakie
using SparseArrays
using LinearAlgebra

function gridlaplacian(m,n)
    S = sparse(0.0I,n*m,n*m)
    linear = LinearIndices((1:m,1:n))

    for i ∈ 1:m
        for j ∈ 1:n
            for (i2,j2) ∈ ((i+1,j), (i,j+1))
                if i2 <= m && j2 <= n
                    S[linear[i,j],linear[i2,j2]] -= 1
                    S[linear[i2,j2],linear[i,j]] -= 1
                    S[linear[i,j],linear[i,j]] +=1
                    S[linear[i2,j2],linear[i2,j2]] +=1
                end 
            end 
        end 
    end
    return S
end

d = 120
F = cholesky(gridlaplacian(d,d) + 0.1I) 

z_vector = randn(d*d)
data_observable = Observable(zeros(d, d))
current_field = zeros(d, d)
fig = Figure(size = (1000, 800), backgroundcolor = :black)

ax = Axis3(fig[1, 1], 
    aspect = (1, 1, 0.4), # Increased z-aspect slightly so waves are more visible
    azimuth = 1.2,
    elevation = 0.4,
    backgroundcolor = :black
)
hidedecorations!(ax)
hidespines!(ax)

wireframe!(ax, 1:d, 1:d, data_observable; 
    linewidth = 0.1,       # Keep lines thin for dense 200x200 grids
    color = :grey,         # High contrast against black background
    transparency = true    # Helps with visual density
)

display(fig)

# 4. Simulation Parameters
α = 0.8             
β = sqrt(1 - α^2)      
t = 0.0                
speed = 0.05           
freq = 0.075           
amp_wave = 1.75       
amp_noise = 0.5   
temporal_smoothness = 0.1
x_grid = repeat(1:d, 1, d) 

# 5. Loop
@async begin
    while isopen(fig.scene)
        global z_vector = α .* z_vector .+ β .* randn(d*d)
        
        target_noise_field = reshape(F \ z_vector, d, d)

        global t += speed
        rolling_wave = @. amp_wave * sin(freq * x_grid - t)
        target_state = rolling_wave .+ (amp_noise .* target_noise_field)
        current_field .= (1 - temporal_smoothness) .* current_field .+ temporal_smoothness .* target_state

        data_observable[] = 0.15 .* current_field

        sleep(1/60) 
    end
end