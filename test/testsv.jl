using StateVectors
using DifferentialEquations
using Quats
using Test
using BenchmarkTools
using KalmanFilterEngine

function f!(dx, x, p, t)
    dx.pos .= x.vel
    dx.vel .= -p.k .* x.pos
    dx.E = sum(abs2, x.vel)      # scalar component: plain =
    return nothing
end

x0 = StateVector{Float64}(pos = 2, vel = 2, E = 1, q=4, θ=2)   # zeros
x0.pos .= [1.0, 0.0]
x0.vel .= [0.0, 1.0]
x0.q .= q_identity()
sol = solve(ODEProblem(f!, x0, (0.0, 10.0), (k=2.0,)), Tsit5())
sol.u[end].pos

q1 = q_random()
q2 = q_random()
q_multiply!(x0.q, q1, q2)

@btime q_multiply!($x0.q, $q1, $q2)
# using Test
# mk() = StateVector{Float64}(pos = 2, vel = 2, E = 1)
# @inferred mk()

M = StateMatrix{Float64}(x0)
M[:pos, :q] .= randn.()

@inferred KalmanFilterEngine.NavStateEKF(0.0, x0, M, 6)
