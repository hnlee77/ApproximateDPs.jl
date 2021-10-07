using ApproximateDPs
const ADP = ApproximateDPs
using FlightSims
const FS = FlightSims
using Transducers
using Plots
using Random, LinearAlgebra, ComponentArrays
using DynamicPolynomials, UnPack
using DifferentialEquations, DataFrames
using NumericalIntegration: integrate


function initialise()
    # setting
    A, B = [-10 1; -0.002 -2], [0; 2]
    Q, R = I, I
    env = LinearSystem(A, B)
    __t = 0.0
    irl = CTLinearValueIterationIRL(Q, R)
    env, irl
end

function train!(env, irl; Δt=0.01, tf=3.0, w_tol=1e-3)
    stop_conds = function(w_diff_norm)
        stop_conds_dict = Dict(
                               :w_tol => w_diff_norm < w_tol,
                              )
    end
    @unpack A, B = env
    args_linearsystem = (A, B)
    linearsystem, integ = LinearSystem_SingleIntegrator(args_linearsystem)  # integrated system with scalar integrator ∫r
    x0 = State(linearsystem, integ)([0.4, 4.0])
    irl.V̂.param = zeros(size(irl.V̂.param))  # zero initialisation
    û = ADP.ApproximateOptimalInput(irl, B)
    _û = (X, p, t) -> û(X.x, p, t)  # for integrated system

    cb_train = ADP.update_params_callback(irl, tf, stop_conds)
    cb = CallbackSet(cb_train)
    running_cost = ADP.RunningCost(irl)
    prob, df = sim(
                   x0,
                   apply_inputs(Dynamics!(linearsystem, integ, running_cost); u=_û);
                   tf=tf,
                   callback=cb,
                   savestep=Δt
                  )
    ts = df.time
    xs = df.sol |> Map(datum -> datum.x) |> collect
    plot(ts, hcat(xs...)')
    # ∫rs = df.sol |> Map(X -> X.∫r) |> collect
    # plot(hcat(∫rs...)')
end

function main(; seed=1)
    Random.seed!(seed)
    env, irl = initialise()
    train!(env, irl; w_tol = 1e-3)
end
