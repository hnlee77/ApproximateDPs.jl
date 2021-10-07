module ApproximateDPs

import FSimZoo: SingleIntegrator
using FSimZoo
using FSimBase # included in FSimZoo?

using DifferentialEquations
using ComponentArrays, UnPack
using JLD2, FileIO, DataFrames
using DynamicPolynomials: @polyvar, PolyVar, AbstractPolynomialLike
using Combinatorics: multiexponents
using Transducers: Map
using LinearAlgebra
using NLopt

## algorithms
# adp
export CTValueIterationADP
# irl
export CTLinearValueIterationIRL
## utils
export AbstractApproximator, LinearApproximator
export PolynomialBasis
## environments
# integrated environments
export LinearSystem_SingleIntegrator

include("utils/utils.jl")
include("algorithms/algorithms.jl")
include("environments/environments.jl")

end
