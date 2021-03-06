struct PolynomialBasis
    polynomial::AbstractVector{T} where T <: AbstractPolynomialLike
    x::AbstractVector{T} where T <: AbstractPolynomialLike
end


"""
# Notes
`with_bias=true` provides polynomials with the degree of from 0 to `d`.
`with_bias=false` provides polynomials with the degree of `d`.
# Variables
n â N: length of array, i.e., x â Râ¿
d â N: degree
"""
function PolynomialBasis(n::Int, d::Int; with_bias=true)
    @assert n >= 1 && d >= 0
    _n = with_bias ? n+1 : n
    exponents = multiexponents(_n, d)
    @polyvar x[1:n]
    _x = with_bias ? [x..., 1] : x
    polynomial = exponents |> Map(exponent -> prod(_x.^exponent)) |> collect
    PolynomialBasis(polynomial, x)
end

function (a::PolynomialBasis)(x::AbstractArray)
    a.polynomial |> Map(poly -> poly(a.x => x)) |> collect
end
