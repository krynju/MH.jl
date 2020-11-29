module MH

using Distributions

function ff(x)
    x /= 100
    pdf(Normal(0, 1), x) + pdf(Normal(3, 1), x) + pdf(Normal(6, 1), x)
end


function mh_naive(x₀::T, N::Integer, burn_N::Integer, f::Function) where {T<:AbstractFloat}

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)

    TARGET = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::T, σ²::T, f::Function)

        g = Normal(xₜ, σ²)
        x′ = rand(g)
        α = f(x′) / f(xₜ)
        x′::T, α
    end

    for i = 1:burn_N
        x′, α = _gen_candidate(xₜ, σ², ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            accepted += 1
        end
        σ² += P * (convert(T, accepted / i) - TARGET)
    end

    for t = 1:N
        x′, α = _gen_candidate(xₜ, σ², ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
        end
        x[t] = xₜ
    end

    return x
end


function mh_optimized1(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function,
) where {T<:AbstractFloat}

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)

    TARGET = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f::Function)

        g = Normal(xₜ, σ²)
        x′ = rand(g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end

    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += P * (convert(T, accepted / i) - TARGET)
    end

    for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end

    return x
end

function mh_optimized1(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function,
) where {T<:AbstractFloat}

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)

    TARGET = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f::Function)

        g = Normal(xₜ, σ²)
        x′ = rand(g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end

    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += P * (convert(T, accepted / i) - TARGET)
    end

    for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end

    return x
end

export mh_naive, ff, mh_optimized1

end
