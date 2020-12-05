module MH

using Distributions
using Random

const SEED = 1337

function ff(x::T) where T <: AbstractFloat
    x /= 100.0
    pdf(Normal(0, 1), x) + pdf(Normal(3, 1), x) + pdf(Normal(6, 1), x)
end


function mh_naive(x₀::T, N::Integer, burn_N::Integer, f::Function) where {T <: AbstractFloat}

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
) where {T <: AbstractFloat}

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)
    rng = MersenneTwister(SEED)

    TARGET = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f::Function)
        g = Normal(xₜ, σ²)
        x′ = rand(rng, g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end

    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += P * (convert(T, accepted / i) - TARGET)
    end

    for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end

    return x
end

function mh_threaded(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, seed=10
) where {T <: AbstractFloat}

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)
    rng = MersenneTwister(SEED)

    TARGET = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f::Function)
        g = Normal(xₜ, σ²)
        x′ = rand(rng, g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end


    
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += P * (convert(T, accepted / i) - TARGET)
    end

    Threads.@threads for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end


    return x
end

 

@inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f::Function, rng) where T <: AbstractFloat
    g = Normal(xₜ, σ²)
    x′ = rand(rng, g)
    f_x′ = f(x′)
    α = f_x′ / f_xₜ
    x′, α, f_x′
end

function _inner_loop(_x, range, xₜ, ff, σ², seed)
    _rng = MersenneTwister(seed)
    f_xₜ = ff(xₜ)
    for t in range
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff, _rng)
        u = rand(_rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        _x[t] = xₜ
    end
end

function _burn_loop0(xₜ::T, ff, σ²::T, rng, burn_N,  target) where T<:AbstractFloat
    P=convert(T,0.1)
    TARGET =  convert(T,0.3)
    accepted = 0
    f_xₜ = ff(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, ff, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += convert(T, 1/σ² * 1000( (accepted / i) - TARGET))
    end
    σ², xₜ, accepted/burn_N
end


function mh_threaded2(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, 
    _seed::Integer
) where {T <: AbstractFloat}
    SEED = _seed
    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
    P = convert(T, 0.1)
    rng = MersenneTwister(SEED)

    TARGET=0.3
    σ², xₜ = _burn_loop0(xₜ, ff, σ², rng, burn_N, TARGET)


    N_THR = Threads.nthreads()

    tasks = Vector{Task}(undef, N_THR)
    ranges = collect(Iterators.partition(1:N, (N + N_THR) ÷ N_THR))
    for i in 1:N_THR
        tasks[i] = Threads.@spawn _inner_loop(x, ranges[i], xₜ, ff, σ², rng.seed[1] * i)
    end
    wait.(tasks)

    return x
end

export mh_naive, ff, mh_optimized1, mh_threaded, mh_threaded2

end
