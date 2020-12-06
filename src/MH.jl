module MH

using Distributions
using Random



function ff(x::T) where T <: AbstractFloat
    x /= 100.0
    pdf(Normal(0, 1), x) + pdf(Normal(3, 1), x) + pdf(Normal(6, 1), x)
end


function mh_serial_naive(x₀::T, N::Integer, burn_N::Integer, f::Function) where T <: AbstractFloat
    x = Vector{T}(undef, N)
    xₜ = x₀::T

    @inline function _gen_candidate(xₜ::Y, σ²::Y, f::Function) where Y <: AbstractFloat
        g = Normal(xₜ, σ²)
        x′ = rand(g)
        α = f(x′) / f(xₜ)
        x′, α
    end

    σ² = one(T)
    target = 0.3
    accepted = 0
    for i = 1:burn_N
        x′, α = _gen_candidate(xₜ, σ², f)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            accepted += 1
        end
        σ² += convert(T, 1 / σ² * 1000( (accepted / i) - target))
    end

    for t = 1:N
        x′, α = _gen_candidate(xₜ, σ², f)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = x′
        end
        x[t] = xₜ
    end

    return x
end


function mh_serial_optimized(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function,
    seed::Integer
) where {T <: AbstractFloat}
    # optymalizacja algorytmu poprzez zapamiętywanie poprzednich ewaluacji funkcji aproksymującej
    # optymalizacja wykonania przez jawne podanie generatora liczb losowych (globalny bywa wolniejszy)

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)
   
    rng = MersenneTwister(seed)

    @inline function _gen_candidate(xₜ::Y, σ²::Y, f_xₜ, f::Function) where Y <: AbstractFloat
        g = Normal(xₜ, σ²)
        x′ = rand(rng, g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end

    target = 0.3
    accepted = 0
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += convert(T, 1 / σ² * 1000( (accepted / i) - target))
    end

    for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end

    return x
end

function mh_threaded_naive(
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

    target = convert(T, 0.3)
    accepted = 0

    @inline function _gen_candidate(xₜ::Y, σ²::Y, f_xₜ, f::Function) where Y <: AbstractFloat
        g = Normal(xₜ, σ²)
        x′ = rand(rng, g)
        f_x′ = f(x′)
        α = f_x′ / f_xₜ
        x′, α, f_x′
    end
    
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += convert(T, 1 / σ² * 1000( (accepted / i) - target))
    end

    Threads.@threads for t = 1:N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end

    return x
end

 

@inline function _gen_candidate(xₜ::T, σ²::T, f_xₜ, f, rng) where T <: AbstractFloat
    g = Normal(xₜ, σ²)
    x′ = rand(rng, g)
    f_x′ = f(x′)
    α = f_x′ / f_xₜ
    x′, α, f_x′
end

function __generate_loop(x::Vector{T}, xₜ::T, σ²::T, range::R, f, rng::RNG) where {T <: AbstractFloat,R <: AbstractRange,RNG <: AbstractRNG} 
    f_xₜ = f(xₜ)
    for t in range
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
        end
        x[t] = xₜ
    end
end

function __burn_loop(xₜ::T, σ²::T, burn_N::Integer, target, f, rng::RNG) where  {T <: AbstractFloat,RNG <: AbstractRNG} 
    accepted = 0
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        x′, α, f_x′ = _gen_candidate(xₜ, σ², f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = x′
            f_xₜ = f_x′
            accepted += 1
        end
        σ² += convert(T, 1 / σ² * 1000( (accepted / i) - target))
    end
    σ², xₜ
end


function mh_threaded_optimized(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, 
    seed::Integer
) where {T <: AbstractFloat}
    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ² = one(T)

    rng = MersenneTwister(seed)

    TARGET = 0.3
    σ², xₜ = __burn_loop(xₜ, σ², burn_N, TARGET, f, rng)

    N_THR = Threads.nthreads()

    ranges = collect(Iterators.partition(1:N, (N + N_THR) ÷ N_THR))

    _task = (id) -> Threads.@spawn __generate_loop(x, xₜ, σ², ranges[id], f,  MersenneTwister(rng.seed[1] * id))
    wait.(map(id -> _task(id), 1:N_THR))  

    return x
end

export ff, mh_serial_naive, mh_serial_optimized, mh_threaded_naive, mh_threaded_optimized 

end
