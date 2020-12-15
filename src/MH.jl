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
    σ = one(T)
   
    @inline function _gen_candidate(xₜ::Y, σ::Y, f::Function) where Y <: AbstractFloat
        g = Normal(xₜ, σ)
        xc = rand(g)
        f_xc = f(xc)
        α = f_xc / f(xₜ)
        xc, α
    end

    target = 0.3
    accepted = 0
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        xc, α = _gen_candidate(xₜ, σ, f)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = xc
            accepted += 1
        end
        σ += convert(T, 1 / σ * 1000( (accepted / i) - target))
    end

    for t = 1:N
        xc, α = _gen_candidate(xₜ, σ, f)
        u = rand(Uniform(0, 1))
        if (u <= α)
            xₜ = xc
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
    σ = one(T)
   
    rng = MersenneTwister(seed)

    @inline function _gen_candidate(xₜ::Y, σ::Y, f_xₜ, f::Function, rng::RNG) where {Y <: AbstractFloat,RNG <: AbstractRNG}
        g = Normal(xₜ, σ)
        xc = rand(rng, g)
        f_xc = f(xc)
        α = f_xc / f_xₜ
        xc, α, f_xc
    end

    target = 0.3
    accepted = 0
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = xc
            f_xₜ = f_xc
            accepted += 1
        end
        σ += convert(T, 1 / σ * 1000( (accepted / i) - target))
    end

    for t = 1:N
        xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = xc
            f_xₜ = f_xc
        end
        x[t] = xₜ
    end

    return x
end

function mh_threaded_naive(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, seed
) where {T <: AbstractFloat}

    
    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ = one(T)

    rng = MersenneTwister(seed)

    @inline function _gen_candidate(xₜ::Y, σ::Y, f_xₜ, f::Function, rng::RNG) where {Y <: AbstractFloat,RNG <: AbstractRNG}
        g = Normal(xₜ, σ)
        xc = rand(rng, g)
        f_xc = f(xc)
        α = f_xc / f_xₜ
        xc, α, f_xc
    end

    target = 0.3
    accepted = 0
    f_xₜ = f(xₜ)
    for i = 1:burn_N
        xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = xc
            f_xₜ = f_xc
            accepted += 1
        end
        σ += convert(T, 1 / σ * 1000( (accepted / i) - target))
    end

    Threads.@threads for t = 1:N
        xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
        u = rand(rng, Uniform(0, 1))
        if (u <= α)
            xₜ = xc
            f_xₜ = f_xc
        end
        x[t] = xₜ
    end

    return x
end

 



function mh_threaded_optimized(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, 
    seed::Integer
) where {T <: AbstractFloat}


    @inline function _gen_candidate(xₜ::T, σ::T, f_xₜ, f, rng) where T <: AbstractFloat
        g = Normal(xₜ, σ)
        xc = rand(rng, g)
        f_xc = f(xc)
        α = f_xc / f_xₜ
        xc, α, f_xc
    end

    function __generate_loop(x::Vector{T}, xₜ::T, σ::T, range::R, f, rng::RNG) where {T <: AbstractFloat,R <: AbstractRange,RNG <: AbstractRNG} 
        f_xₜ = f(xₜ)
        for t in range
            xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
            u = rand(rng, Uniform(0, 1))
            if (u <= α)
                xₜ = xc
                f_xₜ = f_xc
            end
            @inbounds x[t] = xₜ
        end
    end

    function __burn_loop(xₜ::T, σ::T, burn_N::Integer, target, f, rng::RNG) where  {T <: AbstractFloat,RNG <: AbstractRNG} 
        accepted = 0
        f_xₜ = f(xₜ)
        for i = 1:burn_N
            xc, α, f_xc = _gen_candidate(xₜ, σ, f_xₜ, f, rng)
            u = rand(rng, Uniform(0, 1))
            if (u <= α)
                xₜ = xc
                f_xₜ = f_xc
                accepted += 1
            end
            σ += convert(T, 1 / σ * 1000( (accepted / i) - target))
        end
        σ, xₜ
    end

    x = Vector{T}(undef, N)
    xₜ = x₀::T
    σ = one(T)

    rng = MersenneTwister(seed)

    TARGET = 0.3
    σ, xₜ = __burn_loop(xₜ, σ, burn_N, TARGET, f, rng)

    N_THR = Threads.nthreads()

    ranges = collect(Iterators.partition(1:N, (N + N_THR) ÷ N_THR))

    _task = (id) -> Threads.@spawn __generate_loop(x, $xₜ, $σ, $ranges[id], f,  MersenneTwister(rng.seed[1] * id))
    wait.(map(id -> _task(id), 1:N_THR))

    return x
end

export ff, mh_serial_naive, mh_serial_optimized, mh_threaded_naive, mh_threaded_optimized 

end
