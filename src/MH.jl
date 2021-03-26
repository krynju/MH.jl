module MH

using Distributions
using Random



function ff(x::T) where T <: AbstractFloat
    x /= 100.0
    convert(T, pdf(Normal(0, 1), x) + pdf(Normal(3, 1), x) + pdf(Normal(6, 1), x))
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

using CUDA, Random123, StructArrays






function mh_gpu(
    x₀::T,
    N::Integer,
    burn_N::Integer,
    f::Function, 
    seed::Integer
) where {T <: AbstractFloat}

    SAMPLES_PER_THREAD = 100000
    N_THR_PER_BLOCK = 1024


    @inline function _gen_candidate(xₜ::T, σ::T, f_xₜ, f, rng) where T <: AbstractFloat
        g = Normal(xₜ, σ)
        xc = rand(rng, g)
        f_xc = f(xc)
        α = f_xc / f_xₜ
        xc, α, f_xc
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

    xₜ = x₀::T
    σ = one(T)

    rng = MersenneTwister(seed)

    TARGET = 0.3
    σ, xₜ = __burn_loop(xₜ, σ, burn_N, TARGET, f, rng)


    function d_pdf(mean, std, x)
        0.3989481634448608f0 / std * exp(-0.5 * (((x - mean) / std)^2))
    end
    
    function d_ff(x)
        x /= 100.0
        d_pdf(0.0, 1.0, x) + d_pdf(3.0, 1.0, x) + d_pdf(6.0, 1.0, x)
    end

    

    function d_generate_loop(x, xₜ, σ, N, rng)
        id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        range = (1 + (id - 1) * SAMPLES_PER_THREAD):(id * SAMPLES_PER_THREAD) % (N + 1)
        if range.start <= range.stop
            r =  rng[id]
            set_counter!(r, (id - 1) * 4 * SAMPLES_PER_THREAD)
            f_xₜ = d_ff(xₜ)
            for t in range
                n = CUDA.sqrt(-2f0 * CUDA.log(rand(r, Float32))) *  CUDA.cos(2f0 * π * rand(r, Float32))
              
                xc = xₜ  +  σ  * n
                f_xc = d_ff(xc)
                α = f_xc / f_xₜ
                u = rand(r, Float32)
                if ( u <= α)
                    xₜ = xc
                    f_xₜ = f_xc
                end
                @inbounds x[t] = xₜ
            end
        end
        return nothing
    end

    _n = (_a, _b) -> (_a + _b - 1) ÷ _b

    
    N_BLOCKS = _n(N, SAMPLES_PER_THREAD * N_THR_PER_BLOCK)
    N_THR = _n(N, SAMPLES_PER_THREAD)

    d_x = CuArray{T}(undef, N)
    generators = StructArray([Philox2x(UInt32, seed + i) for i in 1:N_THR])
    d_generators = replace_storage(CuArray, generators)

    CUDA.@sync begin
        @cuda threads = N_THR_PER_BLOCK blocks = N_BLOCKS d_generate_loop(d_x, xₜ, σ, N, d_generators)
    end
    
    return Vector{T}(d_x)
end

export ff, mh_serial_naive, mh_serial_optimized, mh_threaded_naive, mh_threaded_optimized , mh_gpu

end
