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

using CUDA, Random123,  FLoops, FoldsCUDA, StructArrays





function mh_gpu(
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

    x = CuArray{T}(undef, N)
    xₜ = x₀::T
    σ = one(T)

    rng = MersenneTwister(seed)

    TARGET = 0.3
    σ, xₜ = __burn_loop(xₜ, σ, burn_N, TARGET, f, rng)

    N_THR = 1024

    ranges = collect(Iterators.partition(1:N, (N + N_THR) ÷ N_THR))

    d_x = CuArray(x)
    d_ranges= CuArray(hcat(map(x-> [ x.start x.stop ], ranges )...))

    function my_pdf(mean, std, x)
        return convert(Float32, 0.3989481634448608f0 / std * exp(-0.5 * (((x - mean) / std)^2)));
    end
    
    function d_ff(x)
        x /= 100.0
        my_pdf(0.0, 1.0, x) + my_pdf(3.0, 1.0, x) + my_pdf(6.0, 1.0, x)
    end

    
    function d__generate_loop(x, xₜ, σ, ranges)
        id = threadIdx().x
        #rng = Philox2x(0)
        range = ranges[id*2+1]:ranges[id*2+2]
        #f_xₜ = d_ff(xₜ)
        for t in 1:100 
           
            #xc = rand(rng, Float32)
            xc = 0.0f0
            f_xc = d_ff(xc)
            xc = 1000f0
            α = f_xc / f_xₜ
            #u = rand(rng, Float32)
            u = 0.5f0
            if (u <= α)
                xₜ = xc
                f_xₜ = f_xc
            end
            x[t] = xₜ
        end
        return nothing
    end


   

    function d__generate_loop2(x, xₜ, σ, ranges,rng)
        
        id = threadIdx().x
        r =  rng[id]
        

        range = ranges[2id-1]:ranges[2id]
        f_xₜ = d_ff(xₜ)
        for t in range
            ll = -2.0f0 * CUDA.log( rand(r, Float32))
            cc = CUDA.cos(2.0f0*π* rand(r, Float32))
            n = CUDA.sqrt(ll )*cc
            
            xc = xₜ  +  σ  *n
            f_xc = d_ff(xc)
            α = f_xc / f_xₜ
            u = rand(r, Float32)
            if ( u <= α)
                xₜ = xc
                f_xₜ = f_xc
            end
            x[t] = xₜ
         end
        return nothing
    end

    a = StructArray([Philox2x(UInt32) for _ in 1:N_THR])
    b = replace_storage(CuArray, a)

    CUDA.@sync begin
    @cuda threads=N_THR d__generate_loop2(d_x, xₜ, σ, d_ranges, b)
    end
    
    return x
end

export ff, mh_serial_naive, mh_serial_optimized, mh_threaded_naive, mh_threaded_optimized , mh_gpu

end
