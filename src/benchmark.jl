
using MH, BenchmarkTools

burn_N = 100_000
SEED = 1111
for N in [100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    println("===================")
    println("N = $N")
    @benchmark mh_serial_naive(Float32(0.0), N, burn_N, ff) 
    @benchmark mh_serial_optimized(Float32(0.0), N, burn_N, ff, SEED) 
    
    @benchmark mh_threaded_optimized(Float32(0.0), N, burn_N, ff, SEED) 
    @benchmark mh_threaded_naive(Float32(0.0), N, burn_N, ff, SEED) 

end
