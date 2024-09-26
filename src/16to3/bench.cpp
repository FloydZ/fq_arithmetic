#include <benchmark/benchmark.h>
#include "arith.h"

static void BM_gf16to3v_mul(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf16to3v_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

BENCHMARK(BM_gf16to3v_mul);
BENCHMARK_MAIN();

