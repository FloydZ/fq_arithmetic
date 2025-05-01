// run with
//  nvcc -O3 clock.cu -o clock && optirun ./clock
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <math.h>
#include "../src/repeat.h"

#define NOMAIN

#include "../src/arith.cuh"

__constant__ const uint32_t __p32[8] = { 0xFFFFFFFD, 0xAC0E7A06, 0x00000012, 0xffffffff, 0xFFFFFFFD, 0xAC0E7A06, 0x00000012, 0xffffffff };
__constant__ const uint32_t __mu32[8] = { 0x1, 0xeeeeee, 0xadddddd, 0x121212, 0x1, 0xeeeeee, 0xadddddd, 0x121212 };

// NOT CORRECT
__constant__ uint64_t __p64[4] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF7BC6BFFF, 0x8AF43417A1087212, 0x01248A1B };
__constant__ uint64_t __mu64[4] = { 0x0000000000000001, 0x000000007BC6C000, 0x8AF43417A1087212, 0x91248A1B };

__device__ __forceinline__ void fp_add96_32(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_add96_32(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2]);
}
__device__ __forceinline__ void fp_add128_32(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_add128_32(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3], __p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_add160_32(uint32_t out[5], const uint32_t arg1[5], const uint32_t arg2[5]) {
    __fp_add160_32(out[0], out[1], out[2], out[3], out[4], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4]);
}
__device__ __forceinline__ void fp_add192_32(uint32_t out[6], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __fp_add192_32(out[0], out[1], out[2], out[3], out[4], out[5], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5]);
}
__device__ __forceinline__ void fp_add224_32(uint32_t out[7], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __fp_add224_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5], __p32[6]);
}
__device__ __forceinline__ void fp_add256_32(uint32_t out[8], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __fp_add256_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], arg2[7], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5], __p32[6], __p32[7]);
}

__device__ __forceinline__ void fp_sub96_32(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_sub96_32(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2]);
}
__device__ __forceinline__ void fp_sub128_32(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_sub128_32(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3], __p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_sub160_32(uint32_t out[5], const uint32_t arg1[5], const uint32_t arg2[5]) {
    __fp_sub160_32(out[0], out[1], out[2], out[3], out[4], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4]);
}
__device__ __forceinline__ void fp_sub192_32(uint32_t out[6], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __fp_sub192_32(out[0], out[1], out[2], out[3], out[4], out[5], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5]);
}
__device__ __forceinline__ void fp_sub224_32(uint32_t out[7], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __fp_sub224_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5], __p32[6]);
}
__device__ __forceinline__ void fp_sub256_32(uint32_t out[8], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __fp_sub256_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], arg2[7], __p32[0], __p32[1], __p32[2], __p32[3], __p32[4], __p32[5], __p32[6], __p32[7]);
}


__device__ __forceinline__ void fp_add192_64(uint64_t out[3], const uint64_t arg1[3], const uint64_t arg2[3]) {
    __fp_add192_64(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p64[0], __p64[1], __p64[2]);
}
__device__ __forceinline__ void fp_sub192_64(uint64_t out[3], const uint64_t arg1[3], const uint64_t arg2[3]) {
    __fp_sub192_64(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p64[0], __p64[1], __p64[2]);
}
__device__ __forceinline__ void fp_add256_64(uint64_t out[4], const uint64_t arg1[4], const uint64_t arg2[4]) {
    __fp_add256_64(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg1[2], arg1[3],arg2[0], arg2[1], arg2[2], arg2[3], __p64[0], __p64[1], __p64[2], __p64[3]);
}
__device__ __forceinline__ void fp_sub256_64(uint64_t out[4], const uint64_t arg1[4], const uint64_t arg2[4]) {
    __fp_sub256_64(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3], __p64[0], __p64[1], __p64[2], __p64[3]);
}

__device__ __forceinline__ void fp_karatsuba1x1_32(uint32_t out[2], const uint32_t arg1[1], const uint32_t arg2[1]) {
    __karatsuba32_1x1(out[0], out[1], arg1[0], arg2[0]);
}
__device__ __forceinline__ void fp_karatsuba1x2_32(uint32_t out[3], const uint32_t arg1[1], const uint32_t arg2[2]) {
    __karatsuba32_1x2(out[0], out[1], out[2], arg1[0], arg2[0], arg2[1]);
}
__device__ __forceinline__ void fp_karatsuba1x2_32_v2(uint32_t out[3], const uint32_t arg1[1], const uint32_t arg2[2]) {
    __karatsuba32_1x2_v2(out[0], out[1], out[2], arg1[0], arg2[0], arg2[1]);
}
__device__ __forceinline__ void fp_karatsuba1x3_32(uint32_t out[4], const uint32_t arg1[1], const uint32_t arg2[3]) {
    __karatsuba32_1x3(out[0], out[1], out[2], out[3], arg1[0], arg2[0], arg2[1], arg2[2]);
}
__device__ __forceinline__ void fp_karatsuba1x4_32(uint32_t out[5], const uint32_t arg1[1], const uint32_t arg2[4]) {
    __karatsuba32_1x4(out[0], out[1], out[2], out[3], out[4], arg1[0], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void fp_karatsuba1x5_32(uint32_t out[6], const uint32_t arg1[1], const uint32_t arg2[5]) {
    __karatsuba32_1x5(out[0], out[1], out[2], out[3], out[4], out[5], arg1[0], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4]);
}
__device__ __forceinline__ void fp_karatsuba1x6_32(uint32_t out[7], const uint32_t arg1[1], const uint32_t arg2[6]) {
    __karatsuba32_1x6(out[0], out[1], out[2], out[3], out[4], out[5], out[6], arg1[0], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5]);
}
__device__ __forceinline__ void fp_karatsuba1x7_32(uint32_t out[8], const uint32_t arg1[1], const uint32_t arg2[7]) {
    __karatsuba32_1x7(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], arg1[0], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6]);
}


__device__ __forceinline__ void fp_karatsuba2x2_64(uint64_t out[2], const uint64_t arg1[2], const uint64_t arg2[2]) {
    __fp_mul64_2x2(out[0], out[1], arg1[0], arg1[1], arg2[0], arg2[1], __p64[0], __p64[1], __mu64[0], __mu64[1]);
}
__device__ __forceinline__ void fp_karatsuba3x3_64(uint64_t out[3], const uint64_t arg1[3], const uint64_t arg2[3]) {
    __fp_mul3x3_64(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p64[0], __p64[1], __p64[2], __mu64[0], __mu64[1], __mu64[2]);
}
__device__ __forceinline__ void fp_karatsuba3x3_32(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_school3x3_32(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_school3x3_32_v2(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32_v2(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_school3x3_32_v3(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32_v4(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_school3x3_32_v4(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32_v4(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_school3x3_32_v5(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __fp_mul3x3_32_v5(out[0], out[1], out[2], arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2], __p32[0], __p32[1], __p32[2], __mu32[0], __mu32[1], __mu32[2]);
}
__device__ __forceinline__ void fp_karatsuba4x4_64(uint64_t out[4], const uint64_t arg1[4], const uint64_t arg2[4]) {
    __fp_mul4x4_64(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu64[0], __mu64[1], __mu64[2], __mu64[3], __p64[0], __p64[1], __p64[2], __p64[3]);
}
__device__ __forceinline__ void fp_karatsuba4x4_32(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], __p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school4x4_32(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], 
			__p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school4x4_32_v2(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32_v2(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], 
			__p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school4x4_32_v3(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32_v3(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], 
			__p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school4x4_32_v4(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32_v4(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], 
			__p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school4x4_32_v7(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_mul4x4_32_v7(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], 
			__p32[0], __p32[1], __p32[2], __p32[3]);
}
__device__ __forceinline__ void fp_school5x5_32(uint32_t out[10], const uint32_t arg1[5], const uint32_t arg2[5]) {
    __fp_mul5x5_32(out[0], out[1], out[2], out[3], out[4],
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4],
			arg2[0], arg2[1], arg2[3], arg2[3], arg2[4],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4]);
}
__device__ __forceinline__ void fp_karatsuba6x6_32(uint32_t out[12], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __fp_mul6x6_32(out[0], out[1], out[2], out[3], out[4], out[5], 
		arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5],  arg2[0], 
		arg2[1], arg2[3], arg2[3], arg2[4], arg2[5],
		__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],
		__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5]);
}
__device__ __forceinline__ void fp_school6x6_32(uint32_t out[12], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __fp_mul6x6_32_v2(out[0], out[1], out[2], out[3], out[4], out[5], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5],  
			arg2[0], arg2[1], arg2[3], arg2[3], arg2[4], arg2[5],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5]);
}
__device__ __forceinline__ void fp_karatsuba7x7_32(uint32_t out[17], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __fp_mul7x7_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], 
			arg2[0], arg2[1], arg2[3], arg2[3], arg2[4], arg2[5], arg2[6],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],__mu32[6],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5],__p32[6]);
}
__device__ __forceinline__ void fp_school7x7_32(uint32_t out[14], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __fp_mul7x7_32_v2(out[0], out[1], out[2], out[3], out[4], out[5], out[6], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6],
			arg2[0], arg2[1], arg2[3], arg2[3], arg2[4], arg2[5], arg2[6],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],__mu32[6],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5],__p32[6]);
}
__device__ __forceinline__ void fp_karatsuba8x8_32(uint32_t out[16], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __fp_mul8x8_32(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg2[6], arg2[7],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],__mu32[6],__mu32[7],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5],__p32[6],__p32[7]);
}
__device__ __forceinline__ void fp_school8x8_32(uint32_t out[16], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __fp_mul8x8_32_v2(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg2[6], arg2[7],
			__mu32[0],__mu32[1],__mu32[2],__mu32[3],__mu32[4],__mu32[5],__mu32[6],__mu32[7],
			__p32[0],__p32[1],__p32[2],__p32[3],__p32[4],__p32[5],__p32[6],__p32[7]);
}

__device__ __forceinline__ void fp_sqr4x4_32(uint32_t out[4], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __fp_sqr4x4_32(out[0], out[1], out[2], out[3], 
			arg1[0], arg1[1], arg1[2], arg1[3],
			__mu32[0], __mu32[1], __mu32[2], __mu32[3], __p32[0], __p32[1], __p32[2], __p32[3]);
}

__device__ __forceinline__ void karatsuba_mul2x2_32(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __karatsuba32_2x2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void karatsuba_mul2x2_32_v2(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __karatsuba32_2x2_v2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_32(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __school32_2x2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_32_v2(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __school32_2x2_v2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_32_v3(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __school32_2x2_v3(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_32_v6(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __school32_2x2_v6(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_32_v7(uint32_t out[4], const uint32_t arg1[2], const uint32_t arg2[2]) {
    __school32_2x2_v7(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1], out[0], out[1], out[2], out[3]);
}
__device__ __forceinline__ void karatsuba_mul3x3_32(uint32_t out[6], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __karatsuba32_3x3(out[0], out[1], out[2], out[3], out[4], out[5], 
			arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2]);
}
__device__ __forceinline__ void school_mul3x3_32(uint32_t out[6], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __school32_3x3(out[0], out[1], out[2], out[3], out[4], out[5],
			arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2]);
}
__device__ __forceinline__ void school_mul3x3_32_v2(uint32_t out[6], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __school32_3x3_v2(out[0], out[1], out[2], out[3], out[4], out[5],
			arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2]);
}
__device__ __forceinline__ void karatsuba_mul4x4_32(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __karatsuba32_4x4(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void school_mul4x4_32(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __school32_4x4(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void school_mul4x4_32_v2(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __school32_4x4_v2(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void school_mul4x4_32_v3(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __school32_4x4_v3(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void school_mul4x4_32_v7(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __school32_4x4_v7(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3]);
}
__device__ __forceinline__ void school_mul5x5_32(uint32_t out[10], const uint32_t arg1[5], const uint32_t arg2[5]) {
    __school32_5x5(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9],
			arg1[0], arg1[1], arg1[2], arg1[3], arg2[4], arg2[0], arg2[1], arg2[2], arg2[3], arg2[4]);
}
__device__ __forceinline__ void karatsuba_mul6x6_32(uint32_t out[12], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __karatsuba32_6x6(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5]);
}
__device__ __forceinline__ void school_mul6x6_32(uint32_t out[12], const uint32_t arg1[6], const uint32_t arg2[6]) {
    __school32_6x6(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11],
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5]);
}
__device__ __forceinline__ void karatsuba_mul7x7_32(uint32_t out[14], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __karatsuba32_7x7(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13],
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6]);
}
__device__ __forceinline__ void school_mul7x7_32(uint32_t out[14], const uint32_t arg1[7], const uint32_t arg2[7]) {
    __school32_7x7(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6]);
}
__device__ __forceinline__ void karatsuba_mul8x8_32(uint32_t out[16], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __karatsuba32_8x8(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], arg2[7]);
}
__device__ __forceinline__ void school_mul8x8_32(uint32_t out[16], const uint32_t arg1[8], const uint32_t arg2[8]) {
    __school32_8x8(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15], 
			arg1[0], arg1[1], arg1[2], arg1[3], arg1[4], arg1[5], arg1[6], arg1[7], 
			arg2[0], arg2[1], arg2[2], arg2[3], arg2[4], arg2[5], arg2[6], arg2[7]);
}

__device__ __forceinline__ void school_sqr4x4_32(uint32_t out[8], const uint32_t arg1[4], const uint32_t arg2[4]) {
    __school_sqr32_4x4(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			arg1[0], arg1[1], arg1[2], arg1[3]);
}


// low
__device__ __forceinline__ void school_mul3x3_low_32(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __school32_3x3_low(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2]);
}
__device__ __forceinline__ void school_mul3x3_low_32_v2(uint32_t out[3], const uint32_t arg1[3], const uint32_t arg2[3]) {
    __school32_3x3_low_v2(out[0], out[1], out[2],
			arg1[0], arg1[1], arg1[2], arg2[0], arg2[1], arg2[2]);
}


// 64BIT
__device__ __forceinline__ void school_mul1x1_64_v2(uint64_t out[2], const uint64_t arg1[1], const uint64_t arg2[1]) {
    __school64_1x1_v2(out[0], out[1], arg1[0], arg2[0]);
}
__device__ __forceinline__ void school_mul1x1_64_v3(uint64_t out[2], const uint64_t arg1[1], const uint64_t arg2[1]) {
    __school64_1x1_v3(out[0], out[1], arg1[0], arg2[0]);
}

__device__ __forceinline__ void school_mul2x2_64(uint64_t out[4], const uint64_t arg1[2], const uint64_t arg2[2]) {
    __school64_2x2_v2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}
__device__ __forceinline__ void school_mul2x2_64_v2(uint64_t out[4], const uint64_t arg1[2], const uint64_t arg2[2]) {
    __school64_2x2_v2(out[0], out[1], out[2], out[3], arg1[0], arg1[1], arg2[0], arg2[1]);
}




__global__ void kclock(uint32_t *ts) {
	uint64_t start_time = 0, stop_time = 0;

	start_time = clock();
	// Measure something here
	stop_time = clock();
	
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time;
}


#define ATOMICADD32(a, b) 	atomicAdd((uint32_t *)&a, b)
#define ADD32(a, b) 		asm volatile("{ add.u32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define ADDC32(a, b) 		asm volatile("{ addc.u32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define ADDCCC32(a, b) 		asm volatile("{ addc.cc.u32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define SUB32(a, b)    	 	asm volatile("{ sub.u32 %0, %1, %2; }" : "=r"(b) : "r"(b), "r"(a))		
#define MUL32(a, b)    	 	asm volatile("{ mul.lo.u32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define MULWIDE(a,b,c) 		asm volatile("{ mul.wide.u32 %0, %1, %2; }" : "=l"(a) : "r"(b), "r"(c))		
#define MAD32(a, b)    	 	asm volatile("{ mad.lo.u32 %0, %1, %2, %0; }" : "=r"(a) : "r"(b), "r"(a))		
#define MADCCC32(a, b) 		asm volatile("{ madc.lo.cc.u32 %0, %1, %2, %0; }" : "=r"(a) : "r"(b), "r"(a))		
#define MULADDCCC32(a, b)	asm volatile("{ mul.lo.u32 %0, %1, %2; addc.cc.u32 %0, %0, %0; }" : "=r"(a) : "r"(b), "r"(a))

// src: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
// b must be a valid mem addr
#define LOADCONST32(a, b)   asm volatile("{ ld.const.u32  %0, [%1]; }" : "=r"(a) : "l"(b))
#define LOADGLOBAL32(a, b)  asm volatile("{ ld.global.u32 %0, [%1]; }" : "=r"(a) : "l"(b))
#define LOADLOCAL32(a, b)   asm volatile("{ ld.local.u32  %0, [%1]; }" : "=r"(a) : "l"(b))
#define LOADPARAM32(a, b)   asm volatile("{ ld.param.u32  %0, [%1]; }" : "=r"(a) : "l"(b))
#define LOADSHARED32(a, b)  asm volatile("{ ld.shared.u32 %0, [%1]; }" : "=r"(a) : "l"(b))

#define MOV32(a, b) 		asm volatile("{ mov.u32 %0, %1; }" : "=r"(a) : "r"(b))		
#define AND32(a, b) 		asm volatile("{ and.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		

#define OR32(a, b) 			asm volatile("{ or.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define XOR32(a, b) 		asm volatile("{ xor.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define NOT32(a, b) 		asm volatile("{ not.b32 %0, %1; }" : "=r"(a) : "r"(b))		
#define CNOT32(a, b) 		asm volatile("{ cnot.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
//#define LOP332(a, b) 		asm volatile("{ lop3.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define SHFLWRAP32(a, b) 	asm volatile("{ shf.l.wrap.b32 %0, %1, %2, %3; }" : "=r"(a) : "r"(b), "r"(a), "r"(b))		
#define SHFLCLAMP32(a, b) 	asm volatile("{ shf.l.clamp.b32 %0, %1, %2, %4; }" : "=r"(a) : "r"(b), "r"(a), "r"(b))		
#define SHL32(a, b) 		asm volatile("{ shl.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define SHR32(a, b) 		asm volatile("{ shr.b32 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		

#define ADD64(a, b) 		asm volatile("{ add.u64 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define ADDC64(a, b) 		asm volatile("{ addc.u64 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define ADDCCC64(a, b) 		asm volatile("{ addc.cc.u64 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define SUB64(a, b)    	 	asm volatile("{ sub.u64 %0, %1, %2; }" : "=r"(b) : "r"(b), "r"(a))		
#define MUL64(a, b)    	 	asm volatile("{ mul.lo.u64 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		
#define MAD64(a, b)    	 	asm volatile("{ mad.lo.u64 %0, %1, %2, %0; }" : "=r"(a) : "r"(b), "r"(a))		
#define MADCCC64(a, b) 		asm volatile("{ madc.lo.cc.u64 %0, %1, %2, %0; }" : "=r"(a) : "r"(b), "r"(a))		
#define MULADDCCC64(a, b)	asm volatile("{ mul.lo.u64 %0, %1, %2; addc.cc.u64 %0, %0, %0; }" : "=r"(a) : "r"(b), "r"(a))		
#define MOV64(a, b) 		asm volatile("{ mov.u64 %0, %1; }" : "=r"(a) : "r"(b))		
#define AND64(a, b) 		asm volatile("{ and.b64 %0, %1, %2; }" : "=r"(a) : "r"(b), "r"(a))		

/// TODO describe
#define INSTRUCTION_BENCH(OP, MULT) 																	\
__global__ 																								\
void kernel_instr_clock_##OP(uint32_t *ts, uint32_t *out, uint32_t p1, uint32_t p2, uint32_t its) {  	\
	uint32_t start_time = 0, stop_time = 0; 															\
	uint32_t tmp = p2; 																					\
	for (size_t i = 0; i < its; i++) { 																	\
		__syncthreads(); 																				\
		start_time = clock(); 																			\
		repeat##MULT(OP(tmp, p1); OP(p1, tmp);) 														\
		stop_time = clock(); 																			\
	} 																									\
	out[0] = tmp; 																						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time; 											\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time; 											\
}

#define INSTRUCTION_MEM_BENCH(OP, MULT) 																\
__global__ 																								\
void kernel_instr_clock_##OP(uint32_t *ts, uint32_t *out, uint32_t p1, uint32_t p2, uint32_t its) {  	\
	uint32_t start_time = 0, stop_time = 0; 															\
	uint32_t tmp1 = p1, tmp2 = p2; 																		\
	for (size_t i = 0; i < its; i++) { 																	\
		__syncthreads(); 																				\
		start_time = clock(); 																			\
		repeat##MULT(OP(out[0], tmp2); OP(out[0], tmp1);) 													\
		stop_time = clock(); 																			\
	} 																									\
	out[0] += tmp1 + tmp2; 																				\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time; 											\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time; 											\
}
// TODO describe
#define INSTRUCTION_BENCH_WIDE(OP, MULT)																\
__global__ 																								\
void kernel_instr_clock_##OP(uint32_t *ts, uint32_t *out, uint32_t p1, uint32_t p2, uint32_t its) {  	\
	uint32_t start_time = 0, stop_time = 0; 															\
	uint32_t tmp = p2; uint64_t tmp2 = p1;																\
	for (size_t i = 0; i < its; i++) { 																	\
		__syncthreads(); 																				\
		start_time = clock(); 																			\
		repeat##MULT(OP(tmp2, uint32_t(tmp2), p1); OP(tmp2, tmp, uint32_t(tmp2));) 						\
		stop_time = clock(); 																			\
	} 																									\
	out[0] = tmp2 ^ (tmp2 >> 32u); 																		\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time; 											\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time; 											\
} 											

// TODO descrine
#define FUNCTION_BENCH_IMPL(OP, MULT, SIZE, T)												\
__global__ void kernel_function_clock_##OP(uint32_t *ts, T *out, T p1, T p2, uint32_t its){	\
	T t1[SIZE] = {0};																		\
	T t2[SIZE] = {0};																		\
	T t3[SIZE] = {0};																		\
	for (uint32_t i = 0; i < SIZE; i++) {													\
		t1[i] = p1; t2[i] = p2;	t3[i] = p1+p2;												\
	}																						\
	uint32_t start_time = 0, stop_time = 0; 												\
	for (uint32_t i = 0; i < its; i++) {													\
		__syncthreads();																	\
		start_time = clock();																\
		repeat##MULT(OP(t3, t1, t2); OP(t1, t2, t3); OP(t2, t3, t1);) 						\
		stop_time = clock();																\
	}																						\
	for (uint32_t i = 0; i < SIZE; i++) {													\
		out[i] = t1[i] + t2[i];																\
	}																						\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;								\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time;								\
}


#define FUNCTION_BENCH(OP, MULT, SIZE)   FUNCTION_BENCH_IMPL(OP,MULT,SIZE,uint32_t)
#define FUNCTION_BENCH64(OP, MULT, SIZE) FUNCTION_BENCH_IMPL(OP,MULT,SIZE,uint64_t)


// FUNC: kernel to call
// NUM:  number of threads in each Block
#define MEASURE_THROUGHPUT(FUNC, NUM, ITERS)											\
do {																					\
	Db.x = NUM; 																		\
																						\
	printf (" %s \tthroughput:\t", #FUNC);												\
	FUNC<<<Dg, Db>>>(d_ts, d_out, rand(), rand(), ITERS);								\
	cudaDeviceSynchronize();															\
	cudaError_t error;																	\
	if ((error = cudaGetLastError()) != cudaSuccess) {									\
		printf("  failed. %s\n\n", cudaGetErrorString(error));							\
		break;																			\
	}																					\
	cudaDeviceSynchronize();															\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);							\
	uint32_t min_t=uint32_t(-1), max_t=0;												\
	for (int i=0;i< Db.x*2;i++) {														\
		min_t = min(min_t, ts[i]);														\
		max_t = max(max_t, ts[i]);														\
	}																					\
																						\
	const double ops_clk = 	(Db.x*ITERS)/(double)(max_t-min_t); 						\
	printf("%9u clk (%.3f ops/clk, %.3f clk/ops), ", max_t-min_t, ops_clk, 1./ops_clk);	\
	printf("threads: %d, #ops: %d\n", NUM, ITERS);										\
} while (0)	

/// TODO preheating flag
/// Needs the following variables locally defined"
/// 	d_out, d_ts
/// FUNC: 		kernel to run
/// ITERS: 		nr of tries to averae over (32)
/// LTHREADS: 	number of threads to start (1)
/// HTHREADS: 	number of threads to end   (8192)
/// BITS:		number of bits in the argument of the kernel
#define PRINT_HISTOGRAM_IMPL(FUNC, ITERS, LTHREADS, HTHREADS, BITS)																\
do {																															\
	double tries = double(ITERS); 	 																							\
	printf ("\nPipeline latency/throughput with multiple warps (%d iterations of %d ops)\n", tries, ITERS);						\
	printf ("  %s:\n", #FUNC);																									\
	cudaError_t code;  																											\
	uint32_t threads = LTHREADS, blocks = 1, tpb = 1;	  																		\
	for (; threads <= HTHREADS; threads += (threads < 4) ? 1 : (threads < 8) ? 2 : 												\
			(threads < 32) ? 8 : (threads < 128) ? 32 : (threads < 1024) ? 128: 1024) { 										\
		uint32_t histogram[1024] = {0};																							\
		uint32_t sum_time = 0, max_time, min_time, sum_max_time = 0; 															\
		bool failed = false;																									\
		blocks = (threads + 1023)/1024;  																						\
		tpb = threads >= 1024 ? 1024 : threads; 																				\
		for (uint32_t i=0; i < uint32_t(tries) && !failed; i++) {																\
			cudaGetLastError();		/* Clear previous error code, if any */														\
			FUNC <<<blocks, tpb>>>(d_ts, d_out##BITS, rand(), rand(), iterations);												\
			code = cudaGetLastError();  																						\
			if (code != cudaSuccess) { 																							\
				failed = true;																									\
				break;																											\
			}																													\
																																\
			cudaDeviceSynchronize();																							\
			cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);															\
																																\
			max_time = 0;																										\
			min_time = (unsigned)-1;																							\
			/* Compute histogram.	*/																							\
			for (int j=0; j< threads*2; j+= 64) {																				\
				sum_time += (ts[j+1] - ts[j]);																					\
				max_time = max(max_time, ts[j+1]);																				\
				min_time = min(min_time, ts[j]);																				\
				histogram[(ts[j+1] - ts[j])/ITERS]++;																			\
			}																													\
			sum_max_time += max_time-min_time;																					\
		}																														\
																																\
		if (failed) {																											\
			printf ("    %3d warp%c (%2d block%c, %4d tbp, %4d thread%c)  failed: %s", 											\
					(threads+31)/32, threads>=64 ? 's' : ' ', blocks, blocks > 1 ? 's' : ' ', tpb,    							\
					threads, threads > 1 ? 's' : ' ', cudaGetErrorString(code));												\
		} else { 																												\
			/* Compute average latency over the lifetime of each warp (sum_time), 												\
			   and average throughput of the kernel (sum_max_time). */ 															\
			const double clk_warp = sum_time / tries / double(ITERS)/((threads+31)/32); 										\
			const double ops_clk  = ITERS*tries*threads/sum_max_time; 															\
			printf ("    %3d warp%c (%2d blk, %4d tpb, %4d thr) %9u clk (%.3f clk/warp, %.3f ops/clk, %.3f clk/ops) ", 			\
						(threads+31)/32, threads>=64 ? 's' : ' ', blocks, tpb, threads, sum_max_time,							\
						clk_warp, ops_clk, 1./ops_clk);																			\
		}																														\
																																\
		printf ("\n");																											\
	}																															\
																																\
    printf ("\n");																												\
} while(0)

//#define PRINT_HISTOGRAM(FUNC, ITERS)  	PRINT_HISTOGRAM_IMPL(FUNC, ITERS, 1, 8192, 32)
//#define PRINT_HISTOGRAM64(FUNC, ITERS)  PRINT_HISTOGRAM_IMPL(FUNC, ITERS, 1, 8192, 64)
#define PRINT_HISTOGRAM(FUNC, ITERS)  	PRINT_HISTOGRAM_IMPL(FUNC, ITERS, 1, 1024, 32)
#define PRINT_HISTOGRAM64(FUNC, ITERS)  PRINT_HISTOGRAM_IMPL(FUNC, ITERS, 1, 1024, 64)

#define MULTIPLIER 1024*2
//INSTRUCTION_MEM_BENCH(ATOMICADD32, 1024)
//INSTRUCTION_BENCH(ADD32, 1024)
//INSTRUCTION_BENCH(ADDC32, 1024)
//INSTRUCTION_BENCH(ADDCCC32, 1024)
//INSTRUCTION_BENCH(SUB32, 1024)
//INSTRUCTION_BENCH(MUL32, 1024)
//INSTRUCTION_BENCH_WIDE(MULWIDE, 1024)
//INSTRUCTION_BENCH(MAD32, 1024)
//INSTRUCTION_BENCH(MADCCC32, 1024)
//INSTRUCTION_BENCH(MULADDCCC32, 1024)
//INSTRUCTION_BENCH(MOV32, 1024)
//INSTRUCTION_BENCH(AND32, 1024)

//INSTRUCTION_BENCH(ADD64, 1024)
//INSTRUCTION_BENCH(ADDC64, 1024)
//INSTRUCTION_BENCH(ADDCCC64, 1024)
//INSTRUCTION_BENCH(SUB64, 1024)
//INSTRUCTION_BENCH(MUL64, 1024)
//INSTRUCTION_BENCH_WIDE(MULWIDE, 1024)
//INSTRUCTION_BENCH(MAD64, 1024)
//INSTRUCTION_BENCH(MADCCC64, 1024)
//INSTRUCTION_BENCH(MULADDCCC64, 1024)
//INSTRUCTION_BENCH(MOV64, 1024)
//INSTRUCTION_BENCH(AND64, 1024)

//FUNCTION_BENCH(fp_add96_32, 32, 3)
//FUNCTION_BENCH(fp_add128_32, 32, 4)
//FUNCTION_BENCH(fp_add160_32, 32, 5)
//FUNCTION_BENCH(fp_add192_32, 32, 6)
//FUNCTION_BENCH(fp_add224_32, 32, 7)
//FUNCTION_BENCH(fp_add256_32, 32, 8)

//FUNCTION_BENCH(fp_sub96_32, 32, 3)
//FUNCTION_BENCH(fp_sub128_32, 32, 4)
//FUNCTION_BENCH(fp_sub160_32, 32, 5)
//FUNCTION_BENCH(fp_sub192_32, 32, 6)
//FUNCTION_BENCH(fp_sub224_32, 32, 7)
//FUNCTION_BENCH(fp_sub256_32, 32, 8)

//FUNCTION_BENCH64(fp_add192_64, 32, 6)
//FUNCTION_BENCH64(fp_add256_64, 32, 6)
//FUNCTION_BENCH64(fp_sub192_64, 32, 6)
//FUNCTION_BENCH64(fp_sub256_64, 32, 6)


//FUNCTION_BENCH(karatsuba_mul2x2_32, 32, 4)
//FUNCTION_BENCH(karatsuba_mul2x2_32_v2, 32, 4)
//FUNCTION_BENCH(school_mul2x2_32, 32, 4)
//FUNCTION_BENCH(school_mul2x2_32_v6, 32, 4)
//FUNCTION_BENCH(school_mul2x2_32_v7, 32, 4)
//FUNCTION_BENCH(school_mul2x2_32_v2, 32, 4)
//FUNCTION_BENCH(school_mul2x2_32_v3, 32, 4)
//FUNCTION_BENCH(karatsuba_mul3x3_32, 32, 6)
//FUNCTION_BENCH(school_mul3x3_32, 32, 6)
//FUNCTION_BENCH(school_mul3x3_32_v2, 32, 6)
//FUNCTION_BENCH(karatsuba_mul4x4_32, 32, 8)
//FUNCTION_BENCH(school_sqr4x4_32, 32, 8)
//FUNCTION_BENCH(school_mul4x4_32, 32, 8)
//FUNCTION_BENCH(school_mul4x4_32_v7, 32, 8)
//FUNCTION_BENCH(school_mul4x4_32_v2, 32, 8)
//FUNCTION_BENCH(school_mul4x4_32_v3, 32, 8)
//FUNCTION_BENCH(school_mul5x5_32, 32, 10)
//FUNCTION_BENCH(karatsuba_mul6x6_32, 32, 12)
//FUNCTION_BENCH(school_mul6x6_32, 32, 12)
//FUNCTION_BENCH(karatsuba_mul7x7_32, 32, 14)
//FUNCTION_BENCH(school_mul7x7_32, 32, 14)
//FUNCTION_BENCH(karatsuba_mul8x8_32, 32, 16)
//FUNCTION_BENCH(school_mul8x8_32, 32, 16)


//FUNCTION_BENCH(school_mul3x3_low_32, 32, 3)
//FUNCTION_BENCH(school_mul3x3_low_32_v2, 32, 3)


//FUNCTION_BENCH(fp_karatsuba1x1_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x2_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x2_32_v2, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x3_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x4_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x5_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x6_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba1x7_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba2x2_32, 32, 4)
//FUNCTION_BENCH(fp_karatsuba3x3_32, 32, 3)
//FUNCTION_BENCH(fp_school3x3_32, 32, 3)
//FUNCTION_BENCH(fp_school3x3_32_v2, 32, 3)
//FUNCTION_BENCH(fp_school3x3_32_v3, 32, 3)
//FUNCTION_BENCH(fp_school3x3_32_v4, 32, 3)
//FUNCTION_BENCH(fp_school3x3_32_v5, 32, 3)
//FUNCTION_BENCH(fp_karatsuba4x4_32, 32, 4)
//FUNCTION_BENCH(fp_school4x4_32, 32, 4)
//FUNCTION_BENCH(fp_school4x4_32_v2, 32, 4)
//FUNCTION_BENCH(fp_school4x4_32_v3, 32, 4)
//FUNCTION_BENCH(fp_school4x4_32_v7, 32, 4)
//FUNCTION_BENCH(fp_sqr4x4_32, 32, 4)
//FUNCTION_BENCH(fp_school5x5_32, 32, 5)
FUNCTION_BENCH(fp_karatsuba6x6_32, 32, 6)
FUNCTION_BENCH(fp_school6x6_32, 32, 6)
//FUNCTION_BENCH(fp_karatsuba7x7_32, 32, 7)
//FUNCTION_BENCH(fp_school7x7_32, 32, 7)
//FUNCTION_BENCH(fp_karatsuba8x8_32, 32, 8)
//FUNCTION_BENCH(fp_school8x8_32, 32, 8)

//FUNCTION_BENCH64(fp_karatsuba2x2_64, 64, 2)
//FUNCTION_BENCH64(fp_karatsuba3x3_64, 64, 3)
FUNCTION_BENCH64(fp_karatsuba4x4_64, 64, 4)


//FUNCTION_BENCH64(school_mul1x1_64_v2, 64, 2)
//FUNCTION_BENCH64(school_mul1x1_64_v3, 64, 2)
//FUNCTION_BENCH64(school_mul2x2_64, 64, 4)
//FUNCTION_BENCH64(school_mul2x2_64_v2, 64, 4)


	__constant__ uint32_t constloadbenchdata[100*32] = {1};
__global__ uint32_t globalloadbenchdata[100*32] = {1};
__global__ void kernel_instr_clock_LOAD32(uint32_t *ts, uint32_t *out, uint32_t *extdata, uint32_t its){
	uint32_t data = 0, tmp;
	uint32_t start_time = 0, stop_time = 0;
	const size_t index = threadIdx.x + blockIdx.x*blockDim.x;
	const size_t blocksize = blockIdx.x*blockDim.x;

	for (uint32_t i = 0; i < its; i++) {
		__syncthreads();
		start_time = clock();
		repeat128(LOADCONST32(tmp, extdata+i*blocksize+index); ADD32(data, tmp);)
		//LOADCONST32(tmp, extdata+i+index);
		stop_time = clock();
		data += tmp;
	}

	out[0] = data;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stop_time;
}

void print_information(uint32_t *ts, const dim3 Dg, const uint32_t mult, const uint32_t empty_cycles){
	int m = 0;
	const uint32_t off = 1;
	const double MULT = mult;
	for (uint32_t i = 0; i < off; i++) {	
		for (uint32_t j = 0; j < Dg.x/off; j++, m++) {
			uint32_t start = ts[(off*j+i)*2];
			uint32_t end   = ts[(off*j+i)*2+1];
			printf("  Block %02d: start: %08u, stop: %08u, diff: %u, clock: %lf\n", off*j+i, start, end, end-start, double(end-start)/MULT);
		}
	}	
}

int main() {
	srand(time(NULL));
	uint32_t ts[1024], out32[128];
	uint32_t *d_ts,*d_ts2,*d_out32;

	uint64_t out64[128];
	uint64_t *d_out64;
	
	dim3 Db = dim3(1);
    dim3 Dg = dim3(1,1,1);
	
	Dg.x = 1; 
	Db.x = 32;

	// Allocate device array.
	cudaError_t errcode;
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts)))) {
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts2, sizeof(ts)))) {
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_out32, 4 * 128))) {
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_out64, 8 * 128))) {
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}

	// clear any error
	cudaGetLastError();
	printf ("\nkclock (preheating): ");
	for (uint64_t i = 0; i < 1000; i++) {
		kclock <<<Dg, Db>>> (d_ts);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);

	printf("\n");
	//for (int i=0; i< Dg.x*Db.x; i++)
	//	printf ("   (%8u, %8u): %u\n", ts[i*2], ts[i*2+1], ts[i*2+1]-ts[i*2]);
    //	printf ("\n");	

	// meassures the number of cycles for doing nothing
	const uint32_t empty_cycles = ts[1]-ts[0];
	const uint32_t iterations = 16;
	
//	printf ("\nkernel_instr_clock_LOADCONST32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_instr_clock_LOAD32<<<Dg, Db>>>(d_ts, d_out, constloadbenchdata, 2);
//	cudaGetLastError();
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 128, empty_cycles);
//
//	printf ("\nkernel_instr_clock_LOADGLOBAL32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_instr_clock_LOAD32<<<Dg, Db>>>(d_ts, d_out, globalloadbenchdata, 2);
//	cudaGetLastError();
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 128, empty_cycles);
//
//	printf ("\nkernel_instr_clock_ADD32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_instr_clock_ADD32<<<Dg, Db>>>(d_ts, d_out, 268324234u, 876218736u, iterations);
//	cudaGetLastError();
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 2*128, empty_cycles);

//	printf ("\nkernel_instr_clock_ATOMICADD32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_instr_clock_ATOMICADD32<<<Dg, Db>>>(d_ts, d_out, 268324234u, 876218736u, iterations);
//	cudaGetLastError();
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 2*128, empty_cycles);


//	MEASURE_THROUGHPUT(kernel_instr_clock_ADD32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_ATOMICADD32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_ADDC32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_ADD32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_SUB32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_MUL32, 32, 1024*2);
//  MEASURE_THROUGHPUT(kernel_instr_clock_MULWIDE, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_MAD32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_MADCCC32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_MULADDCCC32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_MOV32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_instr_clock_AND32, 32, 1024*2);

// 	PRINT_HISTOGRAM(kernel_instr_clock_ADD32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_ATOMICADD32, 1024*2);
//  PRINT_HISTOGRAM(kernel_instr_clock_ADDC32, 1024*2);
//  PRINT_HISTOGRAM(kernel_instr_clock_ADDCCC32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_SUB32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_MUL32, 1024*2);
//  PRINT_HISTOGRAM(kernel_instr_clock_MULWIDE, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_MAD32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_MADCCC32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_MULADDCCC32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_MOV32, 1024*2);
//	PRINT_HISTOGRAM(kernel_instr_clock_AND32, 1024*2);


//	PRINT_HISTOGRAM(kernel_function_clock_fp_add96_32,  3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_add128_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_add160_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_add192_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_add224_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_add256_32, 3*32);
	
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub96_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub128_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub160_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub192_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub224_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sub256_32, 3*32);


//	PRINT_HISTOGRAM64(kernel_function_clock_fp_add192_64, 3*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_fp_add256_64, 3*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_fp_sub192_64, 3*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_fp_sub256_64, 3*32);

//	MEASURE_THROUGHPUT(kernel_function_clock_fp_karatsuba1x2_32, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_function_clock_fp_karatsuba1x2_32_v2, 32, 1024*2);
//	MEASURE_THROUGHPUT(kernel_function_clock_fp_karatsuba1x2_32, 128, 1024*2);
//	MEASURE_THROUGHPUT(kernel_function_clock_fp_karatsuba1x2_32_v2, 128, 1024*2);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba1x2_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba1x2_32_v2, 3*32);

//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul2x2_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul2x2_32_v2, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul2x2_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul2x2_32_v6, 32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul2x2_32_v7, 32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul2x2_32_v2, 32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul2x2_32_v3, 32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul3x3_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul3x3_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul3x3_32_v2, 32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul4x4_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul4x4_32,    32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul4x4_32_v7, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul4x4_32_v2, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul4x4_32_v3, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_sqr4x4_32,    32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul5x5_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul6x6_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul6x6_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul7x7_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul7x7_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_karatsuba_mul8x8_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul8x8_32,    32*32);
	

//	PRINT_HISTOGRAM(kernel_function_clock_school_mul3x3_low_32,    32*32);	
//	PRINT_HISTOGRAM(kernel_function_clock_school_mul3x3_low_32_v2, 32*32);	


//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba3x3_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school3x3_32,    32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school3x3_32_v2, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school3x3_32_v3, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school3x3_32_v4, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school3x3_32_v5, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba4x4_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school4x4_32,    32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school4x4_32_v2, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school4x4_32_v3, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school4x4_32_v7, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_sqr4x4_32,       32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school5x5_32,    32*32);
	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba6x6_32, 32*32);
	PRINT_HISTOGRAM(kernel_function_clock_fp_school6x6_32,    32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba7x7_32, 32*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school7x7_32,    32*32);
	
//	PRINT_HISTOGRAM(kernel_function_clock_fp_karatsuba_mul8x8_32, 3*32);
//	PRINT_HISTOGRAM(kernel_function_clock_fp_school_mul8x8_32, 3*32);


//	PRINT_HISTOGRAM64(kernel_function_clock_fp_karatsuba2x2_64, 32*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_fp_karatsuba3x3_64, 32*32);
	PRINT_HISTOGRAM64(kernel_function_clock_fp_karatsuba4x4_64, 32*32);

//	PRINT_HISTOGRAM64(kernel_function_clock_school_mul1x1_64_v2, 32*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_school_mul1x1_64_v3, 32*32);
	
//	PRINT_HISTOGRAM64(kernel_function_clock_school_mul2x2_64, 32*32);
//	PRINT_HISTOGRAM64(kernel_function_clock_school_mul2x2_64_v2, 32*32);

//  printf ("\nkernel_function_clock_fp_karatsuba_mul128_32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//  kernel_function_clock_fp_karatsuba_mul128_32<<<Dg, Db>>>(d_ts, d_out, 268324234u, 876218736u, iterations);
//  cudaDeviceSynchronize();
//  cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//  print_information(ts, Dg, 3*128, empty_cycles);

//	printf ("\nkernel_clock_fpadd32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_clock_fpadd32<<<Dg, Db>>>(d_ts, d_out, 268324234u, 876218736u, iterations);
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 3*128, empty_cycles);
//
//	printf ("\nkernel_clock_fpadd64: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_clock_fpadd<<<Dg, Db>>>(d_ts, d_out, 198743268324234ull, 876218736ull, iterations);
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 3*128, empty_cycles);
//
//	printf ("\nkernel_clock_fpmul32: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_clock_fpmul32<<<Dg, Db>>>(d_ts, d_out, 268324234u, 876218736u, iterations);
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 3*128, empty_cycles);
//
//	printf ("\nkernel_clock_fpmul64: [%d blocks, %d thread(s)/block]\n", Dg.x, Db.x);
//	kernel_clock_fpmul<<<Dg, Db>>>(d_ts, d_out, 198743268324234ull, 876218736ull, iterations);
//	cudaDeviceSynchronize();
//	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);
//	print_information(ts, Dg, 3*128, empty_cycles);

	// make some how sure that the compile does not throw away the instruction, 
	// want to benchmark.
	cudaDeviceSynchronize();
	cudaMemcpy(out32, d_out32, sizeof(out32), cudaMemcpyDeviceToHost);
	cudaMemcpy(out64, d_out64, sizeof(out64), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaFree (d_ts);
	cudaFree (d_ts2);
	return 0;
}
