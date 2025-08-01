#pragma once

#include <stdint.h>
#include <arm_neon.h>
#include "arith.h"

#ifdef __ARM_FEATURE_CRYPTO

// ARM NEON implementation of GF(2^256) multiplication using PMULL instructions
// This is a translation of the AVX2 implementation in arith.h

// Helper function to perform carryless multiplication of two 64-bit values
static inline
poly128_t clmul_neon(uint64_t a, uint64_t b) {
    return vmull_p64((poly64_t)a, (poly64_t)b);
}

// Helper function to XOR two 128-bit values and return a 128-bit result
static inline
uint64x2_t xor_u128(uint64x2_t a, uint64x2_t b) {
    return veorq_u64(a, b);
}

// Helper function to perform 128-bit shift left
static inline
uint64x2_t slli_u128(uint64x2_t a, int count) {
    if (count == 8) {
        // Shift left by 8 bytes (64 bits)
        return vextq_u64(vdupq_n_u64(0), a, 1);
    }
    // Other shift amounts would require more complex handling
    // Not implemented for simplicity
    return a;
}

// Helper function to perform 128-bit shift right
static inline
uint64x2_t srli_u128(uint64x2_t a, int count) {
    if (count == 8) {
        // Shift right by 8 bytes (64 bits)
        return vextq_u64(a, vdupq_n_u64(0), 1);
    }
    // Other shift amounts would require more complex handling
    // Not implemented for simplicity
    return a;
}

/// Multiply two field elements using ARM NEON instructions (PMULL)
/// Uses Karatsuba-like approach for multiplying 4-limb numbers
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_neon(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    // Load the input values as 128-bit NEON vectors
    uint64x2_t a_low = vld1q_u64(&a[0]);
    uint64x2_t a_high = vld1q_u64(&a[2]);
    uint64x2_t b_low = vld1q_u64(&b[0]);
    uint64x2_t b_high = vld1q_u64(&b[2]);
    
    // Prepare for Karatsuba-like multiplication with 9 multiplications
    // Extract individual 64-bit limbs for easier access
    uint64_t a0 = vgetq_lane_u64(a_low, 0);
    uint64_t a1 = vgetq_lane_u64(a_low, 1);
    uint64_t a2 = vgetq_lane_u64(a_high, 0);
    uint64_t a3 = vgetq_lane_u64(a_high, 1);
    
    uint64_t b0 = vgetq_lane_u64(b_low, 0);
    uint64_t b1 = vgetq_lane_u64(b_low, 1);
    uint64_t b2 = vgetq_lane_u64(b_high, 0);
    uint64_t b3 = vgetq_lane_u64(b_high, 1);
    
    // Compute the 9 multiplications needed for Karatsuba
    // 1. c0 = a0 * b0
    poly128_t c0 = clmul_neon(a0, b0);
    uint64x2_t c0v = vreinterpretq_u64_p128(c0);
    
    // 2. c6 = a3 * b3
    poly128_t c6 = clmul_neon(a3, b3);
    uint64x2_t c6v = vreinterpretq_u64_p128(c6);
    
    // 3. t = a1 * b1
    poly128_t t = clmul_neon(a1, b1);
    uint64x2_t tv = vreinterpretq_u64_p128(t);
    
    // 4. u = a2 * b2
    poly128_t u = clmul_neon(a2, b2);
    uint64x2_t uv = vreinterpretq_u64_p128(u);
    
    // 5. c1 = (a0 + a1) * (b0 + b1) - c0 - t
    poly128_t tmp = clmul_neon(a0 ^ a1, b0 ^ b1);
    uint64x2_t c1v = xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c0v), tv);
    
    // 6. c2 = (a0 + a2) * (b0 + b2) - c0 + t - u
    tmp = clmul_neon(a0 ^ a2, b0 ^ b2);
    uint64x2_t c2v = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c0v), tv), uv);
    
    // 7. c5 = (a2 + a3) * (b2 + b3) - c6 - u
    tmp = clmul_neon(a2 ^ a3, b2 ^ b3);
    uint64x2_t c5v = xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c6v), uv);
    
    // 8. c4 = (a1 + a3) * (b1 + b3) - c6 + u - t
    tmp = clmul_neon(a1 ^ a3, b1 ^ b3);
    uint64x2_t c4v = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c6v), uv), tv);
    
    // 9. c3 = (a0 + a1 + a2 + a3) * (b0 + b1 + b2 + b3) - c0 - c1 - c2 - c4 - c5 - c6
    tmp = clmul_neon(a0 ^ a1 ^ a2 ^ a3, b0 ^ b1 ^ b2 ^ b3);
    uint64x2_t c3v = xor_u128(xor_u128(xor_u128(xor_u128(xor_u128(
                     xor_u128(vreinterpretq_u64_p128(tmp), c0v), c1v), c2v), c4v), c5v), c6v);
    
    // Merge the 6 overlapping 128-bit limbs into non-overlapping 128-bit limbs
    uint64x2_t d0 = xor_u128(c0v, slli_u128(c1v, 8));
    uint64x2_t d1 = xor_u128(xor_u128(c2v, srli_u128(c1v, 8)), slli_u128(c3v, 8));
    uint64x2_t d2 = xor_u128(xor_u128(c4v, srli_u128(c3v, 8)), slli_u128(c5v, 8));
    uint64x2_t d3 = xor_u128(c6v, srli_u128(c5v, 8));
    
    // Perform reduction with the GF(2^256) modulus
    const uint64_t modulus = MODULUS; // 0b10000100101
    
    // Reduce w.r.t. high half of d3
    poly128_t tmp1 = clmul_neon(vgetq_lane_u64(d3, 1), modulus);
    uint64x2_t tmp1v = vreinterpretq_u64_p128(tmp1);
    d2 = xor_u128(d2, srli_u128(tmp1v, 8));
    d1 = xor_u128(d1, slli_u128(tmp1v, 8));
    
    // Reduce w.r.t. low half of d3
    poly128_t tmp2 = clmul_neon(vgetq_lane_u64(d3, 0), modulus);
    uint64x2_t tmp2v = vreinterpretq_u64_p128(tmp2);
    d1 = xor_u128(d1, tmp2v);
    
    // Reduce w.r.t. high half of d2
    poly128_t tmp3 = clmul_neon(vgetq_lane_u64(d2, 1), modulus);
    uint64x2_t tmp3v = vreinterpretq_u64_p128(tmp3);
    d1 = xor_u128(d1, srli_u128(tmp3v, 8));
    d0 = xor_u128(d0, slli_u128(tmp3v, 8));
    
    // Reduce w.r.t. low half of d2
    poly128_t tmp4 = clmul_neon(vgetq_lane_u64(d2, 0), modulus);
    uint64x2_t tmp4v = vreinterpretq_u64_p128(tmp4);
    d0 = xor_u128(d0, tmp4v);
    
    // Store the final result
    vst1q_u64(&r[0], d0);
    vst1q_u64(&r[2], d1);
}

// Optimized version that uses the Polynomial Multiplication NEON intrinsics
static inline
void gf2to256v_mul_neon(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    gf2to256_mul_neon(r, a, b);
}

// Alternative version using naive multiplication approach with NEON
static inline
void gf2to256v_mul_neon_v2(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    // Load input values
    uint64x2_t a_low = vld1q_u64(&a[0]);
    uint64x2_t a_high = vld1q_u64(&a[2]);
    uint64x2_t b_low = vld1q_u64(&b[0]);
    uint64x2_t b_high = vld1q_u64(&b[2]);
    
    // Extract individual 64-bit limbs
    uint64_t a0 = vgetq_lane_u64(a_low, 0);
    uint64_t a1 = vgetq_lane_u64(a_low, 1);
    uint64_t a2 = vgetq_lane_u64(a_high, 0);
    uint64_t a3 = vgetq_lane_u64(a_high, 1);
    
    uint64_t b0 = vgetq_lane_u64(b_low, 0);
    uint64_t b1 = vgetq_lane_u64(b_low, 1);
    uint64_t b2 = vgetq_lane_u64(b_high, 0);
    uint64_t b3 = vgetq_lane_u64(b_high, 1);
    
    // Compute all 16 pairwise products with PMULL
    poly128_t m00 = clmul_neon(a0, b0);
    poly128_t m01 = clmul_neon(a0, b1);
    poly128_t m02 = clmul_neon(a0, b2);
    poly128_t m03 = clmul_neon(a0, b3);
    poly128_t m10 = clmul_neon(a1, b0);
    poly128_t m11 = clmul_neon(a1, b1);
    poly128_t m12 = clmul_neon(a1, b2);
    poly128_t m13 = clmul_neon(a1, b3);
    poly128_t m20 = clmul_neon(a2, b0);
    poly128_t m21 = clmul_neon(a2, b1);
    poly128_t m22 = clmul_neon(a2, b2);
    poly128_t m23 = clmul_neon(a2, b3);
    poly128_t m30 = clmul_neon(a3, b0);
    poly128_t m31 = clmul_neon(a3, b1);
    poly128_t m32 = clmul_neon(a3, b2);
    poly128_t m33 = clmul_neon(a3, b3);
    
    // Combine products to get overlapping 128-bit limbs
    uint64x2_t c0 = vreinterpretq_u64_p128(m00);
    uint64x2_t c1 = xor_u128(vreinterpretq_u64_p128(m01), vreinterpretq_u64_p128(m10));
    uint64x2_t c2 = xor_u128(xor_u128(vreinterpretq_u64_p128(m02), vreinterpretq_u64_p128(m11)), 
                            vreinterpretq_u64_p128(m20));
    uint64x2_t c3 = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(m03), vreinterpretq_u64_p128(m12)), 
                            vreinterpretq_u64_p128(m21)), vreinterpretq_u64_p128(m30));
    uint64x2_t c4 = xor_u128(xor_u128(vreinterpretq_u64_p128(m13), vreinterpretq_u64_p128(m22)), 
                            vreinterpretq_u64_p128(m31));
    uint64x2_t c5 = xor_u128(vreinterpretq_u64_p128(m23), vreinterpretq_u64_p128(m32));
    uint64x2_t c6 = vreinterpretq_u64_p128(m33);
    
    // Merge into non-overlapping 128-bit limbs
    uint64x2_t d0 = xor_u128(c0, slli_u128(c1, 8));
    uint64x2_t d1 = xor_u128(xor_u128(c2, srli_u128(c1, 8)), slli_u128(c3, 8));
    uint64x2_t d2 = xor_u128(xor_u128(c4, srli_u128(c3, 8)), slli_u128(c5, 8));
    uint64x2_t d3 = xor_u128(c6, srli_u128(c5, 8));
    
    // Reduction with the GF(2^256) modulus
    const uint64_t modulus = MODULUS;
    
    // Reduce with respect to each part of d2 and d3
    poly128_t tmp;
    uint64x2_t tmpv;
    
    // Reduce w.r.t. high half of d3
    tmp = clmul_neon(vgetq_lane_u64(d3, 1), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d2 = xor_u128(d2, srli_u128(tmpv, 8));
    d1 = xor_u128(d1, slli_u128(tmpv, 8));
    
    // Reduce w.r.t. low half of d3
    tmp = clmul_neon(vgetq_lane_u64(d3, 0), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d1 = xor_u128(d1, tmpv);
    
    // Reduce w.r.t. high half of d2
    tmp = clmul_neon(vgetq_lane_u64(d2, 1), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d1 = xor_u128(d1, srli_u128(tmpv, 8));
    d0 = xor_u128(d0, slli_u128(tmpv, 8));
    
    // Reduce w.r.t. low half of d2
    tmp = clmul_neon(vgetq_lane_u64(d2, 0), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d0 = xor_u128(d0, tmpv);
    
    // Store the final result
    vst1q_u64(&r[0], d0);
    vst1q_u64(&r[2], d1);
}

#endif // __ARM_FEATURE_CRYPTO