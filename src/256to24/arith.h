#ifndef FQ_ARITHMETIC_GF256TO24_ARITH_H
#define FQ_ARITHMETIC_GF256TO24_ARITH_H
#include <stdint.h>
#include <string.h>
#include "../256to16/arith.h"
#include "../256/arith.h"

/// in total we use the following representation for the field extension tower
///     GF256to24:uint8_t[24]: X^3 + x + 1;
/// i.e. three limbs in GF(256^8) (stored as 3 little endian uint64_t)
typedef uint8_t  gf256to24[24];

static inline void gf256to24_zero(gf256to24 res) {
    memset(res, 0, 24);
}

static inline void gf256to24_set(gf256to24 res,
                                 const uint8_t *b) {
    memmove(res, b, 24);
}

static inline void gf256to24_set_(gf256to24 *res,
                                  const uint8_t *b) {
    memmove(*res, b, 24);
}

/// multiplication using karatsuba, reduced by X^3 + X + 1:
///     c_0 = p0, c_1 = p01 + p0 + p1, c_2 = p012 + p01 + p12,
///     c_3 = p12 + p1 + p2, c_4 = p2
///     r_0 = c_0 + c_3       = p0 + p1 + p2 + p12
///     r_1 = c_1 + c_3 + c_4 = p0 + p01 + p12
///     r_2 = c_2 + c_4       = p012 + p01 + p12 + p2
/// \param res
/// \param a
/// \param b
/// \reutrn
static inline void gf256to24_mul(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
    // NOTE: memcpy instead of casts to `uint64_t *`, which break strict aliasing
    uint64_t a_[3], b_[3], r_[3];
    memcpy(a_, a, 24);
    memcpy(b_, b, 24);

    const uint64_t p0 = gf256to8_mul(a_[0], b_[0]);
    const uint64_t p1 = gf256to8_mul(a_[1], b_[1]);
    const uint64_t p2 = gf256to8_mul(a_[2], b_[2]);

    const uint64_t a01 = gf256to8_add(a_[0], a_[1]);
    const uint64_t a12 = gf256to8_add(a_[1], a_[2]);
    const uint64_t b01 = gf256to8_add(b_[0], b_[1]);
    const uint64_t b12 = gf256to8_add(b_[1], b_[2]);
    const uint64_t p01 = gf256to8_mul(a01, b01);
    const uint64_t p12 = gf256to8_mul(a12, b12);

    const uint64_t a012= gf256to8_add(a01, a_[2]);
    const uint64_t b012= gf256to8_add(b01, b_[2]);
    const uint64_t p012= gf256to8_mul(a012, b012);

    r_[0] = gf256to8_add(p1, p2);
    r_[0] = gf256to8_add(r_[0], p12);
    r_[0] = gf256to8_add(r_[0], p0);

    r_[1] = gf256to8_add(p0, p01);
    r_[1] = gf256to8_add(r_[1], p12);

    r_[2] = gf256to8_add(p012, p01);
    r_[2] = gf256to8_add(r_[2], p12);
    r_[2] = gf256to8_add(r_[2], p2);
    memcpy(res, r_, 24);
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_add(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
    for (uint32_t i = 0; i < 24; i++){
        res[i] = a[i] ^ b[i];
    }
}

/// res = a - b
static inline void gf256to24_sub(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
    gf256to24_add(res, a, b);
}

/// res = -a = a (characteristic 2)
static inline void gf256to24_neg(gf256to24 res,
                                 const gf256to24 a) {
    memmove(res, a, 24);
}

/// Itoh-Tsujii inversion over GF(q), q = 256^8:
///     u = a^(q + q^2), N(a) = u*a \in GF(q), a^-1 = u * N(a)^-1
/// NOTE: inv(0) = 0
/// \param res = a^-1
static inline void gf256to24_inv(gf256to24 res,
                                 const gf256to24 a) {
    gf256to24 t1, t2, u, n;
    uint64_t u_[3], n0;

    // t1 = a^q
    memcpy(t1, a, 24);
    for (uint32_t i = 0; i < 64; i++) {
        gf256to24_mul(t1, t1, t1);
    }

    // t2 = a^(q^2)
    memcpy(t2, t1, 24);
    for (uint32_t i = 0; i < 64; i++) {
        gf256to24_mul(t2, t2, t2);
    }

    gf256to24_mul(u, t1, t2);
    gf256to24_mul(n, u, a);
    memcpy(&n0, n, 8);
    const gf256to8 ninv = gf256to8_inv(n0);

    memcpy(u_, u, 24);
    for (uint32_t i = 0; i < 3; i++) {
        u_[i] = gf256to8_mul(u_[i], ninv);
    }
    memcpy(res, u_, 24);
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf256to2(gf256to24 res,
                                          const gf256to24 a,
                                          const gf256to2 b) {
    gf256to2 a_[12];
    memcpy(a_, a, 24);
    for (uint32_t i = 0; i < 12; ++i) {
        a_[i] = gf256to2_mul(a_[i], b);
    }
    memcpy(res, a_, 24);
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf256(gf256to24 res,
                                       const gf256to24 a,
                                       const gf256 b) {
    for (uint32_t i = 0; i < 24; ++i) {
        res[i] = gf256_mul(a[i], b);
    }
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf16(gf256to24 res,
                                      const gf256to24 a,
                                      const gf16 b) {
    const gf256 t = gf256_expand_tab[b];
    gf256to24_mul_gf256(res, a, t);
}

#ifdef USE_AVX2
/// a single element per register: [a_0, a_1, a_2, *] (the 3 limbs in GF(256^8))
/// \return a+b
static inline __m256i gf256to24v_add_u256(const __m256i a,
                                          const __m256i b) {
    return a ^ b;
}

/// \return a-b
static inline __m256i gf256to24v_sub_u256(const __m256i a,
                                          const __m256i b) {
    return a ^ b;
}

/// same as `gf256to24_mul`, but the 6 products are computed in 2 vectorized
/// multiplications.
/// NOTE: the 4th limb of a and b is ignored, the 4th limb of the output is zero
/// \param a: [a_0, a_1, a_2, *]
/// \param b: [b_0, b_1, b_2, *]
/// \return a*b
static inline __m256i gf256to24v_mul_u256(const __m256i a,
                                          const __m256i b) {
    const __m256i zero = _mm256_setzero_si256();

    // q = [p0, p1, p2, *]
    const __m256i q = gf256to8v_mul_u256(a, b);

    // [a0^a1, a1^a2, a0^a1^a2, 0]
    const __m256i a1 = _mm256_blend_epi32(zero, _mm256_permute4x64_epi64(a, 0x10), 0x30);
    const __m256i b1 = _mm256_blend_epi32(zero, _mm256_permute4x64_epi64(b, 0x10), 0x30);
    const __m256i x = a ^ _mm256_permute4x64_epi64(a, 0xC9) ^ a1;
    const __m256i y = b ^ _mm256_permute4x64_epi64(b, 0xC9) ^ b1;

    // w = [p01, p12, p012, *]
    const __m256i w = gf256to8v_mul_u256(x, y);

    // s = p0 + p1 + p2 in every limb
    const __m256i s = _mm256_permute4x64_epi64(q, 0x00) ^
                      _mm256_permute4x64_epi64(q, 0x55) ^
                      _mm256_permute4x64_epi64(q, 0xAA);

    const __m256i w1 = _mm256_permute4x64_epi64(w, 0xC1); // [p12, p01, p01, *]
    const __m256i w2 = _mm256_permute4x64_epi64(w, 0xD4); // [*, p12, p12, *]
    const __m256i q1 = _mm256_permute4x64_epi64(q, 0xE0); // [*, p0, p2, *]

    const __m256i r0 = s ^ w1;              // [r0, *, *, *]
    const __m256i r1 = w1 ^ w2 ^ q1;        // [*, r1, p2 + p01 + p12, *]
    const __m256i r2 = r1 ^ w;              // [*, *, r2, *]

    __m256i r = _mm256_blend_epi32(r0, r1, 0x0C);
    r = _mm256_blend_epi32(r, r2, 0x30);
    return _mm256_blend_epi32(r, zero, 0xC0);
}
#endif

#ifdef USE_NEON
/// a single element is held in two registers:
///     val[0] = [a_0, a_1], val[1] = [a_2, *]
/// \return a+b
static inline uint64x2x2_t gf256to24v_add_u128(const uint64x2x2_t a,
                                               const uint64x2x2_t b) {
    uint64x2x2_t r;
    r.val[0] = veorq_u64(a.val[0], b.val[0]);
    r.val[1] = veorq_u64(a.val[1], b.val[1]);
    return r;
}

/// \return a-b
static inline uint64x2x2_t gf256to24v_sub_u128(const uint64x2x2_t a,
                                               const uint64x2x2_t b) {
    return gf256to24v_add_u128(a, b);
}

/// NEON version of `gf256to24_mul`: the 6 products are computed with 3
/// 2-way vectorized multiplications.
/// NOTE: the 4th limb of a and b is ignored, the 4th limb of the output is zero
/// \return a*b
static inline uint64x2x2_t gf256to24v_mul_u128(const uint64x2x2_t a,
                                               const uint64x2x2_t b) {
    const uint64x1_t a0 = vget_low_u64(a.val[0]), a1 = vget_high_u64(a.val[0]), a2 = vget_low_u64(a.val[1]);
    const uint64x1_t b0 = vget_low_u64(b.val[0]), b1 = vget_high_u64(b.val[0]), b2 = vget_low_u64(b.val[1]);
    const uint64x1_t a01 = veor_u64(a0, a1), b01 = veor_u64(b0, b1);
    const uint64x1_t a12 = veor_u64(a1, a2), b12 = veor_u64(b1, b2);

    // [p0, p1], [p2, p01], [p12, p012]
    const uint64x2_t x = gf256to8v_mul_u128(a.val[0], b.val[0]);
    const uint64x2_t y = gf256to8v_mul_u128(vcombine_u64(a2, a01), vcombine_u64(b2, b01));
    const uint64x2_t z = gf256to8v_mul_u128(vcombine_u64(a12, veor_u64(a01, a2)),
                                            vcombine_u64(b12, veor_u64(b01, b2)));

    const uint64x1_t p0  = vget_low_u64(x), p1  = vget_high_u64(x);
    const uint64x1_t p2  = vget_low_u64(y), p01 = vget_high_u64(y);
    const uint64x1_t p12 = vget_low_u64(z), p012= vget_high_u64(z);

    const uint64x1_t r0 = veor_u64(veor_u64(p0, p1), veor_u64(p2, p12));
    const uint64x1_t r1 = veor_u64(veor_u64(p0, p01), p12);
    const uint64x1_t r2 = veor_u64(veor_u64(p012, p01), veor_u64(p12, p2));

    uint64x2x2_t r;
    r.val[0] = vcombine_u64(r0, r1);
    r.val[1] = vcombine_u64(r2, vdup_n_u64(0));
    return r;
}
#endif // USE_NEON

#ifdef USE_AVX512
/// two elements per register, each in its own 256-bit lane:
///     [a_0, a_1, a_2, *, a'_0, a'_1, a'_2, *]
/// \return a+b
static inline __m512i gf256to24v_add_u512(const __m512i a,
                                          const __m512i b) {
    return a ^ b;
}

/// \return a-b
static inline __m512i gf256to24v_sub_u512(const __m512i a,
                                          const __m512i b) {
    return a ^ b;
}

/// same as `gf256to24v_mul_u256`, but for both 256-bit lanes
/// NOTE: the 4th limb of each lane is ignored, the 4th limb of the output is zero
/// \return a*b
static inline __m512i gf256to24v_mul_u512(const __m512i a,
                                          const __m512i b) {
    // q = [p0, p1, p2, *]
    const __m512i q = gf256to8v_mul_u512(a, b);

    // [a0^a1, a1^a2, a0^a1^a2, 0]
    const __m512i a1 = _mm512_maskz_permutex_epi64(0x44, a, 0x10);
    const __m512i b1 = _mm512_maskz_permutex_epi64(0x44, b, 0x10);
    const __m512i x = a ^ _mm512_permutex_epi64(a, 0xC9) ^ a1;
    const __m512i y = b ^ _mm512_permutex_epi64(b, 0xC9) ^ b1;

    // w = [p01, p12, p012, *]
    const __m512i w = gf256to8v_mul_u512(x, y);

    // s = p0 + p1 + p2 in every limb
    const __m512i s = _mm512_permutex_epi64(q, 0x00) ^
                      _mm512_permutex_epi64(q, 0x55) ^
                      _mm512_permutex_epi64(q, 0xAA);

    const __m512i w1 = _mm512_permutex_epi64(w, 0xC1); // [p12, p01, p01, *]
    const __m512i w2 = _mm512_permutex_epi64(w, 0xD4); // [*, p12, p12, *]
    const __m512i q1 = _mm512_permutex_epi64(q, 0xE0); // [*, p0, p2, *]

    const __m512i r0 = s ^ w1;
    const __m512i r1 = w1 ^ w2 ^ q1;
    const __m512i r2 = r1 ^ w;

    __m512i r = _mm512_mask_blend_epi64(0x22, r0, r1);
    r = _mm512_mask_blend_epi64(0x44, r, r2);
    return _mm512_maskz_mov_epi64(0x77, r);
}
#endif // USE_AVX512
#endif
