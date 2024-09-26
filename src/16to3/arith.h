#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "../helper.h"
#include "../16/arith.h"

/// Representation: a single F16to3 elements is stored in uin16_t
///  GF16to3:uint16_t: X^3 + x + 1;
typedef uint16_t gf16to3;

/// \return a+b
static inline gf16to3 gf16to3_add(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a+b for 5 elements in parallel
static inline gf16to3 gf16to3v_add(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// the same as add: char = 2 = best char
/// \return a-b = a+b
static inline gf16to3 gf16to3_sub(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a-b for 5 elements in parallel
static inline gf16to3 gf16to3v_sub(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// vector multiplication
/// \return a*b
static inline gf16to3 gf16to3_mul_gf16(const gf16to3 a,
                                       const gf16 b) {
    return gf16_mul(a&0xF, b) ^
          (gf16_mul((a>>4)&0xF, b) << 4u) ^
          (gf16_mul((a>>8)&0xF, b) << 8u);
}

/// implemented using karatsuba
/// \return a*b
static inline gf16to3 gf16to3_mul(const gf16to3 a,
                                  const gf16to3 b) {
    gf16to3 r;

    const gf16 a0 = a&0xF, a1 = (a>>4)&0XF, a2 = (a>>8)&0xF;
    const gf16 b0 = b&0xF, b1 = (b>>4)&0XF, b2 = (b>>8)&0xF;

    const gf16 p0 = gf16_mul(a0, b0);
    const gf16 p1 = gf16_mul(a1, b1);
    const gf16 p2 = gf16_mul(a2, b2);

    const gf16 a01 = gf16_add(a0, a1);
    const gf16 a12 = gf16_add(a1, a2);
    const gf16 b01 = gf16_add(b0, b1);
    const gf16 b12 = gf16_add(b1, b2);
    const gf16 p01 = gf16_mul(a01, b01);
    const gf16 p12 = gf16_mul(a12, b12);

    const gf16 a012 = gf16_add(a01, a2);
    const gf16 b012 = gf16_add(b01, b2);
    const gf16 p012 = gf16_mul(a012, b012);

    // compute lowest limb
    r = gf16_add(p1, p2);
    r = gf16_add(r, p12);
    r = gf16_add(r, p0);

    r^= p2 << 4;
    r^= p01 << 4;
    r^= p0 << 4;
    r^= p1 << 4;

    r^= p012 << 8;
    r^= p01 << 8;
    r^= p12 << 8;
    return r;
}

static inline gf16to3 gf16to3v_mul(const uint64_t a,
                                   const uint64_t b) {
    const uint64_t m0 = 0x00F00F00F00F, m1 = 0x0F00F00F00F0,
                   m2 = 0xF00F00F00F00;
    uint64_t r;

    // TODO

    return r;
}


#ifdef USE_AVX2

/// vector multiplication
/// \param a
/// \param b
/// \return a*b;
static inline __m256i gf16to3v_mul_gf16(const __m256i a,
                                        const uint8_t b) {
   const __m256i bb = _mm256_set1_epi8(b);
   return gf16v_mul_u256(a, bb);
}

/// TODO there are multiple ways to implement this:
///     - assumes that each 1.5byte element is stored in 2 bytes,
///         thus there are 16 elements in a register
///     - or that that the elements are compressed together, meaning
///         we have 21 elements in a register
static inline __m256i gf16to3v_mul_u256(const __m256i a,
                                        const __m256i b) {
    return a;
}

/// compressed representations
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    size_t i = (size_t)((double)d * 1.5) + d&1u;

    uint8_t *o8 = (uint8_t *)out;
    uint8_t *i8 = (uint8_t *)in1;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)o8,
                            _mm256_loadu_si256((__m256i *)o8) ^
                            _mm256_loadu_si256((__m256i *)i8));
        i   -= 32u;
        in1 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)o8,
                         _mm_loadu_si128((__m128i *)o8) ^
                         _mm_loadu_si128((__m128i *)i8));
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *o8++ ^= *i8++;
    }
}
#endif


#undef MODULUS
