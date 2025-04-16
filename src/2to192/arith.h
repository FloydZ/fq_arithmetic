#include <stdint.h>
#include <stdlib.h>

#if USE_AVX2 
#include <immintrin.h>
#endif

// GF(2)/(x^192 + x^7 + x^2 + x + 1).
#define MODULUS 0b10000111

typedef uint64_t gf2to192[3];

void gf2to192_add(gf2to192 r, 
                  const gf2to192 a,
                  const gf2to192 b) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
    r[2] = a[2] ^ b[2];
}

void gf2to192_sub(gf2to192 r, 
                  const gf2to192 a,
                  const gf2to192 b) {
    gf2to192_add(r, a, b);
}

void gf2to192_mul(gf2to192 r, 
                  const gf2to192 a,
                  const gf2to192 b) {
#if USE_AVX2 
    /* load the two operands and the modulus into 128-bit registers.
       we load corresponding limbs of both operands into a single register,
       because it lets us implement Karatsuba (see below) with fewer 128-bit
       xors. */
    const __m128i ab0 = _mm_set_epi64x(a[0], b[0]);
    const __m128i ab1 = _mm_set_epi64x(a[1], b[1]);
    const __m128i ab2 = _mm_set_epi64x(a[2], b[2]);
    const __m128i modulus = _mm_set_epi64x(MODULUS, MODULUS);

    /* here we implement a Karatsuba-like approach for multiplying 3-limb numbers.
    given
      a = a0 + B * a1 + B^2 * a2
      b = b0 + B * b1 + B^2 * b2
    we can compute
      c = c0 + ... + B^4 * c4
    (where ai and bi are < B, but ci are < B^2)
    with 6 multiplications as follows:
      1. c0 = a0 * b0
      2. c4 = a2 * b2
      3. t  = a1 * b1
      4. c1 = (a0 + a1) * (b0 + b1) - c0 - t
      5. c3 = (a1 + a2) * (b1 + b2) - c4 - t
      6. c2 = (a0 + a2) * (b0 + b2) - c0 - c4 + t */
    __m128i c0 = _mm_clmulepi64_si128(ab0, ab0, 0x01); /* multiply low and high halves */
    __m128i c4 = _mm_clmulepi64_si128(ab2, ab2, 0x01);

    __m128i t = _mm_clmulepi64_si128(ab1, ab1, 0x01);

    __m128i xor01 = _mm_xor_si128(ab0, ab1);
    __m128i c1 = _mm_clmulepi64_si128(xor01, xor01, 0x01);
    c1 = _mm_xor_si128(_mm_xor_si128(c1, c0), t);

    __m128i xor12 = _mm_xor_si128(ab1, ab2);
    __m128i c3 = _mm_clmulepi64_si128(xor12, xor12, 0x01);
    c3 = _mm_xor_si128(_mm_xor_si128(c3, c4), t);

    __m128i xor02 = _mm_xor_si128(ab0, ab2);
    __m128i c2 = _mm_clmulepi64_si128(xor02, xor02, 0x01);
    c2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(c2, c0), c4), t);

    /* now let's compute
         d = d0 + B^2 * d1 + B^4 d2
       where d = c and di < B^2 */
    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d2 = _mm_xor_si128(c4, _mm_srli_si128(c3, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)),
                                _mm_slli_si128(c3, 8));

    /* done with the multiplication, time to reduce */

    /* reduce w.r.t. high half of d2 */
    __m128i tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, tmp);

    /* reduce w.r.t. low half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. high half of d1 */
    tmp = _mm_clmulepi64_si128(d1, modulus, 0x01);
    d0 = _mm_xor_si128(d0, tmp);

    /* done, now just store everything back into this->value_ */
    _mm_storeu_si128((__m128i*) &r[0], d0);
    _mm_storel_epi64((__m128i*) &r[2], d1);
#else 
    /* Slow, but straight-forward */
    uint64_t shifted[3] = {a[0], a[1], a[2]};

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            if (b[i] & (1ull << j)) {
                r[0] ^= shifted[0];
                r[1] ^= shifted[1];
                r[2] ^= shifted[2];
            }

            if (shifted[2] & (1ull << 63)) {
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = (shifted[0] << 1) ^ MODULUS;
            } else {
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = shifted[0] << 1;
            }
        }

    }
#endif
}


#ifdef USE_AVX2
#include <immintrin.h>

static inline
__m256i gf192v_add_u256(const __m256i a,
                        const __m256i b) {
    // TODO 
    return a;
}

#endif
