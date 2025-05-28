#include <stdint.h>

/// main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
// GF(2)/[x^32 + x^22 + x^2 + x^1 + 1]
//#define MODULUS 0b100000000010000000000000000000111
// GF(2)/[x^32 + x^15 + x^7 + x^1 + 1]
#define MODULUS 0b000000000000000010000000100000011ull
typedef uint32_t gf2to32;

///
/// \param a
/// \param b
/// \return
static inline
gf2to32 gf2to32_add(const gf2to32 a,
                 const gf2to32 b) {
    return a ^ b;
}

///
/// \param a
/// \param b
/// \return
static inline
gf2to32 gf2to32_sub(const gf2to32 a,
                 const gf2to32 b) {
    return a ^ b;
}

/// NOTE: non ct
/// \param a
/// \param b
/// \return
static inline
gf2to32 gf2to32_mul_v2(const gf2to32 a,
                 const gf2to32 b) {
    uint32_t result = 0;
    uint32_t shifted = a;

    for (uint32_t i = 0; i < 32; ++i) {
        if ((b & (1u << i)) != 0u) {
            result ^= shifted;
        }

        if ((shifted & (1u << 31u)) != 0u) {
            shifted <<= 1u;
            shifted ^= MODULUS;
        } else {
            shifted <<= 1u;
        }
    }

    return result;
}

/// this is ct
/// \param a
/// \param b
/// \return
static inline
gf2to32 gf2to32_mul(const gf2to32 a,
                       const gf2to32 b) {
    uint32_t r;
    r = (-(b>>31u     ) & a);
    for (int i = 30; i >= 0; --i) {
        r = (-(b>>i & 1u) & a) ^ (-(r>>31) & MODULUS) ^ (r+r);
    }
    return r;
}

///
/// \param a
/// \return
static inline
gf2to32 gf2to32_neg(const gf2to32 a) {
    return ~a;
}

#ifdef USE_AVX2
#include <immintrin.h>

///
/// \param a
/// \param b
/// \return
__m256i gf2to32v_add_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

///
/// \param a
/// \param b
/// \return
__m256i gf2to32v_sub_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYM9DgDJ4GmADl3ACNMYhAAZlIAB1QFQlsGZzcPPVj4mwFffyCWUPCoy0xrRKECJmICZPdPLgtMK0yGMoqCbMCQsMiLcsrq1LqFXra/DryuiIBKC1RXYmR2DgBSACYIv2Q3LABqJYjHPBYWPwJiPwA6BD3sJY0AQVWIrBp/bYBZAHkAETlvOSFthpglwNKCwWCQeCodCIVwuLcHvcEaYWCsAKwANjw22AVBWRAiKwAbsYWK5aMZXOiMRA0AwhtsUdTsUxSAjthzOVzuTzuXSGUzMdjgpNdgB2ABC7M5/IIjNJzO2AC8wqhdhEvoyjtTjEoCFxjJhongABwQDSTPZS%2B5c2Xy1FC7YsDDqzWkh0Y3WYfWG42EqAnSafH5/ISWiLWu5cwVY7nEV3bCAxvCTd064L0BjoEnRBRQGOTFXEVCkRMF1llhWYyYiq3SjnJ7aiYjEACeCaTVaxqe1mOMGYcObznY9hdVpZH1MmzvQE4LxHDkfr2wA9Cvthjtn4hsRXCUBApttEwttaKhUNFl/x4xATluExENFb79c9pqnxHtgBaL8p8WR3kmwPOVG31Dtkx7D1%2B0zbNjFzfMuzHYs50QitJ2rNM%2BwUWhaDwX08H9YJSwiLhf0mRdl1tYD7UVAgVgTMCljRbAmwqNs6xtXk7VAiIGPopiWMwz0mHQdB8P9MhtgXDio15eM322AgIhkwDmzbcCuxTIToMHODhwLIsS0rUcZxQ0dpIjZcljFL4EWXYhvTmBgpJk6yvg4aZaE4NFeE8DgtFIVBOEcbYFFmeZMF2NYeFIAhNA86YAGsQEkABOc4xQ0OETQiDE0RWDQVlSjF9E4SRfPiwLOF4BQQA0WL4umOBYCQNAWGNegyAoWlUHaugumQTZDGAAqQT4OgCDCWqIGCSrgj8CpW04GL5uYNsPmCbRiji7heDathBA%2BBhaCW/zeCwYJXGARwxFoWrdtILAWGG8QzsevAHJKIlMHugLMFUYpXEm5beBOBpKtw4JiEW5wsEq05DhB0hvuIYI4kwL5MGeoxcKMRq%2BAMYAFAANTwTAAHcPmPPyYv4QQRDEdgpBkQRFBUdQ3t0FZ9GGkBTGMcxIdqyBpgvJp7q/D5eK/Z6iVUb9ZdUejeFQFGziwYWIGmIp908CAHH6WpSB8UZcnyNI4gSARDYtjJEnaM2ukGBpttKYYbedxo3daB3OnCQZ3ZcGo9CGH3Tb9iRtfChY9HJwwCCpggAEkGH4EHPO8iq3qCjhtlUE0MS/DFJCbAwjG2Arzi4c4NETXBCBIKKSMmXgdq0cjSGStENHOIqTRNLgMQiSQxVSzLJEH0qOHK0g/ICnOarqhqzo7ryOBWLP5%2Bq5f2%2BmFH4jsSQgA%3D%3D%3D
/// instructions 2 + 6*31 = 188
/// \param a
/// \param b
/// \return
__m256i gf2to32v_mul_u256(const __m256i a,
                          const __m256i b) {
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i mod = _mm256_set1_epi32((int)MODULUS);
    __m256i     r = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)a, (__m256)b);
    __m256i carry = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)mod, (__m256)r);

    // 6 instructions per loop
    for (int i = 30; i >= 0; --i) {
        const __m256i t1 = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)a, (__m256)_mm256_slli_epi32(b, 31-i));
        const __m256i t2 = t1 ^ carry;
        const __m256i t3 = t2 ^ _mm256_add_epi32(r, r);
        r = t3;
        carry = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)mod, (__m256)r);
    }

    return r;
}

///
/// @param a1
/// @param c1
/// @return
static inline
__m128i gf2to32v_mul_u256_helper(__m128i a1,
                                 __m128i c1) {
    const __m128i mod = _mm_setr_epi64((__m64)MODULUS, (__m64)0ll);
    const __m128i m1  = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);

    __m128i a1c1t0 = _mm_clmulepi64_si128(a1, c1, 0);
    a1c1t0 ^= _mm_bslli_si128(_mm_clmulepi64_si128(a1, c1, 0x11), 8);
    __m128i a1c1t1 = _mm_srli_epi64(a1c1t0, 32);
    a1c1t0 = a1c1t0&m1;
    __m128i a1c1t2 = _mm_clmulepi64_si128(mod, a1c1t1, 0);
    a1c1t2 ^= _mm_bslli_si128(_mm_clmulepi64_si128(mod, a1c1t1, 0x10), 8);
    __m128i a1c1t3 = (a1c1t0 ^ a1c1t2) & m1;

    __m128i a1c1t4 = _mm_srli_epi64(a1c1t2, 32);
    __m128i a1c1t5 = _mm_clmulepi64_si128(mod, a1c1t4, 0);
    a1c1t5 ^= _mm_bslli_si128(_mm_clmulepi64_si128(mod, a1c1t4, 0x10), 8);
    return a1c1t3 ^ a1c1t5;
}

/// based on clmulepi_64
/// \param a
/// \param b
/// \return
static inline
__m256i gf2to32v_mul_u256_v2(const __m256i a,
                             const __m256i b) {
    const __m128i m1  = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);

    const __m128i al1 = _mm256_castsi256_si128(a);
    const __m128i bl1 = _mm256_castsi256_si128(b);
    const __m128i ah1 = _mm256_extracti128_si256(a, 0x1);
    const __m128i bh1 = _mm256_extracti128_si256(b, 0x1);

    const __m128i a1 = al1 & m1;
    const __m128i a3 = _mm_srli_epi64(al1, 32) & m1;

    const __m128i b1 = ah1 & m1;
    const __m128i b3 = _mm_srli_epi64(ah1, 32) & m1;


    const __m128i c1 = bl1 & m1;
    const __m128i c3 = _mm_srli_epi64(bl1, 32) & m1;

    const __m128i d1 = bh1 & m1;
    const __m128i d3 = _mm_srli_epi64(bh1, 32) & m1;


    __m128i a1c1 = gf2to32v_mul_u256_helper(a1, c1);
    __m128i a3c3 = _mm_slli_epi64(gf2to32v_mul_u256_helper(a3, c3), 32);
    __m128i b1d1 = gf2to32v_mul_u256_helper(b1, d1);
    __m128i b3d3 = _mm_slli_epi64(gf2to32v_mul_u256_helper(b1, d1), 32);

    a1c1 ^= a3c3;
    b1d1 ^= b3d3;
    const __m256i ret = _mm256_set_m128i(b1d1, a1c1);
    return ret;
}
#endif
