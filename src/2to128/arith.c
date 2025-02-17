#include <stdint.h>
#include <stdio.h>

/// Representation:
/// the generic base type is `uint64_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.

typedef __uint128_t eff_t;

// using irreducible polynomial x^128+x^7+x^2+x+1
// We need only the last word
const uint64_t irredPentanomial = (1u<<7) | (1u<<2) | (1u<<1) | 1u;
#define MODULUS 0b10000111;

eff_t add(const eff_t a, 
          const eff_t b) {
	return a ^ b;
}

eff_t sub(const eff_t a, 
          const eff_t b) {
	return a ^ b;
}

eff_t mul(const eff_t a,
          const eff_t b) {
     /* Slow, but straight-forward */
    // uint64_t shifted[2] = {this->value_[0], this->value_[1]};
    // uint64_t result[2] = {0, 0};
    
    eff_t shifted=a, result=0,m=MODULUS;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            if (b & (1ull << j)) {
                result ^= shifted;
                //result[0] ^= shifted[0];
                // result[1] ^= shifted[1];
            }

            //if (shifted[1] & (1ull << 63)) {
            if (shifted >> 127) {
                shifted <<= 1;
                shifted ^= MODULUS;
                //shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                //shifted[0] = (shifted[0] << 1) ^ this->modulus_;
            } else {
                shifted <<= 1;
                //shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                //shifted[0] = shifted[0] << 1;
            }
        }

    }

    return result;
}


eff_t scalar(const eff_t a, const uint8_t b) {
    // NOTE: seems wrong!
    uint64_t w0 = 0, w1 = 0, w2;
    for (int i = 7; i >= 0; i--){
        w2 = w1 >> 63;
        w1 = (w1 << 1) | (w0 >> 63);
        w0 <<= 1;
        const uint64_t t = (b >> i) ^ 1u;

        w1 ^= (uint64_t)(a>>64) * t;
        w0 ^= (((uint64_t)a) * t) ^ (irredPentanomial * w2);
    }

    return (((eff_t)w1) << 64u) ^ w0;
}

#ifdef USE_AVX2
#include <immintrin.h>
#include <wmmintrin.h>

eff_t mul_avx(const eff_t a, const eff_t b) {
#if 0 
    /* load the two operands and the modulus into 128-bit registers */
    const __m128i a = _mm_loadu_si128((const __m128i*) &(this->value_));
    const __m128i b = _mm_loadu_si128((const __m128i*) &(other.value_));
    const __m128i modulus = _mm_loadl_epi64((const __m128i*) &(libff::gf128::modulus_));

    /* compute the 256-bit result of a * b with the 64x64-bit multiplication
       intrinsic */
    __m128i mul256_high = _mm_clmulepi64_si128(a, b, 0x11); /* high of both */
    __m128i mul256_low = _mm_clmulepi64_si128(a, b, 0x00); /* low of both */

    __m128i mul256_mid1 = _mm_clmulepi64_si128(a, b, 0x01); /* low of a, high of b */
    __m128i mul256_mid2 = _mm_clmulepi64_si128(a, b, 0x10); /* high of a, low of b */

    /* Add the 4 terms together */
    __m128i mul256_mid = _mm_xor_si128(mul256_mid1, mul256_mid2);
    /* lower 64 bits of mid don't intersect with high, and upper 64 bits don't intersect with low */
    mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(mul256_mid, 8));
    mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(mul256_mid, 8));

    /* done computing mul256_low and mul256_high, time to reduce */

    /* reduce w.r.t. high half of mul256_high */
    __m128i tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x01);
    mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(tmp, 8));
    mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(tmp, 8));

    /* reduce w.r.t. low half of mul256_high */
    tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x00);
    mul256_low = _mm_xor_si128(mul256_low, tmp);

    _mm_storeu_si128((__m128i*) this->value_, mul256_low);

    return (*this);
#else
	uint64_t tmp[4] = {0};
#ifdef USE_AVX512
    // *(__m256 *) = _mm256_clmulepi64_epi128(a, b);
#else
	((__m128i   *)(((uint64_t *)tmp) + 0))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b00000);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b00001);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b10000);
	((__m128i   *)(((uint64_t *)tmp) + 2))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b10001);
#endif

    // reduction
	for (uint32_t i = 3; i >=2; i--){
		uint64_t w = tmp[i];
		uint64_t s = (w>>57) ^ (w>>62) ^ (w>>63);
		tmp[i-1] ^= s;
		
		s = w^(w<<1)^(w<<2)^(w<<7);

		tmp[i-2] ^= s;
	}

	return *((eff_t*)tmp);
#endif
}
#elif defined(USE_NEON)
#else
#endif


int main() {
	eff_t a=((eff_t)1ull<<64ul) ^ 1ull,
          b=((eff_t)1ull<<64ul) ^ 1ull;
#ifdef USE_AVX2
	eff_t c1 = mul_avx(a, b);
	printf("%llx\n", c1);
#endif

    // eff_t c2 = mul(a, 2);
    // printf("%llx\n", c2);
	printf("all good\n");
	return 1;
}
