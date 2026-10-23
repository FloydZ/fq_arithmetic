#include <stdint.h>
#include <assert.h>

/// Define your Modulus
#define MODULUS 11

/// you have to define the base type `ff_t`. This type must be capable of 
/// holding ceil(log2(MODULUS)) bits. Additionally `eff_t` must be defined 
/// which must capable of holding 2*sizeof(ff_t) bytes. This is needed for 
/// carry computations.
typedef uint8_t ff_t;
typedef uint16_t eff_t;


#define LOG2(n) ( \
    (n) < 2 ? 0 : \
    (n) < 4 ? 1 : \
    (n) < 8 ? 2 : \
    (n) < 16 ? 3 : \
    (n) < 32 ? 4 : \
    (n) < 64 ? 5 : \
    (n) < 128 ? 6 : \
    (n) < 256 ? 7 : \
    (n) < 512 ? 8 : \
    (n) < 1024 ? 9 : \
    (n) < 2048 ? 10 : \
    (n) < 4096 ? 11 : \
    (n) < 8192 ? 12 : \
    (n) < 16384 ? 13 : \
    (n) < 32768 ? 14 : \
    (n) < 65536 ? 15 : \
    (n) < 131072 ? 16 : \
    (n) < 262144 ? 17 : \
    (n) < 524288 ? 18 : \
    (n) < 1048576 ? 19 : \
    (n) < 2097152 ? 20 : \
    (n) < 4194304 ? 21 : \
    (n) < 8388608 ? 22 : \
    (n) < 16777216 ? 23 : \
    (n) < 33554432 ? 24 : \
    (n) < 67108864 ? 25 : \
    (n) < 134217728 ? 26 : \
    (n) < 268435456 ? 27 : \
    (n) < 536870912 ? 28 : \
    (n) < 1073741824 ? 29 : \
    (n) < 2147483648U ? 30 : 31 \
)

/// \param a[in]: value to reduce
/// \return a+b % q
ff_t gf_mod(const eff_t a) {
	return a % MODULUS;
}

/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b % q
ff_t gf_add(const ff_t a,
            const ff_t b) {
	return ((eff_t)a + b) % MODULUS;
}

/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b % MODULUS
ff_t gf_sub(const ff_t a,
              const ff_t b) {
	return ((eff_t)a - b) % MODULUS;
}


/// \param a[in]: value
/// \return a**2 % MODULUS
ff_t gf_sqr(const ff_t a) {
	return ((eff_t)a * a) % MODULUS;
}

/// \param a[in]: first modulus
/// \param b[in]: second modulud
/// \return a*b % MODULUS
ff_t gf_mul(const ff_t a,
            const ff_t b) {
	return ((eff_t)a * b) % MODULUS;
}

/// \param p[in]: the modules
/// \return: p_inv * p == -1 (mod 2**(LOG2(MODULUS) * 8))
uint64_t gf_montgomery_init(const uint64_t p) {
    uint64_t p_inv = 1;
    for (int i = 0; i < LOG2(MODULUS); i++) {
          p_inv *= 2ull + p_inv * p;
    }
    assert(p_inv * p == (uint64_t)(-1));
    p_inv %= 1ull << (LOG2(MODULUS) * 8);
    return p_inv;
}

/// \param a[in]: \in [0, 2*p)
/// \param b[in]: \in [0, 2*p)
/// \param p[in]: prime/modulus
/// \return: p_inv * p == -1 (mod 2**(LOG2(MODULUS) * 8))
/// \return a*b \in [0, p)
uint64_t gf_montgomery_mul(const uint64_t a,
                           const uint64_t b,
                           const uint64_t p_inv) {
    const eff_t val = (eff_t)a * b;
    uint64_t res = (val + ((ff_t)(val) * p_inv & ((1ULL << (LOG2(MODULUS) * 8)) - 1)) * (eff_t)(MODULUS)) >> (LOG2(MODULUS) * 8);
    res = gf_mod(res);
    return res;
}

/// \param b[in]: input value
/// \return: b_inv = b << (sizeof(ff_t) * 8)
uint64_t gf_barrett_init(const uint64_t b) {
    return (eff_t)b << (sizeof(ff_t)*8)/MODULUS;
}

/// \param a[in]: \in [0, p)
/// \param b[in]: \in [0, p)
/// \param p[in]: prime or modulus
/// \return: b_inv = b << (sizeof(ff_t) * 8)
/// \return a*b \in [0, p)
uint64_t gf_barrett_mul(const uint64_t a,
                        const uint64_t b,
                        const uint64_t b_inv) {
    const uint64_t Q = ((eff_t)b_inv * a) >> (sizeof(ff_t) * 8);
    return gf_mod(b * a - Q * MODULUS);
}
