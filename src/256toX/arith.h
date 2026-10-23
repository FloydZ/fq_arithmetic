#ifndef FQ_ARITHMETIC_GF256TOX_ARITH_H
#define FQ_ARITHMETIC_GF256TOX_ARITH_H

#include <stdint.h>
#include "../256/arith.h"
#include "../16/arith.h"

#define EXTENSION_DEGREE 48
#define NR_REDUCTION_ENTRIES 4

// describes the reduction polynomial:
// 	[0,1,2,5] with the coefficients [9,1,1,1], becomes:
//      X^(EXTENSION_DEGREE) + X^5 + X^2 + X + 9
// NOTE: a polynomial with only 0/1 coefficients (e.g. the former constant
//  term 1) is never irreducible over GF(256) for EXTENSION_DEGREE = 48, as it
//  splits into gcd(48, 8) factors.
static const uint8_t __gf256toX_red_pol[] __attribute__((aligned(32))) = {0,1,2,5};
static const uint8_t __gf256toX_red_coef[] __attribute__((aligned(32))) = {9,1,1,1};
static const uint8_t __gf256toX_red_pol_full[][EXTENSION_DEGREE] __attribute__((aligned(32))) = {
        {0,0,0,0,0,0,0,0,0,0,1}
};
typedef uint8_t  gf256toX[EXTENSION_DEGREE];


static inline void gf256toX_set(gf256toX res,
                                const gf256toX a) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i];
    }
}

static inline void gf256toX_zero(gf256toX res) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = 0;
    }
}

static inline void gf256toX_add(gf256toX res,
                                const gf256toX a,
                                const gf256toX b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i] ^ b[i];
    }
}

static inline void gf256toX_scalar_mul(gf256toX res,
                                       const gf256toX a,
                                       const gf256 b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = gf256_mul(a[i], b);
    }
}

static inline void gf256toX_add_scalar_mul(gf256toX res,
                                           const gf256toX a,
                                           const gf256 b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] ^= gf256_mul(a[i], b);
    }
}

static inline void gf256toX_mul_gf256(gf256toX res,
                                      const gf256toX a,
                                      const gf256 b) {
    gf256toX_scalar_mul(res, a, b);
}

static inline void gf256toX_mul_gf16(gf256toX res,
                                     const gf256toX a,
                                     const gf16 b) {
    gf256toX_scalar_mul(res, a, gf256_expand_tab[b]);
}

static inline void gf256toX_mul(gf256toX res,
                                const gf256toX a,
                                const gf256toX b) {
    gf256 product[2*EXTENSION_DEGREE - 1u] = {0};
    gf256toX tmp;
    gf256toX_scalar_mul(product, a, b[0]);

    // mul
    for(uint32_t i = 1 ; i < EXTENSION_DEGREE ; i++) {
        gf256toX_scalar_mul(tmp, a, b[i]);
        gf256toX_add(product + i, product + i, tmp);
    }

    // Red
    for(uint32_t i = 2*EXTENSION_DEGREE - 2; i>=EXTENSION_DEGREE ; i--) {
        for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
            product[i - (EXTENSION_DEGREE - __gf256toX_red_pol[j])] ^= gf256_mul(product[i], __gf256toX_red_coef[j]);
        }
    }

    // copy back
    for(uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = product[i];
    }
}

#endif