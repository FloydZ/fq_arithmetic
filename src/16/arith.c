#include <stdint.h>
/// Representation: two F16 elements are stored in a single `uint8_t`
// 	gf16 := gf2[x]/ (x^4+x+1)

typedef uint8_t ff_t;

/// M[i, j] = 16**i * j mod 16 for i = row, j = column
const uint8_t __gf16_mulbase[128] __attribute__((aligned(32))) = {
	0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f, 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f, 
	0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e, 0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d, 0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e, 0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d, 
	0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f, 0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09, 0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f, 0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09, 
	0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d, 0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01, 0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d, 0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01, 
};

const uint8_t __gf16_mulbase2[64] = {
	0x00,0x00,0x00,0x00,
	0x01,0x02,0x04,0x08,
	0x02,0x04,0x08,0x03,
	0x03,0x06,0x0c,0x0b,
	0x04,0x08,0x03,0x06,
	0x05,0x0a,0x07,0x0e,
	0x06,0x0c,0x0b,0x05,
	0x07,0x0e,0x0f,0x0d,
	0x08,0x03,0x06,0x0c,
	0x09,0x01,0x02,0x04,
	0x0a,0x07,0x0e,0x0f,
	0x0b,0x05,0x0a,0x07,
	0x0c,0x0b,0x05,0x0a,
	0x0d,0x09,0x01,0x02,
	0x0e,0x0f,0x0d,0x09,
	0x0f,0x0d,0x09,0x01,
};

const uint8_t mult_table[256] = {
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f, 
    0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d, 
    0x00,0x03,0x06,0x05,0x0c,0x0f,0x0a,0x09,0x0b,0x08,0x0d,0x0e,0x07,0x04,0x01,0x02, 
    0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f,0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09, 
    0x00,0x05,0x0a,0x0f,0x07,0x02,0x0d,0x08,0x0e,0x0b,0x04,0x01,0x09,0x0c,0x03,0x06, 
    0x00,0x06,0x0c,0x0a,0x0b,0x0d,0x07,0x01,0x05,0x03,0x09,0x0f,0x0e,0x08,0x02,0x04, 
    0x00,0x07,0x0e,0x09,0x0f,0x08,0x01,0x06,0x0d,0x0a,0x03,0x04,0x02,0x05,0x0c,0x0b, 
    0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d,0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01, 
    0x00,0x09,0x01,0x08,0x02,0x0b,0x03,0x0a,0x04,0x0d,0x05,0x0c,0x06,0x0f,0x07,0x0e, 
    0x00,0x0a,0x07,0x0d,0x0e,0x04,0x09,0x03,0x0f,0x05,0x08,0x02,0x01,0x0b,0x06,0x0c, 
    0x00,0x0b,0x05,0x0e,0x0a,0x01,0x0f,0x04,0x07,0x0c,0x02,0x09,0x0d,0x06,0x08,0x03, 
    0x00,0x0c,0x0b,0x07,0x05,0x09,0x0e,0x02,0x0a,0x06,0x01,0x0d,0x0f,0x03,0x04,0x08, 
    0x00,0x0d,0x09,0x04,0x01,0x0c,0x08,0x05,0x02,0x0f,0x0b,0x06,0x03,0x0e,0x0a,0x07, 
    0x00,0x0e,0x0f,0x01,0x0d,0x03,0x02,0x0c,0x09,0x07,0x06,0x08,0x04,0x0a,0x0b,0x05, 
    0x00,0x0f,0x0d,0x02,0x09,0x06,0x04,0x0b,0x01,0x0e,0x0c,0x03,0x08,0x07,0x05,0x0a, 
};

const uint8_t gf16_inverse_tab[16] = {
    0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8
};

uint8_t inv(const uint8_t a) {
    return gf16_inverse_tab[a];
}

uint8_t add(const uint8_t a, const uint8_t b) { 
	return a ^ b;
}
uint8_t sub(const uint8_t a, const uint8_t b) {
	return a ^ b;
}
uint8_t mul(const uint8_t a, const uint8_t b) { 
	return mult_table[(a&0xffff)*16 + (b&0xffff)] ^
		   (mult_table[(a >> 4)*16   + (b >> 4u)] << 4);
}
uint8_t addmul(const uint8_t a, const uint8_t b, const uint8_t c) { 
	return a^mul(b, c);
}

// NOTE: b must be a single number
uint8_t mul2(const uint8_t a, const uint8_t b) {
	const uint8_t *p = __gf16_mulbase2 + b*4;
	const uint8_t tmp1 = ((a & 0x11) >> 0) * p[0];
	const uint8_t tmp2 = ((a & 0x22) >> 1) * p[1];
	const uint8_t tmp3 = ((a & 0x44) >> 2) * p[2];
	const uint8_t tmp4 = ((a & 0x88) >> 3) * p[3];
	return tmp1 ^ tmp2 ^ tmp3 ^ tmp4;
}

#ifdef USE_AVX2
#include <immintrin.h>


/// Full multiplication
/// NOTE: assumes that in every byte the two nibbles are the same in b
/// \return a*b \in \F_16 for all 64 nibbles in the
__m256i mul_simd(const __m256i a, const __m256i b) {
    const __m256i mask_lvl2 = _mm256_load_si256((__m256i const *) (__gf16_mulbase +   32));
    const __m256i mask_lvl3 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2));
    const __m256i mask_lvl4 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3));
    const __m256i zero = _mm256_setzero_si256();

    __m256i low_lookup = b;
    __m256i high_lookup = _mm256_slli_epi16(low_lookup, 4);
    __m256i tmp1l = _mm256_slli_epi16(a, 7);
    __m256i tmp2h = _mm256_slli_epi16(a, 3);
    __m256i tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    __m256i tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    __m256i tmp = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    __m256i tmp1;

    /// 1
    low_lookup = _mm256_shuffle_epi8(mask_lvl2, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 6);
    tmp2h = _mm256_slli_epi16(a, 2);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 2
    low_lookup = _mm256_shuffle_epi8(mask_lvl3, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 5);
    tmp2h = _mm256_slli_epi16(a, 1);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 3
    low_lookup = _mm256_shuffle_epi8(mask_lvl4, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 4);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, a );
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);
    return tmp;
}
//// Caution: multabs are different from ssse3 version
//// ssse3:  [multab_low] [ multab_high ]
////         <-   16  ->  <-    16     ->
//// avx2:   [         multab_low       ]
////         <---        32          --->
void gf16v_generate_multab_16_avx2(__m256i *multabs,
									__m128i a,
									unsigned len ) {
    __m256i broadcast_x1 = _mm256_set_epi8( 0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16 );
    __m256i broadcast_x2 = _mm256_set_epi8( 0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16 );
    __m256i broadcast_x4 = _mm256_set_epi8( 0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16 );
    __m256i broadcast_x8 = _mm256_set_epi8( 0,0,0,0, 0,0,0,0,  -16,-16,-16,-16, -16,-16,-16,-16,  0,0,0,0, 0,0,0,0,  -16,-16,-16,-16, -16,-16,-16,-16 );

    //__m256i bx1 = _mm256_inserti128_si256( _mm256_castsi128_si256(a), a , 1 );
    __m256i bx1 = _mm256_setr_m128i( a , a );
    __m256i bx2 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32 ))   , bx1 );
    __m256i bx4 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2 )) , bx1 );
    __m256i bx8 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3 )) , bx1 );


    multabs[0] =  _mm256_shuffle_epi8(bx1,broadcast_x1) ^ _mm256_shuffle_epi8(bx2,broadcast_x2)
                  ^ _mm256_shuffle_epi8(bx4,broadcast_x4) ^ _mm256_shuffle_epi8(bx8,broadcast_x8);
    for(uint32_t i=1;i<len;i++) {
        bx1 = _mm256_srli_si256(bx1, 1);
        bx2 = _mm256_srli_si256(bx2, 1);
        bx4 = _mm256_srli_si256(bx4, 1);
        bx8 = _mm256_srli_si256(bx8, 1);

        multabs[i] = _mm256_shuffle_epi8(bx1,broadcast_x1) ^
					 _mm256_shuffle_epi8(bx2,broadcast_x2) ^
					 _mm256_shuffle_epi8(bx4,broadcast_x4) ^
					 _mm256_shuffle_epi8(bx8,broadcast_x8);
    }
}

// 
void mat_simd_16x16(uint8_t *c, 
					const uint8_t *matA,
					const __m256i * multab_b){
    __m256i mask_f = _mm256_set1_epi8(0xf);
    __m256i ma0123 = _mm256_loadu_si256((const __m256i*) (matA) );    // 4 rows
    __m256i ma4567 = _mm256_loadu_si256((const __m256i*) (matA+32) ); // 4 rows
    __m256i mb02 = _mm256_permute2x128_si256(multab_b[0], multab_b[2] , 0x20 );
    __m256i mb13 = _mm256_permute2x128_si256(multab_b[1], multab_b[3] , 0x20 );
    __m256i mb46 = _mm256_permute2x128_si256(multab_b[4], multab_b[6] , 0x20 );
    __m256i mb57 = _mm256_permute2x128_si256(multab_b[5], multab_b[7] , 0x20 );
    __m256i ma0123l = ma0123&mask_f;
    __m256i ma0123h = _mm256_srli_epi16(ma0123,4)&mask_f;
    __m256i ma4567l = ma4567&mask_f;
    __m256i ma4567h = _mm256_srli_epi16(ma4567,4)&mask_f;
    __m256i ma02 = _mm256_unpacklo_epi64(ma0123l, ma0123h);
    __m256i ma13 = _mm256_unpackhi_epi64(ma0123l, ma0123h);
    __m256i ma46 = _mm256_unpacklo_epi64(ma4567l, ma4567h);
    __m256i ma57 = _mm256_unpackhi_epi64(ma4567l, ma4567h);
    __m256i r = _mm256_shuffle_epi8(mb02,ma02) ^
				_mm256_shuffle_epi8(mb13,ma13) ^
				_mm256_shuffle_epi8(mb46,ma46) ^
				_mm256_shuffle_epi8(mb57,ma57);

    __m256i ma89ab = _mm256_loadu_si256((const __m256i*) (matA+64) ); // 4 rows
    __m256i macdef = _mm256_loadu_si256((const __m256i*) (matA+96) ); // 4 rows
    __m256i mb8a = _mm256_permute2x128_si256(multab_b[8], multab_b[10], 0x20);
    __m256i mb9b = _mm256_permute2x128_si256(multab_b[9], multab_b[11], 0x20);
    __m256i mbce = _mm256_permute2x128_si256(multab_b[12], multab_b[14], 0x20);
    __m256i mbdf = _mm256_permute2x128_si256(multab_b[13], multab_b[15], 0x20);
    __m256i ma89abl = ma89ab&mask_f;
    __m256i ma89abh = _mm256_srli_epi16(ma89ab,4)&mask_f;
    __m256i macdefl = macdef&mask_f;
    __m256i macdefh = _mm256_srli_epi16(macdef,4)&mask_f;
    __m256i ma8a = _mm256_unpacklo_epi64(ma89abl,ma89abh);
    __m256i ma9b = _mm256_unpackhi_epi64(ma89abl,ma89abh);
    __m256i mace = _mm256_unpacklo_epi64(macdefl,macdefh);
    __m256i madf = _mm256_unpackhi_epi64(macdefl,macdefh);
    r ^= _mm256_shuffle_epi8(mb8a,ma8a) ^
		 _mm256_shuffle_epi8(mb9b,ma9b) ^
		 _mm256_shuffle_epi8(mbce,mace) ^
		 _mm256_shuffle_epi8(mbdf,madf);

    __m128i rr = _mm256_castsi256_si128(r) ^ _mm256_extractf128_si256(r, 1);
    __m128i rr2 = rr ^ _mm_srli_si128(_mm_slli_epi16(rr, 4), 8);

    uint8_t temp[16] __attribute__((aligned(16)));
    _mm_store_si128( (__m128i*)temp , rr2 );
    for(int i=0;i<8;i++) c[i] = temp[i];
}

static void gf16mat_prod_16x16_avx2( uint8_t * c , const uint8_t * matA , const uint8_t * b )
{
    __m256i multabs[16];
    uint8_t temp[16] __attribute__((aligned(16)));
    for(int i=0;i<8;i++) temp[i]=b[i];
    __m128i x0 = _mm_load_si128((const __m128i*)temp);
    xmmx2_t xx = gf16v_split_16to32_sse2( x0 );
    gf16v_generate_multab_16_avx2( multabs    , xx.v0 , 16 );
    mat_simd_16x16(c, matA, multabs);
}



/// around 30-40% faster than the normal approach
/// Implements the full 16x16 matrix-matrix multiplication
/// NOTE: eventhough this implementation assumes 16columns on 16 elements input matrix, its ok to input
///         matrices of dimension 16x15 (e.g. only 15 elements per column).
/// \param c out matrix
/// \param a in matrix
/// \param b in matrix
void gf16mat_prod_16x16_avx2_wrapper_v2(uint8_t *__restrict c,
                                        const uint8_t *__restrict__ a,
                                        unsigned int height_A,
                                        const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    const uint32_t *b32 = (const uint32_t *) b;
    uint64_t *c64 = (uint64_t *) c;

    const uint32_t mask = 0x0f0f0f0f;
    const __m256i Amask = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
    const __m256i perm  = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);

    __m256i tmp,acc;
    const __m256i Al1 = _mm256_i32gather_epi32((const int *)a32 +  0u, Amask, 4u);
    const __m256i Al2 = _mm256_i32gather_epi32((const int *)a32 + 16u, Amask, 4u);
    const __m256i Ah1 = _mm256_i32gather_epi32((const int *)a32 +  1u, Amask, 4u);
    const __m256i Ah2 = _mm256_i32gather_epi32((const int *)a32 + 17u, Amask, 4u);

    for (uint32_t i = 0; i < height_A; ++i) {
        // load first 4bytes
        uint32_t d11 = b32[2*i+0] & mask;
        uint32_t d12 = (b32[2*i+0] >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); //0b10101010

        // load the next 4 bytes
        uint32_t d21 = b32[2*i+1] & mask;
        uint32_t d22 = (b32[2*i+1] >> 4) & mask;

        __m256i B21 = _mm256_set1_epi32(d21);
        __m256i B22 = _mm256_set1_epi32(d22);
        B21 = _mm256_shuffle_epi8(B21, perm);
        B22 = _mm256_shuffle_epi8(B22, perm);
        const __m256i B2 = _mm256_blend_epi32(B21, B22, 0xaa);


        acc = gf16_mult_avx_compressed_2(Al1, B1);
        tmp = gf16_mult_avx_compressed_2(Al2, B2);
        __m256i tmp1 = _mm256_xor_si256(acc, tmp);

        acc = gf16_mult_avx_compressed_2(Ah1, B1);
        tmp = gf16_mult_avx_compressed_2(Ah2, B2);
        __m256i tmp2 = _mm256_xor_si256(acc, tmp);

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        c64[i] = _mm256_extract_epi64(ret, 0);
    }
}

/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2` which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
void gf16mat_prod_le8xle8_avx2_wrapper_v2(uint8_t *__restrict c,
                                          const uint8_t *__restrict__ a,
                                          const unsigned int nr_cols_B,
                                          const unsigned int column_A_bytes,
                                          const unsigned int nr_cols_A,
                                          const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    uint64_t *c64 = (uint64_t *) c;
    assert(column_A_bytes <= 8);
    assert(nr_cols_A <= 8);

    /// make sure this number is smaller/equal than 4.
    const uint32_t nr_bytes_B_col = nr_cols_A>>1;
    const uint32_t mask = (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const __m256i Amask = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i zero = _mm256_setzero_si256();

    const uint32_t n = nr_cols_A;
    const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 < n ? -1 : 0, 2 < n ? -1 : 0, 3 < n ? -1 : 0, 4 < n ? -1 : 0, 5 < n ? -1 : 0, 6 < n ? -1 : 0, 7 < n ? -1 : 0);
    __m256i tmp1, tmp2, Al1, Ah1;

    const uint32_t BMask = (1u << (8u*nr_bytes_B_col)) - 1u;
    Al1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 0u, Amask, gather_mask, 4u);
    if (column_A_bytes > 4)
        Ah1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 1u, Amask, gather_mask, 4u);


    uint64_t rdata;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B-1) {
            for (uint32_t i = 0; i < (nr_cols_B * nr_bytes_B_col) % 4; i++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + i] << (i * 8);
            }
        }
         b_data = *((uint32_t *)(b+(nr_bytes_B_col * i))) & BMask;
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16_mult_avx_compressed_2(Al1, B1);
        if (column_A_bytes > 4)
            tmp2 = gf16_mult_avx_compressed_2(Ah1, B1);
        else
            tmp2 = zero;

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        rdata = _mm256_extract_epi64(ret, 0);
        if (i < (nr_cols_B-1))
            *((uint64_t *) (c + i * column_A_bytes)) = rdata;
    }


    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}


/// this function differs from `gf16mat_prod_le8xle8_avx2_wrapper_v2`, by allowing the matrix A to have a column length
/// greater than 8.
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
void gf16mat_prod_gr8xle8_avx2_wrapper(uint8_t *__restrict c,
                                       const uint8_t *__restrict__ a,
                                       const unsigned int nr_cols_B,
                                       const unsigned int column_A_bytes,
                                       const unsigned int nr_cols_A,
                                       const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes > 8);
    assert(column_A_bytes < 12);
    assert(column_A_bytes == 10 || column_A_bytes == 11);

    const uint32_t nr_bytes_B_col = nr_cols_A>>1;
    const uint32_t mask = nr_bytes_B_col%4==0 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    __m256i tmp1, tmp2, tmp3;

    // old and original code
    //const __m256i Ashfl = _mm256_setr_epi32(0, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u);
    //const __m256i Ashft = _mm256_setr_epi32(0, 8*off, 8*off, 8*off, 8*off, 8*off, 8*off, 8*off);
    //const __m256i Abla = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    //const __m256i A3ma = _mm256_set1_epi32(((uint32_t)(1u)<<(8*(column_A_bytes % 4))) - 1u);

    //__m256i A1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 0u, Amask, gather_mask, 4u);
    //__m256i A11 = _mm256_sllv_epi32(A1, Ashft);
    //__m256i A12 = _mm256_srli_epi32(A1, 8*(column_A_bytes % 4));

    //__m256i A2 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 1u, Amask, gather_mask, 4u);
    //__m256i A21 = _mm256_sllv_epi32(A2, Ashft);
    //__m256i A22 = _mm256_srli_epi32(A2, 8*(column_A_bytes % 4));

    //__m256i A3 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 2u, Amask, gather_mask, 4u);
    //__m256i A31 = _mm256_sllv_epi32(_mm256_and_si256(A3ma, A3), Ashft);
    //__m256i A32 = _mm256_permutevar8x32_epi32(_mm256_srli_epi32(A3, 8*(column_A_bytes % 4)), Abla);

    __m256i Asel, Ashup, Ashdo, Ashfl, Ash, Abla, A1, A2, A3;
    Abla = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    Asel = _mm256_setr_epi32(0, 3, 6, 9, 11, 14, 17, 20);
    Ashup = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
    Ashdo = _mm256_setr_epi32(0, 24, 16, 8, 0, 24, 16, 8);
    Ashfl = _mm256_setr_epi32(0, (1u << 8u) - 1u, (1u << 16u) - 1u, (1u << 24u) - 1u, 0, (1u << 8u) - 1u,(1u << 16u) - 1u, (1u << 24u) - 1u);
    A1 = _mm256_i32gather_epi32((const int *)a32 + 0u, Asel, 4u);
    __m256i A11 = _mm256_sllv_epi32(A1, Ashup);
    __m256i A12 = _mm256_srlv_epi32(A1, Ashdo);

    A2 = _mm256_i32gather_epi32((const int *)a32 + 1u, Asel, 4u);
    __m256i A21 = _mm256_sllv_epi32(A2, Ashup);
    __m256i A22 = _mm256_srlv_epi32(A2, Ashdo);

    A3 = _mm256_i32gather_epi32((const int *)a32 + 2u, Asel, 4u);
    __m256i A31 = _mm256_sllv_epi32(A3, Ashup);
    __m256i A32 = _mm256_permutevar8x32_epi32(A3, Abla);
    A32 = _mm256_srlv_epi32(A32, Ashdo);

    A1 = _mm256_blendv_epi8(A11, A32, Ashfl);
    A2 = _mm256_blendv_epi8(A21, A12, Ashfl);
    A3 = _mm256_blendv_epi8(A31, A22, Ashfl);

    //
    if (column_A_bytes == 10) {
        Asel = _mm256_setr_epi32(0, 2, 5, 7, 10, 12, 15, 17);
        Ash = _mm256_setr_epi32(0, 16, 0, 16, 0, 16, 0, 16);
        Ashfl = _mm256_setr_epi32(0, (-1u)<<16u, 0, (-1u)<<16u, 0, (-1u)<<16u,0, (-1u)<<16u);
        Abla = _mm256_setr_epi32(1, 1, 2, 3, 4, 5, 6, 7);

        A1 = _mm256_i32gather_epi32((const int *)a32 + 0u, Asel, 4u);
        __m256i A11 = _mm256_srlv_epi32(A1, Ash);
        //__m256i A12 = _mm256_sllv_epi32(A1, Ash);

        A2 = _mm256_i32gather_epi32((const int *)a32 + 1u, Asel, 4u);
        __m256i A21 = _mm256_srlv_epi32(A2, Ash);
        __m256i A22 = _mm256_permutevar8x32_epi32(_mm256_sllv_epi32(A2, Ash), Abla);

        A3 = _mm256_i32gather_epi32((const int *)a32 + 2u, Asel, 4u);
        __m256i A31 = _mm256_srlv_epi32(A3, Ash);
        __m256i A32 = _mm256_sllv_epi32(A3, Ash);

        A1 = _mm256_blendv_epi8(A11, A22, Ashfl);
        A2 = _mm256_blendv_epi8(A21, A32, Ashfl);
        A3 = A31;
    }

    uint32_t rdata2;

    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = *((uint32_t *) (b + (nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16_mult_avx_compressed_2(A1, B1);
        tmp2 = gf16_mult_avx_compressed_2(A2, B1);
        tmp3 = gf16_mult_avx_compressed_2(A3, B1);

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        uint64_t rdata = _mm256_extract_epi64(ret, 0);
        *((uint64_t *) (c + i * column_A_bytes)) = rdata;

        rdata2 = _mm256_extract_epi32(gf16_hadd_avx2_32_v2(tmp3), 0);
        if (i < nr_cols_B - 1) {
            *((uint32_t *) (c + i * column_A_bytes + 8)) = rdata2;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes%4; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + 8 + i] = (uint8_t)(rdata2>>(i*8));
    }
}


/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2` which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
void gf16mat_prod_le4xle4_avx2_wrapper(uint8_t *__restrict c,
                                       const uint8_t *__restrict__ a,
                                       const unsigned int nr_cols_B,
                                       const unsigned int column_A_bytes,
                                       const unsigned int nr_cols_A,
                                       const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes <= 4);
    assert(nr_cols_A <= 16);
    __m256i Al1, Al2, tmp1, tmp2;

    const __m256i zero = _mm256_setzero_si256();
    const uint32_t nr_bytes_B_col = nr_cols_A >> 1;
    const uint32_t mask = nr_bytes_B_col == 4 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const uint32_t n = ((nr_cols_A + 1) * column_A_bytes) >> 2;

    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                          3,3, 3, 3, 3, 3, 3);
    if (column_A_bytes == 4) {
        Al1 = _mm256_loadu_si256((__m256i *)a);
        if (nr_cols_A > 8) {
            Al2 = _mm256_loadu_si256((__m256i *)(a + 32));
        }
    } else if (column_A_bytes == 3) {
        const __m256i Amask = _mm256_setr_epi32(0, 1, 2, 2, 3, 4, 5, 5);

        const __m256i Ashup = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
        const __m256i Ashdo = _mm256_setr_epi32(24, 16, 8, 8, 24, 16, 8, 8);
        const __m256i Abla = _mm256_setr_epi32(0, 0, 1, 2, 0, 4, 5, 6);
        const __m256i Ashfl = _mm256_setr_epi32(0, (1 << 8) - 1, (1 << 16) - 1, (1u << 24) - 1, 0, (1 << 8) - 1,
                                                (1 << 16) - 1, (1u << 24) - 1);
        const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
                                                      3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
                                                      6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

        Al1 = _mm256_mask_i32gather_epi32(zero, (const int *)(a32 + 0u), Amask, gather_mask, 4u);
        __m256i A11 = _mm256_sllv_epi32(Al1, Ashup);
        __m256i A12 = _mm256_srlv_epi32(Al1, Ashdo);
        __m256i A13 = _mm256_permutevar8x32_epi32(A12, Abla);

        Al1 = _mm256_blendv_epi8(A11, A13, Ashfl);

        if (nr_cols_A > 8) {
            Al2 = _mm256_mask_i32gather_epi32(zero, (const int *)(a32 + 6u), Amask, gather_mask, 4u);
            __m256i A21 = _mm256_sllv_epi32(Al2, Ashup);
            __m256i A22 = _mm256_srlv_epi32(Al2, Ashdo);
            __m256i A23 = _mm256_permutevar8x32_epi32(A22, Abla);

            Al2 = _mm256_blendv_epi8(A21, A23, Ashfl);
        }
    } else if (column_A_bytes == 2) {
        const __m256i Amask = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

        const __m256i Ashdo = _mm256_set1_epi32(8);
        const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
                                                      3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
                                                      6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

        Al1 = _mm256_mask_i32gather_epi32(zero, (const int *) a32 + 0u, Amask, gather_mask, 4u);
        Al1 = _mm256_srlv_epi32(Al1, Ashdo);
    }



    uint32_t rdata;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B - 1) {
            for (uint32_t j = 0; j < (nr_cols_B * nr_bytes_B_col) % 4; j++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + j] << (j * 8);
            }
        }
        b_data = *((uint32_t *)(b+(nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        if (nr_cols_A > 8) {
            b_data = *((uint32_t *)(b+(nr_bytes_B_col * i + 4)));
            uint32_t d11 = b_data & mask;
            uint32_t d12 = (b_data >> 4) & mask;

            __m256i B11 = _mm256_set1_epi32(d11);
            __m256i B12 = _mm256_set1_epi32(d12);
            B11 = _mm256_shuffle_epi8(B11, perm);
            B12 = _mm256_shuffle_epi8(B12, perm);
            const __m256i B2 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

            tmp1 = gf16_mult_avx_compressed_2(Al1, B1);
            tmp2 = gf16_mult_avx_compressed_2(Al2, B2);
            tmp1 = _mm256_xor_si256(tmp1, tmp2);
        } else {
            tmp1 = gf16_mult_avx_compressed_2(Al1, B1);
        }

        rdata = _mm256_extract_epi32(gf16_hadd_avx2_32_v2(tmp1), 0);
        if (i < nr_cols_B - 1) {
            *((uint32_t *) (c + i * column_A_bytes)) = rdata;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}



/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2` which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
void gf16mat_prod_le8xle8_avx2_wrapper_v3(uint8_t *__restrict c,
                                          const uint8_t *__restrict__ a,
                                          const unsigned int nr_cols_B,
                                          const unsigned int column_A_bytes,
                                          const unsigned int nr_cols_A,
                                          const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    uint64_t *c64 = (uint64_t *) c;
    assert(column_A_bytes < 8);
    assert(column_A_bytes > 4);
    assert(column_A_bytes == 5);
    assert(nr_cols_A <= 16);
    __m256i A1, A2, A3, A4, tmp1, tmp2;

    const __m256i zero = _mm256_setzero_si256();
    const uint32_t nr_bytes_B_col = nr_cols_A >> 1;
    const uint32_t mask =nr_bytes_B_col%4==0 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const uint32_t n = ((nr_cols_A + 1) * column_A_bytes) >> 2;
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);

    const __m256i Amask = _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 8);
    const __m256i Ashdo = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
    const __m256i Ashup = _mm256_setr_epi32(0, 24, 16, 8, 0, 24, 16, 8);
    const __m256i Ashfl = _mm256_setr_epi32(0, ((1<<8)-1)<<24, ((1<<16)-1)<<16, ((1<<24)-1)<<8,
                                            0, ((1<<8)-1)<<24, ((1<<16)-1)<<16, ((1<<24)-1)<<8);
    const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
                                                  3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
                                                  6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

    A1 = _mm256_i32gather_epi32((const int *)(a32 + 0u), Amask, 4u);
    A2 = _mm256_i32gather_epi32((const int *)(a32 + 1u), Amask, 4u);
    __m256i A11 = _mm256_srlv_epi32(A1, Ashdo);
    __m256i A22 = _mm256_sllv_epi32(A2, Ashup);
    A2 = _mm256_srlv_epi32(A2, Ashdo);
    A1 = _mm256_blendv_epi8(A11, A22, Ashfl);

    if (nr_cols_A > 8) {
        A3 = _mm256_i32gather_epi32((const int *)(a32 + 10u), Amask, 4u);
        A4 = _mm256_i32gather_epi32((const int *)(a32 + 11u), Amask, 4u);
        __m256i A31 = _mm256_srlv_epi32(A3, Ashdo);
        __m256i A42 = _mm256_sllv_epi32(A4, Ashup);
        A4 = _mm256_srlv_epi32(A4, Ashdo);
        A3 = _mm256_blendv_epi8(A31, A42, Ashfl);
    }

    uint64_t rdata;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B - 1) {
            for (uint32_t j = 0; j < (nr_cols_B * nr_bytes_B_col) % 4; j++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + j] << (j * 8);
            }
        }

        b_data = *((uint32_t *)(b+(nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16_mult_avx_compressed_2(A1, B1);
        tmp2 = gf16_mult_avx_compressed_2(A2, B1);

        if (nr_cols_A > 8) {
            b_data = *((uint32_t *)(b+(nr_bytes_B_col * i + 4)));
            d11 = b_data & mask;
            d12 = (b_data >> 4) & mask;

            B11 = _mm256_set1_epi32(d11);
            B12 = _mm256_set1_epi32(d12);
            B11 = _mm256_shuffle_epi8(B11, perm);
            B12 = _mm256_shuffle_epi8(B12, perm);
            const __m256i B2 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010


            tmp1 ^= gf16_mult_avx_compressed_2(A3, B2);
            tmp2 ^= gf16_mult_avx_compressed_2(A4, B2);
        }

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        rdata = _mm256_extract_epi64(ret, 0);
        if (i < nr_cols_B - 1) {
            *((uint64_t *) (c + i * column_A_bytes)) = rdata;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}

#elif defined(USE_NEON)
uint8x16_t gf16v_mul_neon2( uint8x16_t a , uint8x16_t b ) {
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t bp = vdupq_n_u8(b);

    uint8x16_t al0 = a&mask_f;
    uint8x16_t ah0 = vshrq_n_u8( a , 4 );
	// mul
    poly8x16_t abl = vmulq_p8( al0 , bp );
    poly8x16_t abh = vmulq_p8( ah0 , bp );

    poly8x16_t rl = abl ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abl,4) );
    poly8x16_t rh = abh ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abh,4) );

    return vsliq_n_u8( rl , rh , 4 );
}
#else
#endif


/// A, B, C are in row-majow form
void matrix_mul(uint8_t *C, uint8_t *A, uint8_t *B,
				const size_t nrows_A, const size_t ncols_A, const size_t ncols_B) {
	for (uint32_t i = 0; i < nrows_A; i++) {
		for (uint32_t j = 0; j < ncols_B; j++) {
			ff_t t = 0;
			for (uint32_t k = 0; k < ncols_A; k++) {
				t = add(t, mul(A[i*ncols_A + k], B[ncols_A*k + j]));
			}

			C[i*ncols_B + j] = t;
		}
	}
}

int main() {
	return 1;
}
