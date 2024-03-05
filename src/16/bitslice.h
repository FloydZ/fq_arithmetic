
/// transforms 4 input registers into their bitsliced version written into out
/// Pictures are for a `uint16_t`
///        31     28     3      0
/// inX  = [ inX_7, ..., inX_0 ] , where each inX_i is a 4 bit element of F16
///         so there are in total 8 elements in an uint32_t
///         31,               4        3        2       1        0
/// out0 = [in3_7_0, ..., in0_1_0, in3_0_0, in2_0_0, in1_0_0, in0_0_0]
/// out1 = [in3_7_1, ..., in0_1_1, in3_0_1, in2_0_1, in1_0_1, in0_0_1]
/// out2 = [in3_7_2, ..., in0_1_2, in3_0_2, in2_0_2, in1_0_2, in0_0_2]
/// out3 = [in3_7_3, ..., in0_1_3, in3_0_3, in2_0_3, in1_0_3, in0_0_3]
///     where inX_Y_Z is the Z bit from inX_Y (NOTE: there are only 4 bits in each F16 element)
/// dont remove tmp
/// clobbert: out0-3 and tmp. in0-3 remain untouched
/// NOTE: inputs can be `uint8_t`...`uint64_t`
#define gf16v_bitslice(out0, out1, out2, out3, in0, in1, in2, in3, tmp) \
	out0 = in0 & 0x11111111; \
	out3 = in1 & 0x11111111; \
	out0 = out0 | (out3<<1u);\
	out3 = in2 & 0x11111111; \
	out0 = out0 | (out3<<2u);\
	out3 = in3 & 0x11111111; \
	out0 = out0 | (out3<<3u);\
							 \
	out1 = in1 & 0x22222222; \
	out3 = in0 & 0x22222222; \
	out1 = out1 | (out3>>1u);\
	out3 = in2 & 0x22222222; \
	out1 = out1 | (out3<<1u);\
	out3 = in3 & 0x22222222; \
	out1 = out1 | (out3<<2u);\
							 \
	out2 = in2 & 0x44444444; \
	out3 = in0 & 0x44444444; \
	out2 = out2 | (out3>>2u);\
	out3 = in1 & 0x44444444; \
	out2 = out2 | (out3>>1u);\
	out3 = in3 & 0x44444444; \
	out2 = out2 | (out3<<1u);\
							 \
	out3 = in3 & 0x88888888; \
	tmp  = in0 & 0x88888888; \
	out3 = out3 | (tmp>>3u); \
	tmp  = in1 & 0x88888888; \
	out3 = out3 | (tmp>>2u); \
	tmp  = in2 & 0x88888888; \
	out3 = out3 | (tmp>>1u);


/// clobbert: acc0-acc3,t0-t3
/// NOTE: only the lower 4 bits of b are used.
/// NOTE: t0-t3 are tmp registers/variables
/// acc += in*b
#define gf16v_madd_bitsclice(acc0, acc1, acc2, acc3, in0, in1, in2, in3, b, t0, t1, t2) \
	t0 = in0 ^ in3;    \
	t1 = in2 ^ in3;    \
	t2 = in1 ^ in2;    \
	if (b&1u){		   \
	acc0 = acc0 ^ in0; \
	acc1 = acc1 ^ in1; \
	acc2 = acc2 ^ in2; \
	acc3 = acc3 ^ in3; \
	}				   \
	if (b&2u) {		   \
	acc0 = acc0 ^ in3; \
	acc1 = acc1 ^ tmp0;\
	acc2 = acc2 ^ in1; \
	acc3 = acc3 ^ in2; \
	}				   \
 	if (b&4u){		   \
	acc0 = acc0 ^ in2; \
	acc1 = acc1 ^ tmp1;\
	acc2 = acc2 ^ tmp0;\
	acc3 = acc3 ^ in1; \
	}				   \
 	if (b&8u){		   \
	acc0 = acc0 ^ in1; \
	acc1 = acc1 ^ tmp2;\
	acc2 = acc2 ^ tmp1;\
	acc3 = acc3 ^ tmp0;\
	}



/// src: https://github.com/pqov/pqov-paper
/// same as `gf16v_bitslice`
/// TODO: does this even make sense?
#define gf16v_bitslice_avx2(out0, out1, out2, out3, in0, in1, in2, in3, tmp, mask1, mask2, mask4, mask8) \
	out0 = _mm256_and_si256(in0, mask1);                        \
	out3 = _mm256_and_si256(in1, mask1);                        \
	out0 = _mm256_or_si256(out0, _mm256_slli_epi64(out3, 1u));  \
	out3 = _mm256_and_si256(in2, mask1);                        \
	out0 = _mm256_or_si256(out0, _mm256_slli_epi64(out3, 2u));  \
	out3 = _mm256_and_si256(in3, mask1);                        \
	out0 = _mm256_or_si256(out0, _mm256_slli_epi64(out3, 3u));  \
							                                    \
	out1 = _mm256_and_si256(in1, mask2);                        \
	out3 = _mm256_and_si256(in0, mask2);                        \
	out1 = _mm256_or_si256(out1, _mm256_srli_epi64(out3, 1u));  \
	out3 = _mm256_and_si256(in2, mask2);                        \
	out1 = _mm256_or_si256(out1, _mm256_slli_epi64(out3, 1u));  \
	out3 = _mm256_and_si256(in3, mask2);                        \
	out1 = _mm256_or_si256(out1, _mm256_slli_epi64(out3, 2u));  \
							                                    \
	out2 = _mm256_and_si256(in2, mask4);                        \
	out3 = _mm256_and_si256(in0, mask4);                        \
	out2 = _mm256_or_si256(out2, _mm256_srli_epi64(out3, 2u));  \
	out3 = _mm256_and_si256(in1, mask4);                        \
	out2 = _mm256_or_si256(out2, _mm256_srli_epi64(out3, 1u));  \
	out3 = _mm256_and_si256(in3, mask4);                        \
	out2 = _mm256_or_si256(out2, _mm256_slli_epi64(out3, 1u));  \
                                                                \
	out3 = _mm256_and_si256(in3, mask8);                        \
	tmp  = _mm256_and_si256(in0, mask8);                        \
	out3 = _mm256_or_si256(out3, _mm256_srli_epi64(tmp, 3u));   \
	tmp  = _mm256_and_si256(in1, mask8);                        \
	out3 = _mm256_or_si256(out3, _mm256_srli_epi64(tmp, 2u));   \
	tmp  = _mm256_and_si256(in2, mask8);                        \
	out3 = _mm256_or_si256(out3, _mm256_srli_epi64(tmp, 1u));


/// SRC: https://github.com/pqov/pqov-paper
/// clobbert: acc0-acc3,t0-t3
/// NOTE: only the lower 4 bits of b are used
/// acc += in*b
#define gf16v_madd_bitsclice_avx2(acc0, acc1, acc2, acc3, in0, in1, in2, in3, b, t0, t1, t2) \
	t0 = _mm256_xor_si256(in0, in3);    \
	t1 = _mm256_xor_si256(in2, in3);    \
	t2 = _mm256_xor_si256(in1, in2);    \
	if (b&1u){		                    \
	acc0 = _mm256_xor_si256(cc0, in0);  \
	acc1 = _mm256_xor_si256(cc1, in1);  \
	acc2 = _mm256_xor_si256(cc2, in2);  \
	acc3 = _mm256_xor_si256(cc3, in3);  \
	}				                    \
	if (b&2u) {		                    \
	acc0 = _mm256_xor_si256(cc0, in3);  \
	acc1 = _mm256_xor_si256(cc1, tmp0); \
	acc2 = _mm256_xor_si256(cc2, in1);  \
	acc3 = _mm256_xor_si256(cc3, in2);  \
	}				                    \
 	if (b&4u){		                    \
	acc0 = _mm256_xor_si256(cc0, in2);  \
	acc1 = _mm256_xor_si256(cc1, tmp1); \
	acc2 = _mm256_xor_si256(cc2, tmp0); \
	acc3 = _mm256_xor_si256(cc3, in1);  \
	}				                    \
 	if (b&8u){		                    \
	acc0 = _mm256_xor_si256(cc0, in1);  \
	acc1 = _mm256_xor_si256(cc1, tmp2); \
	acc2 = _mm256_xor_si256(cc2, tmp1); \
	acc3 = _mm256_xor_si256(cc3, tmp0); \
	}

/// src: https://github.com/PQCMayo/MAYO-C/blob/main/src/AVX2/bitsliced_arithmetic_64.h
/// NOTE: only read the lower 4 bits of each input byte
///       31                  0
/// out =[[in0_0, ... , in31_0] , with inX_Y is the Y bit of in[X]
///               ...
///       [in{32*31}_0, ... , in{31*32+31}_0] // assume there are m_legs=32 many inputs
///                                <- m_legs many `uint32_t`
///       [in0_1, ... , in31_1]
///               ...
///       [in0_2, ... , in31_2]
///               ...
///       [in0_3, ... , in31_3]
///               ...
///
///
/// \param m_legs = len(in) / 4
/// \param in
/// \param out
void bitslice_m_vec(const uint32_t m_legs,
                    const unsigned char *in,
                    uint32_t *out) {
    uint32_t *out0 = out;
    uint32_t *out1 = out + m_legs;
    uint32_t *out2 = out + 2 * m_legs;
    uint32_t *out3 = out + 3 * m_legs;

    for (uint32_t leg = 0; leg < m_legs; leg++) {
        uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        // NOTE: the int is important
        for (int i = 31; i >= 0; i--) {
            d0 = (d0 << 1) ^  (in[32 * leg + i] & 1);
            d1 = (d1 << 1) ^ ((in[32 * leg + i] & 2) >> 1);
            d2 = (d2 << 1) ^ ((in[32 * leg + i] & 4) >> 2);
            d3 = (d3 << 1) ^ ((in[32 * leg + i] & 8) >> 3);
        }
        out0[leg] = d0;
        out1[leg] = d1;
        out2[leg] = d2;
        out3[leg] = d3;
    }
}

///
/// \param out 		output: bitsliced row
/// 						MUST be an array of 4 `uint32_t`
///          0                             31
/// out[0] = [A[row][0]_0, ..., A[row][31]_0]
/// out[1] = [A[row][0]_1, ..., A[row][31]_1]
/// out[2] = [A[row][0]_2, ..., A[row][31]_2]
/// out[3] = [A[row][0]_3, ..., A[row][31]_3]
/// where A[i][j] is the i-th row and j-th column
/// and   A[i][j]_k is the k-th bit of  i-th row and j-th column
/// \param A        input: full matrix
/// \param nr_rows  input: nr of rows in A
/// \param nr_cols  input: nr of cols in A
/// \param row      input: row to bitslice
void bitslice_row(uint32_t *out,
                  const uint8_t *A,
				  const uint32_t nr_rows,
                  const uint32_t nr_cols,
				  const uint32_t row) {
	assert(nr_cols <= 32);
	out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;

    for (uint32_t i = 0; i < nr_cols; i++) {
		uint8_t e = get_entry(A, nr_rows, row, i);
    	out[0] ^= ((e >> 0u)&1u) << i;
    	out[1] ^= ((e >> 1u)&1u) << i;
    	out[2] ^= ((e >> 2u)&1u) << i;
    	out[3] ^= ((e >> 3u)&1u) << i;
    }
}

/// inverse operation of `bitslice_row`
void unbitslice_row(uint8_t *A,
                    const uint32_t *bitslice,
                    const uint32_t nr_rows,
                    const uint32_t nr_cols,
                    const uint32_t row) {
    assert(nr_cols <= 32);
    for (uint32_t i = 0; i < nr_cols; i++) {
        uint8_t e = 0;
        e ^= ((bitslice[0] >> i) & 1u) << 0u;
        e ^= ((bitslice[1] >> i) & 1u) << 1u;
        e ^= ((bitslice[2] >> i) & 1u) << 2u;
        e ^= ((bitslice[3] >> i) & 1u) << 3u;

        set_entry(A, nr_rows, row, i, e);
    }
}


/// bitslices a full matrix row wise
/// \param out      output must be at least 16*4*2 bytes
/// 0bit	  0                                 16
/// out[ 0] = [A[ 0][0]_0, A[ 0][ 1]_0, ..., A[ 0][15]_0]
/// out[ 1] = [A[ 1][0]_0, A[ 1][ 1]_0, ..., A[ 1][15]_0]
///   					....
/// out[15] = [A[15][0]_0, A[ 1][ 1]_0, ..., A[15][15]_0]
/// 1bit
/// out[16] = [A[ 0][0]_1, A[ 1][ 1]_1, ..., A[ 0][15]_1]
/// 					....
/// out[31] = [A[15][0]_1, A[15][ 1]_1, ..., A[15][15]_1]
/// 2bit
/// out[32] = [A[ 0][0]_2, A[ 1][ 1]_2, ..., A[ 0][15]_2]
/// 					....
/// out[47] = [A[15][0]_2, A[15][ 1]_2, ..., A[15][15]_2]
/// 3bit
/// out[48] = [A[ 0][0]_3, A[ 1][ 1]_3, ..., A[ 0][15]_3]
/// 					....
/// out[63] = [A[15][0]_3, A[15][ 1]_3, ..., A[15][15]_3]
///
/// where A[i][j] is the i-th row and j-th column of A.
/// and   A[i][j]_k is the k-th bit in the i-th row and j-th column of A.
/// \param A        input: matrix, at most 16 cols and rows
/// \param nr_rows  input: number of rows in A
/// \param nr_cols  input: number of cols in A
void bitslice_row_matrix16(uint16_t *out,
                    	   const uint8_t *A,
                       	   const uint32_t nr_rows,
                       	   const uint32_t nr_cols) {
    assert(nr_cols <= 16);
    assert(nr_rows <= 16);

    /// NOTE: offset of 16, thus its loadable into a avx register
    uint16_t *out0 = out +  0;
    uint16_t *out1 = out + 16;
    uint16_t *out2 = out + 32;
    uint16_t *out3 = out + 48;

    for (uint32_t row = 0; row < nr_rows; ++row) {
        for (uint32_t i = 0; i < nr_cols; i++) {
            uint8_t e = get_entry(A, nr_rows, row, i);
            out0[row] ^= ((e >> 0u)&1u) << i;
            out1[row] ^= ((e >> 1u)&1u) << i;
            out2[row] ^= ((e >> 2u)&1u) << i;
            out3[row] ^= ((e >> 3u)&1u) << i;
        }
    }
}

/// inverse operation of `bitslice_matrix16`
void unbitslice_row_matrix16(uint8_t *A,
                    		 const uint16_t *bitslice,
                         	 const uint32_t nr_rows,
                         	 const uint32_t nr_cols) {
    assert(nr_cols <= 16);
    assert(nr_rows <= 16);

    /// NOTE: offset of 16, thus its loadable into a avx register
    const uint16_t *bitslice0 = bitslice +  0;
    const uint16_t *bitslice1 = bitslice + 16;
    const uint16_t *bitslice2 = bitslice + 32;
    const uint16_t *bitslice3 = bitslice + 48;

    for (uint32_t row = 0; row < nr_rows; ++row) {
        for (uint32_t i = 0; i < nr_cols; i++) {
            uint8_t e = 0;
            e ^= ((bitslice0[row] >> i) & 1u) << 0u;
            e ^= ((bitslice1[row] >> i) & 1u) << 1u;
            e ^= ((bitslice2[row] >> i) & 1u) << 2u;
            e ^= ((bitslice3[row] >> i) & 1u) << 3u;

            set_entry(A, nr_rows, row, i, e);
        }
    }
}

/// bitslices a full matrix col wise
/// \param out      output must be at least 16*4*2 bytes
/// 0bit	  0                                 16
/// out[ 0] = [A[ 0][ 0]_0, A[ 1][ 0]_0, ..., A[15][ 0]_0]
/// out[ 1] = [A[ 0][ 1]_0, A[ 1][ 1]_0, ..., A[15][ 1]_0]
///   					....
/// out[15] = [A[ 0][15]_0, A[ 1][ 1]_0, ..., A[15][15]_0]
/// 1bit
/// out[16] = [A[ 0][ 0]_1, A[ 1][ 1]_1, ..., A[15][ 0]_1]
/// 				     ....
/// out[31] = [A[ 0][15]_1, A[ 1][15]_1, ..., A[15][15]_1]
/// 2bit
/// out[32] = [A[ 0][ 0]_2, A[ 1][ 0]_2, ..., A[15][ 0]_2]
/// 				     ....
/// out[47] = [A[ 0][15]_2, A[ 1][15]_2, ..., A[15][15]_2]
/// 3bit
/// out[48] = [A[ 0][ 0]_3, A[ 1][ 1]_3, ..., A[15][ 0]_3]
/// 				     ....
/// out[63] = [A[ 0][15]_3, A[ 1][15]_3, ..., A[15][15]_3]
///
/// where A[i][j] is the i-th row and j-th column of A.
/// and   A[i][j]_k is the k-th bit in the i-th row and j-th column of A.
/// \param A        input: matrix, at most 16 cols and rows
/// \param nr_rows  input: number of rows in A
/// \param nr_cols  input: number of cols in A
void bitslice_col_matrix16(uint16_t *out,
                    	   const uint8_t *A,
                       	   const uint32_t nr_rows,
                       	   const uint32_t nr_cols) {
    assert(nr_cols <= 16);
    assert(nr_rows <= 16);

    /// NOTE: offset of 16, thus its loadable into a avx register
    uint16_t *out0 = out +  0;
    uint16_t *out1 = out + 16;
    uint16_t *out2 = out + 32;
    uint16_t *out3 = out + 48;

    for (uint32_t col = 0; col < nr_cols; ++col) {
        out0[col] = 0; out1[col] = 0; out2[col] = 0; out3[col] = 0;
        for (uint32_t row = 0; row < nr_rows; row++) {
            uint8_t e = get_entry(A, nr_rows, row, col);
            out0[col] ^= ((e >> 0u)&1u) << row;
            out1[col] ^= ((e >> 1u)&1u) << row;
            out2[col] ^= ((e >> 2u)&1u) << row;
            out3[col] ^= ((e >> 3u)&1u) << row;
        }

        assert(out0[col] < (1u << nr_rows));
        assert(out1[col] < (1u << nr_rows));
        assert(out2[col] < (1u << nr_rows));
        assert(out3[col] < (1u << nr_rows));
    }
}

/// inverse function of `bitslice_col_matrix16`.
void unbitslice_col_matrix16(uint8_t *A,
                    		 const uint16_t *bitslice,
                         	 const uint32_t nr_rows,
                         	 const uint32_t nr_cols) {
    assert(nr_cols <= 16);
    assert(nr_rows <= 16);

    /// NOTE: offset of 16, thus its loadable into a avx register
    const uint16_t *bitslice0 = bitslice +  0;
    const uint16_t *bitslice1 = bitslice + 16;
    const uint16_t *bitslice2 = bitslice + 32;
    const uint16_t *bitslice3 = bitslice + 48;

    for (uint32_t col = 0; col < nr_cols; ++col) {
        assert(bitslice0[col] < (1u << nr_rows));
        assert(bitslice1[col] < (1u << nr_rows));
        assert(bitslice2[col] < (1u << nr_rows));
        assert(bitslice3[col] < (1u << nr_rows));
        for (uint32_t row = 0; row < nr_rows; row++) {
            uint8_t e = 0;
            e ^= ((bitslice0[col] >> row) & 1u) << 0u;
            e ^= ((bitslice1[col] >> row) & 1u) << 1u;
            e ^= ((bitslice2[col] >> row) & 1u) << 2u;
            e ^= ((bitslice3[col] >> row) & 1u) << 3u;

            set_entry(A, nr_rows, row, col, e);
        }
    }
}

///
void bitslice_col_matrix_16_madd(uint16_t *acc,
							     uint16_t *in,
					      	     const uint8_t *b,
							     const uint8_t nr_cols_b,
                                 const uint8_t nr_bytes_in_b) {
	const __m256i lut_b0 = _mm256_setr_epi8(
        0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
        0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1
    );
    const __m256i lut_b1 = _mm256_setr_epi8(
        0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1,
        0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1
    );
    const __m256i lut_b2 = _mm256_setr_epi8(
        0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1,
        0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1
    );
    const __m256i lut_b3 = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1
    );

	assert(nr_bytes_in_b <= 8);
	uint8_t data[32] = {0};

    /// NOTE: unaligned, but I think there is not much i can do.
    __m256i ACC[4];
	__m256i_u* IN  = (__m256i_u *) in;
    __m256i IN0=IN[0],IN1=IN[1],IN2=IN[2],IN3=IN[3];
    __m256i x = IN0 ^ IN3;
    __m256i y = IN3 ^ IN2;
    __m256i z = IN2 ^ IN1;

	/// per loop 35 operations + memory operations
	for (uint8_t i = 0; i < nr_cols_b; i++) {
        for (uint8_t j = 0; j < nr_bytes_in_b; j++) {
            const uint8_t d1 = b[i*nr_bytes_in_b + j] & 0xf;
            data[4*j+0] = d1;
            data[4*j+1] = d1;

            const uint8_t d2 = b[i*nr_bytes_in_b + j] >> 4u;
            data[4*j+2] = d2;
            data[4*j+3] = d2;
        }

        const __m256i bb = _mm256_loadu_si256((__m256i *)data);
        const __m256i b0 = _mm256_shuffle_epi8(lut_b0, bb);
        const __m256i b1 = _mm256_shuffle_epi8(lut_b1, bb);
        const __m256i b2 = _mm256_shuffle_epi8(lut_b2, bb);
        const __m256i b3 = _mm256_shuffle_epi8(lut_b3, bb);

		// deg 0 term of b
		ACC[0] = b0 & IN0;
		ACC[1] = b0 & IN1;
		ACC[2] = b0 & IN2;
		ACC[3] = b0 & IN3;

		// deg 1 term of b
		ACC[0] ^= b1 & IN3;
		ACC[1] ^= b1 & x;
		ACC[2] ^= b1 & IN1;
		ACC[3] ^= b1 & IN2;
		
		// deg 2 term of b
		ACC[0] ^= b2 & IN2;
		ACC[1] ^= b2 & y;
		ACC[2] ^= b2 & x;
		ACC[3] ^= b2 & IN1;
		
		// deg 3 term of b
		ACC[0] ^= b3 & IN1;
		ACC[1] ^= b3 & z;
		ACC[2] ^= b3 & y;
		ACC[3] ^= b3 & x;

        acc[i +  0] = _mm256_extract_epi16(gf16_hadd_avx2_16(ACC[0]), 0);
        acc[i + 16] = _mm256_extract_epi16(gf16_hadd_avx2_16(ACC[1]), 0);
        acc[i + 32] = _mm256_extract_epi16(gf16_hadd_avx2_16(ACC[2]), 0);
        acc[i + 48] = _mm256_extract_epi16(gf16_hadd_avx2_16(ACC[3]), 0);
	}
}


/// src: https://github.com/PQCMayo/MAYO-C/blob/main/src/AVX2/bitsliced_arithmetic_64.h
/// computes the sum of the two input vectors
static inline void bitsliced_64_vec_sum(const uint64_t *_in1,
                                        const uint64_t *_in2,
                                        uint64_t *_out) {
    const __m256i* in1 = (const __m256i*) _in1;
    const __m256i* in2 = (const __m256i*) _in2;
    __m256i* out = (__m256i*) _out;
    out[0] = in1[0] ^ in2[0];
}

/// src: https://github.com/PQCMayo/MAYO-C/blob/main/src/AVX2/bitsliced_arithmetic_64.h
/// acc += in
static inline void bitsliced_64_vec_add(const uint64_t *_in,
                                        uint64_t *_acc) {
    const __m256i* in = (const __m256i*) _in;
    __m256i* acc = (__m256i*)_acc;
    acc[0] ^= in[0];
}

/// src: https://github.com/PQCMayo/MAYO-C/blob/main/src/AVX2/bitsliced_arithmetic_64.h
///
static inline void bitsliced_64_vec_mul_add_x(const uint64_t* _in,
                                              uint64_t* _acc) {
    const __m256i* in = (const __m256i*) _in;
    __m256i* acc = (__m256i*)_acc;

    const __m256i andmask = _mm256_set_epi64x(0, 0, 0xffffffffffffffff, 0);
    acc[0] ^= _mm256_permute4x64_epi64(in[0], 0b10011111) ^ (_mm256_shuffle_epi32(in[0], 0b01001110) & andmask);
}

static
inline void bitsliced_64_vec_mul_add_x_inv(const uint64_t *_in, uint64_t *_acc) {
    const __m256i* in = (const __m256i*) _in;
    __m256i* acc = (__m256i*)_acc;

    const __m256i andmask = _mm256_set_epi64x(0, 0, 0, 0xffffffffffffffff);
    acc[0] ^= _mm256_permute4x64_epi64(in[0], 0b00111001) ^ (in[0] & andmask);
}

/// src: https://github.com/PQCMayo/MAYO-C/blob/main/src/AVX2/bitsliced_arithmetic_64.h
///
/// \param _in
/// \param a
/// \param _acc
static
inline void bitsliced_64_vec_mul_add(const uint64_t *_in,
                                     unsigned char a,
                                     uint64_t *_acc) {
    const __m256i* in = (const __m256i*) _in;
    __m256i* acc = (__m256i*) _acc;

    const __m256i lut_a = _mm256_setr_epi8(
            0x0, 0x13, 0x26, 0x35, 0x4c, 0x5f, 0x6a, 0x79, 0x98, 0x8b, 0xbe, 0xad, 0xd4, 0xc7, 0xf2, 0xe1,
            0x0, 0x13, 0x26, 0x35, 0x4c, 0x5f, 0x6a, 0x79, 0x98, 0x8b, 0xbe, 0xad, 0xd4, 0xc7, 0xf2, 0xe1
    );


    __m256i inshuf2 = _mm256_permute4x64_epi64(in[0], 0b01001110);
    __m256i inshuf1 = _mm256_shuffle_epi32(in[0], 0b01001110);
    __m256i inshuf3 = _mm256_shuffle_epi32(inshuf2, 0b01001110);

    const __m256i mask1 = _mm256_set_epi64x(16,  16, 16, 1  );
    const __m256i mask2 = _mm256_set_epi64x(32,  8,  32, 128);
    const __m256i mask3 = _mm256_set_epi64x(64,  64, 4,  64 );
    const __m256i mask4 = _mm256_set_epi64x(128, 32, 8,  32 );

    __m256i aaaa = _mm256_shuffle_epi8(lut_a, _mm256_set1_epi8(a));

    acc[0] ^= in[0]   & _mm256_cmpeq_epi64(aaaa & mask1, mask1);
    acc[0] ^= inshuf1 & _mm256_cmpeq_epi64(aaaa & mask2, mask2);
    acc[0] ^= inshuf2 & _mm256_cmpeq_epi64(aaaa & mask3, mask3);
    acc[0] ^= inshuf3 & _mm256_cmpeq_epi64(aaaa & mask4, mask4);
}
