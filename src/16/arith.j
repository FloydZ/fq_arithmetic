u8[256] gf16_mult_table = {
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
    0x00,0x0f,0x0d,0x02,0x09,0x06,0x04,0x0b,0x01,0x0e,0x0c,0x03,0x08,0x07,0x05,0x0a
};

// table[i] = i^{-1} mod GF(16)
u8[16] gf16_inverse_tab = {
    0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8
};

inline 
fn gf16_add(reg u8 a, reg u8 b) -> reg u8 {
	reg u8 c;
	c = a ^ b;
	return c;
}

inline 
fn gf16_sub(reg u8 a, reg u8 b) -> reg u8 {
	reg u8 c;
	c = gf16_add(a, b);
	return c;
}

export
fn gf16_mul_table(reg u8 a, reg u8 b) -> reg u8 {
    reg u8 c;
	stack u8[256] T;
	T = gf16_mult_table;
	return c; 
}

export
fn gf16_mul(reg u8 a, reg u8 b) -> reg u8 {
	reg u8 r a1 b1;
	b1 = b;
	b1 >>= 3;
	a1 = a;
    r = b1 & a1;

	return r;
}
