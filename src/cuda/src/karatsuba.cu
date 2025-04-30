#include <stdlib.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "karatsuba64.cu"
#include "arith.cuh"

//__device__ const uint64_t __p[2]   = { 0xAC0E7A06FFFFFFFD, 0x0000000000000012 };
//__device__ const uint32_t __p32[4] = { 0xFFFFFFFD, 0xAC0E7A06, 0x00000012, 0x0 };
//
//__device__ const uint64_t __mu[2]   = { 0x1, 0x0000000000000012 };
//__device__ const uint32_t __mu32[4] = { 0x1, 0x1, 0x1, 0x0 };

//__constant__ limb_t __p[NWORDS_FIELD] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7BC6BFFF, 0x8AF43417, 0xA1087212, 0x01248A1B };
//__constant__ limb_t __mu[NWORDS_FIELD] = { 0x00000001, 0x00000000, 0x00000000, 0x7BC6C000, 0x8AF43417, 0xA1087212, 0x91248A1B };

// this are the 6 limb p and mu value
//__constant__ uint32_t __p[8]  = { 0xFFFFFFFF, 0xFFFFFFFF, 0x17FFFFFF, 0xD9A6C7B1, 0x20D5040B, 0x002895AC, 0, 0 };
//__constant__ uint32_t __mu[8] = { 0x00000001, 0x00000000, 0x18000000, 0xD9A6C7B1, 0x20D5040B, 0x326895AC, 0, 0 };

// these are the 8 limb p and mu value 
__constant__ uint32_t __p[8] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7BC6BFFF, 0x8AF43417, 0xA1087212, 0x01248A1B, 0 };
__constant__ uint32_t __mu[8] = { 0x00000001, 0x00000000, 0x00000000, 0x7BC6C000, 0x8AF43417, 0xA1087212, 0x91248A1B, 0 };

// just testing
//uint32_t __cpu_p[8]  = { 0xFFFFFFFF, 0xFFFFFFFF, 0x17FFFFFF, 0xD9A6C7B1, 0x20D5040B, 0x002895AC, 0, 0 };
//uint32_t __cpu_mu[8] = { 0x00000001, 0x00000000, 0x18000000, 0xD9A6C7B1, 0x20D5040B, 0x326895AC, 0, 0 };
uint32_t __cpu_p[8] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7BC6BFFF, 0x8AF43417, 0xA1087212, 0x01248A1B };
uint32_t __cpu_mu[8] = { 0x00000001, 0x00000000, 0x00000000, 0x7BC6C000, 0x8AF43417, 0xA1087212, 0x91248A1B };





// first part of montgomery reduction 
// essential its a multiplication a*mu mod 2**64
// This version does not use the 'mad' instructions
#define __school32_inplace_2x2_low_v2(a0,a1,mu0,mu1)\
	asm volatile ("{\n\t"                       \
			".reg .u32 t10;\n\t"                \
			/* t0-t2: m = a * mu mod 2^64 */    \
			/* Multiplication involving a0 */   \
			"mul.lo.u32     %0,  %4, %2;\n\t"	\
			"mul.hi.u32     %1,  %4, %2;\n\t"   \
			"mul.lo.u32     t10, %4, %3;\n\t"   \
			/* now fixing the carry chains */   \
  			"add.cc.u32     %1,  %1,  t10;\n\t" \
			/*Multiplication involving a3 second (low) diagonal*/   		\
			"mul.lo.u32     t10, %5,  %3;\n\t"  \
			"add.u32         %1, %1, t20;\n\t"  \
			"}\n"		                          		\
			: "=r"(a0),"=r"(a1),"=r"(a2)   /*0 , 1*/ 	\
			: "r"(mu0),"r"(mu1),"r"(mu2)   /*2 , 3*/ 	\
			  "r"(a0),"r"(a1),"r"(a2)  	   /*4 , 5*/	\
		);


// second part of the montgomery reduction written by floyd
#define __reduce32_sec10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,p0,p1,p2,p3,p4,i0,i1,i2,i3,i4,i5,i6,i7,i8,i9)	 	\
	asm volatile ("{\n\t"                   						\
			/*%0-%5: r= (a + u) div 2^(5**32)*/      				\
			".reg.u32 		t0, t5,t6,t7,t8,t9;\n\t"				\
			"mov.b32        t0, 0;\n\t"             				\
			/*a + u*/												\
			"add.cc.u32     %0, %20, %10;\n\t"        				\
			"addc.cc.u32    %1, %21, %11;\n\t"        				\
			"addc.cc.u32    %2, %22, %12;\n\t"        				\
			"addc.cc.u32    %3, %23, %13;\n\t"        				\
			"addc.cc.u32    %4, %24, %14;\n\t"       				\
			"addc.cc.u32    t5, %25, %15;\n\t"       				\
			"addc.cc.u32    t6, %26, %16;\n\t"       				\
			"addc.cc.u32    t7, %27, %17;\n\t"       				\
			"addc.cc.u32    t8, %28, %18;\n\t"       				\
			"addc.cc.u32    t9, %29, %19;\n\t"       				\
			"addc.u32       t0, t0, 0;\n\t"         				\
			/*-p*/													\
			"sub.cc.u32     t5, t5, %5;\n\t"       					\
			"subc.cc.u32    t6, t6, %6;\n\t"       					\
			"subc.cc.u32    t7, t7, %7;\n\t"       					\
			"subc.cc.u32    t8, t8, %8;\n\t"       					\
			"subc.cc.u32    t9, t9, %9;\n\t"       					\
			"subc.u32       t0, t0, 0;\n\t"         				\
			/*&p*/													\
			"and.b32        %0, t0, %5;\n\t"       					\
			"and.b32        %1, t0, %6;\n\t"       					\
			"and.b32        %2, t0, %7;\n\t"       					\
			"and.b32        %3, t0, %8;\n\t"       					\
			"and.b32        %4, t0, %9;\n\t"       					\
			"add.cc.u32     %0, %0, t5;\n\t"        				\
			"addc.cc.u32    %1, %1, t6;\n\t"        				\
			"addc.cc.u32    %2, %2, t7;\n\t"        				\
			"addc.cc.u32    %3, %3, t8;\n\t"        				\
			"addc.u32       %4, %4, t9;\n\t"       					\
			"}\n"                                     				\
	: "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4)	/*0 , 5*/ 		\
	: "r"(p0),"r"(p1),"r"(p2),"r"(p3),"r"(p4),	 	/*5 , 9*/  		\
	  "r"(i0),"r"(i1),"r"(i2),"r"(i3),"r"(i4), 		/*10,14*/  		\
	  "r"(i5),"r"(i6),"r"(i7),"r"(i8),"r"(i9),      /*15,19*/  		\
	  "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4), 		/*20,24*/  		\
	  "r"(a5),"r"(a6),"r"(a7),"r"(a8),"r"(a9)       /*25,29*/  		\
	);


// montgomery reuction for 10 32bit limbs. 
#define __reduce32_inplace10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,mu0,mu1,mu2,mu3,mu4,p0,p1,p2,p3,p4)		\
	uint32_t t0,t1,t2,t2,t3,t4,t5,t6,t7,t8,t9;														\
	__school32_inplace_5x5_low_v2(a0,a1,a2,a3,a4,mu0,mu1,mu2,mu3,mu4)								\
	__school32_5x5(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,a0,a1,a2,a3,a4,p0,p1,p2,p3,p4)   					\
	__reduce32_sec10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,p0,p1,p2,p3,p4,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9)


// TODO describe
#define __reduce32_sec16(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, p0,p1,p2,p3,p4,p5,p6,p7, i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15)	 	 \
        asm volatile ("{\n\t"						\
			/* r = (a + u) div 2^(8*32)*/      		\
       		 ".reg .u32		 t0, t8,t9,t10,t11,t12,t13,t14,t15;\n\t"\
			 "mov.u32        t0, 0;\n\t"            \
       		 "add.cc.u32     %0, %32, %16;\n\t"     \
       		 "addc.cc.u32    %1, %33, %17;\n\t"     \
       		 "addc.cc.u32    %2, %34, %18;\n\t"     \
       		 "addc.cc.u32    %3, %35, %19;\n\t"     \
       		 "addc.cc.u32    %4, %36, %20;\n\t"     \
       		 "addc.cc.u32    %5, %37, %21;\n\t"     \
       		 "addc.cc.u32    %6, %38, %22;\n\t"     \
       		 "addc.cc.u32    %7, %39, %23;\n\t"     \
       		 "addc.cc.u32    t8, %40, %24;\n\t"     \
       		 "addc.cc.u32    t9, %41, %25;\n\t"     \
       		 "addc.cc.u32    t10, %42, %26;\n\t"    \
       		 "addc.cc.u32    t11, %43, %27;\n\t"    \
       		 "addc.cc.u32    t12, %44, %28;\n\t"    \
       		 "addc.cc.u32    t13, %45, %29;\n\t"    \
       		 "addc.cc.u32    t14, %46, %30;\n\t"    \
       		 "addc.cc.u32    t15, %47, %31;\n\t"    \
       		 "addc.u32       t0, t0, 0;\n\t"        \
       		 "sub.cc.u32     t8, t8, %8;\n\t"       \
       		 "subc.cc.u32    t9, t9, %9;\n\t"       \
       		 "subc.cc.u32    t10, t10, %10;\n\t"    \
       		 "subc.cc.u32    t11, t11, %11;\n\t"    \
       		 "subc.cc.u32    t12, t12, %12;\n\t"    \
       		 "subc.cc.u32    t13, t13, %13;\n\t"    \
       		 "subc.cc.u32    t14, t14, %14;\n\t"    \
       		 "subc.cc.u32    t15, t15, %15;\n\t"    \
			 "subc.u32       t0, t0, 0;\n\t"        \
       		 "and.b32        %0, t0, %8;\n\t"       \
       		 "and.b32        %1, t0, %9;\n\t"       \
       		 "and.b32        %2, t0, %10;\n\t"      \
       		 "and.b32        %3, t0, %11;\n\t"      \
       		 "and.b32        %4, t0, %12;\n\t"      \
       		 "and.b32        %5, t0, %13;\n\t"      \
       		 "and.b32        %6, t0, %14;\n\t"      \
       		 "and.b32        %7, t0, %15;\n\t"      \
       		 "add.cc.u32     %0, %0, t8;\n\t"       \
       		 "addc.cc.u32    %1, %1, t9;\n\t"       \
       		 "addc.cc.u32    %2, %2, t10;\n\t"      \
       		 "addc.cc.u32    %3, %3, t11;\n\t"      \
       		 "addc.cc.u32    %4, %4, t12;\n\t"      \
       		 "addc.cc.u32    %5, %5, t13;\n\t"      \
       		 "addc.cc.u32    %6, %6, t14;\n\t"      \
       		 "addc.u32       %7, %7, t15;\n\t"      \
        "}\n\t"                                     \
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),"=r"(a5),"=r"(a5),"=r"(a7) 					/*0 - 7*/\
        : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6), "r"(p7), 					/*8 -15*/\
          "r"(i0), "r"(i1), "r"(i2), "r"(i3), "r"(i4), "r"(i5), "r"(i6), "r"(i7),               	/*16-23*/\
          "r"(i8), "r"(i9), "r"(i10), "r"(i11), "r"(i12), "r"(i13), "r"(i14), "r"(i15),				/*24-31*/\
          "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),               	/*32-39*/\
          "r"(a8), "r"(a9), "r"(a10), "r"(a11), "r"(a12), "r"(a13), "r"(a14), "r"(a15) 				/*40-47*/\
		  )


// TODO describe
#define __reduce32_inplace16(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,mu0,mu1,mu2,mu3,mu4,mu5,mu6,mu7,p0,p1,p2,p3,p4,p5,p6,p7)	\
	uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15;																	\
	__school32_inplace_8x8_low_v2(a0,a1,a2,a3,a4,a5,a6,a7,mu0,mu1,mu2,mu3,mu4,mu5,mu6,mu7) \
	__school32_8x8(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,a0,a1,a2,a3,a4,a5,a6,a7,p0,p1,p2,p3,p4,p5,p6,p7) 			\
	__reduce32_sec16(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, p0,p1,p2,p3,p4,p5,t6,p7, t0,t1,t2,t3,t4,t5,t6,t8,t8,t9,t10,t11,t12,t13,t14,t15)



// TODO describe
#define __reduce32_16(a0,a1,a2,a3,a4,a5,a6,a7, c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15, mu0,mu1,mu2,mu3,mu4,mu5,mu6,mu7, p0,p1,p2,p3,p4,p5,p6,p7)	\
//

// TODO describe
#define __fp_karatsuba_mul128_32(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,b2,b3,m0,m1,m2,m3,p0,p1,p2,p3)  	\
	uint32_t t0,t1,t2,t3,t4,t5,t6,t7; 															\
	__karatsuba_4x4(t0,t1,t2,t3,t4,t5,t6,t7, a0,a1,a2,a3, b0,b1,b2,b3)							\
	__reduce8(c0,c1,c2,c3, t0,t1,t2,t3,t4,t5,t6,t7, m0,m1,m2,m3, p0,p1,p2,p3)

// TODO describe
#define __fp_karatsuba_mul192_32(c0,c1,c2,c3,c4,c5,a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,m0,m1,m2,m3,m4,m5,p0,p1,p2,p3,p4,p5)  	\
	uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11; 																			\
	__karatsuba_6x6(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11, a0,a1,a2,a3,a4,a5, b0,b1,b2,b3,b4,b5)								\
	__reduce12(c0,c1,c2,c3,c4,c5, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11, m0,m1,m2,m3,m4,m5, p0,p1,p2,p3,p4,p5)

// TODO describe:
#define __fp_karatsuba_mul224_32(c0,c1,c2,c3,c4,c5,c6,a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,m0,m1,m2,m3,m4,m5,m6,p0,p1,p2,p3,p4,p5,p6)  	\
	uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13; 																						\
	__karatsuba_7x7(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13, a0,a1,a2,a3,a4,a5,a6, b0,b1,b2,b3,b4,b5,b6)								\
	__reduce16(c0,c1,c2,c3,c4,c5,c6,c7, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14, m0,m1,m2,m3,m4,m5,m6, p0,p1,p2,p3,p4,p5,p6) 	

// TODO describe:
#define __fp_karatsuba_mul256_32(c0,c1,c2,c3,c4,c5,c6,c7,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,m0,m1,m2,m3,m4,m5,m6,m7,p0,p1,p2,p3,p4,p5,p6,p7)  	\
	uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15; 																						\
	__karatsuba_8x8(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15, a0,a1,a2,a3,a4,a5,a6,a7, b0,b1,b2,b3,b4,b5,b6,b7)								\
	__reduce16(c0,c1,c2,c3,c4,c5,c6,c7, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15, m0,m1,m2,m3,m4,m5,m6,m7, p0,p1,p2,p3,p4,p5,p6,p7) 	

//__global__ void simple_test1(uint32_t *out, uint32_t p1){
//	uint32_t t1[2] = {p1, 0};
//	uint32_t t2[2] = {p1, p1};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_2x2(out[0], out[1], out[2], out[3], t1[0], t1[1], t2[0], t2[1]);
//}
//
//__global__ void simple_test2(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[2] = {p1, p1};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x2(out[0], out[1], out[2], t1[0], t2[0], t2[1]);
//}
//
//__global__ void simple_test2_v2(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[2] = {p1, p1};
//	for (size_t i = 0; i < 16; i++) { out[i] = 0; }
//	__karatsuba32_1x2_v2(out[0], out[1], out[2], t1[0], t2[0], t2[1]);
//}
//
//__global__ void simple_test3(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[3] = {1, 0, 0};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x3(out[0], out[1], out[2], out[3], t1[0], t2[0], t2[1], t2[2]);
//}
//
//__global__ void simple_test4(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[4] = {1, 0, 0, 0};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x4(out[0], out[1], out[2], out[3], out[4], t1[0], t2[0], t2[1], t2[2], t2[3]);
//}
//
//__global__ void simple_test5(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[5] = {1, 0, 0, 0, 0};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x5(out[0], out[1], out[2], out[3], out[4], out[5], t1[0], t2[0], t2[1], t2[2], t2[3], t2[4]);
//}
//
//__global__ void simple_test6(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[6] = {1, 0, 0, 0, 0, 0};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x6(out[0], out[1], out[2], out[3], out[4], out[5], out[6], t1[0], t2[0], t2[1], t2[2], t2[3], t2[4], t2[5]);
//}
//
//__global__ void simple_test7(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[7] = {1, 0, 0, 0, 0, 0, 0};
//	for (size_t i = 0; i < 8; i++) { out[i] = 0; }
//	__karatsuba32_1x7(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], t1[0], t2[0], t2[1], t2[2], t2[3], t2[4], t2[5], t2[6]);
//}
//
//__global__ void simple_test8(uint32_t *out, uint32_t p1){
//	uint32_t t1[1] = {p1};
//	uint32_t t2[8] = {1, 0, 0, 0, 0, 0, 0, 0};
//	for (size_t i = 0; i < 9; i++) { out[i] = 0; }
//	__karatsuba32_1x8(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], t1[0], t2[0], t2[1], t2[2], t2[3], t2[4], t2[5], t2[6], t2[7]);
//}
//
//__global__ void simple_test6444(uint64_t *out, uint64_t p1){
//	uint64_t t1[4] = {p1, 0, 0, 0};
//	uint64_t t2[4] = {1, 0, 0, 0};
//	for(uint64_t i = 0; i < 8; i++){out[i] = 0;}
//	__karatsuba64_4x4(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], t1[0], t1[1], t1[2], t1[3], t2[0], t2[1], t2[2], t2[3]);
//}
//__global__ void simple_test44(uint32_t *out, uint32_t p1){
//	uint32_t t1[4] = {p1, 0, 0, 0};
//	uint32_t t2[4] = {1, 0, 0, 0};
//	for(uint32_t i = 0; i < 8; i++){out[i] = 0;}
//	__karatsuba32_4x4(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], t1[0], t1[1], t1[2], t1[3], t2[0], t2[1], t2[2], t2[3]);
//}
//
//__global__ void simple_test6433(uint64_t *out, uint64_t p1){
//	uint64_t t1[3] = {p1, 0, 0};
//	uint64_t t2[3] = {1, 0, 0};
//	for(uint64_t i = 0; i < 8; i++){out[i] = 0;}
//	__karatsuba64_3x3(out[0], out[1], out[2], out[3], out[4], out[5], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2]);
//}
//__global__ void simple_test33(uint32_t *out, uint32_t p1){
//	uint32_t t1[3] = {p1, 0, 0};
//	uint32_t t2[3] = {1, 0, 0};
//	for(uint32_t i = 0; i < 8; i++){out[i] = 0;}
//	__karatsuba32_3x3(out[0], out[1], out[2], out[3], out[4], out[5], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2]);
//}
//
//__global__ void simple_test6x6(uint32_t *out, uint32_t p1){
//	uint32_t t1[6] = {p1,0,0,0,0,0};
//	uint32_t t2[6] = {1,0,0,0,0,0};
//	for(uint32_t i = 0; i < 12; i++){out[i] = 0;}
//	__karatsuba32_6x6(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5]);
//}
//
//__global__ void simple_test8x8(uint32_t *out, uint32_t p1){
//	uint32_t t1[8] = {p1,0,0,0,0,0,0,0};
//	uint32_t t2[8] = {1,0,0,0,0,0,0,0};
//	for(uint32_t i = 0; i < 16; i++){out[i] = 0;}
//	__karatsuba32_8x8(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5],t1[6],t1[7], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5],t2[6],t2[7]);
//}
//
//__global__ void simple_test_school8x8(uint32_t *out, uint32_t p1){
//	uint32_t t1[8] = {p1,0,0,0,0,0,0,0};
//	uint32_t t2[8] = {1,0,0,0,0,0,0,0};
//	for(uint32_t i = 0; i < 16; i++){out[i] = 0;}
//	__school32_8x8(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5],t1[6],t1[7], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5],t2[6],t2[7]);
//}
//__global__ void simple_test_school7x7(uint32_t *out, uint32_t p1){
//	uint32_t t1[7] = {p1,0,0,0,0,0,0};
//	uint32_t t2[7] = {1,0,0,0,0,0,0};
//	for(uint32_t i = 0; i < 16; i++){out[i] = 0;}
//	__school32_7x7(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5],t1[6], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5],t2[6]);
//}
//__global__ void simple_test_school6x6(uint32_t *out, uint32_t p1){
//	uint32_t t1[6] = {p1,0,0,0,0,0};
//	uint32_t t2[6] = {1,0,0,0,0,0};
//	for(uint32_t i = 0; i < 16; i++){out[i] = 0;}
//	__school32_6x6(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5]);
//}
//
//__global__ void simple_test_fp_school6x6(uint32_t *out, uint32_t p1){
//	uint32_t t1[6] = {p1,0,0,0,0,0};
//	uint32_t t2[6] = {1,0,0,p1,0,0};
//	for(uint32_t i = 0; i < 6; i++){out[i] = 0;}
//	__school32_6x6(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], t1[0],t1[1],t1[2],t1[3],t1[4],t1[5], t2[0],t2[1],t2[2],t2[3],t2[4],t2[5]);
//	__reduce32_6x6(out[0],out[1],out[2],out[3],out[4],out[5], out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], __mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5], __p[0],__p[1],__p[2],__p[3],__p[4],__p[5]); 
//}
//
//
//__global__ void simple_test_school_inplace_3x3_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_3x3_low(out[0],out[1],out[2], __mu[0],__mu[1],__mu[2]);
//}
//__global__ void simple_test_school_inplace_4x4_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_4x4_low(out[0],out[1],out[2],out[3], __mu[0],__mu[1],__mu[2],__mu[3]);
//}
//__global__ void simple_test_school_inplace_5x5_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_5x5_low(out[0],out[1],out[2],out[3],out[4], __mu[0],__mu[1],__mu[2],__mu[3],__mu[4]);
//}
//__global__ void simple_test_school_inplace_6x6_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_6x6_low(out[0],out[1],out[2],out[3],out[4],out[5], __mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5]);
//}
//__global__ void simple_test_school_inplace_7x7_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_7x7_low(out[0],out[1],out[2],out[3],out[4],out[5],out[6], __mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6]);
//}
//__global__ void simple_test_school_inplace_8x8_low_v2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 1;
//	__school32_inplace_8x8_low(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7], __mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6],__mu[7]);
//}
//
//
//__global__ void simple_test_reduction_inplace2(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
////	__reduce32_inplace4(out[0],out[1],out[2],out[3],
////			__mu[0],__mu[1],
////			__p[0],__p[1]);
//}
//__global__ void simple_test_reduction_inplace3(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
////	__reduce32_inplace6(out[0],out[1],out[2],out[3],out[4],out[5],
////			__mu[0],__mu[1],__mu[2],
////			__p[0],__p[1],__p[2]);
//}
//__global__ void simple_test_reduction_inplace4(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
////	__reduce32_inplace8(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],
////			__mu[0],__mu[1],__mu[2],__mu[3],
////			__p[0],__p[1],__p[2],__p[3]);
//}
//__global__ void simple_test_reduction_inplace5(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
////	__reduce32_inplace10(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],
////			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4], 
////			__p[0],__p[1],__p[2],__p[3],__p[4]);
//}
//__global__ void simple_test_reduction_inplace6(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
//	__reduce32_inplace6x6(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5], 
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5]);
//}
//__global__ void simple_test_reduction_inplace7(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
//	__reduce32_inplace7x7(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6], 
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5],__p[6]);
//}
//__global__ void simple_test_reduction_inplace8(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	out[0] = 0; 
//	__reduce32_inplace16(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6],__mu[7],
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5],__p[6],__p[7]);
//}
//
//__global__ void simple_test_reduction4(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
//	__reduce32_8(out[0],out[1],out[2],out[3],
//			out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],
//			__mu[0],__mu[1],__mu[2],__mu[3], 
//			__p[0],__p[1],__p[2],__p[3]);
//}
//
//__global__ void simple_test_reduction6(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
//	__reduce32_12(out[0],out[1],out[2],out[3],out[4],out[5],
//			out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5], 
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5]);
//}
//
//__global__ void simple_test_reduction7(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }
//	
//	out[0] = 0;
//	__reduce32_14(out[0],out[1],out[2],out[3],out[4],out[5],out[6],
//			out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6], 
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5],__p[6]);
//}
//
//__global__ void simple_test_reduction8(uint32_t *out, uint32_t p1){
//	for(uint32_t i = 0; i < 16; i++){ out[i] = 0; }	
//	out[0] = 0; 
//	__reduce32_16(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],
//			out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15], 
//			__mu[0],__mu[1],__mu[2],__mu[3],__mu[4],__mu[5],__mu[6],__mu[7], 
//			__p[0],__p[1],__p[2],__p[3],__p[4],__p[5],__p[6],__p[7]);
//}
//
//
//#ifndef NOMAIN
//int main() { 
//	constexpr uint32_t size = 16;
//	uint32_t out[size]  = {0};
//	uint32_t out2[size] = {0};
//	uint32_t out3[size] = {0};
//	uint32_t out4[size] = {0};
//	uint32_t out5[size] = {0};
//	uint32_t out6[size] = {0};
//	uint32_t out7[size] = {0};
//	uint32_t out8[size] = {0};
//	uint32_t out9[size] = {0};
//
//	uint32_t out10[size] = {0};
//	uint32_t out11[size] = {0};
//	uint32_t out12[size] = {0};
//	uint32_t out13[size] = {0};
//	uint32_t out14[size] = {0};
//	
//	uint32_t out15[size] = {0};
//	uint32_t out16[size] = {0};
//	uint32_t out17[size] = {0};
//	uint32_t out18[size] = {0};
//	uint32_t out19[size] = {0};
//
//	uint32_t out20[size] = {0};
//	uint32_t out21[size] = {0};
//	uint32_t out22[size] = {0};
//	
//	uint32_t out23[size] = {0};
//	uint32_t out24[size] = {0};
//
//	uint32_t out25[size] = {0};
//	
//	uint32_t out26[size] = {0};
//	uint32_t out27[size] = {0};
//
//	uint32_t out28[size] = {0};
//	uint32_t out29[size] = {0};
//	uint32_t out30[size] = {0};
//	
//	uint32_t out31[size] = {0};
//	uint32_t out32[size] = {0};
//	uint32_t out33[size] = {0};
//	uint32_t out34[size] = {0};
//	uint32_t out35[size] = {0};
//	uint32_t *d_out;
//
//	cudaError_t errcode;
//	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_out, sizeof(uint32_t) * size))) {
//		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
//		printf ("   %s\n", cudaGetErrorString(errcode));
//		return;
//	}
//
//	const uint32_t p1 = 1u << 31u; 
//	// simple_test1<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	//      
//	// simple_test2<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out2, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test2_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out26, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	// simple_test3<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out3, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	// 
//	// simple_test4<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out4, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	//     	
//	// simple_test5<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out5, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	//	
//	// simple_test6<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out6, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	//	
//	// simple_test7<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out7, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test8<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out25, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	  
//	// simple_test44<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out8, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	//     	
//	// simple_test33<<<1,1>>>(d_out, p1);
//	// cudaDeviceSynchronize();
//	// cudaMemcpy(out9, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test6444<<<1,1>>>((uint64_t *)d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out23, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	    	
//	simple_test6433<<<1,1>>>((uint64_t *)d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out24, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//
//	simple_test_school_inplace_3x3_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out28, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 3; i > 0; i--){
//		printf("%x ", out28[i-1]);
//	}
//	printf(" school_inplace_3x3_low\n");	
//
//	simple_test_school_inplace_4x4_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out29, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 4; i > 0; i--){
//		printf("%x ", out29[i-1]);
//	}
//	printf(" school_inplace_4x4_low\n");	
//
//	simple_test_school_inplace_5x5_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out30, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 5; i > 0; i--){
//		printf("%x ", out30[i-1]);
//	}
//	printf(" school_inplace_5x5_low\n");
//
//
//	simple_test_school_inplace_6x6_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out15, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 6; i > 0; i--){
//		printf("%x ", out15[i-1]);
//	}
//	printf(" school_inplace_6x6_low\n");
//
//	for(uint32_t i = 6; i > 0; i--){
//		printf("%x ", __cpu_mu[i-1]);
//	}
//	printf(" mu\n");
//
//	simple_test_school_inplace_7x7_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out16, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 7; i > 0; i--){
//		printf("%x ", out16[i-1]);
//	}
//	printf(" school_inplace_7x7_low\n");
//
//	for(uint32_t i = 7; i > 0; i--){
//		printf("%x ", __cpu_mu[i-1]);
//	}
//	printf(" mu\n");
//	simple_test_school_inplace_8x8_low_v2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out17, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	for(uint32_t i = 8; i > 0; i--){
//		printf("%x ", out17[i-1]);
//	}
//	printf(" school_inplace_8x8_low\n");
//
//	for(uint32_t i = 8; i > 0; i--){
//		printf("%x ", __cpu_mu[i-1]);
//	}
//	printf(" mu\n");
//
//	simple_test_school5x5<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out31, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_school6x6<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out10, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_school7x7<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out18, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
////	simple_test_fp_school6x6<<<1,1>>>(d_out, p1);
////	cudaDeviceSynchronize();
////	cudaMemcpy(out11, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_school8x8<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out12, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//
//	simple_test_reduction_inplace2<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out32, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_reduction_inplace3<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out33, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_reduction_inplace4<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out34, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_reduction_inplace5<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out35, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_reduction_inplace6<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out13, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_reduction_inplace7<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out19, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_reduction_inplace8<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out14, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_reduction4<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out27, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	simple_test_reduction6<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out20, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_reduction7<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out21, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//	
//	simple_test_reduction8<<<1,1>>>(d_out, p1);
//	cudaDeviceSynchronize();
//	cudaMemcpy(out22, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out31[i-1]);
//	}
//	printf(" school 5x5\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out10[i-1]);
//	}
//	printf(" school 6x6\n");
//
////	for(uint32_t i = size; i > 0; i--){
////		printf("%u ", out11[i-1]);
////	}
////	printf(" fp school 6x6\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out18[i-1]);
//	}
//	printf(" school 7x7\n");
//	
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out12[i-1]);
//	}
//	printf(" school 8x8\n");
//	
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out32[i-1]);
//	}
//	printf(" reduce inplace 2\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out33[i-1]);
//	}
//	printf(" reduce inplace 3\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out34[i-1]);
//	}
//	printf(" reduce inplace 4\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out35[i-1]);
//	}
//	printf(" reduce inplace 5\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out13[i-1]);
//	}
//	printf(" reduce inplace 6\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out19[i-1]);
//	}
//	printf(" reduce inplace 7\n");
//	
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out14[i-1]);
//	}
//	printf(" reduce inplace 8\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out27[i-1]);
//	}
//	printf(" reduce 4\n");
//	
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out20[i-1]);
//	}
//	printf(" reduce 6\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out21[i-1]);
//	}
//	printf(" reduce 7\n");
//	
//	for(uint32_t i = size; i > 0; i--){
//		printf("%x ", out22[i-1]);
//	}
//	printf(" reduce 8\n");
//
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out23[i-1]);
//	}
//	printf(" 4x4 64\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out24[i-1]);
//	}
//	printf(" 3x3 64\n");
//
//
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out26[i-1]);
//	}
//	printf(" 1x2 v2\n");
//
//
//
//
//	return 0;
//	
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out[i-1]);
//	}
//	printf(" 2x2\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out2[i-1]);
//	}
//	printf(" 1x2\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out3[i-1]);
//	}
//	printf(" 1x3\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out4[i-1]);
//	}
//	printf(" 1x4\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out5[i-1]);
//	}
//	printf(" 1x5\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out6[i-1]);
//	}
//	printf(" 1x6\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out7[i-1]);
//	}
//	printf(" 1x7\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out25[i-1]);
//	}
//	printf(" 1x25\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out8[i-1]);
//	}
//	printf(" 4x4\n");
//
//	for(uint32_t i = size; i > 0; i--){
//		printf("%u ", out9[i-1]);
//	}
//	printf(" 3x3\n");
//
//	cudaFree (d_out);
//	return 0;
//}
//#endif
