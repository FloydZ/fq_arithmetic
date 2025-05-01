#include <stdlib.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>

// TODO not really corrext
__constant__ uint64_t __p64[4] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF7BC6BFFF, 0x8AF43417A1087212, 0x01248A1B00000000 };
__constant__ uint64_t __mu64[4] = { 0x0000000100000000, 0x000000007BC6C000, 0x8AF43417A1087212, 0x91248A1B00000000 };

// just testing
uint64_t __cpu_p64[4] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF7BC6BFFF, 0x8AF43417A1087212, 0x01248A1B00000000 };
uint64_t __cpu_mu64[4] = { 0x0000000100000000, 0x000000007BC6C000, 0x8AF43417A1087212, 0x91248A1B00000000 };


// jeah i mean not really a karatsuba multiplication, but never the less a usefull
// function. c0c1 = a0*b0
#define __karatsuba64_1x1(c0,c1,a0,b0)		\
	asm volatile (  "{\n\t"					\
			"mul.lo.u64 %0, %2, %3;\n\t"	\
			"mul.hi.u64 %1, %2, %3;\n\t"	\
			"}\n\t" 						\
		: "=l"(c0), "=l"(c1)				\
		: "l"(a0), "l"(b0));					

// 2limbs times 2limbs karatsuba multiplication 
#define __karatsuba64_2x2(c0,c1,c2,c3,a0,a1,b0,b1) 					\
	asm volatile("{\n\t"                                            \
			".reg .u64 Al, Bl, Ah, Bh, t0h, t1l, t1m, t2l;\n\t" 	\
			".reg .u64 t1m1, t1m2;\n\t" 	 						\
			"add.cc.u64	   	   Al, %4, %5 ;\n\t"					\
			"addc.u64          Ah,  0,  0 ;\n\t"					\
			"add.cc.u64	       Bl, %6, %7 ;\n\t"					\
			"addc.u64          Bh,  0,  0 ;\n\t"					\
			"mul.lo.u64        t1l, Al, Bl;\n\t" 					\
			"mul.hi.u64        t1m, Al, Bl;\n\t" 					\
			"mul.lo.u64        t1m1, Ah, Bl;\n\t"	 				\
			"mul.lo.u64        t1m2, Al, Bh;\n\t"					\
			"add.u64           t1m, t1m, t1m1;\n\t" 				\
			"add.u64           t1m, t1m, t1m2;\n\t" 				\
			"mul.lo.u64        %0,  %4, %6;\n\t"                    \
			"mul.hi.u64        t0h, %4, %6;\n\t" 					\
			"mul.lo.u64        t2l, %5, %7;\n\t" 					\
			"mul.hi.u64        %3,  %5, %7;\n\t" 					\
			/*AB - a1b1*/ 											\
			"sub.cc.u64        t1l, t1l, t2l;\n\t"                  \
			"subc.u64          t1m, t1m, %3 ;\n\t"                  \
			/*AB - a1b1 - a0b0*/ 									\
			"sub.cc.u64        t1l, t1l, %0 ;\n\t"                  \
			"subc.u64          t1m, t1m, t0h;\n\t"                  \
			"add.cc.u64 	   %1, t1l, t0h ;\n\t"					\
			"addc.u64 	   %2, t1m, t2l ;\n\t"						\
			"}\n"                                                   \
		: "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3) 	/*0,3*/     	\
		: "l"(a0), "l"(a1),    			 			/*4,5*/     	\
		  "l"(b0), "l"(b1),  			 			/*6,7*/     	\
		  "l"(c0), "l"(c1), "l"(c2), "l"(c3));   	/*8,11*/

// 1limb times 2limbs karatsuba multiplication 
#define __karatsuba64_1x2(c0,c1,c2,a0,b0,b1)\
	asm volatile("{\n\t"                                    \
			".reg .u64 Bl, Bh, t0h, t1l, t1m;\n\t" 			\
			"add.cc.u64	       Bl, %4, %5 ;\n\t"			\
			"addc.u64          Bh,  0,  0 ;\n\t"			\
			"mul.lo.u64        t1l, %3, Bl;\n\t" 			\
			"mul.hi.u64        t1m, %3, Bl;\n\t" 			\
			"mad.lo.cc.u64     t1m, %3, Bh, t1m;\n\t"		\
			"madc.hi.u64       %2,  %3, Bh, 0;\n\t" 		\
			"mul.lo.u64        %0,  %3, %4;\n\t"            \
			"mul.hi.u64        t0h, %3, %4;\n\t" 			\
			/*AB - a0b0*/ 									\
			"sub.cc.u64        t1l, t1l, %0 ;\n\t"          \
			"subc.cc.u64       t1m, t1m, t0h;\n\t"          \
			"subc.u64          %2, %8, 0    ;\n\t"          \
			"add.cc.u64 	   %1, t1l, t0h ;\n\t"			\
			"addc.u64 	       %2, t1m, 0   ;\n\t"			\
			"}\n"                                           \
		: "=l"(c0), "=l"(c1), "=l"(c2) 	/*0,2*/         	\
		: "l"(a0),    			 		/*3  */         	\
		  "l"(b0), "l"(b1) 			 	/*4,5*/ 			\
		  "l"(c0), "l"(c1), "l"(c2));   /*6,8*/  

// 1limb times 3limbs karatsuba multiplication 
#define __karatsuba64_1x3(c0,c1,c2,c3,a0,b0,b1,b2) 	\
	__karatsuba64_1x2(c0,c1,c2,a0,b0,b1) 			\
	asm volatile("{\n\t" 							\
			".reg .u64 t0, t1;\n\t" 				\
			"mul.lo.u64 t0, %2, %3;\n\t" 			\
			"mul.hi.u64 t1, %2, %3;\n\t" 			\
			"add.cc.u64 %0, %4, t0;\n\t" 			\
			"addc.u64   %1, %5, t1;\n\t" 			\
			"}\n" 									\
		: "=l"(c2), "=l"(c3) /*0,1*/ 				\
		: "l"(a0), "l"(b2),  /*2,3*/ 				\
		  "l"(c2), "l"(c3)); /*3,5*/

// 1limb times 4limbs karatsuba multiplication 
#define __karatsuba64_1x4(c0,c1,c2,c3,c4,a0,b0,b1,b2,b3) 	\
	__karatsuba64_1x3(c0,c1,c2,c3,a0,b0,b1,b2) 				\
	asm volatile("{\n\t" 									\
			".reg .u64 t0, t1;\n\t" 						\
			"mul.lo.u64 t0, %2, %3;\n\t" 					\
			"mul.hi.u64 t1, %2, %3;\n\t" 					\
			"add.cc.u64 %0, %4, t0;\n\t" 					\
			"addc.u64   %1, %5, t1;\n\t" 					\
			"}\n" 											\
		: "=l"(c3), "=l"(c4) /*0,1*/ 						\
		: "l"(a0), "l"(b3),  /*2,3*/ 						\
		  "l"(c3), "l"(c4)); /*4,5*/


// adds c0c1 = a0+b0b1
// needed for the karatsuba multiplication 3x3
#define __karatsuba64_3x3_2add_helper(c0,c1,a0,b0,b1) 	\
	asm volatile (	"{\n\t" 							\
			"add.cc.u64 	%0, %2, %3;\n\t"			\
			"addc.u64       %1, %4, 0 ;\n\t"			\
			"}\n\t" 									\
		: "=l"(c0), "=l"(c1) 			/*0,1*/			\
		: "l"(a0), "l"(b0), "l"(b1));   /*2,3,4*/

// karatsuba 3x3 multiplication without reduction
#define __karatsuba64_helper_3x3(c0,c1,c2,c3,c4,c5,a0,a1,a2,b0,b1,b2,kt0,kt1,st0,st1,rt0,rt1,rt2,rt3) \
	__karatsuba64_1x1(c0,c1,a0,b0)											\
	__karatsuba64_2x2(c2,c3,c4,c5,a1,a2,b1,b2) 								\
	__karatsuba64_3x3_2add_helper(kt0,kt1,a0,a1,a2)							\
	__karatsuba64_3x3_2add_helper(st0,st1,b0,b1,b2)							\
	__karatsuba64_2x2(rt0,rt1,rt2,rt3,kt0,kt1,st0,st1)						\
	/*UsVs -= U1V1*/ 														\
	asm volatile (	"{\n\t" 												\
			"sub.cc.u64  %0, %8,  %4;\n\t"  								\
			"subc.cc.u64 %1, %9,  %5;\n\t"									\
			"subc.cc.u64 %2, %10, %6;\n\t"									\
			"subc.u64    %3, %11, %7;\n\t"									\
			"}\n\t"															\
		: "=l"(rt0),"=l"(rt1),"=l"(rt2),"=l"(rt3) 	/*output 0,3 */     	\
		: "l"(c2),"l"(c3),"l"(c4),"l"(c5),        	/*U1V1   4,7 */ 		\
		  "l"(rt0),"l"(rt1),"l"(rt2),"l"(rt3)     	/*output 8,11*/   		\
	);																		\
	/*UsVs -= U0V0*/ 														\
	asm volatile (	"{\n\t" 												\
			"sub.cc.u64  %0, %6,  %4;\n\t"									\
			"subc.cc.u64 %1, %7,  %5;\n\t"									\
			"subc.cc.u64 %2, %8,   0;\n\t"									\
			"subc.u64    %3, %9,   0;\n\t"									\
			"}\n\t"															\
		: "=l"(rt0),"=l"(rt1),"=l"(rt2),"=l"(rt3) 	/*output 0,3*/   	  	\
		: "l"(c0),"l"(c1),                 	  		/*U0V0   4,5*/   	 	\
		  "l"(rt0),"l"(rt1),"l"(rt2),"l"(rt3)     	/*output 6,9*/   		\
	);																		\
	/*Final addition and result is written */ 								\
	asm volatile("{\n\t" 													\
			"add.cc.u64  %0, %8,  %4;\n\t" 									\
			"addc.cc.u64 %1, %9,  %5;\n\t" 									\
			"addc.cc.u64 %2, %10, %6;\n\t" 									\
			"addc.u64    %3, %11, %7;\n\t" 									\
			"}\n" 															\
		: "=l"(c1),"=l"(c2),"=l"(c3),"=l"(c4) /*0, 3*/    					\
		: "l"(rt0),"l"(rt1),"l"(rt2),"l"(rt3),/*4, 7*/						\
		  "l"(c1),"l"(c2),"l"(c3),"l"(c4));   /*8,11*/

// TODO explain
#define __karatsuba64_3x3(c0,c1,c2,c3,c4,c5,a0,a1,a2,b0,b1,b2) 	\
	uint64_t kt0,kt1,st0,st1,rt0,rt1,rt2,rt3;					\
	__karatsuba64_helper_3x3(c0,c1,c2,c3,c4,c5,a0,a1,a2,b0,b1,b2,kt0,kt1,st0,st1,rt0,rt1,rt2,rt3)

					
// c = a+b, where a and b are only two limb wide
#define __karatsuba64_4x4_2add_helper(c0,c1,c2,a0,a1,b0,b1) 	\
	asm volatile("{\n\t" 										\
			"add.cc.u64  %0, %3, %5;\n\t" 						\
			"addc.cc.u64 %1, %4, %6;\n\t" 						\
			"addc.u64    %2,  0,  0;\n\t" 						\
			"}\n" 												\
		: "=l"(c0), "=l"(c1), "=l"(c2)         /*0,2*/			\
		: "l"(a0), "l"(a1), "l"(b0), "l"(b1)); /*3,6*/

// karatsuba 4x4 multiplication without reduction
#define __karatsuba64_helper_4x4(c0,c1,c2,c3,c4,c5,c6,c7,a0,a1,a2,a3,b0,b1,b2,b3,ktmp0,ktmp1,ktmp2,stmp0,stmp1,stmp2,rtmp0,rtmp1,rtmp2,rtmp3) 	\
	__karatsuba64_2x2(c0,c1,c2,c3,a0,a1,b0,b1) 									\
	__karatsuba64_2x2(c4,c5,c6,c7,a2,a3,b2,b3) 									\
	__karatsuba64_4x4_2add_helper(ktmp0,ktmp1,ktmp2,a0,a1,a2,a3) 				\
	__karatsuba64_4x4_2add_helper(stmp0,stmp1,stmp2,b0,b1,b2,b3) 				\
	__karatsuba64_2x2(rtmp0,rtmp1,rtmp2,rtmp3,ktmp0,ktmp1,stmp0,stmp1)			\
	asm volatile("{\n\t"  														\
			/*UsVs - U1V1 - U0V0*/ 												\
			"sub.cc.u64  %0, %12, %4;\n\t" 										\
			"subc.cc.u64 %1, %13, %5;\n\t" 										\
			"subc.cc.u64 %2, %14, %6;\n\t" 										\
			"subc.u64    %3, %15, %7;\n\t" 										\
			"sub.cc.u64  %0, %0,  %8;\n\t" 										\
			"subc.cc.u64 %1, %1,  %9;\n\t" 										\
			"subc.cc.u64 %2, %2, %10;\n\t" 										\
			"subc.u64    %3, %3, %11;\n\t"	 									\
		     "}\n" 																\
		: "=l"(rtmp0), "=l"(rtmp1), "=l"(rtmp2), "=l"(rtmp3) /*output 0,3*/     \
		: "l"(c4), "l"(c5), "l"(c6), "l"(c7),                /*U1V1   4,7*/ 	\
		  "l"(c0), "l"(c1), "l"(c2), "l"(c3)                 /*U0V0   8,11*/    \
		  "l"(rtmp0), "l"(rtmp1), "l"(rtmp2), "l"(rtmp3)     /*output 12,15*/   \
		); 																		\
	/*Final addition and result is written */ 									\
	asm volatile("{\n\t" 														\
			"add.cc.u64  %0, %10, %6;\n\t" 										\
			"addc.cc.u64 %1, %11, %7;\n\t" 										\
			"addc.cc.u64 %2, %12, %8;\n\t" 										\
			"addc.cc.u64 %3, %13, %9;\n\t" 										\
			"addc.cc.u64 %4, %14, 0 ;\n\t" 										\
			"addc.u64    %5, %15, 0 ;\n\t" 										\
		     "}\n" 																\
	: "=l"(c2), "=l"(c3), "=l"(c4), "=l"(c5), "=l"(c6), "=l"(c7) 	/*0,5*/ \
	: "l"(rtmp0), "l"(rtmp1), "l"(rtmp2), "l"(rtmp3) 	     		/*6,9*/	\
	  "l"(c2), "l"(c3), "l"(c4), "l"(c5), "l"(c6), "l"(c7));     	/*10,15*/


// wrapper function for the karatsube 4x4 64bit limbs multiplication, allocating the needed helper registers
#define __karatsuba64_4x4(c0,c1,c2,c3,c4,c5,c6,c7,a0,a1,a2,a3,b0,b1,b2,b3) 	\
	uint64_t ktmp0,ktmp1,ktmp2,stmp0,stmp1,stmp2,rtmp0,rtmp1,rtmp2,rtmp3; 	\
	__karatsuba64_helper_4x4(c0,c1,c2,c3,c4,c5,c6,c7,a0,a1,a2,a3,b0,b1,b2,b3,ktmp0,ktmp1,ktmp2,stmp0,stmp1,stmp2,rtmp0,rtmp1,rtmp2,rtmp3)

			
