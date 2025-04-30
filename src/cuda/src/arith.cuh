// Generic three limb addition base on 32bit operations.
#define __fp_add64_32(c0, c1, a0, a1, b0, b1, p0, p1)    \
asm volatile("{\n\t"                            \
         ".reg .u32     t0, t1;     \n\t"       \
         "add.cc.u32    %0, %2, %4; \n\t"       \
         "addc.cc.u32   %1, %3, %5; \n\t"       \
         "addc.u32      t0,  0,  0; \n\t"       \
         "sub.cc.u32    %0, %0, %6; \n\t"       \
         "subc.cc.u32   %1, %1, %7; \n\t"       \
         "subc.u32      t0, t0,  0; \n\t"       \
         "and.b32       t1, t0, %7; \n\t"       \
         "and.b32       t0, t0, %6; \n\t"       \
         "add.cc.u32    %0, %0, t0; \n\t"       \
         "addc.u32      %1, %1, t1; \n\t"       \
         "}"                                    \
    : "=r"(c0), "=r"(c1)  /*0,1*/               \
    : "r"(a0), "r"(a1),   /*2,3*/               \
      "r"(b0), "r"(b1),   /*4,5*/               \
      "r"(p0), "r"(p1));  /*6,7*/

// Generic three limb addition base on 32bit operations.
#define __fp_add96_32(c0, c1, c2, a0, a1, a2, b0, b1, b2, p0, p1, p2)          \
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2; \n\t"                           \
         "add.cc.u32    %0, %3, %6; \n\t"                           \
         "addc.cc.u32   %1, %4, %7; \n\t"                           \
         "addc.cc.u32   %2, %5, %8; \n\t"                           \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %9; \n\t"                           \
         "subc.cc.u32   %1, %1, %10;\n\t"                           \
         "subc.cc.u32   %2, %2, %11;\n\t"                           \
         "subc.u32      t0, t0,  0; \n\t"                           \
         "and.b32       t1, t0, %10;\n\t"                           \
         "and.b32       t2, t0, %11;\n\t"                           \
         "and.b32       t0, t0, %9; \n\t"                           \
         "add.cc.u32    %0, %0, t0; \n\t"                           \
         "addc.cc.u32   %1, %1, t1; \n\t"                           \
         "addc.u32      %2, %2, t2; \n\t"                           \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2) /*0,2*/                          \
    : "r"(a0), "r"(a1), "r"(a2),   /*3,5*/                          \
      "r"(b0), "r"(b1), "r"(b2),   /*6,8*/                          \
      "r"(p0), "r"(p1), "r"(p2));  /*9,11*/

// Generic four limb addition base on 32bit operations.
#define __fp_add128_32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, p0, p1, p2, p3)\
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2, t3;\n\t"                        \
         "add.cc.u32    %0, %4, %8; \n\t"                           \
         "addc.cc.u32   %1, %5, %9; \n\t"                           \
         "addc.cc.u32   %2, %6, %10;\n\t"                           \
         "addc.cc.u32   %3, %7, %11;\n\t"                           \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %12;\n\t"                           \
         "subc.cc.u32   %1, %1, %13;\n\t"                           \
         "subc.cc.u32   %2, %2, %14;\n\t"                           \
         "subc.cc.u32   %3, %3, %15;\n\t"                           \
         "subc.u32      t0, t0,  0;\n\t"                            \
         "and.b32       t1, t0, %13;\n\t"                           \
         "and.b32       t2, t0, %14;\n\t"                           \
         "and.b32       t3, t0, %15;\n\t"                           \
         "and.b32       t0, t0, %12;\n\t"                           \
         "add.cc.u32    %0, %0, t0;\n\t"                            \
         "addc.cc.u32   %1, %1, t1;\n\t"                            \
         "addc.cc.u32   %2, %2, t2;\n\t"                            \
         "addc.u32      %3, %3, t3;\n\t"                            \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0,3*/                \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),    /*4,7*/                \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3),    /*8,11*/               \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3));   /*12,15*/

// Generic 160bit (5 limb) addition base on 32bit operations.
#define __fp_add160_32(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, p0, p1, p2, p3, p4)\
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2, t3, t4;\n\t"                    \
         "add.cc.u32    %0, %5, %10; \n\t"                          \
         "addc.cc.u32   %1, %6, %11; \n\t"                          \
         "addc.cc.u32   %2, %7, %12;\n\t"                           \
         "addc.cc.u32   %3, %8, %13;\n\t"                           \
         "addc.cc.u32   %4, %9, %14;\n\t"                           \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %15;\n\t"                           \
         "subc.cc.u32   %1, %1, %16;\n\t"                           \
         "subc.cc.u32   %2, %2, %17;\n\t"                           \
         "subc.cc.u32   %3, %3, %18;\n\t"                           \
         "subc.cc.u32   %4, %4, %19;\n\t"                           \
         "subc.u32      t0, t0,  0;\n\t"                            \
         "and.b32       t1, t0, %16;\n\t"                           \
         "and.b32       t2, t0, %17;\n\t"                           \
         "and.b32       t3, t0, %18;\n\t"                           \
         "and.b32       t4, t0, %19;\n\t"                           \
         "and.b32       t0, t0, %15;\n\t"                           \
         "add.cc.u32    %0, %0, t0;\n\t"                            \
         "addc.cc.u32   %1, %1, t1;\n\t"                            \
         "addc.cc.u32   %2, %2, t2;\n\t"                            \
         "addc.cc.u32   %3, %3, t3;\n\t"                            \
         "addc.u32      %4, %4, t4;\n\t"                            \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4)  /*0,4*/     \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4),      /*5,9*/     \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4),      /*10,14*/   \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4));     /*15,19*/

// Generic 192bit (6 limb) addition base on 32bit operations.
#define __fp_add192_32(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, p0, p1, p2, p3, p4, p5)\
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2, t3, t4, t5;\n\t"                \
         "add.cc.u32    %0, %6, %12;\n\t"                           \
         "addc.cc.u32   %1, %7, %13;\n\t"                           \
         "addc.cc.u32   %2, %8, %14;\n\t"                           \
         "addc.cc.u32   %3, %9, %15;\n\t"                           \
         "addc.cc.u32   %4, %10, %16;\n\t"                          \
         "addc.cc.u32   %5, %11, %17;\n\t"                          \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %18;\n\t"                           \
         "subc.cc.u32   %1, %1, %19;\n\t"                           \
         "subc.cc.u32   %2, %2, %20;\n\t"                           \
         "subc.cc.u32   %3, %3, %21;\n\t"                           \
         "subc.cc.u32   %4, %4, %22;\n\t"                           \
         "subc.cc.u32   %5, %5, %23;\n\t"                           \
         "subc.u32      t0, t0,  0; \n\t"                           \
         "and.b32       t1, t0, %19;\n\t"                           \
         "and.b32       t2, t0, %20;\n\t"                           \
         "and.b32       t3, t0, %21;\n\t"                           \
         "and.b32       t4, t0, %22;\n\t"                           \
         "and.b32       t5, t0, %23;\n\t"                           \
         "and.b32       t0, t0, %18;\n\t"                           \
         "add.cc.u32    %0, %0, t0;\n\t"                            \
         "addc.cc.u32   %1, %1, t1;\n\t"                            \
         "addc.cc.u32   %2, %2, t2;\n\t"                            \
         "addc.cc.u32   %3, %3, t3;\n\t"                            \
         "addc.cc.u32   %4, %4, t4;\n\t"                            \
         "addc.u32      %5, %5, t5;\n\t"                            \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5) /*0,5*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5),      /*6,11*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5),      /*12,17*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5));     /*18,23*/

// Generic 224bit (7 limb) addition base on 32bit operations.
#define __fp_add224_32(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6, p0, p1, p2, p3, p4, p5, p6)\
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2, t3, t4, t5, t6;\n\t"            \
         "add.cc.u32    %0, %7, %14; \n\t"                          \
         "addc.cc.u32   %1, %8, %15; \n\t"                          \
         "addc.cc.u32   %2, %9, %16; \n\t"                          \
         "addc.cc.u32   %3, %10, %17;\n\t"                          \
         "addc.cc.u32   %4, %11, %18;\n\t"                          \
         "addc.cc.u32   %5, %12, %19;\n\t"                          \
         "addc.cc.u32   %6, %13, %20;\n\t"                          \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %21;\n\t"                           \
         "subc.cc.u32   %1, %1, %22;\n\t"                           \
         "subc.cc.u32   %2, %2, %23;\n\t"                           \
         "subc.cc.u32   %3, %3, %24;\n\t"                           \
         "subc.cc.u32   %4, %4, %25;\n\t"                           \
         "subc.cc.u32   %5, %5, %26;\n\t"                           \
         "subc.cc.u32   %6, %6, %27;\n\t"                           \
         "subc.u32      t0, t0,  0;\n\t"                            \
         "and.b32       t1, t0, %22;\n\t"                           \
         "and.b32       t2, t0, %23;\n\t"                           \
         "and.b32       t3, t0, %24;\n\t"                           \
         "and.b32       t4, t0, %25;\n\t"                           \
         "and.b32       t5, t0, %26;\n\t"                           \
         "and.b32       t6, t0, %27;\n\t"                           \
         "and.b32       t0, t0, %21;\n\t"                           \
         "add.cc.u32    %0, %0, t0;\n\t"                            \
         "addc.cc.u32   %1, %1, t1;\n\t"                            \
         "addc.cc.u32   %2, %2, t2;\n\t"                            \
         "addc.cc.u32   %3, %3, t3;\n\t"                            \
         "addc.cc.u32   %4, %4, t4;\n\t"                            \
         "addc.cc.u32   %5, %5, t5;\n\t"                            \
         "addc.u32      %6, %6, t6;\n\t"                            \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), "=r"(c6) /*0,6*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6),       /*7,13*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6),       /*14,20*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6));      /*21,27*/

// Generic 256bit (8 limb) addition base on 32bit operations.
#define __fp_add256_32(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, p0, p1, p2, p3, p4, p5, p6, p7)\
    asm volatile("{\n\t"                                            \
         ".reg .u32     t0, t1, t2, t3, t4, t5, t6, t7;\n\t"        \
         "add.cc.u32    %0, %8,  %16;\n\t"                          \
         "addc.cc.u32   %1, %9,  %17;\n\t"                          \
         "addc.cc.u32   %2, %10, %18;\n\t"                          \
         "addc.cc.u32   %3, %11, %19;\n\t"                          \
         "addc.cc.u32   %4, %12, %20;\n\t"                          \
         "addc.cc.u32   %5, %13, %21;\n\t"                          \
         "addc.cc.u32   %6, %14, %22;\n\t"                          \
         "addc.cc.u32   %7, %15, %23;\n\t"                          \
         "addc.u32      t0,  0,  0; \n\t"                           \
         "sub.cc.u32    %0, %0, %24;\n\t"                           \
         "subc.cc.u32   %1, %1, %25;\n\t"                           \
         "subc.cc.u32   %2, %2, %26;\n\t"                           \
         "subc.cc.u32   %3, %3, %27;\n\t"                           \
         "subc.cc.u32   %4, %4, %28;\n\t"                           \
         "subc.cc.u32   %5, %5, %29;\n\t"                           \
         "subc.cc.u32   %6, %6, %30;\n\t"                           \
         "subc.cc.u32   %7, %7, %31;\n\t"                           \
         "subc.u32      t0, t0,  0;\n\t"                            \
         "and.b32       t1, t0, %25;\n\t"                           \
         "and.b32       t2, t0, %26;\n\t"                           \
         "and.b32       t3, t0, %27;\n\t"                           \
         "and.b32       t4, t0, %28;\n\t"                           \
         "and.b32       t5, t0, %29;\n\t"                           \
         "and.b32       t6, t0, %30;\n\t"                           \
         "and.b32       t7, t0, %31;\n\t"                           \
         "and.b32       t0, t0, %24;\n\t"                           \
         "add.cc.u32    %0, %0, t0;\n\t"                            \
         "addc.cc.u32   %1, %1, t1;\n\t"                            \
         "addc.cc.u32   %2, %2, t2;\n\t"                            \
         "addc.cc.u32   %3, %3, t3;\n\t"                            \
         "addc.cc.u32   %4, %4, t4;\n\t"                            \
         "addc.cc.u32   %5, %5, t5;\n\t"                            \
         "addc.cc.u32   %6, %6, t6;\n\t"                            \
         "addc.u32      %7, %7, t7;\n\t"                            \
         "}"                                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)/*0,7*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),       /*8,15*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7),       /*16,23*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6), "r"(p7));      /*24,31*/


/* Addition in Fp */
#define __fp_add192_64(c0, c1, c2, a0, a1, a2, b0, b1, b2, p0, p1, p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;\n\t"                     \
        "add.cc.u64     %0, %3, %6;\n\t"        \
        "addc.cc.u64    %1, %4, %7;\n\t"        \
        "addc.cc.u64    %2, %5, %8;\n\t"        \
        "addc.u64       t0, 0, 0;\n\t"          \
        "sub.cc.u64     %0, %0, %9;\n\t"        \
        "subc.cc.u64    %1, %1, %10;\n\t"       \
        "subc.cc.u64    %2, %2, %11;\n\t"       \
        "subc.u64       t0, t0, 0;\n\t"         \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "and.b64        t0, t0, %9;\n\t"        \
        "and.b64        t1, t1, %10;\n\t"       \
        "and.b64        t2, t2, %11;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.u64       %2, %2, t2;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)          \
        : "l"(a0), "l"(a1), "l"(a2), "l"(b0), "l"(b1), "l"(b2),\
            "l"(p0), "l"(p1), "l"(p2) )

/* Addition in Fp */
#define __fp_add256_64(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, p0, p1, p2, p3)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        "add.cc.u64     %0, %4, %8;\n\t"        \
        "addc.cc.u64    %1, %5, %9;\n\t"        \
        "addc.cc.u64    %2, %6, %10;\n\t"       \
        "addc.cc.u64    %3, %7, %11;\n\t"       \
        "addc.u64       t0, 0, 0;\n\t"          \
        "sub.cc.u64     %0, %0, %12;\n\t"       \
        "subc.cc.u64    %1, %1, %13;\n\t"       \
        "subc.cc.u64    %2, %2, %14;\n\t"       \
        "subc.cc.u64    %3, %3, %15;\n\t"       \
        "subc.u64       t0, t0, 0;\n\t"         \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "mov.u64        t3, t0;\n\t"            \
        "and.b64        t0, t0, %12;\n\t"       \
        "and.b64        t1, t1, %13;\n\t"       \
        "and.b64        t2, t2, %14;\n\t"       \
        "and.b64        t3, t3, %15;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.cc.u64    %2, %2, t2;\n\t"        \
        "addc.u64       %3, %3, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
            "l"(p0), "l"(p1), "l"(p2), "l"(p3) )


// Generic 96bit, 2 limb subtraction base on 32bit operations.
#define __fp_sub64_32(c0, c1, a0, a1, b0, b1, p0, p1)\
    asm volatile ("{\n\t"                       \
    ".reg .u32      t0, t1;     \n\t"           \
    "sub.cc.u32     %0, %2, %4; \n\t"           \
    "subc.cc.u32    %1, %3, %5; \n\t"           \
    "subc.u32       t0, 0, 0;   \n\t"           \
    "and.b32        t1, t0, %7 ;\n\t"           \
    "and.b32        t0, t0, %6 ;\n\t"           \
    "add.cc.u32     %0, %0, t0; \n\t"           \
    "addc.u32       %1, %1, t1; \n\t"           \
    "}"                                         \
    : "=r"(c0), "=r"(c1)  /*0-1*/               \
    : "r"(a0), "r"(a1),   /*2-3*/               \
      "r"(b0), "r"(b1),   /*4-5*/               \
      "r"(p0), "r"(p1));  /*6-7*/

// Generic 96bit, 3 limb subtraction base on 32bit operations.
#define __fp_sub96_32(c0, c1, c2, a0, a1, a2, b0, b1, b2, p0, p1, p2)\
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2; \n\t"                           \
    "sub.cc.u32     %0, %3, %6; \n\t"                           \
    "subc.cc.u32    %1, %4, %7; \n\t"                           \
    "subc.cc.u32    %2, %5, %8; \n\t"                           \
    "subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %10;\n\t"                           \
    "and.b32        t2, t0, %11;\n\t"                           \
    "and.b32        t0, t0, %9 ;\n\t"                           \
    "add.cc.u32     %0, %0, t0;\n\t"                            \
    "addc.cc.u32    %1, %1, t1;\n\t"                            \
    "addc.u32       %2, %2, t2;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2) /*0-2*/                      \
    : "r"(a0), "r"(a1), "r"(a2),   /*3-5*/                      \
      "r"(b0), "r"(b1), "r"(b2),   /*6-8*/                      \
      "r"(p0), "r"(p1), "r"(p2));  /*9-11*/

// Generic 128bit  4 limb subtraction base on 32bit operations.
#define __fp_sub128_32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, p0, p1, p2, p3)\
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2, t3;\n\t"                        \
    "sub.cc.u32     %0, %4, %8; \n\t"                           \
    "subc.cc.u32    %1, %5, %9; \n\t"                           \
    "subc.cc.u32    %2, %6, %10;\n\t"                           \
    "subc.cc.u32    %3, %7, %11;\n\t"                           \
    "subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %13;\n\t"                           \
    "and.b32        t2, t0, %14;\n\t"                           \
    "and.b32        t3, t0, %15;\n\t"                           \
    "and.b32        t0, t0, %12;\n\t"                           \
    "add.cc.u32     %0, %0, t0;\n\t"                            \
    "addc.cc.u32    %1, %1, t1;\n\t"                            \
    "addc.cc.u32    %2, %2, t2;\n\t"                            \
    "addc.u32       %3, %3, t3;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0-3*/            \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),    /*4-7*/            \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3),    /*8-11*/           \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3));   /*12-15*/

// Generic 160bit, 5 limb subtraction base on 32bit operations.
#define __fp_sub160_32(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, p0, p1, p2, p3, p4)\
    asm volatile ("{\n\t"                                           \
    ".reg .u32      t0, t1, t2, t3, t4;\n\t"                        \
    "sub.cc.u32     %0, %5, %10;\n\t"                               \
    "subc.cc.u32    %1, %6, %11;\n\t"                               \
    "subc.cc.u32    %2, %7, %12;\n\t"                               \
    "subc.cc.u32    %3, %8, %13;\n\t"                               \
    "subc.cc.u32    %4, %9, %14;\n\t"                               \
    "subc.u32       t0, 0, 0;\n\t"                                  \
    "and.b32        t1, t0, %16;\n\t"                               \
    "and.b32        t2, t0, %17;\n\t"                               \
    "and.b32        t3, t0, %18;\n\t"                               \
    "and.b32        t4, t0, %19;\n\t"                               \
    "and.b32        t0, t0, %15;\n\t"                               \
    "add.cc.u32     %0, %0, t0;\n\t"                                \
    "addc.cc.u32    %1, %1, t1;\n\t"                                \
    "addc.cc.u32    %2, %2, t2;\n\t"                                \
    "addc.cc.u32    %3, %3, t3;\n\t"                                \
    "addc.u32       %4, %4, t4;\n\t"                                \
    "}"                                                             \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4)  /*0,4*/     \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4),      /*5,9*/     \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4),      /*10,14*/   \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4));     /*15,19*/

// Generic 192bit, 6 limb subtraction base on 32bit operations.
#define __fp_sub192_32(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, p0, p1, p2, p3, p4, p5)\
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2, t3, t4, t5;\n\t"                \
    "sub.cc.u32     %0, %6, %12 ;\n\t"                          \
    "subc.cc.u32    %1, %7, %13 ;\n\t"                          \
    "subc.cc.u32    %2, %8, %14 ;\n\t"                          \
    "subc.cc.u32    %3, %9, %15 ;\n\t"                          \
    "subc.cc.u32    %4, %10, %16;\n\t"                          \
    "subc.cc.u32    %5, %11, %17;\n\t"                          \
    "subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %19;\n\t"                           \
    "and.b32        t2, t0, %20;\n\t"                           \
    "and.b32        t3, t0, %21;\n\t"                           \
    "and.b32        t4, t0, %22;\n\t"                           \
    "and.b32        t5, t0, %23;\n\t"                           \
    "and.b32        t0, t0, %18;\n\t"                           \
    "add.cc.u32     %0, %0, t0;\n\t"                            \
    "addc.cc.u32    %1, %1, t1;\n\t"                            \
    "addc.cc.u32    %2, %2, t2;\n\t"                            \
    "addc.cc.u32    %3, %3, t3;\n\t"                            \
    "addc.cc.u32    %4, %4, t4;\n\t"                            \
    "addc.u32       %5, %5, t5;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5) /*0,5*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5),      /*6,11*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5),      /*12,17*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5));     /*18,23*/

// Generic 224bit, 7 limb subtraction base on 32bit operations.
#define __fp_sub224_32(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6, p0, p1, p2, p3, p4, p5, p6)\
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2, t3, t4, t5, t6;\n\t"            \
    "sub.cc.u32     %0, %7, %14 ;\n\t"                          \
    "subc.cc.u32    %1, %8, %15 ;\n\t"                          \
    "subc.cc.u32    %2, %9, %16 ;\n\t"                          \
    "subc.cc.u32    %3, %10, %17;\n\t"                          \
    "subc.cc.u32    %4, %11, %18;\n\t"                          \
    "subc.cc.u32    %5, %12, %19;\n\t"                          \
    "subc.cc.u32    %6, %13, %20;\n\t"                          \
    "subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %22;\n\t"                           \
    "and.b32        t2, t0, %23;\n\t"                           \
    "and.b32        t3, t0, %24;\n\t"                           \
    "and.b32        t4, t0, %25;\n\t"                           \
    "and.b32        t5, t0, %26;\n\t"                           \
    "and.b32        t6, t0, %27;\n\t"                           \
    "and.b32        t0, t0, %21;\n\t"                           \
    "add.cc.u32     %0, %0, t0;\n\t"                            \
    "addc.cc.u32    %1, %1, t1;\n\t"                            \
    "addc.cc.u32    %2, %2, t2;\n\t"                            \
    "addc.cc.u32    %3, %3, t3;\n\t"                            \
    "addc.cc.u32    %4, %4, t4;\n\t"                            \
    "addc.cc.u32    %5, %5, t5;\n\t"                            \
    "addc.u32       %6, %6, t6;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), "=r"(c6) /*0,6*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6),       /*7,13*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6),       /*14,20*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6));      /*21,27*/

// Generic 256bit, 8 limb subtraction base on 32bit operations.
#define __fp_sub256_32(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, p0, p1, p2, p3, p4, p5, p6, p7)\
    asm volatile ("{\n\t"                                       \
    ".reg .u32      t0, t1, t2, t3, t4, t5, t6, t7;\n\t"        \
    "sub.cc.u32     %0, %8,  %16;\n\t"                          \
    "subc.cc.u32    %1, %9,  %17;\n\t"                          \
    "subc.cc.u32    %2, %10, %18;\n\t"                          \
    "subc.cc.u32    %3, %11, %19;\n\t"                          \
    "subc.cc.u32    %4, %12, %20;\n\t"                          \
    "subc.cc.u32    %5, %13, %21;\n\t"                          \
    "subc.cc.u32    %6, %14, %22;\n\t"                          \
    "subc.cc.u32    %7, %15, %23;\n\t"                          \
    "subc.u32       t0, 0, 0;\n\t"                              \
    "and.b32        t1, t0, %25;\n\t"                           \
    "and.b32        t2, t0, %26;\n\t"                           \
    "and.b32        t3, t0, %27;\n\t"                           \
    "and.b32        t4, t0, %28;\n\t"                           \
    "and.b32        t5, t0, %29;\n\t"                           \
    "and.b32        t6, t0, %30;\n\t"                           \
    "and.b32        t7, t0, %31;\n\t"                           \
    "and.b32        t0, t0, %24;\n\t"                           \
    "add.cc.u32     %0, %0, t0;\n\t"                            \
    "addc.cc.u32    %1, %1, t1;\n\t"                            \
    "addc.cc.u32    %2, %2, t2;\n\t"                            \
    "addc.cc.u32    %3, %3, t3;\n\t"                            \
    "addc.cc.u32    %4, %4, t4;\n\t"                            \
    "addc.cc.u32    %5, %5, t5;\n\t"                            \
    "addc.cc.u32    %6, %6, t6;\n\t"                            \
    "addc.u32       %7, %7, t7;\n\t"                            \
    "}"                                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7) /*0,7*/   \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),        /*8,15*/  \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7),        /*16,23*/ \
      "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6), "r"(p7));       /*24,31*/


/* Subtraction in Fp */
#define __fp_sub192_64(c0, c1, c2, a0, a1, a2, b0, b1, b2, p0, p1, p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;\n\t"                     \
        "sub.cc.u64     %0, %3, %6;\n\t"        \
        "subc.cc.u64    %1, %4, %7;\n\t"        \
        "subc.cc.u64    %2, %5, %8;\n\t"        \
        "subc.u64       t0, 0, 0;\n\t"          \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "and.b64        t0, t0, %9;\n\t"        \
        "and.b64        t1, t1, %10;\n\t"       \
        "and.b64        t2, t2, %11;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.u64       %2, %2, t2;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2)          \
        : "l"(a0), "l"(a1), "l"(a2), "l"(b0), "l"(b1), "l"(b2),\
            "l"(p0), "l"(p1), "l"(p2) )

/* Subtraction in Fp */
#define __fp_sub256_64(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, p0, p1, p2, p3)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        "sub.cc.u64     %0, %4, %8;\n\t"        \
        "subc.cc.u64    %1, %5, %9;\n\t"        \
        "subc.cc.u64    %2, %6, %10;\n\t"       \
        "subc.cc.u64    %3, %7, %11;\n\t"       \
        "subc.u64       t0, 0, 0;\n\t"          \
        "mov.u64        t1, t0;\n\t"            \
        "mov.u64        t2, t0;\n\t"            \
        "mov.u64        t3, t0;\n\t"            \
        "and.b64        t0, t0, %12;\n\t"       \
        "and.b64        t1, t1, %13;\n\t"       \
        "and.b64        t2, t2, %14;\n\t"       \
        "and.b64        t3, t3, %15;\n\t"       \
        "add.cc.u64     %0, %0, t0;\n\t"        \
        "addc.cc.u64    %1, %1, t1;\n\t"        \
        "addc.cc.u64    %2, %2, t2;\n\t"        \
        "addc.u64       %3, %3, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
            "l"(p0), "l"(p1), "l"(p2), "l"(p3) )



////////////////////////////////////////////////////////////////////////////////
// 									2x2										  //
////////////////////////////////////////////////////////////////////////////////
// 2x2 schoolbook multiplicaiton NOT using `mad instructions`
// im not adapting a mad version, beause no one will use it
// result is a 4 limb value
// needs 3 helper registers
// c3c2c1c0 = a1a0 * b1b0
#define __school32_2x2(c0, c1, c2, c3, a0, a1, b0, b1)      \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t0,t1,tc;\n\t"                       \
            /*Multiplication involving a0*/                 \
            "mul.lo.u32     %0, %4, %6;\n\t"                \
            "mul.hi.u32     t0, %4, %6;\n\t"                \
            "mul.lo.u32  	%1, %4, %7;\n\t"                \
            "mul.hi.u32     t1, %4, %7;\n\t"                \
            "add.cc.u32		t0, t0, %1;\n\t"                \
            "addc.cc.u32    t1, t1, 0;\n\t"                 \
            "addc.u32       tc, 0, 0;\n\t"                  \
            /*Multiplication involving a1 first diagonal*/  \
            "mul.lo.u32  	%1, %5, %6;\n\t"                \
            "mul.hi.u32 	%2, %5, %6;\n\t"                \
            "mul.hi.u32     %3, %5, %7;\n\t"                \
            "add.cc.u32		%1, %1, t0;\n\t"                \
            "addc.cc.u32	t1, t1, %2;\n\t"                \
            "addc.u32    	t0, %3, tc;\n\t"                \
            /*Multiplication involving a1 second diagonal*/ \
            "mul.lo.u32		%3, %5, %7;\n\t"                \
            "add.cc.u32		%2, %3, t1;\n\t"                \
            "addc.u32       %3, t0 , 0;\n\t"                \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1),"=r"(c2), "=r"(c3)    /*0,  3*/    \
    : "r"(a0), "r"(a1),                        /*4 , 5*/    \
      "r"(b0), "r"(b1));                       /*6 , 7*/


// 2x2 school book with `mad` instruction
#define __school32_2x2_v4(c0, c1, c2, c3, a0, a1, b0, b1)   \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t0,t1,tc;\n\t"                       \
            ".reg .u64 a;\n\t"                              \
            /*Multiplication involving a0*/                 \
            "mul.wide.u32 	a, %4, %6;\n\t"                 \
            "mov.b64		{t0, %0}, a;\n\t"               \
            "mad.lo.cc.u32  t0, %4, %7, t0;\n\t"            \
            "madc.hi.u32	t1, %4, %7, 0;\n\t"             \
            "addc.u32	    tc, 0, 0;\n\t"                  \
            /*Multiplication involving a1 first diagonal*/  \
            "mad.lo.cc.u32	%1, %5, %6, t0;\n\t"            \
            "madc.hi.cc.u32 t0, %5, %6, t1;\n\t"            \
            "madc.hi.u32 	t1, %5, %7, tc;\n\t"            \
            /*Multiplication involving a1 second diagonal*/ \
            "mad.lo.cc.u32  %2, %5, %7, t0;\n\t"            \
            "addc.u32			%3, t1, 0;\n\t"             \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1),"=r"(c2), "=r"(c3)     /*0,  3*/   \
    : "r"(a0), "r"(a1),                         /*4 , 5*/   \
      "r"(b0), "r"(b1));                        /*6 , 7*/


// 2x2 comba Multiplication with `mul.wide` instruction
// NOT  WORKING
#define __school32_2x2_v5(c0, c1, c2, c3, a0, a1, b0, b1)   \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t0;\n\t"                             \
            ".reg .u64 a0;\n\t"                             \
            "mul.wide.u32 	a0, %4, %6;\n\t"                \
            "mov.b64 		{t0, %0}, a0;\n\t"              \
            "mul.wide.u32 	a0, %5, %6;\n\t"                \
            "mov.b64 		{%2, %1}, a0;\n\t"              \
            "add.cc.u32     %1, %1, t0;\n\t"                \
            "addc.u32       %2, %2,  0;\n\t"                \
            "mul.wide.u32 	a0, %4, %7;\n\t"                \
            "mov.b64 		{%3, t0}, a0;\n\t"              \
            "add.cc.u32	    %1, %1, t0;\n\t"                \
            "addc.cc.u32    %2, %2, %3;\n\t"                \
            "mul.wide.u32   a0, %5, %7;\n\t"                \
            "mov.b64        {%3, t0}, a0;\n\t"              \
            "add.cc.u32     %2, %2, t0;\n\t"                \
            "addc.cc.u32    %3, %3,  0;\n\t"                \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1),"=r"(c2), "=r"(c3)     /*0,  3*/   \
    : "r"(a0), "r"(a1),                         /*4 , 5*/   \
      "r"(b0), "r"(b1));                        /*6 , 7*/


// 2x2 comba Multiplication with `mad` instruction
// no additional regisdters needed.
#define __school32_2x2_v6(c0, c1, c2, c3, a0, a1, b0, b1)   \
    asm ("{\n\t"                                            \
            "mul.lo.u32  	%0, %4, %6;\n\t"                \
            "mul.hi.u32  	%1, %4, %6;\n\t"                \
            "mad.lo.cc.u32  %1, %5, %6, %1;\n\t"            \
            "madc.hi.u32    %2, %5, %6, 0;\n\t"             \
            "mad.lo.cc.u32  %1, %4, %7, %1;\n\t"            \
            "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"            \
            "addc.u32       %3, 0, 0;\n\t"                  \
            "mad.lo.cc.u32  %2, %5, %7, %2;\n\t"            \
            "madc.hi.u32    %3, %5, %7, %3;\n\t"            \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1),"=r"(c2), "=r"(c3)     /*0 , 3*/   \
    : "r"(a0), "r"(a1),                         /*4 , 5*/   \
      "r"(b0), "r"(b1));                        /*6 , 7*/



// 2x2 comba Multiplication with `mad` instruction
// with the little twist, that its accumulative, meaning the result is added
// to all c registers
// needs no additonal registers
#define __school32_2x2_v7(c0, c1, c2, c3, a0, a1, b0, b1, d0, d1, d2, d3)            \
    asm ("{\n\t"                                            \
            "mad.lo.cc.u32  %0, %4, %6, %8;\n\t"            \
            "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"            \
            "addc.u32       %2, %10, 0;\n\t"                \
            "mad.lo.cc.u32  %1, %5, %6, %1;\n\t"            \
            "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"            \
            "addc.u32       %3, %11, 0;\n\t"                \
            "mad.lo.cc.u32  %1, %4, %7, %1;\n\t"            \
            "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"            \
            "addc.u32       %3, %11, 0;\n\t"                \
            "mad.lo.cc.u32  %2, %5, %7, %2;\n\t"            \
            "madc.hi.u32    %3, %5, %7, %3;\n\t"            \
        "}\n"                                               \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)       /*0,  3*/   \
    : "r"(a0), "r"(a1),                         /*4 , 5*/   \
      "r"(b0), "r"(b1),                         /*6 , 7*/   \
      "r"(d0), "r"(d1), "r"(d2), "r"(d3));      /*8, 11*/

// taken from https://cacr.uwaterloo.ca/hac/about/chap14.pdf
#define __school_sqr32_2x2(c0, c1, c2, c3, a0, a1)          \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 			u,v,c;\n\t"                 \
            "mul.lo.u32			%0, %4, %4;\n\t"            \
            "mul.hi.u32		 	c, %4, %4;\n\t"             \
            "mul.wide.u32		{%2,%1}, %4, %5;\n\t"       \
            "shf.l.clamp.b32  	%3, %2, %1, 1;\n\t"         \
            /* tmp = {vu%1} */                              \
            "add.cc.u32 		%1, %1, c;\n\t"             \
            "addc.cc.u32		%2, %2, 0;\n\t"             \
            "addc.u32			%3, %3, 0;\n\t"             \
            /* u lays over two limbs */                     \
            "mul.wide.u32		{u,v}, %5, %5;\n\t"         \
            "add.cc.u32			%2, %2, v;\n\t"             \
            "addc.u32 			%3, %3, v;\n\t"             \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1),"=r"(c2), "=r"(c3) /*0,  3*/       \
    : "r"(a0), "r"(a1));                    /*4 , 5*/       \


// jeah i mean not really a karatsuba multiplication, but never the less a
// usefull function.
// nedds 0 registers
// c0c1 = a0*b0
#define __karatsuba32_1x1(c0, c1, a0, b0)   \
    asm (  "{\n\t"                          \
            "mul.lo.u32 %0, %2, %3;\n\t"    \
            "mul.hi.u32 %1, %2, %3;\n\t"    \
            "}\n\t"                         \
            : "=r"(c0), "=r"(c1)            \
            : "r"(a0), "r"(b0));

// 2limbs times 2limbs karatsuba multiplication
// no expensive `mad` instruction
// needs 5 helper registers
// c3c2c1c0 = a1a0 * b1b0
#define __karatsuba32_2x2(c0, c1, c2, c3, a0, a1, b0, b1)                    \
    asm("{\n\t"                                                        \
            ".reg .u32 Al, Bl, Ah, Bh, t1l, t1m;\n\t"                \
            "add.cc.u32	   	   Al, %4, %5 ;\n\t"                    \
            "addc.u32          Ah,  0,  0 ;\n\t"                    \
            "add.cc.u32	       Bl, %6, %7 ;\n\t"                    \
            "addc.u32          Bh,  0,  0 ;\n\t"                    \
            "mul.lo.u32        t1l, Al, Bl;\n\t"                    \
            "mul.hi.u32        t1m, Al, Bl;\n\t"                    \
            /*"mad.lo.u32        t1m, Ah, Bl, t1m;\n\t"*/            \
            /*"mad.lo.u32        t1m, Al, Bh, t1m;\n\t"*/            \
            "mul.lo.u32        Ah, Ah, Bl;\n\t"                        \
            "add.u32           t1m, t1m, Ah;\n\t"                    \
            "mul.lo.u32        Ah, Al, Bh;\n\t"                        \
            "add.u32           t1m, t1m, Ah;\n\t"                    \
            "mul.lo.u32        %0,  %4, %6;\n\t"                    \
            "mul.hi.u32        Al, %4, %6;\n\t"                    \
            "mul.lo.u32        Ah, %5, %7;\n\t"                    \
            "mul.hi.u32        %3,  %5, %7;\n\t"                    \
            /*AB - a1b1*/                                            \
            "sub.cc.u32        t1l, t1l, Ah;\n\t"                   \
            "subc.u32          t1m, t1m, %3;\n\t"                   \
            /*AB - a1b1 - a0b0*/                                    \
            "sub.cc.u32        t1l, t1l, %0;\n\t"                   \
            "subc.u32          t1m, t1m, Al;\n\t"                   \
            "add.cc.u32 	   %1, t1l,  Al;\n\t"                    \
            "addc.cc.u32 	   %2, t1m,  Ah;\n\t"                    \
            "addc.u32 	   	   %3, %3,    0;\n\t"                    \
        "}\n"                                                    \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0,3*/            \
        : "r"(a0), "r"(a1),                         /*4,5*/            \
          "r"(b0), "r"(b1));                     /*6,11*/


// 2limbs times 2limbs karatsuba multiplication
// expensive `mad` instruction is used
// needs 5 helper registers
// c3c2c1c0 = a1a0 * b1b0
#define __karatsuba32_2x2_v2(c0, c1, c2, c3, a0, a1, b0, b1)                \
    asm("{\n\t"                                                        \
            ".reg .u32 Al, Bl, Ah, Bh, t1l, t1m;\n\t"                \
            "add.cc.u32	   	   Al, %4, %5 ;\n\t"                    \
            "addc.u32          Ah,  0,  0 ;\n\t"                    \
            "add.cc.u32	       Bl, %6, %7 ;\n\t"                    \
            "addc.u32          Bh,  0,  0 ;\n\t"                    \
            "mul.lo.u32        t1l, Al, Bl;\n\t"                    \
            "mul.hi.u32        t1m, Al, Bl;\n\t"                    \
            "mad.lo.u32        t1m, Ah, Bl, t1m;\n\t"                \
            "mad.lo.u32        t1m, Al, Bh, t1m;\n\t"                \
            "mul.lo.u32        %0,  %4, %6;\n\t"                    \
            "mul.hi.u32        Al, %4, %6;\n\t"                    \
            "mul.lo.u32        Ah, %5, %7;\n\t"                    \
            "mul.hi.u32        %3,  %5, %7;\n\t"                    \
            /*AB - a1b1*/                                            \
            "sub.cc.u32        t1l, t1l, Ah;\n\t"                   \
            "subc.u32          t1m, t1m, %3;\n\t"                   \
            /*AB - a1b1 - a0b0*/                                    \
            "sub.cc.u32        t1l, t1l, %0;\n\t"                   \
            "subc.u32          t1m, t1m, Al;\n\t"                   \
            "add.cc.u32 	   %1, t1l,  Al;\n\t"                    \
            "addc.cc.u32 	   %2, t1m,  Ah;\n\t"                    \
            "addc.u32 	   	   %3, %3,    0;\n\t"                    \
        "}\n"                                                    \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0,3*/            \
        : "r"(a0), "r"(a1),                         /*4,5*/            \
          "r"(b0), "r"(b1));                     /*6,11*/

// using the expensive mad instruction
#define __karatsuba32_2x2_v2(c0, c1, c2, c3, a0, a1, b0, b1)                \
    asm("{\n\t"                                                        \
            ".reg .u32 Al, Bl, Ah, Bh, t1l, t1m;\n\t"                \
            "add.cc.u32	   	   Al, %4, %5 ;\n\t"                    \
            "addc.u32          Ah,  0,  0 ;\n\t"                    \
            "add.cc.u32	       Bl, %6, %7 ;\n\t"                    \
            "addc.u32          Bh,  0,  0 ;\n\t"                    \
            "mul.lo.u32        t1l, Al, Bl;\n\t"                    \
            "mul.hi.u32        t1m, Al, Bl;\n\t"                    \
            "mad.lo.u32        t1m, Ah, Bl, t1m;\n\t"                \
            "mad.lo.u32        t1m, Al, Bh, t1m;\n\t"                \
            "mul.lo.u32        %0,  %4, %6;\n\t"                    \
            "mul.hi.u32        Al, %4, %6;\n\t"                    \
            "mul.lo.u32        Ah, %5, %7;\n\t"                    \
            "mul.hi.u32        %3,  %5, %7;\n\t"                    \
            /*AB - a1b1*/                                            \
            "sub.cc.u32        t1l, t1l, Ah;\n\t"                   \
            "subc.u32          t1m, t1m, %3;\n\t"                   \
            /*AB - a1b1 - a0b0*/                                    \
            "sub.cc.u32        t1l, t1l, %0;\n\t"                   \
            "subc.u32          t1m, t1m, Al;\n\t"                   \
            "add.cc.u32 	   %1, t1l,  Al;\n\t"                    \
            "addc.cc.u32 	   %2, t1m,  Ah;\n\t"                    \
            "addc.u32 	   	   %3, %3,    0;\n\t"                    \
        "}\n"                                                    \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0,3*/            \
        : "r"(a0), "r"(a1),                         /*4,5*/            \
          "r"(b0), "r"(b1));                     /*6,11*/

// TODO not finished
#define __fp_mul64_32(c0, c1, c2, c3, a0, a1, b0, b1, mu0, mu1) \
    __karatsuba32_2x2(c0,c1,c2,c3,a0,a1,b0,b1)

// 1limb times 2limbs karatsuba multiplication
#define __karatsuba32_1x2(c0, c1, c2, a0, b0, b1)\
    asm volatile("{\n\t"                                    \
            ".reg .u32 Bl, Bh, t0h, t1l, t1m;\n\t"            \
            "add.cc.u32	   Bl, %4, %5 ;\n\t"                \
            "addc.u32          Bh,  0,  0 ;\n\t"            \
            "mul.lo.u32        t1l, %3, Bl;\n\t"            \
            "mul.hi.u32        t1m, %3, Bl;\n\t"            \
            "mad.lo.cc.u32     t1m, %3, Bh, t1m;\n\t"        \
            "madc.hi.u32       %2,  %3, Bh, 0;\n\t"        \
            "mul.lo.u32        %0,  %3, %4;\n\t"            \
            "mul.hi.u32        t0h, %3, %4;\n\t"            \
            /*AB - a0b0*/                                    \
            "sub.cc.u32        t1l, t1l, %0 ;\n\t"          \
            "subc.cc.u32       t1m, t1m, t0h;\n\t"          \
            "subc.u32          %2, %8, 0    ;\n\t"          \
            "add.cc.u32 	   %1, t1l, t0h ;\n\t"            \
            "addc.u32 	   %2, t1m, 0   ;\n\t"                \
            "}\n"                                           \
            : "=r"(c0), "=r"(c1), "=r"(c2)    /*0,2*/         \
            : "r"(a0),                        /*3  */         \
              "r"(b0), "r"(b1)                /*4,5*/        \
              "r"(c0), "r"(c1), "r"(c2));   /*6,8*/

// same as `__karatsuba32_1x2` except its not using `mad` instructions
// 1limb times 2limbs karatsuba multiplication
#define __karatsuba32_1x2_v2(c0, c1, c2, a0, b0, b1)\
    asm volatile("{\n\t"                                    \
            ".reg .u32 Bl, Bh, t0h, t1l, t1m, t1, t2;\n\t"    \
            "add.cc.u32	   	   Bl, %4, %5 ;\n\t"            \
            "addc.u32          Bh,  0,  0 ;\n\t"            \
            "mul.lo.u32        t1l, %3, Bl;\n\t"            \
            "mul.hi.u32        t1m, %3, Bl;\n\t"            \
            "mul.lo.u32	       t1,  %3, Bh;\n\t"            \
            "mul.hi.u32        t2,  %3, Bh;\n\t"            \
            "add.cc.u32        t1m, t1m,t1;\n\t"            \
            "addc.u32          %2,  %2, t2;\n\t"            \
            /*"mad.lo.cc.u32     t1m, %3, Bh, t1m;\n\t"*/    \
            /*"madc.hi.u32       %2,  %3, Bh, 0;\n\t"  */    \
            "mul.lo.u32        %0,  %3, %4;\n\t"            \
            "mul.hi.u32        t0h, %3, %4;\n\t"            \
            /*AB - a0b0*/                                    \
            "sub.cc.u32        t1l, t1l, %0 ;\n\t"          \
            "subc.cc.u32       t1m, t1m, t0h;\n\t"          \
            "subc.u32          %2, %8, 0    ;\n\t"          \
            "add.cc.u32 	   %1, t1l, t0h ;\n\t"            \
            "addc.u32 	   	   %2, t1m, 0   ;\n\t"            \
            "}\n"                                           \
        : "=r"(c0), "=r"(c1), "=r"(c2)    /*0,2*/            \
        : "r"(a0),                        /*3  */            \
          "r"(b0), "r"(b1)                /*4,5*/            \
          "r"(c0), "r"(c1), "r"(c2));   /*6,8*/

// 1limb times 3limbs karatsuba multiplication
#define __karatsuba32_1x3(c0, c1, c2, c3, a0, b0, b1, b2)    \
    __karatsuba32_1x2(c0,c1,c2,a0,b0,b1)            \
    asm volatile("{\n\t"                            \
            ".reg .u32 t0, t1;\n\t"                \
            "mul.lo.u32 t0, %2, %3;\n\t"            \
            "mul.hi.u32 t1, %2, %3;\n\t"            \
            "add.cc.u32 %0, %4, t0;\n\t"            \
            "addc.u32   %1, %5, t1;\n\t"            \
            "}\n"                                    \
            : "=r"(c2), "=r"(c3) /*0,1*/            \
            : "r"(a0), "r"(b2),  /*2,3*/            \
              "r"(c2), "r"(c3)); /*3,5*/

// 1limb times 4limbs karatsuba multiplication
#define __karatsuba32_1x4(c0, c1, c2, c3, c4, a0, b0, b1, b2, b3)    \
    __karatsuba32_1x3(c0,c1,c2,c3,a0,b0,b1,b2)                \
    asm volatile("{\n\t"                                    \
            ".reg .u32 t0, t1;\n\t"                        \
            "mul.lo.u32 t0, %2, %3;\n\t"                    \
            "mul.hi.u32 t1, %2, %3;\n\t"                    \
            "add.cc.u32 %0, %4, t0;\n\t"                    \
            "addc.u32   %1, %5, t1;\n\t"                    \
            "}\n"                                            \
            : "=r"(c3), "=r"(c4) /*0,1*/                    \
            : "r"(a0), "r"(b3),  /*2,3*/                    \
              "r"(c3), "r"(c4)); /*4,5*/

// 1limb times 5limbs karatsuba multiplication
#define __karatsuba32_1x5(c0, c1, c2, c3, c4, c5, a0, b0, b1, b2, b3, b4)    \
    __karatsuba32_1x4(c0,c1,c2,c3,c4,a0,b0,b1,b2,b3)            \
    asm volatile("{\n\t"                                        \
            ".reg .u32 t0, t1;\n\t"                            \
            "mul.lo.u32 t0, %2, %3;\n\t"                        \
            "mul.hi.u32 t1, %2, %3;\n\t"                        \
            "add.cc.u32 %0, %4, t0;\n\t"                        \
            "addc.u32   %1, %5, t1;\n\t"                        \
            "}\n"                                                \
            : "=r"(c4), "=r"(c5) /*0,1*/                        \
            : "r"(a0), "r"(b4),  /*2,3*/                        \
              "r"(c4), "r"(c5)); /*4,5*/

// 1limb times 6limbs karatsuba multiplication
#define __karatsuba32_1x6(c0, c1, c2, c3, c4, c5, c6, a0, b0, b1, b2, b3, b4, b5)    \
    __karatsuba32_1x5(c0,c1,c2,c3,c4,c5,a0,b0,b1,b2,b3,b4)                \
    asm volatile("{\n\t"                                                \
            ".reg .u32 t0, t1;\n\t"                                    \
            "mul.lo.u32 t0, %2, %3;\n\t"                                \
            "mul.hi.u32 t1, %2, %3;\n\t"                                \
            "add.cc.u32 %0, %4, t0;\n\t"                                \
            "addc.u32   %1, %5, t1;\n\t"                                \
            "}\n"                                                        \
            : "=r"(c5), "=r"(c6) /*0,1*/                                \
            : "r"(a0), "r"(b5),  /*2,3*/                                \
              "r"(c5), "r"(c6)); /*4,5*/

// 1limb times 7limbs karatsuba multiplication
#define __karatsuba32_1x7(c0, c1, c2, c3, c4, c5, c6, c7, a0, b0, b1, b2, b3, b4, b5, b6)    \
    __karatsuba32_1x6(c0,c1,c2,c3,c4,c5,c6,a0,b0,b1,b2,b3,b4,b5)            \
    asm volatile("{\n\t"                                                    \
            ".reg .u32 t0, t1;\n\t"                                        \
            "mul.lo.u32 t0, %2, %3;\n\t"                                    \
            "mul.hi.u32 t1, %2, %3;\n\t"                                    \
            "add.cc.u32 %0, %4, t0;\n\t"                                    \
            "addc.u32   %1, %5, t1;\n\t"                                    \
            "}\n"                                                            \
            : "=r"(c6), "=r"(c7) /*0,1*/                                    \
            : "r"(a0), "r"(b6),  /*2,3*/                                    \
              "r"(c6), "r"(c7)); /*4,5*/

// 1limb times 8limbs karatsuba multiplication
#define __karatsuba32_1x8(c0, c1, c2, c3, c4, c5, c6, c7, c8, a0, b0, b1, b2, b3, b4, b5, b6, b7)    \
    __karatsuba32_1x7(c0,c1,c2,c3,c4,c5,c6,c7,a0,b0,b1,b2,b3,b4,b5,b6)                \
    asm volatile("{\n\t"                                                            \
            ".reg .u32 t0, t1;\n\t"                                                \
            "mul.lo.u32 t0, %2, %3;\n\t"                                            \
            "mul.hi.u32 t1, %2, %3;\n\t"                                            \
            "add.cc.u32 %0, %4, t0;\n\t"                                            \
            "addc.u32   %1, %5, t1;\n\t"                                            \
            "}\n"                                                                    \
            : "=r"(c7), "=r"(c8) /*0,1*/                                            \
            : "r"(a0), "r"(b7),  /*2,3*/                                            \
              "r"(c7), "r"(c7)); /*4,5*/

// taken form the internet, slower than my version
#define __school32_2x2_v2(c0, c1, c2, c3, a0, a1, b0, b1)                \
    asm ("{\n\t"                                                \
         ".reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\n\t"    \
         "mov.b32         alo, %4;\n\t"\
         "mov.b32         ahi, %5;\n\t"\
         "mov.b32         blo, %6;\n\t"\
         "mov.b32         bhi, %7;\n\t"\
         "mul.lo.u32      r0, alo, blo;\n\t"\
         "mul.hi.u32      r1, alo, blo; \n\t"\
         "mad.lo.cc.u32   r1, alo, bhi, r1;\n\t"\
         "madc.hi.u32     r2, alo, bhi, 0;\n\t"\
         "mad.lo.cc.u32   r1, ahi, blo, r1;\n\t"\
         "madc.hi.cc.u32  r2, ahi, blo, r2;\n\t"\
         "madc.hi.u32     r3, ahi, bhi, 0;\n\t"\
         "mad.lo.cc.u32   r2, ahi, bhi, r2;\n\t"\
         "addc.u32        r3, r3, 0;\n\t"\
         "mov.b32         %0, r0;\n\t"\
         "mov.b32         %1, r1;\n\t"\
         "mov.b32         %2, r2;\n\t"\
         "mov.b32         %3, r3;\n\t"\
         "}"\
         : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) \
         : "r"(a0), "r"(a1), "r"(b0), "r"(b1));

//taken from the internet/ slower than my version
#define __school32_2x2_v3(c0, c1, c2, c3, a0, a1, b0, b1)                    \
    asm ("{\n\t"                                                    \
         ".reg .u32       alo, ahi, blo, bhi, r0, r1, r2, r3;\n\t"  \
         ".reg .u32       s0, s1, s2, s3, t0, t1, t2, t3;\n\t"    \
         ".reg .u16       a0, a1, a2, a3, b0, b1, b2, b3;\n\t"      \
         /* split inputs into 16-bit chunks */                      \
         "mov.b32         {a0,a1}, %4;\n\t"\
         "mov.b32         {a2,a3}, %5;\n\t"\
         "mov.b32         {b0,b1}, %6;\n\t"\
         "mov.b32         {b2,b3}, %7;\n\t"\
         /* first partial sum:                          */\
         /* a3b3.wide  a1b3.wide  a0b2.wide  a0b0.wide  */\
         /*     0      a2b2.wide  a1b1.wide             */\
         /*     0      a3b1.wide  a2b0.wide             */\
         "mul.wide.u16    r0, a0, b0;\n\t"\
         "mul.wide.u16    r1, a0, b2;\n\t"\
         "mul.wide.u16    r2, a1, b3;\n\t"\
         "mul.wide.u16    r3, a3, b3;\n\t"\
         "mul.wide.u16    t1, a1, b1;\n\t"\
         "mul.wide.u16    t2, a2, b2;\n\t"\
         "add.cc.u32      r1, r1, t1;\n\t"\
         "addc.cc.u32     r2, r2, t2;\n\t"\
         "addc.u32        r3, r3, 0;\n\t" \
         "mul.wide.u16    t1, a2, b0;\n\t"\
         "mul.wide.u16    t2, a3, b1;\n\t"\
         "add.cc.u32      r1, r1, t1;\n\t"\
         "addc.cc.u32     r2, r2, t2;\n\t"\
         "addc.u32        r3, r3, 0;\n\t" \
         /* second partial sum:             */   \
         /* a2b3.wide  a0b3.wide  a0b1.wide */   \
         /* a3b2.wide  a1b2.wide  a1b0.wide */   \
         /*     0      a2b1.wide            */   \
         /*     0      a3b0.wide            */   \
         "mul.wide.u16    s0, a0, b1;\n\t" \
         "mul.wide.u16    s1, a0, b3;\n\t" \
         "mul.wide.u16    s2, a2, b3;\n\t" \
         "mul.wide.u16    t1, a2, b1;\n\t" \
         "add.cc.u32      s1, s1, t1;\n\t" \
         "addc.u32        s2, s2, 0;\n\t"  \
         "mul.wide.u16    t1, a3, b0;\n\t" \
         "add.cc.u32      s1, s1, t1;\n\t" \
         "addc.u32        s2, s2, 0;\n\t"  \
         "mul.wide.u16    t0, a1, b0;\n\t" \
         "mul.wide.u16    t1, a1, b2;\n\t" \
         "mul.wide.u16    t2, a3, b2;\n\t" \
         "add.cc.u32      s0, s0, t0;\n\t" \
         "addc.cc.u32     s1, s1, t1;\n\t" \
         "addc.cc.u32     s2, s2, t2;\n\t" \
         "addc.u32        s3, 0, 0;\n\t"   \
         /* offset second partial sum by 16 bits to the left*/ \
         "shf.l.clamp.b32 t3, s2, s3, 16;\n\t" \
         "shf.l.clamp.b32 t2, s1, s2, 16;\n\t" \
         "shf.l.clamp.b32 t1, s0, s1, 16;\n\t" \
         "shf.l.clamp.b32 t0,  0, s0, 16;\n\t" \
         /* add first sum in r{0,1,2,3} to second sum in t{0,1,2,3} */\
         "add.cc.u32      r0, r0, t0;\n\t" \
         "addc.cc.u32     r1, r1, t1;\n\t" \
         "addc.cc.u32     r2, r2, t2;\n\t" \
         "addc.u32        r3, r3, t3;\n\t" \
         /* pack outputs */ \
         "mov.b32         %0, r0;\n\t" \
         "mov.b32         %1, r1;\n\t" \
         "mov.b32         %2, r2;\n\t" \
         "mov.b32         %3, r3;\n\t" \
         "}" \
         : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) \
         : "r"(a0), "r"(a1), "r"(b0), "r"(b1));


#define __mul64_32(c0, c1, c2, c3, a0, a1, b0, b1)    \
    __karatsuba32_2x2_v2(c0,c1,c2,c3,a0,a1,b0,b1)

#define __sqr64_32(c0, c1, c2, c3, a0, a1)    \
    __school32_2x2_v2(c0,c1,c2,c3,a0,a1,a0,a1)

#define __fp_mul64_2x2(c0, c1, a0, a1, b0, b1, m0, m1, p0, p1)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        "mul.lo.u64     t0, %2, %4;\n\t"        \
        "mul.hi.u64     t1, %2, %4;\n\t"        \
        "mad.lo.cc.u64  t1, %2, %5, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %2, %5, 0;\n\t"     \
        "mad.lo.cc.u64  t1, %3, %4, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %3, %4, t2;\n\t"    \
        "madc.lo.cc.u64 t2, %3, %5, t2;\n\t"    \
        "madc.hi.u64    t3, %3, %5, 0;\n\t"     \
        "mul.lo.u64     %0, t0, %6;\n\t"/*Montgomery reduction, m = mu*c mod 2^(2*64)*/\
        "mul.hi.u64     %1, t0, %6;\n\t"        \
        "mad.lo.u64     %1, t0, %7, %1;\n\t"    \
        "mad.lo.u64     %1, t1, %6, %1;\n\t"    \
        "mul.lo.u64     t4, %0, %8;\n\t"/* u = m*p */\
        "mul.hi.u64     t5, %0, %8;\n\t"        \
        "mad.lo.cc.u64  t5, %0, %9, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %0, %9, 0;\n\t"     \
        "mad.lo.cc.u64  t5, %1, %8, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %1, %8, t6;\n\t"    \
        "madc.lo.cc.u64 t6, %1, %9, t6;\n\t"    \
        "madc.hi.u64    t7, %1, %9, 0;\n\t"     \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(2*64)*/\
        "add.cc.u64     t0, t4, t0;\n\t"        \
        "addc.cc.u64    t1, t5, t1;\n\t"        \
        "addc.cc.u64    t2, t6, t2;\n\t"        \
        "addc.cc.u64    t3, t7, t3;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t2, t2, %8;\n\t"        \
        "subc.cc.u64    t3, t3, %9;\n\t"        \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "and.b64        %0, %0, %8;\n\t"        \
        "and.b64        %1, %1, %9;\n\t"        \
        "add.cc.u64     %0, %0, t2;\n\t"        \
        "addc.u64       %1, %1, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)\
        : "l"(a0), "l"(a1), "l"(b0), "l"(b1),   \
          "l"(m0), "l"(m1), "l"(p0), "l"(p1) )

#define __fp_sqr64_2x2(c0, c1, a0, a1, m0, m1, p0, p1)\
    asm volatile ("{\n\t"                           \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"         \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"         \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"         \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"         \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"         \
        "mul.lo.u64     t0, %2, %2;\n\t"            \
        "mul.hi.u64     t1, %2, %2;\n\t"            \
        "mul.lo.u64     t4, %2, %3;\n\t"            \
        "mul.hi.u64     t5, %2, %3;\n\t"            \
        "add.cc.u64     t1, t1, t4;\n\t"            \
        "addc.cc.u64    t2, t5, 0;\n\t"             \
        "addc.cc.u64    t1, t1, t4;\n\t"            \
        "addc.cc.u64    t2, t2, t5;\n\t"            \
        "madc.lo.cc.u64 t2, %3, %3, t2;\n\t"        \
        "madc.hi.u64    t3, %3, %3, 0;\n\t"         \
        "mul.lo.u64     %0, t0, %4;\n\t"/*Montgomery reduction, m = mu*c mod 2^(2*64)*/\
        "mul.hi.u64     %1, t0, %4;\n\t"        \
        "mad.lo.u64     %1, t0, %5, %1;\n\t"    \
        "mad.lo.u64     %1, t1, %4, %1;\n\t"    \
        "mul.lo.u64     t4, %0, %6;\n\t"/* u = m*p */\
        "mul.hi.u64     t5, %0, %6;\n\t"        \
        "mad.lo.cc.u64  t5, %0, %7, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %0, %7, 0;\n\t"     \
        "madc.lo.cc.u64 t5, %1, %6, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %1, %6, t6;\n\t"    \
        "madc.lo.cc.u64 t6, %1, %7, t6;\n\t"    \
        "madc.hi.u64    t7, %1, %7, 0;\n\t"     \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(2*64)*/\
        "add.cc.u64     t0, t4, t0;\n\t"        \
        "addc.cc.u64    t1, t5, t1;\n\t"        \
        "addc.cc.u64    t2, t6, t2;\n\t"        \
        "addc.cc.u64    t3, t7, t3;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t2, t2, %6;\n\t"        \
        "subc.cc.u64    t3, t3, %7;\n\t"        \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "and.b64        %0, %0, %6;\n\t"        \
        "and.b64        %1, %1, %7;\n\t"        \
        "add.cc.u64     %0, %0, t2;\n\t"        \
        "addc.u64       %1, %1, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)                    \
        : "l"(a0), "l"(a1), "l"(m0), "l"(m1),   \
        "l"(p0), "l"(p1));

////////////////////////////////////////////////////////////////////////////////
// 									3x3										  //
////////////////////////////////////////////////////////////////////////////////
// 3x3 schoolbook multiplicaiton using `mad` instructions
// uses 4 tmp registers
// result is a 6 limb value
// c5c4c3c2c1c0 = a2a1a0 * b2b1b0
#define __school32_3x3(c0, c1, c2, c3, c4, c5, a0, a1, a2, b0, b1, b2) \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t0,t1,t2,tc;\n\t"                    \
            /*Multiplication involving a0*/                 \
            "mul.lo.u32     %0, %6, %9;\n\t"                \
            "mul.hi.u32     t0, %6, %9;\n\t"                \
            "mad.lo.cc.u32  t0, %6, %10, t0;\n\t"           \
            "mul.hi.u32     t1, %6, %10;\n\t"               \
            "madc.lo.cc.u32 t1, %6, %11, t1;\n\t"           \
            "madc.hi.cc.u32 t2, %6, %11, 0;\n\t"            \
            "addc.u32       tc, 0, 0;\n\t"                  \
            /*Multiplication involving a1 first diagonal*/  \
            "mad.lo.cc.u32  %1, %7, %9, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %7, %9, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %7, %10,t2;\n\t"            \
            "madc.hi.u32 	t0, %7, %11, tc;\n\t"           \
            /*Multiplication involving a1 second diagonal*/ \
            "mad.lo.cc.u32  t1, %7, %10, t1;\n\t"           \
            "madc.lo.cc.u32 t2, %7, %11, t2;\n\t"           \
            "addc.cc.u32    t0, t0 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a2 first diagonal*/  \
            "mad.lo.cc.u32  %2, %8, %9, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %8, %9, t2;\n\t"            \
            "madc.hi.cc.u32 t0, %8, %10,t0;\n\t"            \
            "madc.hi.u32  	t1, %8, %11,tc;\n\t"            \
            /*Multiplication involving a2 second diagonal*/ \
            "mad.lo.cc.u32  %3, %8, %10, t2;\n\t"           \
            "madc.lo.cc.u32 %4, %8, %11, t0;\n\t"           \
            "addc.u32       %5, t1 , 0;\n\t"                \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1), "=r"(c2), /*0,  2*/               \
      "=r"(c3), "=r"(c4), "=r"(c5)  /*3,  5*/               \
    : "r"(a0), "r"(a1), "r"(a2),    /*6 , 8*/               \
      "r"(b0), "r"(b1), "r"(b2));   /*9 ,11*/

// Multiplication from luis, with a few optimisations by floyd
// TODO reduce register count
// fastest of the 3x3 Multiplications
#define __school32_3x3_v2(c0, c1, c2, c3, c4, c5, a0, a1, a2, b0, b1, b2)            \
    asm volatile ("{\n\t"                           \
        ".reg .u32 t0;" ".reg .u32 t1;\n\t"         \
        ".reg .u32 t2;" ".reg .u32 t3;\n\t"         \
        ".reg .u32 t4;" ".reg .u32 t5;\n\t"         \
        ".reg .u32 t6;" ".reg .u32 t7;\n\t"         \
        ".reg .u32 t8;" ".reg .u32 t9;\n\t"         \
        ".reg .u32 t10;" ".reg .u32 t11;\n\t"       \
        "mul.lo.u32     t0, %6, %9;\n\t"            \
        "mul.hi.u32     t1, %6, %9;\n\t"            \
        "mad.lo.cc.u32  t1, %6, %10, t1;\n\t"       \
        "mul.hi.u32     t2, %6, %10;\n\t"           \
        "madc.lo.cc.u32 t2, %6, %11, t2;\n\t"       \
        "madc.hi.u32    t3, %6, %11, 0;\n\t"        \
        "mul.hi.u32     t5, %7, %9;\n\t"            \
        "mad.lo.cc.u32  t5, %7, %10, t5;\n\t"       \
        "mul.hi.u32     t6, %7, %10;\n\t"           \
        "madc.lo.cc.u32 t6, %7, %11, t6;"           \
        "madc.hi.u32    t7, %7, %11, 0;\n\t"        \
        "mad.lo.cc.u32  t1, %7, %9, t1;\n\t"        \
        "addc.cc.u32    t5, t5, t2;\n\t"            \
        "addc.cc.u32    t6, t6, t3;\n\t"            \
        "addc.u32       t7, t7, 0;\n\t"             \
        "mul.lo.u32     t2, %8, %9;\n\t"            \
        "mul.hi.u32     t3, %8, %9;\n\t"            \
        "mad.lo.cc.u32  t3, %8, %10, t3;\n\t"       \
        "mul.hi.u32     t4, %8, %10;\n\t"           \
        "madc.lo.cc.u32 t4, %8, %11, t4;\n\t"       \
        "madc.hi.u32    t8, %8, %11, 0;\n\t"        \
        "add.cc.u32     t2, t2, t5;\n\t"            \
        "addc.cc.u32    t3, t3, t6;\n\t"            \
        "addc.cc.u32    t4, t4, t7;\n\t"            \
        "addc.u32       t5, t8, 0;\n\t"             \
        "mov.u32 		%0, t0;\n\t"                \
        "mov.u32 		%1, t1;\n\t"                \
        "mov.u32 		%2, t2;\n\t"                \
        "mov.u32 		%3, t3;\n\t"                \
        "mov.u32 		%4, t4;\n\t"                \
        "mov.u32 		%5, t5;\n\t"                \
        "}\n\t"                                     \
    : "=r"(c0), "=r"(c1), "=r"(c2), /*0,  2*/       \
      "=r"(c3), "=r"(c4), "=r"(c5)  /*3,  5*/       \
    : "r"(a0), "r"(a1), "r"(a2),    /*6 , 8*/       \
      "r"(b0), "r"(b1), "r"(b2));   /*9 ,11*/


// adds c0c1 = a0+b0b1
// needed for the karatsuba multiplication 3x3
#define __karatsuba32_3x3_2add_helper(c0, c1, a0, b0, b1)   \
    asm (    "{\n\t"                                        \
            "add.cc.u32 	%0, %2, %3;\n\t"                \
            "addc.u32       %1, %4, 0 ;\n\t"                \
        "}\n\t"                                             \
        : "=r"(c0), "=r"(c1)            /*0,1*/             \
        : "r"(a0), "r"(b0), "r"(b1));   /*2,3,4*/

// karatsuba 3x3 multiplication without reduction
// needs 4 + 3 = 7 registers
// c5c4c3c2c1c0 = a2a1a0 * b2b1b0
#define __karatsuba32_helper_3x3(c0, c1, c2, c3, c4, c5, a0, a1, a2, b0, b1, b2, rt0, rt1, rt2, rt3) \
    __karatsuba32_3x3_2add_helper(c0,c1, a0, a1,a2)                             \
    __karatsuba32_3x3_2add_helper(c2,c3, b0, b1,b2)                             \
    __school32_2x2(rt0,rt1,rt2,rt3, c0,c1, c2,c3)                               \
    __karatsuba32_1x1(c0,c1, a0, b0)                                            \
    __school32_2x2(c2,c3,c4,c5, a1,a2, b1,b2)                                   \
    /*UsVs -= U1V1*/                                                            \
    asm ("{\n\t"                                                                \
            "sub.cc.u32  %0, %8,  %4;\n\t"                                      \
            "subc.cc.u32 %1, %9,  %5;\n\t"                                      \
            "subc.cc.u32 %2, %10, %6;\n\t"                                      \
            "subc.u32    %3, %11, %7;\n\t"                                      \
            "}\n\t"                                                             \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3) /*output 0,3 */               \
        : "r"(c2),"r"(c3),"r"(c4),"r"(c5),        /*U1V1   4,7 */               \
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3)     /*output 8,11*/               \
    );                                                                          \
    /*UsVs -= U0V0*/                                                            \
    asm ("{\n\t"                                                                \
            "sub.cc.u32  %0, %6,  %4;\n\t"                                      \
            "subc.cc.u32 %1, %7,  %5;\n\t"                                      \
            "subc.cc.u32 %2, %8,   0;\n\t"                                      \
            "subc.u32    %3, %9,   0;\n\t"                                      \
            "}\n\t"                                                             \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3) /*output 0,3*/                \
        : "r"(c0),"r"(c1),                          /*U0V0   4,5*/              \
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3)     /*output 6,9*/                \
    );                                                                          \
    /*Final addition and result is written */                                   \
    asm ("{\n\t"                                                                \
            "add.cc.u32  %0, %8,  %4;\n\t"                                      \
            "addc.cc.u32 %1, %9,  %5;\n\t"                                      \
            "addc.cc.u32 %2, %10, %6;\n\t"                                      \
            "addc.u32    %3, %11, %7;\n\t"                                      \
            "}\n"                                                               \
        : "=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4) /*0, 3*/                          \
        : "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),/*4, 7*/                          \
          "r"(c1),"r"(c2),"r"(c3),"r"(c4));   /*8,11*/

// 3x3 Karatsuba implementation. Result is a 6 limb value
// No mod reduction calculated
// c5c4c3c2c1c0 = a2a1a0 * b2b1b0
#define __karatsuba32_3x3(c0, c1, c2, c3, c4, c5, a0, a1, a2, b0, b1, b2)    \
    uint32_t rt0,rt1,rt2,rt3;                                    \
    __karatsuba32_helper_3x3(c0,c1,c2,c3,c4,c5, a0,a1,a2,b0,b1,b2, rt0,rt1,rt2,rt3)


// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**96
// This version does not use the 'mad' instructions
// Inplace version a2a1a0 = a2a1a0 * mu2mu1mu0 mod 2**(3*32)
#define __school32_inplace_3x3_low(a0, a1, a2, mu0, mu1, mu2)    \
    asm volatile ("{\n\t"                                        \
            ".reg .u32 t10,t20;\n\t"                             \
            /* t0-t2: m = a * mu mod 2^96 */                     \
            /* Multiplication involving a0 */                    \
            "mul.lo.u32     %0,  %6, %3;\n\t"                    \
            "mul.hi.u32     %1,  %6, %3;\n\t"                    \
            "mul.lo.u32     t10, %6, %4;\n\t"                    \
            "mul.hi.u32     %2,  %6, %4;\n\t"                    \
            "mul.lo.u32     t20, %6, %5;\n\t"                    \
            /* now fixing the carry chains */                    \
            "add.cc.u32     %1,  %1,  t10;\n\t"                  \
            "addc.u32       %2,  %2,  t20;\n\t"                  \
            /*Multiplication involving a1 second (low) diagonal*/\
            "mul.lo.u32     t10, %7, %3;\n\t"                    \
            "mul.lo.u32     t20, %7, %4;\n\t"                    \
            "add.cc.u32     %1, %1, t10;\n\t"                    \
            "addc.u32       %2, %2, t20;\n\t"                    \
            /*Multiplication involving a1 first (high) diagonal*/\
            "mul.hi.u32     t20, %7, %3;\n\t"                    \
            "add.u32        %2, %2, t20;\n\t"                    \
            /*Multiplication involving a3 second (low) diagonal*/\
            "mul.lo.u32     t20, %8,  %3;\n\t"                   \
            "add.u32         %2, %2, t20;\n\t"                   \
            "}\n"                                                \
            : "=r"(a0),"=r"(a1),"=r"(a2)   /*0 , 2*/             \
            : "r"(mu0),"r"(mu1),"r"(mu2)   /*3 , 5*/             \
              "r"(a0),"r"(a1),"r"(a2)       /*6 , 8*/            \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**96
// This version does USE the 'mad' instructions
// needs 2 helper registers
// NonInplace Version:
#define __school32_3x3_low_v2(c0, c1, c2, a0, a1, a2, b0, b1, b2)   \
    asm volatile ("{\n\t"                                           \
            /* t0-t2: m = a * mu mod 2^96 */                        \
            /* Multiplication involving a0 */                       \
            "mul.lo.u32     %0, %6, %3;\n\t"                        \
            "mul.hi.u32     %1, %6, %3;\n\t"                        \
            "mul.hi.u32     %2, %6, %4;\n\t"                        \
            "mad.lo.cc.u32  %1, %6, %4, %1;\n\t"                    \
            "madc.lo.u32    %2, %6, %5, %2;\n\t"                    \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mad.lo.cc.u32  %1, %7, %3, %1;\n\t"                    \
            "madc.lo.u32  	%2, %7, %4, %2;\n\t"                    \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mad.hi.u32  	%2, %7, %3, %2;\n\t"                    \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mad.lo.u32  	%2, %8, %3, %2;\n\t"                    \
            "}\n"                                                   \
        : "=r"(c0),"=r"(c1),"=r"(c2)   /*0 , 2*/                    \
        : "r"(b0),"r"(b1),"r"(b2)      /*3 , 5*/                    \
          "r"(a0),"r"(a1),"r"(a2)       /*6 , 8*/                   \
    );


// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**96
// This version does NOT use the 'mad' instructions
// needs 2 helper registers
// NonInplace Version:
#define __school32_3x3_low(c0, c1, c2, a0, a1, a2, b0, b1, b2)      \
    asm volatile ("{\n\t"                                           \
            ".reg .u32 t10,t20;\n\t"                                \
            /* t0-t2: m = a * mu mod 2^96 */                        \
            /* Multiplication involving a0 */                       \
            "mul.lo.u32     %0,  %6, %3;\n\t"                       \
            "mul.hi.u32     %1,  %6, %3;\n\t"                       \
            "mul.lo.u32     t10, %6, %4;\n\t"                       \
            "mul.hi.u32     %2,  %6, %4;\n\t"                       \
            "mul.lo.u32     t20, %6, %5;\n\t"                       \
            /* now fixing the carry chains */                       \
            "add.cc.u32     %1,  %1,  t10;\n\t"                     \
            "addc.u32       %2,  %2,  t20;\n\t"                     \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mul.lo.u32     t10, %7, %3;\n\t"                       \
            "mul.lo.u32     t20, %7, %4;\n\t"                       \
            "add.cc.u32     %1, %1, t10;\n\t"                       \
            "addc.u32       %2, %2, t20;\n\t"                       \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mul.hi.u32     t20, %7, %3;\n\t"                       \
            "add.u32        %2, %2, t20;\n\t"                       \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mul.lo.u32     t20, %8,  %3;\n\t"                      \
            "add.u32         %2, %2, t20;\n\t"                      \
            "}\n"                                                   \
        : "=r"(c0),"=r"(c1),"=r"(c2)   /*0 , 2*/                    \
        : "r"(b0),"r"(b1),"r"(b2)      /*3 , 5*/                    \
          "r"(a0),"r"(a1),"r"(a2)       /*6 , 8*/                   \
    );
// second part of the montgomery reduction written by floyd
// NonInplace Version
#define __reduce32_sec3x3(c0, c1, c2, a0, a1, a2, a3, a4, a5, p0, p1, p2, i0, i1, i2, i3, i4, i5) \
    asm volatile ("{\n\t"                           \
            /*%0-%3: r= (a + u) div 2^(3**32)*/     \
            ".reg.u32 		t0,t3,t4,t5;\n\t"       \
            /*a + i*/                               \
            "add.cc.u32     %0, %12, %6;\n\t"       \
            "addc.cc.u32    %1, %13, %7;\n\t"       \
            "addc.cc.u32    %2, %14, %8;\n\t"       \
            "addc.cc.u32    t3, %15, %9;\n\t"       \
            "addc.cc.u32    t4, %16, %10;\n\t"      \
            "addc.cc.u32    t5, %17, %11;\n\t"      \
            "addc.u32       t0, 0, 0;\n\t"          \
            /*-p*/                                  \
            "sub.cc.u32     t3, t3, %3;\n\t"        \
            "subc.cc.u32    t4, t4, %4;\n\t"        \
            "subc.cc.u32    t5, t5, %5;\n\t"        \
            "subc.u32       t0, t0, 0;\n\t"         \
            /*&p*/                                  \
            "and.b32        %0, t0, %3;\n\t"        \
            "and.b32        %1, t0, %4;\n\t"        \
            "and.b32        %2, t0, %5;\n\t"        \
            "add.cc.u32     %0, %0, t3;\n\t"        \
            "addc.cc.u32    %1, %1, t4;\n\t"        \
            "addc.u32    	%2, %2, t5;\n\t"        \
            "}\n"                                   \
        : "=r"(c0),"=r"(c1),"=r"(c2)    /*0 , 2*/   \
        : "r"(p0),"r"(p1),"r"(p2),      /*3 , 5*/   \
          "r"(i0),"r"(i1),"r"(i2),      /*6 , 8*/   \
          "r"(i3),"r"(i4),"r"(i5),      /*9, 11*/   \
          "r"(a0),"r"(a1),"r"(a2),      /*12,14*/   \
          "r"(a3),"r"(a4),"r"(a5)       /*15,17*/   \
    );

// montgomery reuction for 6 32bit limbs.
// working inplace
// a0a1a2 = a0a1a2a3a4a5 mod p0p1p2
#define __reduce32_inplace3x3(a0, a1, a2, a3, a4, a5, mu0, mu1, mu2, p0, p1, p2)    \
    uint32_t t0,t1,t2,t3,t4,t5,c0,c1,c2;                                            \
    __school32_3x3_low(c0,c1,c2,a0,a1,a2,mu0,mu1,mu2)                               \
    __school32_3x3(t0,t1,t2,t3,t4,t5,c0,c1,c2,p0,p1,p2)                             \
    __reduce32_sec3x3(a0,a1,a2,a0,a1,a2,a3,a4,a5,p0,p1,p2,t0,t1,t2,t3,t4,t5)

// Momtgomery reduction for 6 32bit limbs
// NonInplace version
// needs 2 + 7 + 5 + 6 = 20 registers
// c2c1c1 = a5a4a3a2a1a0 mod p2p1p0
#define __reduce32_3x3(c0, c1, c2, a0, a1, a2, a3, a4, a5, mu0, mu1, mu2, p0, p1, p2)   \
    uint32_t u0,u1,u2,u3,u4,u5;                                                         \
    __school32_3x3_low(c0,c1,c2, a0,a1,a2, mu0,mu1,mu2)                                 \
    __school32_3x3(u0,u1,u2,u3,u4,u5, c0,c1,c2, p0,p1,p2)                               \
    __reduce32_sec3x3(c0,c1,c2, a0,a1,a2,a3,a4,a5, p0,p1,p2, u0,u1,u2,u3,u4,u5)

// 3x3 schoolbook multiplication over Fp.
#define __fp_mul3x3_32(c0, c1, c2, a0, a1, a2, b0, b1, b2, mu0, mu1, mu2, p0, p1, p2)   \
    uint32_t f0,f1,f2,f3,f4,f5,u0,u1,u2,u3,u4,u5;                                       \
    __school32_3x3(f0,f1,f2,f3,f4,f5, a0,a1,a2, b0,b1,b2)                               \
    __school32_3x3_low(c0,c1,c2, f0,f1,f2, mu0,mu1,mu2)                                 \
    __school32_3x3(u0,u1,u2,u3,u4,u5, c0,c1,c2, p0,p1,p2)                               \
    __reduce32_sec3x3(c0,c1,c2, f0,f1,f2,f3,f4,f5, p0,p1,p2, u0,u1,u2,u3,u4,u5)


// 3x3 schoolbook multiplication over Fp.
#define __fp_mul3x3_32_v2(c0, c1, c2, a0, a1, a2, b0, b1, b2, mu0, mu1, mu2, p0, p1, p2)    \
    uint32_t f0,f1,f2,f3,f4,f5,u0,u1,u2,u3,u4,u5;                                           \
    __school32_3x3(f0,f1,f2,f3,f4,f5, a0,a1,a2, b0,b1,b2)                                   \
    __school32_3x3_low_v2(c0,c1,c2, f0,f1,f2, mu0,mu1,mu2)                                  \
    __school32_3x3(u0,u1,u2,u3,u4,u5, c0,c1,c2, p0,p1,p2)                                   \
    __reduce32_sec3x3(c0,c1,c2, f0,f1,f2,f3,f4,f5, p0,p1,p2, u0,u1,u2,u3,u4,u5)

// not finished
#define __fp_mul3x3_32_v3(c0, c1, c2, a0, a1, a2, b0, b1, b2, mu0, mu1, mu2, p0, p1, p2)    \
    asm volatile ("{\n\t"                                           \
            ".reg .u32 t0,t1,t2,tc;\n\t"                            \
            ".reg .u32 t3,t4,t5,tc;\n\t"                            \
            /* a * b */                                             \
            /*Multiplication involving a0*/                         \
            "mul.lo.u32     %0, %3, %6;\n\t"                        \
            "mul.hi.u32     t0, %3, %6;\n\t"                        \
            "mad.lo.cc.u32  t0, %3, %7, t0;\n\t"                    \
            "mul.hi.u32     t1, %3, %7;\n\t"                        \
            "madc.lo.cc.u32 t1, %3, %8, t1;\n\t"                    \
            "madc.hi.cc.u32 t2, %3, %8, 0;\n\t"                     \
            "addc.u32       tc, 0, 0;\n\t"                          \
            /*Multiplication involving a1 first diagonal*/          \
            "mad.lo.cc.u32  %1, %4, %6, t0;\n\t"                    \
            "madc.hi.cc.u32 t1, %4, %6, t1;\n\t"                    \
            "madc.hi.cc.u32 t2, %4, %7, t2;\n\t"                    \
            "madc.hi.u32 	t0, %4, %8, tc;\n\t"                    \
            /*Multiplication involving a1 second diagonal*/         \
            "mad.lo.cc.u32  t1, %4, %7, t1;\n\t"                    \
            "madc.lo.cc.u32 t2, %4, %8, t2;\n\t"                    \
            "addc.cc.u32    t0, t0 , 0;\n\t"                        \
            "addc.u32       tc, tc , 0;\n\t"                        \
            /*Multiplication involving a2 first diagonal*/          \
            "mad.lo.cc.u32  %2, %5, %6, t1;\n\t"                    \
            "madc.hi.cc.u32 t2, %5, %6, t2;\n\t"                    \
            "madc.hi.cc.u32 t0, %5, %7, t0;\n\t"                    \
            "madc.hi.cc.u32 t1, %5, %8, tc;\n\t"                    \
            "addc.u32       tc, 0 , 0;\n\t"                         \
            /*Multiplication involving a2 second diagonal*/         \
            "mad.lo.cc.u32  %3, %5, %7, t2;\n\t"                    \
            "madc.lo.cc.u32 %4, %5, %8, t0;\n\t"                    \
            "addc.u32       %5, t1 , 0;\n\t"                        \
            /* t1,t0,t2,%2,%1,%0 = a * b finished */                \
            /* */                                                   \
            /* Multiplication involving a0 */                       \
            "mul.lo.u32     %0, %9, %0;\n\t"                        \
            "mul.hi.u32     t3, %9, %0;\n\t"                        \
            "mul.hi.u32     t4, %9, %1;\n\t"                        \
            "mad.lo.cc.u32  %1, %9, %1, t3;\n\t"                    \
            "madc.lo.u32    %2, %9, %2, t4;\n\t"                    \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mad.lo.cc.u32  %1, %10, %0, %1;\n\t"                   \
            "madc.lo.u32  	%2, %10, %1, %2;\n\t"                   \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mad.hi.u32  	%2, %10, %0, %2;\n\t"                   \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mad.lo.u32  	%2, %11, %0, %2;\n\t"                   \
            /*%*/                                                   \
            "}\n"                                                   \
    : "=r"(c0), "=r"(c1), "=r"(c2), /*0,  2*/                       \
    : "r"(a0), "r"(a1), "r"(a2),    /*3,  5*/                       \
      "r"(b0), "r"(b1), "r"(b2),    /*6,  8*/                       \
      "r"(mu0), "r"(mu1), "r"(mu2), /*9, 11*/                       \
      "r"(p0), "r"(p1), "r"(p2));   /*12,14*/

// org luis code, with minor changes by floyd
#define __fp_mul3x3_32_v4(c0, c1, c2, a0, a1, a2, b0, b1, b2, m0, m1, m2, p0, p1, p2)   \
    asm volatile ("{\n\t"                                                               \
        ".reg .u32 t0,t1,t2,t3,t4,t5,t6;"                                               \
        ".reg .u32 t7,t8,t9,t10,t11;"                                                   \
        "mul.lo.u32     t0, %3, %6;\n\t"                                                \
        "mul.hi.u32     t1, %3, %6;\n\t"                                                \
        "mad.lo.cc.u32  t1, %3, %7, t1;\n\t"                                            \
        "mul.hi.u32     t2, %3, %7;\n\t"                                                \
        "madc.lo.cc.u32 t2, %3, %8, t2;\n\t"                                            \
        "madc.hi.u32    t3, %3, %8, 0;\n\t"                                             \
        "mul.hi.u32     t5, %4, %6;\n\t"                                                \
        "mad.lo.cc.u32  t5, %4, %7, t5;\n\t"                                            \
        "mul.hi.u32     t6, %4, %7;\n\t"                                                \
        "madc.lo.cc.u32 t6, %4, %8, t6;"                                                \
        "madc.hi.u32    t7, %4, %8, 0;\n\t"                                             \
        "mad.lo.cc.u32  t1, %4, %6, t1;\n\t"                                            \
        "addc.cc.u32    t5, t5, t2;\n\t"                                                \
        "addc.cc.u32    t6, t6, t3;\n\t"                                                \
        "addc.u32       t7, t7, 0;\n\t"                                                 \
        "mul.lo.u32     t2, %5, %6;\n\t"                                                \
        "mul.hi.u32     t3, %5, %6;\n\t"                                                \
        "mad.lo.cc.u32  t3, %5, %7, t3;\n\t"                                            \
        "mul.hi.u32     t4, %5, %7;\n\t"                                                \
        "madc.lo.cc.u32 t4, %5, %8, t4;\n\t"                                            \
        "madc.hi.u32    t8, %5, %8, 0;\n\t"                                             \
        "add.cc.u32     t2, t2, t5;\n\t"                                                \
        "addc.cc.u32    t3, t3, t6;\n\t"                                                \
        "addc.cc.u32    t4, t4, t7;\n\t"                                                \
        "addc.u32       t5, t8, 0;\n\t"                                                 \
        /*0,5=c= a*b */                                                                 \
        "mov.u32        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(3*32)*/     \
        "mad.lo.cc.u32  %1, %10, t0, t1;\n\t"                                           \
        "madc.hi.u32    %2, %10, t0, t2;\n\t"                                           \
        "mad.lo.u32     %2, %10, t1, %2;\n\t"                                           \
        "mad.lo.u32     %2, %11, t0, %2;\n\t"                                           \
        "mul.lo.u32     t6, %0, %12;\n\t"/* u = m*p */                                  \
        "mul.hi.u32     t7, %0, %12;\n\t"                                               \
        "mad.lo.cc.u32  t7, %0, %13, t7;\n\t"                                           \
        "mul.hi.u32     t8, %0, %13;\n\t"                                               \
        "madc.lo.cc.u32 t8, %0, %14, t8;\n\t"                                           \
        "madc.hi.u32    t9, %0, %14, 0;\n\t"                                            \
        "mul.lo.u32     t10, %1, %12;\n\t"                                              \
        "mul.hi.u32     t11, %1, %12;\n\t"                                              \
        "mad.lo.cc.u32  t11, %1, %13, t11;\n\t"                                         \
        "mul.hi.u32     %0, %1, %13;\n\t"                                               \
        "madc.lo.cc.u32 %0, %1, %14, %0;"                                               \
        "madc.hi.u32    %1, %1, %14, 0;\n\t"                                            \
        "add.cc.u32     t7, t7, t10;\n\t"                                               \
        "addc.cc.u32    t11, t11, t8;\n\t"                                              \
        "addc.cc.u32    %0, %0, t9;\n\t"                                                \
        "addc.u32       %1, %1, 0;\n\t"                                                 \
        "mul.lo.u32     t8, %2, %12;\n\t"                                               \
        "mul.hi.u32     t9, %2, %12;\n\t"                                               \
        "mad.lo.cc.u32  t9, %2, %13, t9;\n\t"                                           \
        "mul.hi.u32     t10, %2, %13;\n\t"                                              \
        "madc.lo.cc.u32 t10, %2, %14, t10;\n\t"                                         \
        "madc.hi.u32    %2, %2, %14, 0;\n\t"                                            \
        "add.cc.u32     t8, t8, t11;\n\t"                                               \
        "addc.cc.u32    t9, t9, %0;\n\t"                                                \
        "addc.cc.u32    t10, t10, %1;\n\t"                                              \
        "addc.u32       t11, %2, 0;\n\t"                                                \
        /* t6-t11 = u */                                                                \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(3*32)*/                         \
        "add.cc.u32     t0, t6, t0;\n\t"                                                \
        "addc.cc.u32    t1, t7, t1;\n\t"                                                \
        "addc.cc.u32    t2, t8, t2;\n\t"                                                \
        "addc.cc.u32    t3, t9, t3;\n\t"                                                \
        "addc.cc.u32    t4, t10, t4;\n\t"                                               \
        "addc.cc.u32    t5, t11, t5;\n\t"                                               \
        "addc.u32       %0, %0, 0;\n\t"                                                 \
        /* subtract p */                                                                \
        "sub.cc.u32     t3, t3, %12;\n\t"                                               \
        "subc.cc.u32    t4, t4, %13;\n\t"                                               \
        "subc.cc.u32    t5, t5, %14;\n\t"                                               \
        "subc.u32       %0, %0, 0;\n\t"                                                 \
        "mov.u32        %1, %0;\n\t"                                                    \
        "mov.u32        %2, %0;\n\t"                                                    \
        "and.b32        %0, %0, %12;\n\t"                                               \
        "and.b32        %1, %1, %13;\n\t"                                               \
        "and.b32        %2, %2, %14;\n\t"                                               \
        "add.cc.u32     %0, %0, t3;\n\t"                                                \
        "addc.cc.u32    %1, %1, t4;\n\t"                                                \
        "addc.u32       %2, %2, t5;\n\t"                                                \
        "}"                                                                             \
    : "=r"(c0), "=r"(c1), "=r"(c2) /*0 , 2*/                                            \
    : "r"(a0), "r"(a1), "r"(a2),   /*3 , 5*/                                            \
      "r"(b0), "r"(b1), "r"(b2),   /*6 , 8*/                                            \
      "r"(m0), "r"(m1), "r"(m2),   /*9 ,11*/                                            \
      "r"(p0), "r"(p1), "r"(p2));  /*12,14*/

// org luis code, a*b mult exchanged with schoolbook mult
// not correct slower than v4
#define __fp_mul3x3_32_v5(c0, c1, c2, a0, a1, a2, b0, b1, b2, m0, m1, m2, p0, p1, p2)\
    asm volatile ("{\n\t"                               \
        ".reg .u32 t0;" ".reg .u32 t1;\n\t"             \
        ".reg .u32 t2;" ".reg .u32 t3;\n\t"             \
        ".reg .u32 t4;" ".reg .u32 t5;\n\t"             \
        ".reg .u32 t6;" ".reg .u32 t7;\n\t"             \
        ".reg .u32 t8;" ".reg .u32 t9;\n\t"             \
        ".reg .u32 t10;" ".reg .u32 t11,tc;\n\t"        \
        /*Multiplication involving a0*/                 \
        "mul.lo.u32     t0, %3, %6;\n\t"                \
        "mul.hi.u32     t5, %3, %6;\n\t"                \
        "mad.lo.cc.u32  t5, %3, %7, t5;\n\t"            \
        "mul.hi.u32     t3, %3, %7;\n\t"                \
        "madc.lo.cc.u32 t3, %3, %8, t3;\n\t"            \
        "madc.hi.cc.u32 t4, %3, %8, 0;\n\t"             \
        "addc.u32       tc, 0, 0;\n\t"                  \
        /*Multiplication involving a1 first diagonal*/  \
        "mad.lo.cc.u32  t1, %4, %6, t5;\n\t"            \
        "madc.hi.cc.u32 t3, %4, %6, t3;\n\t"            \
        "madc.hi.cc.u32 t4, %4, %8, t4;\n\t"            \
        "madc.hi.u32 	t5, %4, %9, tc;\n\t"            \
        /*Multiplication involving a1 second diagonal*/ \
        "mad.lo.cc.u32  t3, %4, %7, t3;\n\t"            \
        "madc.lo.cc.u32 t4, %4, %8, t4;\n\t"            \
        "addc.cc.u32    t5, t5 , 0;\n\t"                \
        "addc.u32       tc, tc , 0;\n\t"                \
        /*Multiplication involving a2 first diagonal*/  \
        "mad.lo.cc.u32  t2, %5, %6, t3;\n\t"            \
        "madc.hi.cc.u32 t4, %5, %6, t4;\n\t"            \
        "madc.hi.cc.u32 t5, %5, %7, t5;\n\t"            \
        "madc.hi.u32 	tc, %5, %8, tc;\n\t"            \
        /*Multiplication involving a2 second diagonal*/ \
        "mad.lo.cc.u32  t3, %5, %7, t4;\n\t"            \
        "madc.lo.cc.u32 t4, %5, %8, t5;\n\t"            \
        "addc.u32       t5, tc , 0;\n\t"                \
        /*0,5=c= a*b */                                 \
        "mov.u32        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(3*32)*/\
        "mad.lo.cc.u32  %1, %10, t0, t1;\n\t"   \
        "madc.hi.u32    %2, %10, t0, t2;\n\t"   \
        "mad.lo.u32     %2, %10, t1, %2;\n\t"   \
        "mad.lo.u32     %2, %11, t0, %2;\n\t"   \
        "mul.lo.u32     t6, %0, %12;\n\t"/* u = m*p */\
        "mul.hi.u32     t7, %0, %12;\n\t"       \
        "mad.lo.cc.u32  t7, %0, %13, t7;\n\t"   \
        "mul.hi.u32     t8, %0, %13;\n\t"       \
        "madc.lo.cc.u32 t8, %0, %14, t8;\n\t"   \
        "madc.hi.u32    t9, %0, %14, 0;\n\t"    \
        "mul.lo.u32     t10, %1, %12;\n\t"      \
        "mul.hi.u32     t11, %1, %12;\n\t"      \
        "mad.lo.cc.u32  t11, %1, %13, t11;\n\t" \
        "mul.hi.u32     %0, %1, %13;\n\t"       \
        "madc.lo.cc.u32 %0, %1, %14, %0;"       \
        "madc.hi.u32    %1, %1, %14, 0;\n\t"    \
        "add.cc.u32     t7, t7, t10;\n\t"       \
        "addc.cc.u32    t11, t11, t8;\n\t"      \
        "addc.cc.u32    %0, %0, t9;\n\t"        \
        "addc.u32       %1, %1, 0;\n\t"         \
        "mul.lo.u32     t8, %2, %12;\n\t"       \
        "mul.hi.u32     t9, %2, %12;\n\t"       \
        "mad.lo.cc.u32  t9, %2, %13, t9;\n\t"   \
        "mul.hi.u32     t10, %2, %13;\n\t"      \
        "madc.lo.cc.u32 t10, %2, %14, t10;\n\t" \
        "madc.hi.u32    %2, %2, %14, 0;\n\t"    \
        "add.cc.u32     t8, t8, t11;\n\t"       \
        "addc.cc.u32    t9, t9, %0;\n\t"        \
        "addc.cc.u32    t10, t10, %1;\n\t"      \
        "addc.u32       t11, %2, 0;\n\t"        \
        /* t6-t11 = u */                        \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(3*32)*/\
        "add.cc.u32     t0, t6, t0;\n\t"        \
        "addc.cc.u32    t1, t7, t1;\n\t"        \
        "addc.cc.u32    t2, t8, t2;\n\t"        \
        "addc.cc.u32    t3, t9, t3;\n\t"        \
        "addc.cc.u32    t4, t10, t4;\n\t"       \
        "addc.cc.u32    t5, t11, t5;\n\t"       \
        "addc.u32       %0, %0, 0;\n\t"         \
        /* subtract p */                        \
        "sub.cc.u32     t3, t3, %12;\n\t"       \
        "subc.cc.u32    t4, t4, %13;\n\t"       \
        "subc.cc.u32    t5, t5, %14;\n\t"       \
        "subc.u32       %0, %0, 0;\n\t"         \
        "mov.u32        %1, %0;\n\t"            \
        "mov.u32        %2, %0;\n\t"            \
        "and.b32        %0, %0, %12;\n\t"       \
        "and.b32        %1, %1, %13;\n\t"       \
        "and.b32        %2, %2, %14;\n\t"       \
        "add.cc.u32     %0, %0, t3;\n\t"        \
        "addc.cc.u32    %1, %1, t4;\n\t"        \
        "addc.u32       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=r"(c0), "=r"(c1), "=r"(c2) /*0 , 2*/\
        : "r"(a0), "r"(a1), "r"(a2),   /*3 , 5*/\
          "r"(b0), "r"(b1), "r"(b2),   /*6 , 8*/\
          "r"(m0), "r"(m1), "r"(m2),   /*9 ,11*/\
          "r"(p0), "r"(p1), "r"(p2));  /*12,14*/


#define __fp_mul2x2_64(c0, c1, a0, a1, b0, b1, m0, m1, p0, p1)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        "mul.lo.u64     t0, %2, %4;\n\t"        \
        "mul.hi.u64     t1, %2, %4;\n\t"        \
        "mad.lo.cc.u64  t1, %2, %5, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %2, %5, 0;\n\t"     \
        "mad.lo.cc.u64  t1, %3, %4, t1;\n\t"    \
        "madc.hi.cc.u64 t2, %3, %4, t2;\n\t"    \
        "madc.lo.cc.u64 t2, %3, %5, t2;\n\t"    \
        "madc.hi.u64    t3, %3, %5, 0;\n\t"     \
        "mul.lo.u64     %0, t0, %6;\n\t"/*Montgomery reduction, m = mu*c mod 2^(2*64)*/\
        "mul.hi.u64     %1, t0, %6;\n\t"        \
        "mad.lo.u64     %1, t0, %7, %1;\n\t"    \
        "mad.lo.u64     %1, t1, %6, %1;\n\t"    \
        "mul.lo.u64     t4, %0, %8;\n\t"/* u = m*p */\
        "mul.hi.u64     t5, %0, %8;\n\t"        \
        "mad.lo.cc.u64  t5, %0, %9, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %0, %9, 0;\n\t"     \
        "mad.lo.cc.u64  t5, %1, %8, t5;\n\t"    \
        "madc.hi.cc.u64 t6, %1, %8, t6;\n\t"    \
        "madc.lo.cc.u64 t6, %1, %9, t6;\n\t"    \
        "madc.hi.u64    t7, %1, %9, 0;\n\t"     \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(2*64)*/\
        "add.cc.u64     t0, t4, t0;\n\t"        \
        "addc.cc.u64    t1, t5, t1;\n\t"        \
        "addc.cc.u64    t2, t6, t2;\n\t"        \
        "addc.cc.u64    t3, t7, t3;\n\t"        \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t2, t2, %8;\n\t"        \
        "subc.cc.u64    t3, t3, %9;\n\t"        \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "and.b64        %0, %0, %8;\n\t"        \
        "and.b64        %1, %1, %9;\n\t"        \
        "add.cc.u64     %0, %0, t2;\n\t"        \
        "addc.u64       %1, %1, t3;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1)\
        : "l"(a0), "l"(a1), "l"(b0), "l"(b1),   \
          "l"(m0), "l"(m1), "l"(p0), "l"(p1) )

// luis code, little modification by floyd
#define __fp_mul3x3_64(c0, c1, c2, a0, a1, a2, b0, b1, b2, m0, m1, m2, p0, p1, p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"     \
        ".reg .u64 t10;" ".reg .u64 t11;\n\t"   \
        "mul.lo.u64     t0, %3, %6;\n\t"        \
        "mul.hi.u64     t1, %3, %6;\n\t"        \
        "mad.lo.cc.u64  t1, %3, %7, t1;\n\t"    \
        "mul.hi.u64     t2, %3, %7;\n\t"        \
        /*"madc.lo.cc.u64 t2, %3, %8, t2;\n\t"*/\
        /*"madc.hi.u64    t3, %3, %8, 0;\n\t" */\
        /*"mul.lo.u64     t4, %4, %6;\n\t" */   \
        "mul.hi.u64     t5, %4, %6;\n\t"        \
        "mad.lo.cc.u64  t5, %4, %7, t5;\n\t"    \
        "mul.hi.u64     t6, %4, %7;\n\t"        \
        "madc.lo.cc.u64 t6, %4, %8, t6;"        \
        "madc.hi.u64    t7, %4, %8, 0;\n\t"     \
        /*modification */ \
        "madc.lo.cc.u64 t1, %4, %6, t1;"        \
        "madc.lo.cc.u64 t5, %3, %8, t2;\n\t"    \
        "madc.hi.cc.u64 t6, %3, %8, t3;\n\t"    \
        /*"add.cc.u64     t1, t1, t4;\n\t" */   \
        /*"addc.cc.u64    t5, t5, t2;\n\t" */   \
        /*"addc.cc.u64    t6, t6, t3;\n\t" */   \
        "addc.u64       t7, t7, 0;\n\t"         \
        "mul.lo.u64     t2, %5, %6;\n\t"        \
        "mul.hi.u64     t3, %5, %6;\n\t"        \
        "mad.lo.cc.u64  t3, %5, %7, t3;\n\t"    \
        "mul.hi.u64     t4, %5, %7;\n\t"        \
        "madc.lo.cc.u64 t4, %5, %8, t4;\n\t"    \
        "madc.hi.u64    t8, %5, %8, 0;\n\t"     \
        "add.cc.u64     t2, t2, t5;\n\t"        \
        "addc.cc.u64    t3, t3, t6;\n\t"        \
        "addc.cc.u64    t4, t4, t7;\n\t"        \
        "addc.u64       t5, t8, 0;\n\t"         \
        /*0,5=c= a*b */                         \
        "mov.u64        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(3*64)*/\
        "mad.lo.cc.u64  %1, %10, t0, t1;\n\t"   \
        "madc.hi.u64    %2, %10, t0, t2;\n\t"   \
        "mad.lo.u64     %2, %10, t1, %2;\n\t"   \
        "mad.lo.u64     %2, %11, t0, %2;\n\t"   \
        "mul.lo.u64     t6, %0, %12;\n\t"/* u = m*p */\
        "mul.hi.u64     t7, %0, %12;\n\t"       \
        "mad.lo.cc.u64  t7, %0, %13, t7;\n\t"   \
        "mul.hi.u64     t8, %0, %13;\n\t"       \
        "madc.lo.cc.u64 t8, %0, %14, t8;\n\t"   \
        "madc.hi.u64    t9, %0, %14, 0;\n\t"    \
        "mul.lo.u64     t10, %1, %12;\n\t"      \
        "mul.hi.u64     t11, %1, %12;\n\t"      \
        "mad.lo.cc.u64  t11, %1, %13, t11;\n\t" \
        "mul.hi.u64     %0, %1, %13;\n\t"       \
        "madc.lo.cc.u64 %0, %1, %14, %0;"       \
        "madc.hi.u64    %1, %1, %14, 0;\n\t"    \
        "add.cc.u64     t7, t7, t10;\n\t"       \
        "addc.cc.u64    t11, t11, t8;\n\t"      \
        "addc.cc.u64    %0, %0, t9;\n\t"        \
        "addc.u64       %1, %1, 0;\n\t"         \
        "mul.lo.u64     t8, %2, %12;\n\t"       \
        "mul.hi.u64     t9, %2, %12;\n\t"       \
        "mad.lo.cc.u64  t9, %2, %13, t9;\n\t"   \
        "mul.hi.u64     t10, %2, %13;\n\t"      \
        "madc.lo.cc.u64 t10, %2, %14, t10;\n\t" \
        "madc.hi.u64    %2, %2, %14, 0;\n\t"    \
        "add.cc.u64     t8, t8, t11;\n\t"       \
        "addc.cc.u64    t9, t9, %0;\n\t"        \
        "addc.cc.u64    t10, t10, %1;\n\t"      \
        "addc.u64       t11, %2, 0;\n\t"        \
        /* t6-t11 = u */                        \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(3*64)*/\
        "add.cc.u64     t0, t6, t0;\n\t"        \
        "addc.cc.u64    t1, t7, t1;\n\t"        \
        "addc.cc.u64    t2, t8, t2;\n\t"        \
        "addc.cc.u64    t3, t9, t3;\n\t"        \
        "addc.cc.u64    t4, t10, t4;\n\t"       \
        "addc.cc.u64    t5, t11, t5;\n\t"       \
        "addc.u64       %0, %0, 0;\n\t"         \
        /* subtract p */                        \
        "sub.cc.u64     t3, t3, %12;\n\t"       \
        "subc.cc.u64    t4, t4, %13;\n\t"       \
        "subc.cc.u64    t5, t5, %14;\n\t"       \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "mov.u64        %2, %0;\n\t"            \
        "and.b64        %0, %0, %12;\n\t"       \
        "and.b64        %1, %1, %13;\n\t"       \
        "and.b64        %2, %2, %14;\n\t"       \
        "add.cc.u64     %0, %0, t3;\n\t"        \
        "addc.cc.u64    %1, %1, t4;\n\t"        \
        "addc.u64       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2) /*0 , 2*/\
        : "l"(a0), "l"(a1), "l"(a2),   /*3 , 5*/\
          "l"(b0), "l"(b1), "l"(b2),   /*6 , 8*/\
          "l"(m0), "l"(m1), "l"(m2),   /*9 ,11*/\
          "l"(p0), "l"(p1), "l"(p2));  /*12,14*/

#define __fp_sqr3x3_32(c0, c1, c2, a0, a1, a2, m0, m1, m2, p0, p1, p2)\
    asm volatile ("{\n\t"                       \
        ".reg .u32 t0;" ".reg .u32 t1;\n\t"     \
        ".reg .u32 t2;" ".reg .u32 t3;\n\t"     \
        ".reg .u32 t4;" ".reg .u32 t5;\n\t"     \
        ".reg .u32 t6;" ".reg .u32 t7;\n\t"     \
        ".reg .u32 t8;" ".reg .u32 t9;\n\t"     \
        ".reg .u32 t10;" ".reg .u32 t11;\n\t"   \
        ".reg .u32 t12;" ".reg .u32 t13;\n\t"   \
        "mul.lo.u32     t0, %3, %3;\n\t"        \
        "mul.hi.u32     t1, %3, %3;\n\t"        \
        "mul.lo.u32     t8, %3, %4;\n\t"        \
        "mul.hi.u32     t9, %3, %4;\n\t"        \
        "mul.lo.u32     t10, %3, %5;\n\t"       \
        "mul.hi.u32     t11, %3, %5;\n\t"       \
        "add.cc.u32     t1, t1, t8;\n\t"        \
        "addc.cc.u32    t2, t9, t10;\n\t"       \
        "addc.u32       t3, t11, 0;\n\t"        \
        "mul.lo.u32     t12, %4, %5;\n\t"       \
        "mul.hi.u32     t13, %4, %5;\n\t"       \
        "mul.hi.u32     t6, %4, %4;\n\t"        \
        "mad.lo.cc.u32  t5, %4, %4, t9;\n\t"    \
        "addc.cc.u32    t6, t6, t12;\n\t"       \
        "addc.u32       t7, t13, 0;\n\t"        \
        "add.cc.u32     t1, t1, t8;\n\t"        \
        "addc.cc.u32    t5, t5, t2;\n\t"        \
        "addc.cc.u32    t6, t6, t3;\n\t"        \
        "addc.u32       t7, t7, 0;\n\t"         \
        "mul.lo.u32     t4, %5, %5;\n\t"        \
        "mul.hi.u32     t8, %5, %5;\n\t"        \
        "add.cc.u32     t3, t11, t12;\n\t"      \
        "addc.cc.u32    t4, t4, t13;\n\t"       \
        "addc.cc.u32    t8, t8, 0;\n\t"         \
        "add.cc.u32     t2, t10, t5;\n\t"       \
        "addc.cc.u32    t3, t3, t6;\n\t"        \
        "addc.cc.u32    t4, t4, t7;\n\t"        \
        "addc.u32       t5, t8, 0;\n\t"         \
        "mov.u32        %0, t0;\n\t"/*Montgomery reduction, m = mu*c mod 2^(32*3)*/\
        "mad.lo.cc.u32  %1, %7, t0, t1;\n\t"    \
        "madc.hi.u32    %2, %7, t0, t2;\n\t"    \
        "mad.lo.u32     %2, %7, t1, %2;\n\t"    \
        "mad.lo.u32     %2, %8, t0, %2;\n\t"    \
        "mul.lo.u32     t6, %0, %9;\n\t"/* u = m*p */\
        "mul.hi.u32     t7, %0, %9;\n\t"        \
        "mad.lo.cc.u32  t7, %0, %10, t7;\n\t"   \
        "mul.hi.u32     t8, %0, %10;\n\t"       \
        "madc.lo.cc.u32 t8, %0, %11, t8;\n\t"   \
        "madc.hi.u32    t9, %0, %11, 0;\n\t"    \
        "mul.lo.u32     t10, %1, %9;\n\t"       \
        "mul.hi.u32     t11, %1, %9;\n\t"       \
        "mad.lo.cc.u32  t11, %1, %10, t11;\n\t" \
        "mul.hi.u32     %0, %1, %10;\n\t"       \
        "madc.lo.cc.u32 %0, %1, %11, %0;"       \
        "madc.hi.u32    %1, %1, %11, 0;\n\t"    \
        "add.cc.u32     t7, t7, t10;\n\t"       \
        "addc.cc.u32    t11, t11, t8;\n\t"      \
        "addc.cc.u32    %0, %0, t9;\n\t"        \
        "addc.u32       %1, %1, 0;\n\t"         \
        "mul.lo.u32     t8, %2, %9;\n\t"        \
        "mul.hi.u32     t9, %2, %9;\n\t"        \
        "mad.lo.cc.u32  t9, %2, %10, t9;\n\t"   \
        "mul.hi.u32     t10, %2, %10;\n\t"      \
        "madc.lo.cc.u32 t10, %2, %11, t10;\n\t" \
        "madc.hi.u32    %2, %2, %11, 0;\n\t"    \
        "add.cc.u32     t8, t8, t11;\n\t"       \
        "addc.cc.u32    t9, t9, %0;\n\t"        \
        "addc.cc.u32    t10, t10, %1;\n\t"      \
        "addc.u32       t11, %2, 0;\n\t"        \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(32*3)*/\
        "add.cc.u32     t0, t6, t0;\n\t"        \
        "addc.cc.u32    t1, t7, t1;\n\t"        \
        "addc.cc.u32    t2, t8, t2;\n\t"        \
        "addc.cc.u32    t3, t9, t3;\n\t"        \
        "addc.cc.u32    t4, t10, t4;\n\t"       \
        "addc.cc.u32    t5, t11, t5;\n\t"       \
        "addc.u32       %0, %0, 0;\n\t"         \
        "sub.cc.u32     t3, t3, %9;\n\t"        \
        "subc.cc.u32    t4, t4, %10;\n\t"       \
        "subc.cc.u32    t5, t5, %11;\n\t"       \
        "subc.cc.u32    %0, %0, 0;\n\t"         \
        "mov.u32        %1, %0;\n\t"            \
        "mov.u32        %2, %0;\n\t"            \
        "and.b32        %0, %0, %9;\n\t"        \
        "and.b32        %1, %1, %10;\n\t"       \
        "and.b32        %2, %2, %11;\n\t"       \
        "add.cc.u32     %0, %0, t3;\n\t"        \
        "addc.cc.u32    %1, %1, t4;\n\t"        \
        "addc.u32       %2, %2, t5;\n\t"        \
        "}"                                     \
        : "=r"(c0), "=r"(c1), "=r"(c2)          \
        : "r"(a0), "r"(a1), "r"(a2),            \
          "r"(m0), "r"(m1), "r"(m2),            \
          "r"(p0), "r"(p1), "r"(p2));           \

// TODO unfinished
// taken from https://cacr.uwaterloo.ca/hac/about/chap14.pdf
#define __school_sqr32_3x3(c0, c1, c2, c3, c4, c5, a0, a1, a2)  \
    asm volatile ("{\n\t"                                       \
            ".reg .u32 			u,v,c;\n\t"                     \
            "mul.lo.u32			%0, %4, %4;\n\t"                \
            "mul.hi.u32		 	c, %4, %4;\n\t"                 \
            "}\n"                                               \
    : "=r"(c0),"=r"(c1),"=r"(c2),    /*0, 5*/                   \
      "=r"(c3),"=r"(c4),"=r"(c5)                                \
    : "r"(a0),"r"(a1),"r"(a2));        /*4 , 5*/

// Helper wraper around the karatsuba 3x3 multiplication macro
#define __mul96_32(c0, c1, c2, c3, c4, c5, a0, a1, a2, b0, b1, b2)  \
    __karatsuba32_3x3(c0,c1,c2,c3,c4,c5, a0,a1,a2, b0,b1,b2)

////////////////////////////////////////////////////////////////////////////////
// 									4x4										  //
////////////////////////////////////////////////////////////////////////////////
// helper function only needed for the 4x4 32bit karatsuba implementation
// c = a+b, where a and b are only two limb wide
#define __karatsuba32_4x4_2add_helper(c0, c1, a0, a1, b0, b1)   \
    asm ("{\n\t"                                                \
            "add.cc.u32  %0, %2, %4;\n\t"                       \
            "addc.u32 	 %1, %3, %5;\n\t"                       \
        "}\n"                                                   \
        : "=r"(c0), "=r"(c1)    /*0,1*/                         \
        : "r"(a0), "r"(a1),     /*2,3*/                         \
          "r"(b0), "r"(b1));    /*4,5*/

// helper function only needed for the 4x4 32bit karatsuba implementation
// c = a+b, where a and b are only two limb wide
#define __karatsuba32_4x4_2add_helper_v2(c0, c1, c2, a0, a1, b0, b1)\
    asm ("{\n\t"                                                    \
            "add.cc.u32  %0, %3, %5;\n\t"                           \
            "addc.cc.u32 %1, %4, %6;\n\t"                           \
            "addc.u32 	 %2,  0, 0;\n\t"                            \
        "}\n"                                                       \
        : "=r"(c0), "=r"(c1), "=r"(c2)  /*0,1*/                     \
        : "r"(a0), "r"(a1),             /*2,3*/                     \
          "r"(b0), "r"(b1));            /*4,5*/

// karatsuba 4x4 multiplication without reduction
// fix multiplication to use `ktmp2` and `stmp2` on the lower limbs
// 	only saves 3 cycles. so who cares
#define __karatsuba32_helper_4x4(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3, rt0, rt1, rt2, rt3, rt4, rt5, rt6, rt7) \
    __karatsuba32_4x4_2add_helper_v2(c0,c1,c2, a0,a1, a2,a3)            \
    __karatsuba32_4x4_2add_helper_v2(c3,c4,c5, b0,b1, b2,b3)            \
    __school32_3x3(rt0,rt1,rt2,rt3,rt4,rt5, c0,c1,c2, c3,c4,c5)         \
    __school32_2x2(c0,c1,c2,c3, a0,a1, b0,b1)                           \
    __school32_2x2(c4,c5,c6,c7, a2,a3, b2,b3)                           \
    asm ("{\n\t"                                                        \
            /*UsVs - U1V1 - U0V0*/                                      \
            "sub.cc.u32  %0, %16, %8 ;\n\t"                             \
            "subc.cc.u32 %1, %17, %9 ;\n\t"                             \
            "subc.cc.u32 %2, %18, %10;\n\t"                             \
            "subc.cc.u32 %3, %19, %11;\n\t"                             \
            "subc.u32 	 %4, %20, 0;\n\t"                               \
            "sub.cc.u32  %0, %0,  %12;\n\t"                             \
            "subc.cc.u32 %1, %1,  %13;\n\t"                             \
            "subc.cc.u32 %2, %2,  %14;\n\t"                             \
            "subc.cc.u32 %3, %3,  %15;\n\t"                             \
            "subc.u32 	 %4, %4,  0;\n\t"                               \
        "}\n"                                                           \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3),  /*output 0, 3 */    \
          "=r"(rt4),"=r"(rt5),"=r"(rt6),"=r"(rt7)   /*output 4, 7 */    \
        : "r"(c4),"r"(c5),"r"(c6),"r"(c7),          /*U1V1   8, 11*/    \
          "r"(c0),"r"(c1),"r"(c2),"r"(c3),          /*U0V0   12,15*/    \
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),      /*output 16,19*/    \
          "r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7)       /*output 20,23*/    \
    );                                                                  \
    /*Final addition and result is written */                           \
    asm ("{\n\t"                                                        \
            "add.cc.u32  %0, %10, %6;\n\t"                              \
            "addc.cc.u32 %1, %11, %7;\n\t"                              \
            "addc.cc.u32 %2, %12, %8;\n\t"                              \
            "addc.cc.u32 %3, %13, %9;\n\t"                              \
            "addc.cc.u32 %4, %14, 0 ;\n\t"                              \
            "addc.u32    %5, %15, 0 ;\n\t"                              \
        "}\n"                                                           \
    : "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)/*0,5*/ \
    : "r"(rt0), "r"(rt1), "r"(rt2), "r"(rt3)                    /*6,9*/ \
      "r"(c2), "r"(c3), "r"(c4), "r"(c5), "r"(c6), "r"(c7));    /*10,15*/

// wrapper function for the karatsube 4x4 32bit limbs multiplication, allocating the needed helper registers
#define __karatsuba32_4x4(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)   \
    uint32_t rtmp0,rtmp1,rtmp2,rtmp3,rtmp4,rtmp5,rtmp6,rtmp7;                               \
    __karatsuba32_helper_4x4(c0,c1,c2,c3,c4,c5,c6,c7, a0,a1,a2,a3, b0,b1,b2,b3, rtmp0,rtmp1,rtmp2,rtmp3,rtmp4,rtmp5,rtmp6,rtmp7)



// 4x4 32bit schoolbook multiplication
// c7c6c5c4c3c2c1c0 = a3a2a1a0 * b3b2b1b0
// using the slow `mad` instruction
// needs 4 tmp registers
#define __school32_4x4(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)    \
    asm volatile ("{\n\t"                                               \
            ".reg .u32 t1,t2,t3,tc;\n\t"                                \
            /*Multiplication involving a0*/                             \
            "mul.hi.u32     %0, %8, %12;\n\t"                           \
            "mad.lo.cc.u32  %0, %8, %13, %0;\n\t"                       \
            "mul.hi.u32     t1, %8, %13;\n\t"                           \
            "madc.lo.cc.u32 t1, %8, %14, t1;\n\t"                       \
            "mul.hi.u32     t2, %8, %14;\n\t"                           \
            "madc.lo.cc.u32 t2, %8, %15, t2;\n\t"                       \
            "madc.hi.cc.u32 t3, %8, %15, 0;\n\t"                        \
            "addc.u32       tc, 0, 0;\n\t"                              \
            /*Multiplication involving a1 first diagonal*/              \
            "mad.lo.cc.u32  %1, %9, %12, %0;\n\t"                       \
            "madc.hi.cc.u32 t1, %9, %12, t1;\n\t"                       \
            "madc.hi.cc.u32 t2, %9, %13, t2;\n\t"                       \
            "madc.hi.cc.u32 t3, %9, %14, t3;\n\t"                       \
            "madc.hi.u32    %0, %9, %15, tc;\n\t"                       \
            /*Multiplication involving a1 second diagonal*/             \
            "mad.lo.cc.u32  t1, %9, %13, t1;\n\t"                       \
            "madc.lo.cc.u32 t2, %9, %14, t2;\n\t"                       \
            "madc.lo.cc.u32 t3, %9, %15, t3;\n\t"                       \
            "addc.cc.u32    %0, %0 , 0;\n\t"                            \
            "addc.u32       tc, tc , 0;\n\t"                            \
            /*Multiplication involving a2 first diagonal*/              \
            "mad.lo.cc.u32  %2, %10, %12, t1;\n\t"                      \
            "madc.hi.cc.u32 t2, %10, %12, t2;\n\t"                      \
            "madc.hi.cc.u32 t3, %10, %13, t3;\n\t"                      \
            "madc.hi.cc.u32 %0, %10, %14, %0;\n\t"                      \
            "madc.hi.cc.u32 t1, %10, %15, tc;\n\t"                      \
            "addc.u32       tc, 0 , 0;\n\t"                             \
            /*Multiplication involving a2 second diagonal*/             \
            "mad.lo.cc.u32  t2, %10, %13, t2;\n\t"                      \
            "madc.lo.cc.u32 t3, %10, %14, t3;\n\t"                      \
            "madc.lo.cc.u32 %0, %10, %15, %0;\n\t"                      \
            "addc.cc.u32    t1, t1 , 0;\n\t"                            \
            "addc.u32       tc, tc , 0;\n\t"                            \
            /*Multiplication involving a3 first diagonal*/              \
            "mad.lo.cc.u32  %3, %11, %12, t2;\n\t"                      \
            "madc.hi.cc.u32 t3, %11, %12, t3;\n\t"                      \
            "madc.hi.cc.u32 %0, %11, %13, %0;\n\t"                      \
            "madc.hi.cc.u32 t1, %11, %14, t1;\n\t"                      \
            "madc.hi.cc.u32 t2, %11, %15, tc;\n\t"                      \
            "addc.u32       tc, 0 , 0;\n\t"                             \
            /*Multiplication involving a3 second diagonal*/             \
            "mad.lo.cc.u32  %4, %11, %13, t3;\n\t"                      \
            "madc.lo.cc.u32 %5, %11, %14, %0;\n\t"                      \
            "madc.lo.cc.u32 %6, %11, %15, t1;\n\t"                      \
            "addc.u32       %7, t2 , tc;\n\t"                           \
            "mul.lo.u32     %0, %8, %12;\n\t"                           \
            "}\n"                                                       \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), /*0,  3*/                 \
      "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  /*4,  7*/                 \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     /*8 ,11*/                 \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3));    /*12,15*/


// org multiplication code code by luis, modified by floyd
#define __school32_4x4_v2(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)    \
    asm volatile ("{\n\t"                               \
        ".reg .u32 t1,t2,t3,t4,t5,t6,t7;\n\t"           \
        ".reg .u32 t8,t9,t10,t11,t12,t13;\n\t"          \
        "mul.lo.u32     %0, %8, %12;\n\t"               \
        "mul.hi.u32     t1, %8, %12;\n\t"               \
        "mad.lo.cc.u32  t1, %8, %13, t1;\n\t"           \
        "mul.hi.u32     t2, %8, %13;\n\t"               \
        "madc.lo.cc.u32 t2, %8, %14, t2;\n\t"           \
        "mul.hi.u32     t3, %8, %14;\n\t"               \
        "madc.lo.cc.u32 t3, %8, %15, t3;\n\t"           \
        "madc.hi.u32    t4, %8, %15, 0;\n\t"            \
        "mul.hi.u32     t6, %9, %12;\n\t"               \
        "mad.lo.cc.u32  t6, %9, %13, t6;\n\t"           \
        "mul.hi.u32     t7, %9, %13;\n\t"               \
        "madc.lo.cc.u32 t7, %9, %14, t7;\n\t"           \
        "mul.hi.u32     t8, %9, %14;\n\t"               \
        "madc.lo.cc.u32 t8, %9, %15, t8;\n\t"           \
        "madc.hi.u32    t9, %9, %15, 0;\n\t"            \
        "mad.lo.cc.u32  %1, %9, %12, t1;\n\t"           \
        "addc.cc.u32    t6, t6, t2;\n\t"                \
        "addc.cc.u32    t7, t7, t3;\n\t"                \
        "addc.cc.u32    t8, t8, t4;\n\t"                \
        "addc.u32       t9, t9, 0;\n\t"                 \
        "mul.hi.u32     t3, %10,%12;\n\t"               \
        "mad.lo.cc.u32  t3, %10,%13, t3;\n\t"           \
        "mul.hi.u32     t4, %10,%13;\n\t"               \
        "madc.lo.cc.u32 t4, %10,%14, t4;\n\t"           \
        "mul.hi.u32     t5, %10,%14;\n\t"               \
        "madc.lo.cc.u32 t5, %10,%15, t5;\n\t"           \
        "madc.hi.u32    t10,%10,%15, 0;\n\t"            \
        "mad.lo.cc.u32  %2, %10,%12, t6;\n\t"           \
        "addc.cc.u32    t3, t3, t7;\n\t"                \
        "addc.cc.u32    t4, t4, t8;\n\t"                \
        "addc.cc.u32    t5, t5, t9;\n\t"                \
        "addc.u32       t10,t10, 0;\n\t"                \
        "mul.hi.u32     t7, %11,%12;\n\t"               \
        "mad.lo.cc.u32  t7, %11,%13, t7;\n\t"           \
        "mul.hi.u32     t8, %11,%13;\n\t"               \
        "madc.lo.cc.u32 t8, %11,%14, t8;\n\t"           \
        "mul.hi.u32     t9, %11,%14;\n\t"               \
        "madc.lo.cc.u32 t9, %11,%15, t9;\n\t"           \
        "madc.hi.u32    t11,%11,%15, 0;\n\t"            \
        "mad.lo.cc.u32  %3, %11,%12, t3;\n\t"           \
        "addc.cc.u32    %4, t4, t7;\n\t"                \
        "addc.cc.u32    %5, t5, t8;\n\t"                \
        "addc.cc.u32    %6, t9, t10;\n\t"               \
        "addc.u32       %7, t11, 0;\n\t"                \
        "}\n\t"                                         \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), /*0,  3*/ \
      "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  /*4,  7*/ \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     /*8 ,11*/ \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3));    /*12,15*/


// comba multiplication 4x4 based on 2x2 comba
// needs no additional registers
#define __school32_4x4_v7(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)    \
    __school32_2x2_v7(c0,c1,c2,c3, a0,a1, b0,b1, c0,c1,c2,c3)\
    __school32_2x2_v7(c2,c3,c4,c5, a2,a3, b0,b1, c2,c3,c4,c5)\
    __school32_2x2_v7(c2,c3,c4,c5, a0,a1, b2,b3, c2,c3,c4,c5)\
    __school32_2x2_v7(c4,c5,c6,c7, a2,a3, b2,b3, c4,c5,c6,c7)\


// comba multiplication 4x4, needs no add. registers
// https://mdpi-res.com/d_attachment/applsci/applsci-10-01539/article_deploy/applsci-10-01539-v2.pdf?version=1582950152
// NOT FINIDHED
#define __school32_4x4_v8(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)    \
    asm volatile ("{\n\t"                                   \
            /*first block*/                                 \
            "mul.lo.u32  	%0, %8, %12;\n\t"               \
            "mul.hi.u32  	%1, %8, %12;\n\t"               \
            "mad.lo.cc.u32  %1, %9, %12, %1;\n\t"           \
            "madc.hi.u32    %2, %9, %12, 0;\n\t"            \
            "mad.lo.cc.u32  %1, %8, %13, %1;\n\t"           \
            "madc.hi.cc.u32 %2, %8, %13, %2;\n\t"           \
            "addc.u32       %3, 0, 0;\n\t"                  \
            "mad.lo.cc.u32  %2, %5, %7, %2;\n\t"            \
            "madc.hi.u32    %3, %5, %7, %3;\n\t"            \
            /*2. block*/                                    \
            "mad.lo.cc.u32  %2, %10, %8;\n\t"               \
            "madc.hi.u32 	%3, %10, %8, %3;\n\t"           \
            "addc.u32 		%4, 0, 0;\n\t"                  \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), /*0,  3*/     \
      "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  /*4,  7*/     \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),     /*8 ,11*/     \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3));    /*12,15*/

// montgomery reduction
// 32 bit limbs, 8 limbs input, 4limbs p and mu
#define __reduce32_8(c0, c1, c2, c3, a0, a1, a2, a3, a4, a5, a6, a7, m0, m1, m2, m3, p0, p1, p2, p3)  \
    asm volatile ("{\n\t"                                       \
        ".reg .u32 t0;" ".reg .u32 t1;\n\t"                     \
        ".reg .u32 t2;" ".reg .u32 t3;\n\t"                     \
        ".reg .u32 t4;" ".reg .u32 t5;\n\t"                     \
        ".reg .u32 t6;" ".reg .u32 t7;\n\t"                     \
        ".reg .u32 t8;" ".reg .u32 t9;\n\t"                     \
        ".reg .u32 t10;" ".reg .u32 t11;\n\t"                   \
        ".reg .u32 t12;" ".reg .u32 t13;\n\t"                   \
        ".reg .u32 t14;" ".reg .u32 t15;\n\t"                   \
        ".reg .u32 t16;" ".reg .u32 t17;\n\t"                   \
        /*TODO this can be optimizes*/                          \
        "mov.u32 		t0, %4;\n\t"                            \
        "mov.u32 		t1, %5;\n\t"                            \
        "mov.u32 		t2, %6;\n\t"                            \
        "mov.u32 		t3, %7;\n\t"                            \
        "mov.u32 		t4, %8;\n\t"                            \
        "mov.u32 		t5, %9;\n\t"                            \
        "mov.u32 		t6, %10;\n\t"                           \
        "mov.u32 		t7, %11;\n\t"                           \
        /*At this point the procedure expects */                \
        /*  the values to reduce in t0-t7	  */                \
        /*Montgomery reduction, m = mu*c mod 2^(32*4)*/         \
        "mul.lo.u32     %0, t0, %12;\n\t"                       \
        "mul.hi.u32     %1, t0, %12;\n\t"                       \
        "mad.lo.cc.u32  %1, t0, %13, %1;\n\t"                   \
        "mul.hi.u32     %2, t0, %13;\n\t"                       \
        "madc.lo.cc.u32 %2, t0, %14, %2;\n\t"                   \
        "mul.hi.u32     %3, t0, %14;\n\t"                       \
        "madc.lo.u32    %3, t0, %15, %3;\n\t"                   \
        "mul.lo.u32     t8, t1, %12;\n\t"                       \
        "mul.hi.u32     t9, t1, %12;\n\t"                       \
        "mad.lo.cc.u32  t9, t1, %13, t9;\n\t"                   \
        "mul.hi.u32     t10, t1, %13;\n\t"                      \
        "madc.lo.u32    t10, t1, %14, t10;\n\t"                 \
        "add.cc.u32     %1, %1, t8;\n\t"                        \
        "addc.cc.u32    t9, t9, %2;\n\t"                        \
        "addc.u32       t0, t10, %3;\n\t"                       \
        "mul.lo.u32     %2, t2, %12;\n\t"                       \
        "mul.hi.u32     %3, t2, %12;\n\t"                       \
        "mad.lo.u32     %3, t2, %13, %3;\n\t"                   \
        "add.cc.u32     %2, %2, t9;\n\t"                        \
        "addc.u32       %3, %3, t10;\n\t"                       \
        "mad.lo.u32     %3, t3, %12, %3;\n\t"                   \
        "mul.lo.u32     t8, %0, %16;\n\t"/* u = m*p */          \
        "mul.hi.u32     t9, %0, %16;\n\t"                       \
        "mad.lo.cc.u32  t9, %0, %17, t9;\n\t"                   \
        "mul.hi.u32     t10, %0, %17;\n\t"                      \
        "madc.lo.cc.u32 t10, %0, %18, t10;\n\t"                 \
        "mul.hi.u32     t11, %0, %18;\n\t"                      \
        "madc.lo.cc.u32 t11, %0, %19, t11;"                     \
        "madc.hi.u32    t12, %0, %19, 0;\n\t"                   \
        "mul.lo.u32     t13, %1, %16;\n\t"                      \
        "mul.hi.u32     t14, %1, %16;\n\t"                      \
        "mad.lo.cc.u32  t14, %1, %17, t14;\n\t"                 \
        "mul.hi.u32     t15, %1, %17;\n\t"                      \
        "madc.lo.cc.u32 t15, %1, %18, t15;\n\t"                 \
        "mul.hi.u32     t16, %1, %18;\n\t"                      \
        "madc.lo.cc.u32 t16, %1, %19, t16;\n\t"                 \
        "madc.hi.u32    t17, %1, %19, 0;\n\t"                   \
        "add.cc.u32     t9, t9, t13;\n\t"                       \
        "addc.cc.u32    t14, t14, t10;\n\t"                     \
        "addc.cc.u32    t15, t15, t11;\n\t"                     \
        "addc.cc.u32    t16, t16, t12;\n\t"                     \
        "addc.u32       t17, t17, 0;\n\t"                       \
        "mul.lo.u32     t10, %2, %16;\n\t"                      \
        "mul.hi.u32     t11, %2, %16;\n\t"                      \
        "mad.lo.cc.u32  t11, %2, %17, t11;\n\t"                 \
        "mul.hi.u32     t12, %2, %17;\n\t"                      \
        "madc.lo.cc.u32 t12, %2, %18, t12;\n\t"                 \
        "mul.hi.u32     t13, %2, %18;\n\t"                      \
        "madc.lo.cc.u32 t13, %2, %19, t13;\n\t"                 \
        "madc.hi.u32    %0, %2, %19, 0;\n\t"                    \
        "add.cc.u32     t10, t10, t14;\n\t"                     \
        "addc.cc.u32    t11, t11, t15;\n\t"                     \
        "addc.cc.u32    t12, t12, t16;\n\t"                     \
        "addc.cc.u32    t13, t13, t17;\n\t"                     \
        "addc.u32       %0, %0, 0;\n\t"                         \
        "mul.lo.u32     t14, %3, %16;\n\t"                      \
        "mul.hi.u32     t15, %3, %16;\n\t"                      \
        "mad.lo.cc.u32  t15, %3, %17, t15;\n\t"                 \
        "mul.hi.u32     t16, %3, %17;\n\t"                      \
        "madc.lo.cc.u32 t16, %3, %18, t16;\n\t"                 \
        "mul.hi.u32     t17, %3, %18;\n\t"                      \
        "madc.lo.cc.u32 t17, %3, %19, t17;\n\t"                 \
        "madc.hi.u32    %1, %3, %19, 0;\n\t"                    \
        "add.cc.u32     t11, t11, t14;\n\t"                     \
        "addc.cc.u32    t12, t12, t15;\n\t"                     \
        "addc.cc.u32    t13, t13, t16;\n\t"                     \
        "addc.cc.u32    t14, %0, t17;\n\t"                      \
        "addc.u32       t15, %1, 0;\n\t"                        \
        "mov.b32        %0, 0;\n\t"                             \
        /*r = (c + u) div 2^(4*32)*/                            \
        "add.cc.u32     t0, t8, t0;\n\t"                        \
        "addc.cc.u32    t1, t9, t1;\n\t"                        \
        "addc.cc.u32    t2, t10, t2;\n\t"                       \
        "addc.cc.u32    t3, t11, t3;\n\t"                       \
        "addc.cc.u32    t4, t12, t4;\n\t"                       \
        "addc.cc.u32    t5, t13, t5;\n\t"                       \
        "addc.cc.u32    t6, t14, t6;\n\t"                       \
        "addc.cc.u32    t7, t15, t7;\n\t"                       \
        "addc.u32       %0, %0, 0;\n\t"                         \
        "sub.cc.u32     t4, t4, %16;\n\t"                       \
        "subc.cc.u32    t5, t5, %17;\n\t"                       \
        "subc.cc.u32    t6, t6, %18;\n\t"                       \
        "subc.cc.u32    t7, t7, %19;\n\t"                       \
        "subc.u32       %0, %0, 0;\n\t"                         \
        "mov.u32        %1, %0;\n\t"                            \
        "mov.u32        %2, %0;\n\t"                            \
        "mov.u32        %3, %0;\n\t"                            \
        "and.b32        %0, %0, %16;\n\t"                       \
        "and.b32        %1, %1, %17;\n\t"                       \
        "and.b32        %2, %2, %18;\n\t"                       \
        "and.b32        %3, %3, %19;\n\t"                       \
        "add.cc.u32     %0, %0, t4;\n\t"                        \
        "addc.cc.u32    %1, %1, t5;\n\t"                        \
        "addc.cc.u32    %2, %2, t6;\n\t"                        \
        "addc.u32       %3, %3, t7;\n\t"                        \
        "}\n"                                                   \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) /*0,3*/        \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),    /*4,7*/        \
          "r"(a4), "r"(a5), "r"(a6), "r"(a7),    /*8,11*/       \
          "r"(m0), "r"(m1), "r"(m2), "r"(m3),    /*12,16*/      \
          "r"(p0), "r"(p1), "r"(p2), "r"(p3))    /*16,19*/

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**128
// This version does not use the 'mad' instructions
// Inplace version
#define __school32_inplace_4x4_low(a0, a1, a2, a3, mu0, mu1, mu2, mu3)    \
    asm volatile ("{\n\t"                                        \
            ".reg .u32 t10,t20,t30;\n\t"                         \
            /* t0-t5: m = a * mu mod 2^128 */                    \
            /* Multiplication involving a0 */                    \
            "mul.lo.u32     %0,  %8, %4;\n\t"                    \
            "mul.hi.u32     %1,  %8, %4;\n\t"                    \
            "mul.lo.u32     t10, %8, %5;\n\t"                    \
            "mul.hi.u32     %2,  %8, %5;\n\t"                    \
            "mul.lo.u32     t20, %8, %6;\n\t"                    \
            "mul.hi.u32     %3,  %8, %6;\n\t"                    \
            "mul.lo.u32     t30, %8, %7;\n\t"                    \
            /* now fixing the carry chains */                    \
            "add.cc.u32     %1,  %1,  t10;\n\t"                  \
            "addc.cc.u32    %2,  %2,  t20;\n\t"                  \
            "addc.u32       %3,  %3,  t30;\n\t"                  \
            /*Multiplication involving a1 second (low) diagonal*/\
            "mul.lo.u32     t10, %9, %4;\n\t"                    \
            "mul.lo.u32     t20, %9, %5;\n\t"                    \
            "mul.lo.u32     t30, %9, %6;\n\t"                    \
            "add.cc.u32     %1, %1, t10;\n\t"                    \
            "addc.cc.u32    %2, %2, t20;\n\t"                    \
            "addc.u32       %3, %3, t30;\n\t"                    \
            /*Multiplication involving a1 first (high) diagonal*/\
            "mul.hi.u32     t20, %9, %4;\n\t"                    \
            "mul.hi.u32     t30, %9, %5;\n\t"                    \
            "add.cc.u32     %2, %2, t20;\n\t"                    \
            "addc.u32       %3, %3, t30;\n\t"                    \
            /*Multiplication involving a2 second (low) diagonal*/\
            "mul.lo.u32     t20, %10, %4;\n\t"                   \
            "mul.lo.u32     t30, %10, %5;\n\t"                   \
            "add.cc.u32     %2, %2, t20;\n\t"                    \
            "addc.u32    	%3, %3, t30;\n\t"                    \
            /*Multiplication involving a2 first (high) diagonal*/\
            "mul.hi.u32     t30, %10, %4;\n\t"                   \
            "add.u32        %3, %3, t30;\n\t"                    \
            /*Multiplication involving a3 second (low) diagonal*/\
            "mul.lo.u32     t30, %11, %4;\n\t"                   \
            "add.u32         %3, %3, t30;\n\t"                   \
        "}\n"                                                    \
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)  /*0 , 3*/         \
        : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3), /*4 , 7*/         \
          "r"(a0),"r"(a1),"r"(a2),"r"(a3)       /*8 ,11*/        \
    );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**128
// This version does not use the 'mad' instructions
// non inplace version
#define __school32_4x4_low(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3)    \
    asm volatile ("{\n\t"                                        \
            ".reg .u32 t10,t20,t30;\n\t"                         \
            /* t0-t5: m = a * mu mod 2^128 */                    \
            /* Multiplication involving a0 */                    \
            "mul.lo.u32     %0,  %8, %4;\n\t"                    \
            "mul.hi.u32     %1,  %8, %4;\n\t"                    \
            "mul.lo.u32     t10, %8, %5;\n\t"                    \
            "mul.hi.u32     %2,  %8, %5;\n\t"                    \
            "mul.lo.u32     t20, %8, %6;\n\t"                    \
            "mul.hi.u32     %3,  %8, %6;\n\t"                    \
            "mul.lo.u32     t30, %8, %7;\n\t"                    \
            /* now fixing the carry chains */                    \
            "add.cc.u32     %1,  %1,  t10;\n\t"                  \
            "addc.cc.u32    %2,  %2,  t20;\n\t"                  \
            "addc.u32       %3,  %3,  t30;\n\t"                  \
            /*Multiplication involving a1 second (low) diagonal*/\
            "mul.lo.u32     t10, %9, %4;\n\t"                    \
            "mul.lo.u32     t20, %9, %5;\n\t"                    \
            "mul.lo.u32     t30, %9, %6;\n\t"                    \
            "add.cc.u32     %1, %1, t10;\n\t"                    \
            "addc.cc.u32    %2, %2, t20;\n\t"                    \
            "addc.u32       %3, %3, t30;\n\t"                    \
            /*Multiplication involving a1 first (high) diagonal*/\
            "mul.hi.u32     t20, %9, %4;\n\t"                    \
            "mul.hi.u32     t30, %9, %5;\n\t"                    \
            "add.cc.u32     %2, %2, t20;\n\t"                    \
            "addc.u32       %3, %3, t30;\n\t"                    \
            /*Multiplication involving a2 second (low) diagonal*/\
            "mul.lo.u32     t20, %10, %4;\n\t"                   \
            "mul.lo.u32     t30, %10, %5;\n\t"                   \
            "add.cc.u32     %2, %2, t20;\n\t"                    \
            "addc.u32    	%3, %3, t30;\n\t"                    \
            /*Multiplication involving a2 first (high) diagonal*/\
            "mul.hi.u32     t30, %10, %4;\n\t"                   \
            "add.u32        %3, %3, t30;\n\t"                    \
            /*Multiplication involving a3 second (low) diagonal*/\
            "mul.lo.u32     t30, %11, %4;\n\t"                   \
            "add.u32        %3, %3, t30;\n\t"                    \
            "}\n"                                                \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)  /*0 , 3*/     \
            : "r"(b0),"r"(b1),"r"(b2),"r"(b3),     /*4 , 7*/     \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3)       /*8 ,11*/    \
        );

// second part of the montgomery reduction written by floyd
// inplace version
#define __reduce32_inplace_sec4x4(a0, a1, a2, a3, a4, a5, a6, a7, p0, p1, p2, p3, i0, i1, i2, i3, i4, i5, i6, i7)    \
    asm volatile ("{\n\t"                                               \
            /*%0-%5: r= (a + u) div 2^(4**32)*/                         \
            ".reg.u32 		t0, t4,t5,t6,t7;\n\t"                       \
            "mov.b32        t0, 0;\n\t"                                 \
            /*a + u*/                                                   \
            "add.cc.u32     %0, %16, %8;\n\t"                           \
            "addc.cc.u32    %1, %17, %9;\n\t"                           \
            "addc.cc.u32    %2, %18, %10;\n\t"                          \
            "addc.cc.u32    %3, %19, %11;\n\t"                          \
            "addc.cc.u32    t4, %20, %12;\n\t"                          \
            "addc.cc.u32    t5, %21, %13;\n\t"                          \
            "addc.cc.u32    t6, %22, %14;\n\t"                          \
            "addc.cc.u32    t7, %23, %15;\n\t"                          \
            "addc.u32       t0, t0, 0;\n\t"                             \
            /*-p*/                                                      \
            "sub.cc.u32     t4, t4, %4;\n\t"                            \
            "subc.cc.u32    t5, t5, %5;\n\t"                            \
            "subc.cc.u32    t6, t6, %6;\n\t"                            \
            "subc.cc.u32    t7, t7, %7;\n\t"                            \
            "subc.u32       t0, t0, 0;\n\t"                             \
            /*&p*/                                                      \
            "and.b32        %0, t0, %4;\n\t"                            \
            "and.b32        %1, t0, %5;\n\t"                            \
            "and.b32        %2, t0, %6;\n\t"                            \
            "and.b32        %3, t0, %7;\n\t"                            \
            "add.cc.u32     %0, %0, t4;\n\t"                            \
            "addc.cc.u32    %1, %1, t5;\n\t"                            \
            "addc.cc.u32    %2, %2, t6;\n\t"                            \
            "addc.u32    %3, %3, t7;\n\t"                               \
            "}\n"                                                       \
    : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)   /*0 , 3*/                   \
    : "r"(p0),"r"(p1),"r"(p2),"r"(p3),      /*4 , 7*/                   \
      "r"(i0),"r"(i1),"r"(i2),"r"(i3),      /*8 ,11*/                   \
      "r"(i4),"r"(i5),"r"(i6),"r"(i7),      /*12,15*/                   \
      "r"(a0),"r"(a1),"r"(a2),"r"(a3),      /*16,19*/                   \
      "r"(a4),"r"(a5),"r"(a6),"r"(a7)       /*20,23*/                   \
    );

// second part of the montgomery reduction written by floyd
#define __reduce32_sec4x4(c0, c1, c2, c3, a0, a1, a2, a3, a4, a5, a6, a7, p0, p1, p2, p3, i0, i1, i2, i3, i4, i5, i6, i7)    \
    asm volatile ("{\n\t"                                           \
            /*%0-%5: r= (a + u) div 2^(4**32)*/                     \
            ".reg.u32 		t0,t4,t5,t6,t7;\n\t"                    \
            "mov.b32        t0, 0;\n\t"                             \
            /*a + u*/                                               \
            "add.cc.u32     %0, %16, %8;\n\t"                       \
            "addc.cc.u32    %1, %17, %9;\n\t"                       \
            "addc.cc.u32    %2, %18, %10;\n\t"                      \
            "addc.cc.u32    %3, %19, %11;\n\t"                      \
            "addc.cc.u32    t4, %20, %12;\n\t"                      \
            "addc.cc.u32    t5, %21, %13;\n\t"                      \
            "addc.cc.u32    t6, %22, %14;\n\t"                      \
            "addc.cc.u32    t7, %23, %15;\n\t"                      \
            "addc.u32       t0, t0, 0;\n\t"                         \
            /*-p*/                                                  \
            "sub.cc.u32     t4, t4, %4;\n\t"                        \
            "subc.cc.u32    t5, t5, %5;\n\t"                        \
            "subc.cc.u32    t6, t6, %6;\n\t"                        \
            "subc.cc.u32    t7, t7, %7;\n\t"                        \
            "subc.u32       t0, t0, 0;\n\t"                         \
            /*&p*/                                                  \
            "and.b32        %0, t0, %4;\n\t"                        \
            "and.b32        %1, t0, %5;\n\t"                        \
            "and.b32        %2, t0, %6;\n\t"                        \
            "and.b32        %3, t0, %7;\n\t"                        \
            "add.cc.u32     %0, %0, t4;\n\t"                        \
            "addc.cc.u32    %1, %1, t5;\n\t"                        \
            "addc.cc.u32    %2, %2, t6;\n\t"                        \
            "addc.u32    %3, %3, t7;\n\t"                           \
        "}\n"                                                       \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)   /*0 , 3*/               \
    : "r"(p0),"r"(p1),"r"(p2),"r"(p3),      /*4 , 7*/               \
      "r"(i0),"r"(i1),"r"(i2),"r"(i3),      /*8 ,11*/               \
      "r"(i4),"r"(i5),"r"(i6),"r"(i7),      /*12,15*/               \
      "r"(a0),"r"(a1),"r"(a2),"r"(a3),      /*16,19*/               \
      "r"(a4),"r"(a5),"r"(a6),"r"(a7)       /*20,23*/               \
    );

// montgomery reduction for 8 32bit limbs.
// inplace version
// a3a2a1a0 = a7a6a5a4a3a2a1a0 mod p3p2p1p0
#define __reduce32_inplace4x4(a0, a1, a2, a3, a4, a5, a6, a7, mu0, mu1, mu2, mu3, p0, p1, p2, p3)   \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7;                                                               \
    __school32_inplace_4x4_low(a0,a1,a2,a3,mu0,mu1,mu2,mu3)                                         \
    __school32_4x4(t0,t1,t2,t3,t4,t5,t6,t7,a0,a1,a2,a3,p0,p1,p2,p3)                                 \
    __reduce32_inplace_sec4x4(a0,a1,a2,a3,a4,a5,a6,a7,p0,p1,p2,p3,t0,t1,t2,t3,t4,t5,t6,t7)

// montgomery reduction for 8 32bit limbs.
// c3c2c1c0 = a7a6a5a4a3a2a1a0 mod p3p2p1p0
#define __reduce32_helper_4x4(c0, c1, c2, c3, a0, a1, a2, a3, a4, a5, a6, a7, mu0, mu1, mu2, mu3, p0, p1, p2, p3, t0, t1, t2, t3, t4, t5, t6, t7, m0, m1, m2, m3, f0, f1, f2, f3, f4, f5, f6, f7)    \
    __school32_4x4_low(m0,m1,m2,m3, a0,a1,a2,a3, mu0,mu1,mu2,mu3)                                           \
    __karatsuba32_helper_4x4(t0,t1,t2,t3,t4,t5,t6,t7, m0,m1,m2,m3, p0,p1,p2,p3, f0,f1,f2,f3,f4,f5,f6,f7)    \
    __reduce32_sec4x4(c0,c1,c2,c3, a0,a1,a2,a3,a4,a5,a6,a7, p0,p1,p2,p3, t0,t1,t2,t3,t4,t5,t6,t7)

#define __reduce32_4x4(c0, c1, c2, c3, a0, a1, a2, a3, a4, a5, a6, a7, mu0, mu1, mu2, mu3, p0, p1, p2, p3)  \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7,m0,m1,m2,m3;                                                           \
    uint32_t k0,k1,k2, s0,s1,s2, f0,f1,f2,f3,f4,f5,f6,f7;    /*karatsuba helper*/                           \
    __reduce32_helper_4x4(c0,c1,c2,c3, a0,a1,a2,a3,a4,a5,a6,a7, mu0,mu1,mu2,mu3, p0,p1,p2,p3, t0,t1,t2,t3,t4,t5,t6,t7, m0,m1,m2,m3, k0,k1,k2, s0,s1,s2, f0,f1,f2,f3,f4,f5,f6,f7)    \

// c3c2c1c0 = a3a2a1a0 * b3b2b1b0 mod p3p2p1p0
#define __fp_mul4x4_32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, mu0, mu1, mu2, mu3, p0, p1, p2, p3)  \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7;                        /*mult helper*/                                \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7;                        /*karatsuba helper*/                           \
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7,m0,m1,m2,m3;            /*reduce help */                               \
    __karatsuba32_helper_4x4(f0,f1,f2,f3,f4,f5,f6,f7, a0,a1,a2,a3, b0,b1,b2,b3, t0,t1,t2,t3,t4,t5,t6,t7)    \
    __reduce32_helper_4x4(c0,c1,c2,c3, f0,f1,f2,f3,f4,f5,f6,f7, mu0,mu1,mu2,mu3, p0,p1,p2,p3, r0,r1,r2,r3,r4,r5,r6,r7, m0,m1,m2,m3,  t0,t1,t2,t3,t4,t5,t6,t7)


// fp 4x4 multiplication based on school book multiplication.
// uses 16 +  helper registers
#define __fp_mul4x4_32_v2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, mu0, mu1, mu2, mu3, p0, p1, p2, p3)   \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7;                        /*mult helper*/                                    \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7;                        /*reduce help */                                   \
    __school32_4x4(f0,f1,f2,f3,f4,f5,f6,f7, a0,a1,a2,a3, b0,b1,b2,b3)                                           \
    __school32_4x4_low(c0,c1,c2,c3, a0,a1,a2,a3, mu0,mu1,mu2,mu3)                                               \
    __school32_4x4(t0,t1,t2,t3,t4,t5,t6,t7, c0,c1,c2,c3, p0,p1,p2,p3)                                           \
    __reduce32_sec4x4(c0,c1,c2,c3, f0,f1,f2,f3,f4,f5,f6,f7, p0,p1,p2,p3, t0,t1,t2,t3,t4,t5,t6,t7)

// fp 4x4 multiplication based on comba multiplication.
// uses 16 helper registers
#define __fp_mul4x4_32_v7(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, mu0, mu1, mu2, mu3, p0, p1, p2, p3)   \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7;                        /*mult helper*/                                    \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7;                        /*reduce help */                                   \
    __school32_4x4_v7(f0,f1,f2,f3,f4,f5,f6,f7, a0,a1,a2,a3, b0,b1,b2,b3)                                        \
    __school32_4x4_low(c0,c1,c2,c3, a0,a1,a2,a3, mu0,mu1,mu2,mu3)                                               \
    __school32_4x4_v7(t0,t1,t2,t3,t4,t5,t6,t7, c0,c1,c2,c3, p0,p1,p2,p3)                                        \
    __reduce32_sec4x4(c0,c1,c2,c3, f0,f1,f2,f3,f4,f5,f6,f7, p0,p1,p2,p3, t0,t1,t2,t3,t4,t5,t6,t7)


// org code by luis, modified by floyd
#define __fp_mul4x4_32_v3(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, m0, m1, m2, m3, p0, p1, p2, p3)  \
    asm volatile ("{\n\t"                       \
        ".reg .u32 t1,t2,t3,t4,t5,t6,t7;\n\t"   \
        ".reg .u32 t8,t9,t10,t11,t12,t13;\n\t"  \
        ".reg .u32 t14,t15,t16,t17;\n\t"        \
        "mul.hi.u32     t1, %4, %8;\n\t"        \
        "mad.lo.cc.u32  t1, %4, %9, t1;\n\t"    \
        "mul.hi.u32     t2, %4, %9;\n\t"        \
        "madc.lo.cc.u32 t2, %4, %10, t2;\n\t"   \
        "mul.hi.u32     t3, %4, %10;\n\t"       \
        "madc.lo.cc.u32 t3, %4, %11, t3;\n\t"   \
        "madc.hi.u32    t4, %4, %11, 0;\n\t"    \
        "mul.hi.u32     t6, %5, %8;\n\t"        \
        "mad.lo.cc.u32  t6, %5, %9, t6;\n\t"    \
        "mul.hi.u32     t7, %5, %9;\n\t"        \
        "madc.lo.cc.u32 t7, %5, %10, t7;\n\t"   \
        "mul.hi.u32     t8, %5, %10;\n\t"       \
        "madc.lo.cc.u32 t8, %5, %11, t8;\n\t"   \
        "madc.hi.u32    t9, %5, %11, 0;\n\t"    \
        "mad.lo.cc.u32  t1, %5, %8, t1;\n\t"    \
        "addc.cc.u32    t6, t6, t2;\n\t"        \
        "addc.cc.u32    t7, t7, t3;\n\t"        \
        "addc.cc.u32    t8, t8, t4;\n\t"        \
        "addc.u32       t9, t9, 0;\n\t"         \
        "mul.hi.u32     t3, %6, %8;\n\t"        \
        "mad.lo.cc.u32  t3, %6, %9, t3;\n\t"    \
        "mul.hi.u32     t4, %6, %9;\n\t"        \
        "madc.lo.cc.u32 t4, %6, %10, t4;\n\t"   \
        "mul.hi.u32     t5, %6, %10;\n\t"       \
        "madc.lo.cc.u32 t5, %6, %11, t5;\n\t"   \
        "madc.hi.u32    t10, %6, %11, 0;\n\t"   \
        "mad.lo.cc.u32  t2, %6, %8, t6;\n\t"    \
        "addc.cc.u32    t3, t3, t7;\n\t"        \
        "addc.cc.u32    t4, t4, t8;\n\t"        \
        "addc.cc.u32    t5, t5, t9;\n\t"        \
        "addc.u32       t10, t10, 0;\n\t"       \
        "mul.hi.u32     t7, %7, %8;\n\t"        \
        "mad.lo.cc.u32  t7, %7, %9, t7;\n\t"    \
        "mul.hi.u32     t8, %7, %9;\n\t"        \
        "madc.lo.cc.u32 t8, %7, %10, t8;\n\t"   \
        "mul.hi.u32     t9, %7, %10;\n\t"       \
        "madc.lo.cc.u32 t9, %7, %11, t9;\n\t"   \
        "madc.hi.u32    t11, %7, %11, 0;\n\t"   \
        "mad.lo.cc.u32  t3, %7, %8, t3;\n\t"    \
        "addc.cc.u32    t4, t4, t7;\n\t"        \
        "addc.cc.u32    t5, t5, t8;\n\t"        \
        "addc.cc.u32    t6, t9, t10;\n\t"       \
        "addc.u32       t7, t11, 0;\n\t"        \
        /*Montgomery reduction, m = mu*c mod 2^(64*4)*/ \
        "mul.lo.u32     t17, %4, %8;\n\t"        \
        "mul.lo.u32     %0, t17, %12;\n\t"       \
        "mul.hi.u32     %1, t17, %12;\n\t"       \
        "mad.lo.cc.u32  %1, t17, %13, %1;\n\t"   \
        "mul.hi.u32     %2, t17, %13;\n\t"       \
        "madc.lo.cc.u32 %2, t17, %14, %2;\n\t"   \
        "mul.hi.u32     %3, t17, %14;\n\t"       \
        "madc.lo.u32    %3, t17, %15, %3;\n\t"   \
        "mul.hi.u32     t9, t1, %12;\n\t"       \
        "mad.lo.cc.u32  t9, t1, %13, t9;\n\t"   \
        "mul.hi.u32     t10, t1, %13;\n\t"      \
        "madc.lo.u32    t10, t1, %14, t10;\n\t" \
        "mad.lo.cc.u32  %1, t1, %12, %1;\n\t"   \
        "addc.cc.u32    t9, t9, %2;\n\t"        \
        "addc.u32       t10, t10, %3;\n\t"      \
        "mul.hi.u32     %3, t2, %12;\n\t"       \
        "mad.lo.u32     %3, t2, %13, %3;\n\t"   \
        "mad.lo.cc.u32  %2, t2, %12, t9;\n\t"   \
        "addc.u32       %3, %3, t10;\n\t"       \
        "mad.lo.u32     %3, t3, %12, %3;\n\t"   \
        "mul.lo.u32     t8, %0, %16;\n\t"/* u = m*p */\
        "mul.hi.u32     t9, %0, %16;\n\t"       \
        "mad.lo.cc.u32  t9, %0, %17, t9;\n\t"   \
        "mul.hi.u32     t10, %0, %17;\n\t"      \
        "madc.lo.cc.u32 t10, %0, %18, t10;\n\t" \
        "mul.hi.u32     t11, %0, %18;\n\t"      \
        "madc.lo.cc.u32 t11, %0, %19, t11;"     \
        "madc.hi.u32    t12, %0, %19, 0;\n\t"   \
        "mul.hi.u32     t14, %1, %16;\n\t"      \
        "mad.lo.cc.u32  t14, %1, %17, t14;\n\t" \
        "mul.hi.u32     t15, %1, %17;\n\t"      \
        "madc.lo.cc.u32 t15, %1, %18, t15;\n\t" \
        "mul.hi.u32     t16, %1, %18;\n\t"      \
        "madc.lo.cc.u32 t16, %1, %19, t16;\n\t" \
        "madc.hi.u32    t17, %1, %19, 0;\n\t"   \
        "mad.lo.cc.u32  t9, %1, %16, t9;\n\t"   \
        "addc.cc.u32    t14, t14, t10;\n\t"     \
        "addc.cc.u32    t15, t15, t11;\n\t"     \
        "addc.cc.u32    t16, t16, t12;\n\t"     \
        "addc.u32       t17, t17, 0;\n\t"       \
        "mul.hi.u32     t11, %2, %16;\n\t"      \
        "mad.lo.cc.u32  t11, %2, %17, t11;\n\t" \
        "mul.hi.u32     t12, %2, %17;\n\t"      \
        "madc.lo.cc.u32 t12, %2, %18, t12;\n\t" \
        "mul.hi.u32     t13, %2, %18;\n\t"      \
        "madc.lo.cc.u32 t13, %2, %19, t13;\n\t" \
        "madc.hi.u32    %0, %2, %19, 0;\n\t"    \
        "madc.lo.cc.u32 t10, %2, %16, t14;\n\t" \
        "addc.cc.u32    t11, t11, t15;\n\t"     \
        "addc.cc.u32    t12, t12, t16;\n\t"     \
        "addc.cc.u32    t13, t13, t17;\n\t"     \
        "addc.u32       %0, %0, 0;\n\t"         \
        "mul.hi.u32     t15, %3, %16;\n\t"      \
        "mad.lo.cc.u32  t15, %3, %17, t15;\n\t" \
        "mul.hi.u32     t16, %3, %17;\n\t"      \
        "madc.lo.cc.u32 t16, %3, %18, t16;\n\t" \
        "mul.hi.u32     t17, %3, %18;\n\t"      \
        "madc.lo.cc.u32 t17, %3, %19, t17;\n\t" \
        "madc.hi.u32    %1, %3, %19, 0;\n\t"    \
        "madc.lo.cc.u32 t11, %3, %16, t11;\n\t" \
        "addc.cc.u32    t12, t12, t15;\n\t"     \
        "addc.cc.u32    t13, t13, t16;\n\t"     \
        "addc.cc.u32    t14, %0, t17;\n\t"      \
        "addc.u32       t15, %1, 0;\n\t"        \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(4*64)*/\
        "mad.lo.cc.u32  t17, %4, %8, t8;\n\t"   \
        "addc.cc.u32    t1, t9, t1;\n\t"        \
        "addc.cc.u32    t2, t10, t2;\n\t"       \
        "addc.cc.u32    t3, t11, t3;\n\t"       \
        "addc.cc.u32    t4, t12, t4;\n\t"       \
        "addc.cc.u32    t5, t13, t5;\n\t"       \
        "addc.cc.u32    t6, t14, t6;\n\t"       \
        "addc.cc.u32    t7, t15, t7;\n\t"       \
        "addc.u32       %0, %0, 0;\n\t"         \
        "sub.cc.u32     t4, t4, %16;\n\t"       \
        "subc.cc.u32    t5, t5, %17;\n\t"       \
        "subc.cc.u32    t6, t6, %18;\n\t"       \
        "subc.cc.u32    t7, t7, %19;\n\t"       \
        "subc.u32       %0, %0, 0;\n\t"         \
        "mov.u32        %1, %0;\n\t"            \
        "mov.u32        %2, %0;\n\t"            \
        "mov.u32        %3, %0;\n\t"            \
        "and.b32        %0, %0, %16;\n\t"       \
        "and.b32        %1, %1, %17;\n\t"       \
        "and.b32        %2, %2, %18;\n\t"       \
        "and.b32        %3, %3, %19;\n\t"       \
        "add.cc.u32     %0, %0, t4;\n\t"        \
        "addc.cc.u32    %1, %1, t5;\n\t"        \
        "addc.cc.u32    %2, %2, t6;\n\t"        \
        "addc.u32       %3, %3, t7;\n\t"        \
        "}"                                     \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)\
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),    \
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),    \
          "r"(m0), "r"(m1), "r"(m2), "r"(m3),    \
          "r"(p0), "r"(p1), "r"(p2), "r"(p3));

// mult is a schoolbook mult by floyd. The reduction is by luis
// NOTE t1 is computed wrong
#define __fp_mul4x4_32_v4(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, m0, m1, m2, m3, p0, p1, p2, p3)  \
    asm volatile ("{\n\t"                               \
        ".reg .u32 t0,t1,t2,t3,t4,t5,t6,t7;\n\t"        \
        ".reg .u32 t8,t9,t10,t11,t12,t13;\n\t"          \
        ".reg .u32 t14,t15,t16,t17,tc;\n\t"             \
        /*Multiplication involving a0*/                 \
        "mul.hi.u32     t0, %4, %8;\n\t"                \
        "mad.lo.cc.u32  t0, %4, %9, t0;\n\t"            \
        "mul.hi.u32     t1, %4, %9;\n\t"                \
        "madc.lo.cc.u32 t1, %4, %10, t1;\n\t"           \
        "mul.hi.u32     t2, %4, %10;\n\t"               \
        "madc.lo.cc.u32 t2, %4, %11, t2;\n\t"           \
        "mul.hi.u32     t3, %4, %11;\n\t"               \
        "addc.cc.u32    t3, t3, 0;\n\t"                 \
        "addc.u32       tc, 0, 0;\n\t"                  \
        /*Multiplication involving a1 first diagonal*/  \
        "mad.lo.cc.u32  %1, %5, %8, t0;\n\t"            \
        "madc.hi.cc.u32 t1, %5, %8, t1;\n\t"            \
        "madc.hi.cc.u32 t2, %5, %9, t2;\n\t"            \
        "madc.hi.cc.u32 t3, %5, %10, t3;\n\t"           \
        "mul.hi.u32     t0, %5, %11;\n\t"               \
        "addc.cc.u32    t0, t0 , tc;\n\t"               \
        /*Multiplication involving a1 second diagonal*/ \
        "mad.lo.cc.u32  t1, %5, %9, t1;\n\t"            \
        "madc.lo.cc.u32 t2, %5, %10, t2;\n\t"           \
        "madc.lo.cc.u32 t3, %5, %11, t3;\n\t"           \
        "addc.cc.u32    t0, t0 , 0;\n\t"                \
        "addc.u32       tc, tc , 0;\n\t"                \
        /*Multiplication involving a2 first diagonal*/  \
        "mad.lo.cc.u32  %2, %6, %8, t1;\n\t"            \
        "madc.hi.cc.u32 t2, %6, %8, t2;\n\t"            \
        "madc.hi.cc.u32 t3, %6, %9, t3;\n\t"            \
        "madc.hi.cc.u32 t0, %6, %10, t0;\n\t"           \
        "mul.hi.u32     t1, %6, %11;\n\t"               \
        "addc.cc.u32    t1, t1 , tc;\n\t"               \
        "addc.u32       tc, 0 , 0;\n\t"                 \
        /*Multiplication involving a2 second diagonal*/ \
        "mad.lo.cc.u32  t2, %6, %9, t2;\n\t"            \
        "madc.lo.cc.u32 t3, %6, %10, t3;\n\t"           \
        "madc.lo.cc.u32 t0, %6, %11, t0;\n\t"           \
        "addc.cc.u32    t1, t1 , 0;\n\t"                \
        "addc.u32       tc, tc , 0;\n\t"                \
        /*Multiplication involving a3 first diagonal*/  \
        "mad.lo.cc.u32  %3, %7, %8, t2;\n\t"            \
        "madc.hi.cc.u32 t3, %7, %8, t3;\n\t"            \
        "madc.hi.cc.u32 t0, %7, %9, t0;\n\t"            \
        "madc.hi.cc.u32 t1, %7, %10, t1;\n\t"           \
        "mul.hi.u32     t2, %7, %11;\n\t"               \
        "addc.cc.u32    t2, t2 , tc;\n\t"               \
        "addc.u32       tc, 0 , 0;\n\t"                 \
        /*Multiplication involving a3 second diagonal*/ \
        "mad.lo.cc.u32  t4, %7, %9, t3;\n\t"            \
        "madc.lo.cc.u32 t5, %7, %19, t0;\n\t"           \
        "madc.lo.cc.u32 t6, %7, %11, t1;\n\t"           \
        "addc.u32       t7, t2 , tc;\n\t"               \
        "mov.u32 		t1, %1;\n\t"                    \
        "mov.u32 		t2, %2;\n\t"                    \
        "mov.u32 		t3, %3;\n\t"                    \
                                                        \
        /*Montgomery reduction, m = mu*c mod 2^(64*4)*/ \
        "mul.lo.u32     t17, %4, %8;\n\t"       \
        "mul.lo.u32     %0, t17, %12;\n\t"      \
        "mul.hi.u32     %1, t17, %12;\n\t"      \
        "mad.lo.cc.u32  %1, t17, %13, %1;\n\t"  \
        "mul.hi.u32     %2, t17, %13;\n\t"      \
        "madc.lo.cc.u32 %2, t17, %14, %2;\n\t"  \
        "mul.hi.u32     %3, t17, %14;\n\t"      \
        "madc.lo.u32    %3, t17, %15, %3;\n\t"  \
        "mul.hi.u32     t9, t1, %12;\n\t"       \
        "mad.lo.cc.u32  t9, t1, %13, t9;\n\t"   \
        "mul.hi.u32     t10, t1, %13;\n\t"      \
        "madc.lo.u32    t10, t1, %14, t10;\n\t" \
        "mad.lo.cc.u32  %1, t1, %12, %1;\n\t"   \
        "addc.cc.u32    t9, t9, %2;\n\t"        \
        "addc.u32       t10, t10, %3;\n\t"      \
        "mul.hi.u32     %3, t2, %12;\n\t"       \
        "mad.lo.u32     %3, t2, %13, %3;\n\t"   \
        "mad.lo.cc.u32  %2, t2, %12, t9;\n\t"   \
        "addc.u32       %3, %3, t10;\n\t"       \
        "mad.lo.u32     %3, t3, %12, %3;\n\t"   \
        "mul.lo.u32     t8, %0, %16;\n\t"/* u = m*p */\
        "mul.hi.u32     t9, %0, %16;\n\t"       \
        "mad.lo.cc.u32  t9, %0, %17, t9;\n\t"   \
        "mul.hi.u32     t10, %0, %17;\n\t"      \
        "madc.lo.cc.u32 t10, %0, %18, t10;\n\t" \
        "mul.hi.u32     t11, %0, %18;\n\t"      \
        "madc.lo.cc.u32 t11, %0, %19, t11;"     \
        "madc.hi.u32    t12, %0, %19, 0;\n\t"   \
        "mul.hi.u32     t14, %1, %16;\n\t"      \
        "mad.lo.cc.u32  t14, %1, %17, t14;\n\t" \
        "mul.hi.u32     t15, %1, %17;\n\t"      \
        "madc.lo.cc.u32 t15, %1, %18, t15;\n\t" \
        "mul.hi.u32     t16, %1, %18;\n\t"      \
        "madc.lo.cc.u32 t16, %1, %19, t16;\n\t" \
        "madc.hi.u32    t17, %1, %19, 0;\n\t"   \
        "mad.lo.cc.u32  t9, %1, %16, t9;\n\t"   \
        "addc.cc.u32    t14, t14, t10;\n\t"     \
        "addc.cc.u32    t15, t15, t11;\n\t"     \
        "addc.cc.u32    t16, t16, t12;\n\t"     \
        "addc.u32       t17, t17, 0;\n\t"       \
        "mul.hi.u32     t11, %2, %16;\n\t"      \
        "mad.lo.cc.u32  t11, %2, %17, t11;\n\t" \
        "mul.hi.u32     t12, %2, %17;\n\t"      \
        "madc.lo.cc.u32 t12, %2, %18, t12;\n\t" \
        "mul.hi.u32     t13, %2, %18;\n\t"      \
        "madc.lo.cc.u32 t13, %2, %19, t13;\n\t" \
        "madc.hi.u32    %0, %2, %19, 0;\n\t"    \
        "madc.lo.cc.u32 t10, %2, %16, t14;\n\t" \
        "addc.cc.u32    t11, t11, t15;\n\t"     \
        "addc.cc.u32    t12, t12, t16;\n\t"     \
        "addc.cc.u32    t13, t13, t17;\n\t"     \
        "addc.u32       %0, %0, 0;\n\t"         \
        "mul.hi.u32     t15, %3, %16;\n\t"      \
        "mad.lo.cc.u32  t15, %3, %17, t15;\n\t" \
        "mul.hi.u32     t16, %3, %17;\n\t"      \
        "madc.lo.cc.u32 t16, %3, %18, t16;\n\t" \
        "mul.hi.u32     t17, %3, %18;\n\t"      \
        "madc.lo.cc.u32 t17, %3, %19, t17;\n\t" \
        "madc.hi.u32    %1, %3, %19, 0;\n\t"    \
        "madc.lo.cc.u32 t11, %3, %16, t11;\n\t" \
        "addc.cc.u32    t12, t12, t15;\n\t"     \
        "addc.cc.u32    t13, t13, t16;\n\t"     \
        "addc.cc.u32    t14, %0, t17;\n\t"      \
        "addc.u32       t15, %1, 0;\n\t"        \
        "mov.b32        %0, 0;\n\t"/*r = (c + u) div 2^(4*64)*/\
        "mad.lo.cc.u32  t17, %4, %8, t8;\n\t"   \
        "addc.cc.u32    t1, t9, t1;\n\t"        \
        "addc.cc.u32    t2, t10, t2;\n\t"       \
        "addc.cc.u32    t3, t11, t3;\n\t"       \
        "addc.cc.u32    t4, t12, t4;\n\t"       \
        "addc.cc.u32    t5, t13, t5;\n\t"       \
        "addc.cc.u32    t6, t14, t6;\n\t"       \
        "addc.cc.u32    t7, t15, t7;\n\t"       \
        "addc.u32       %0, %0, 0;\n\t"         \
        "sub.cc.u32     t4, t4, %16;\n\t"       \
        "subc.cc.u32    t5, t5, %17;\n\t"       \
        "subc.cc.u32    t6, t6, %18;\n\t"       \
        "subc.cc.u32    t7, t7, %19;\n\t"       \
        "subc.u32       %0, %0, 0;\n\t"         \
        "mov.u32        %1, %0;\n\t"            \
        "mov.u32        %2, %0;\n\t"            \
        "mov.u32        %3, %0;\n\t"            \
        "and.b32        %0, %0, %16;\n\t"       \
        "and.b32        %1, %1, %17;\n\t"       \
        "and.b32        %2, %2, %18;\n\t"       \
        "and.b32        %3, %3, %19;\n\t"       \
        "add.cc.u32     %0, %0, t4;\n\t"        \
        "addc.cc.u32    %1, %1, t5;\n\t"        \
        "addc.cc.u32    %2, %2, t6;\n\t"        \
        "addc.u32       %3, %3, t7;\n\t"        \
        "}"                                     \
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)\
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),    \
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),    \
          "r"(m0), "r"(m1), "r"(m2), "r"(m3),    \
          "r"(p0), "r"(p1), "r"(p2), "r"(p3));


/* Multiplication in Fp */
#define __fp_mul4x4_64(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, m0, m1, m2, m3, p0, p1, p2, p3)  \
    asm volatile ("{\n\t"                       \
        ".reg .u64 t0;" ".reg .u64 t1;\n\t"     \
        ".reg .u64 t2;" ".reg .u64 t3;\n\t"     \
        ".reg .u64 t4;" ".reg .u64 t5;\n\t"     \
        ".reg .u64 t6;" ".reg .u64 t7;\n\t"     \
        ".reg .u64 t8;" ".reg .u64 t9;\n\t"     \
        ".reg .u64 t10;" ".reg .u64 t11;\n\t"   \
        ".reg .u64 t12;" ".reg .u64 t13;\n\t"   \
        ".reg .u64 t14;" ".reg .u64 t15;\n\t"   \
        ".reg .u64 t16;" ".reg .u64 t17;\n\t"   \
        "mul.lo.u64     t0, %4, %8;\n\t"        \
        "mul.hi.u64     t1, %4, %8;\n\t"        \
        "mad.lo.cc.u64  t1, %4, %9, t1;\n\t"    \
        "mul.hi.u64     t2, %4, %9;\n\t"        \
        "madc.lo.cc.u64 t2, %4, %10, t2;\n\t"   \
        "mul.hi.u64     t3, %4, %10;\n\t"       \
        "madc.lo.cc.u64 t3, %4, %11, t3;\n\t"   \
        "madc.hi.u64    t4, %4, %11, 0;\n\t"    \
        "mul.lo.u64     t5, %5, %8;\n\t"        \
        "mul.hi.u64     t6, %5, %8;\n\t"        \
        "mad.lo.cc.u64  t6, %5, %9, t6;\n\t"    \
        "mul.hi.u64     t7, %5, %9;\n\t"        \
        "madc.lo.cc.u64 t7, %5, %10, t7;\n\t"   \
        "mul.hi.u64     t8, %5, %10;\n\t"       \
        "madc.lo.cc.u64 t8, %5, %11, t8;\n\t"   \
        "madc.hi.u64    t9, %5, %11, 0;\n\t"    \
        "add.cc.u64     t1, t1, t5;\n\t"        \
        "addc.cc.u64    t6, t6, t2;\n\t"        \
        "addc.cc.u64    t7, t7, t3;\n\t"        \
        "addc.cc.u64    t8, t8, t4;\n\t"        \
        "addc.u64       t9, t9, 0;\n\t"         \
        "mul.lo.u64     t2, %6, %8;\n\t"        \
        "mul.hi.u64     t3, %6, %8;\n\t"        \
        "mad.lo.cc.u64  t3, %6, %9, t3;\n\t"    \
        "mul.hi.u64     t4, %6, %9;\n\t"        \
        "madc.lo.cc.u64 t4, %6, %10, t4;\n\t"   \
        "mul.hi.u64     t5, %6, %10;\n\t"       \
        "madc.lo.cc.u64 t5, %6, %11, t5;\n\t"   \
        "madc.hi.u64    t10, %6, %11, 0;\n\t"   \
        "add.cc.u64     t2, t2, t6;\n\t"        \
        "addc.cc.u64    t3, t3, t7;\n\t"        \
        "addc.cc.u64    t4, t4, t8;\n\t"        \
        "addc.cc.u64    t5, t5, t9;\n\t"        \
        "addc.u64       t10, t10, 0;\n\t"       \
        "mul.lo.u64     t6, %7, %8;\n\t"        \
        "mul.hi.u64     t7, %7, %8;\n\t"        \
        "mad.lo.cc.u64  t7, %7, %9, t7;\n\t"    \
        "mul.hi.u64     t8, %7, %9;\n\t"        \
        "madc.lo.cc.u64 t8, %7, %10, t8;\n\t"   \
        "mul.hi.u64     t9, %7, %10;\n\t"       \
        "madc.lo.cc.u64 t9, %7, %11, t9;\n\t"   \
        "madc.hi.u64    t11, %7, %11, 0;\n\t"   \
        "add.cc.u64     t3, t3, t6;\n\t"        \
        "addc.cc.u64    t4, t4, t7;\n\t"        \
        "addc.cc.u64    t5, t5, t8;\n\t"        \
        "addc.cc.u64    t6, t9, t10;\n\t"       \
        "addc.u64       t7, t11, 0;\n\t"        \
        "mul.lo.u64     %0, t0, %12;\n\t"/*Montgomery reduction, m = mu*c mod 2^(64*4)*/\
        "mul.hi.u64     %1, t0, %12;\n\t"       \
        "mad.lo.cc.u64  %1, t0, %13, %1;\n\t"   \
        "mul.hi.u64     %2, t0, %13;\n\t"       \
        "madc.lo.cc.u64 %2, t0, %14, %2;\n\t"   \
        "mul.hi.u64     %3, t0, %14;\n\t"       \
        "madc.lo.u64    %3, t0, %15, %3;\n\t"   \
        "mul.lo.u64     t8, t1, %12;\n\t"       \
        "mul.hi.u64     t9, t1, %12;\n\t"       \
        "mad.lo.cc.u64  t9, t1, %13, t9;\n\t"   \
        "mul.hi.u64     t10, t1, %13;\n\t"      \
        "madc.lo.u64    t10, t1, %14, t10;\n\t" \
        "add.cc.u64     %1, %1, t8;\n\t"        \
        "addc.cc.u64    t9, t9, %2;\n\t"        \
        "addc.u64       t10, t10, %3;\n\t"      \
        "mul.lo.u64     %2, t2, %12;\n\t"       \
        "mul.hi.u64     %3, t2, %12;\n\t"       \
        "mad.lo.u64     %3, t2, %13, %3;\n\t"   \
        "add.cc.u64     %2, %2, t9;\n\t"        \
        "addc.u64       %3, %3, t10;\n\t"       \
        "mad.lo.u64     %3, t3, %12, %3;\n\t"   \
        "mul.lo.u64     t8, %0, %16;\n\t"/* u = m*p */\
        "mul.hi.u64     t9, %0, %16;\n\t"       \
        "mad.lo.cc.u64  t9, %0, %17, t9;\n\t"   \
        "mul.hi.u64     t10, %0, %17;\n\t"      \
        "madc.lo.cc.u64 t10, %0, %18, t10;\n\t" \
        "mul.hi.u64     t11, %0, %18;\n\t"      \
        "madc.lo.cc.u64 t11, %0, %19, t11;"     \
        "madc.hi.u64    t12, %0, %19, 0;\n\t"   \
        "mul.lo.u64     t13, %1, %16;\n\t"      \
        "mul.hi.u64     t14, %1, %16;\n\t"      \
        "mad.lo.cc.u64  t14, %1, %17, t14;\n\t" \
        "mul.hi.u64     t15, %1, %17;\n\t"      \
        "madc.lo.cc.u64 t15, %1, %18, t15;\n\t" \
        "mul.hi.u64     t16, %1, %18;\n\t"      \
        "madc.lo.cc.u64 t16, %1, %19, t16;\n\t" \
        "madc.hi.u64    t17, %1, %19, 0;\n\t"   \
        "add.cc.u64     t9, t9, t13;\n\t"       \
        "addc.cc.u64    t14, t14, t10;\n\t"     \
        "addc.cc.u64    t15, t15, t11;\n\t"     \
        "addc.cc.u64    t16, t16, t12;\n\t"     \
        "addc.u64       t17, t17, 0;\n\t"       \
        "mul.lo.u64     t10, %2, %16;\n\t"      \
        "mul.hi.u64     t11, %2, %16;\n\t"      \
        "mad.lo.cc.u64  t11, %2, %17, t11;\n\t" \
        "mul.hi.u64     t12, %2, %17;\n\t"      \
        "madc.lo.cc.u64 t12, %2, %18, t12;\n\t" \
        "mul.hi.u64     t13, %2, %18;\n\t"      \
        "madc.lo.cc.u64 t13, %2, %19, t13;\n\t" \
        "madc.hi.u64    %0, %2, %19, 0;\n\t"    \
        "add.cc.u64     t10, t10, t14;\n\t"     \
        "addc.cc.u64    t11, t11, t15;\n\t"     \
        "addc.cc.u64    t12, t12, t16;\n\t"     \
        "addc.cc.u64    t13, t13, t17;\n\t"     \
        "addc.u64       %0, %0, 0;\n\t"         \
        "mul.lo.u64     t14, %3, %16;\n\t"      \
        "mul.hi.u64     t15, %3, %16;\n\t"      \
        "mad.lo.cc.u64  t15, %3, %17, t15;\n\t" \
        "mul.hi.u64     t16, %3, %17;\n\t"      \
        "madc.lo.cc.u64 t16, %3, %18, t16;\n\t" \
        "mul.hi.u64     t17, %3, %18;\n\t"      \
        "madc.lo.cc.u64 t17, %3, %19, t17;\n\t" \
        "madc.hi.u64    %1, %3, %19, 0;\n\t"    \
        "add.cc.u64     t11, t11, t14;\n\t"     \
        "addc.cc.u64    t12, t12, t15;\n\t"     \
        "addc.cc.u64    t13, t13, t16;\n\t"     \
        "addc.cc.u64    t14, %0, t17;\n\t"      \
        "addc.u64       t15, %1, 0;\n\t"        \
        "mov.b64        %0, 0;\n\t"/*r = (c + u) div 2^(4*64)*/\
        "add.cc.u64     t0, t8, t0;\n\t"        \
        "addc.cc.u64    t1, t9, t1;\n\t"        \
        "addc.cc.u64    t2, t10, t2;\n\t"       \
        "addc.cc.u64    t3, t11, t3;\n\t"       \
        "addc.cc.u64    t4, t12, t4;\n\t"       \
        "addc.cc.u64    t5, t13, t5;\n\t"       \
        "addc.cc.u64    t6, t14, t6;\n\t"       \
        "addc.cc.u64    t7, t15, t7;\n\t"       \
        "addc.u64       %0, %0, 0;\n\t"         \
        "sub.cc.u64     t4, t4, %16;\n\t"       \
        "subc.cc.u64    t5, t5, %17;\n\t"       \
        "subc.cc.u64    t6, t6, %18;\n\t"       \
        "subc.cc.u64    t7, t7, %19;\n\t"       \
        "subc.u64       %0, %0, 0;\n\t"         \
        "mov.u64        %1, %0;\n\t"            \
        "mov.u64        %2, %0;\n\t"            \
        "mov.u64        %3, %0;\n\t"            \
        "and.b64        %0, %0, %16;\n\t"       \
        "and.b64        %1, %1, %17;\n\t"       \
        "and.b64        %2, %2, %18;\n\t"       \
        "and.b64        %3, %3, %19;\n\t"       \
        "add.cc.u64     %0, %0, t4;\n\t"        \
        "addc.cc.u64    %1, %1, t5;\n\t"        \
        "addc.cc.u64    %2, %2, t6;\n\t"        \
        "addc.u64       %3, %3, t7;\n\t"        \
        "}"                                     \
        : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)\
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3),\
          "l"(m0), "l"(m1), "l"(m2), "l"(m3), "l"(p0), "l"(p1), "l"(p2), "l"(p3) )



// 4x4 32bit schoolbook squaring
// c7c6c5c4c3c2c1c0 = a3a2a1a0 * b3b2b1b0
#define __school_sqr32_4x4(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3) \
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t1,t2,t3,t4,tc,tp;\n\t"              \
            /* walking down the addition chain*/            \
            /* a0 * a3 */                                   \
            "mul.lo.u32 	t3, %8, %11;\n\t"               \
            "mul.hi.u32 	t4, %8, %11;\n\t"               \
            "add.cc.u32 	t3, t3, t3;\n\t"                \
            "addc.cc.u32 	t4, t4, t4;\n\t"                \
            "addc.u32 		tc, 0, 0;\n\t"                  \
            "add.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            /* a0 * a2 */                                   \
            "mul.lo.u32		t2, %8, %10;\n\t"               \
            "mul.hi.u32		%3, %8, %10;\n\t"               \
            "add.cc.u32		t2, t2, t2;\n\t"                \
            "addc.cc.u32 	%3, %3, %3;\n\t"                \
            "addc.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            "add.cc.u32 	t3, t3, %3;\n\t"                \
            "addc.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            /* a0 * a1 */                                   \
            "mul.lo.u32		t1, %8, %9;\n\t"                \
            "mul.hi.u32		%2, %8, %9;\n\t"                \
            "add.cc.u32		t1, t1, t1;\n\t"                \
            "addc.cc.u32	%2, %2, %2;\n\t"                \
            "addc.cc.u32 	t3, t3, 0;\n\t"                 \
            "addc.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            "add.cc.u32 	t2, t2, %2;\n\t"                \
            "addc.cc.u32 	t3, t3, 0;\n\t"                 \
            "addc.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            /* a0 * a0 */                                   \
            "mul.lo.u32		%0, %8, %8;\n\t"                \
            "mad.hi.cc.u32	%1, %8, %8, t1;\n\t"            \
            "addc.cc.u32 	t2, t2, 0;\n\t"                 \
            "addc.cc.u32 	t3, t3, 0;\n\t"                 \
            "addc.cc.u32 	t4, t4, 0;\n\t"                 \
            "addc.u32 		tc, tc, 0;\n\t"                 \
            /* a1 * a3 */                                   \
            "mul.lo.u32		%4, %9, %11;\n\t"               \
            "mul.hi.u32		%5, %9, %11;\n\t"               \
            "add.cc.u32		%4, %4, %4;\n\t"                \
            "addc.cc.u32 	%5, %5, %5;\n\t"                \
            "addc.u32 		%6,  0,  0;\n\t"                \
            "add.cc.u32		%5, %5,	tc;\n\t"                \
            "addc.u32		%6, %6,  0;\n\t"                \
            /* 2 * a1 * a2 unused: %3, %2, tc, tp*/         \
            "mul.lo.u32		%3, %9, %10;\n\t"               \
            "mul.hi.u32		tp, %9, %10;\n\t"               \
            "add.cc.u32		%3, %3, %3;\n\t"                \
            "addc.cc.u32 	tp, tp, tp;\n\t"                \
            "addc.cc.u32 	%5, %5,  0;\n\t"                \
            "addc.u32 		%6, %6,  0;\n\t"                \
            /* a1 * a1 */                                   \
            "mad.lo.cc.u32	%2, %9, %9, t2;\n\t"            \
            "madc.hi.cc.u32	%3, %9, %9, %3;\n\t"            \
            "addc.cc.u32	%4, %4,  0;\n\t"                \
            "addc.cc.u32 	%5, %5,  0;\n\t"                \
            "addc.u32 		%6, %6,  0;\n\t"                \
            /* fixxinf some missing adds */                 \
            "add.cc.u32		%3, %3, t3;\n\t"                \
            "addc.cc.u32	%4, %4, t4;\n\t"                \
            "addc.cc.u32 	%5, %5,  0;\n\t"                \
            "addc.u32 		%6, %6,  0;\n\t"                \
            /* 2 * a2 * a3 */                               \
            "mul.lo.u32		t1, %10, %11;\n\t"              \
            "mul.hi.u32		t2, %10, %11;\n\t"              \
            "add.cc.u32		t1, t1, t1;\n\t"                \
            "addc.cc.u32	t2, t2, t2;\n\t"                \
            "addc.u32		t3,  0,  0;\n\t"                \
            /* a2 * a2 */                                   \
            "mad.lo.cc.u32	%4, %10, %10, %4;\n\t"          \
            "addc.cc.u32 	%5, %5, 0;\n\t"                 \
            "addc.u32		%6, %6, 0;\n\t"                 \
            "mad.hi.cc.u32	%5, %10, %10, %5;\n\t"          \
            "addc.cc.u32	%6, %6, 0;\n\t"                 \
            "addc.u32		%7, 0, 0;\n\t"                  \
            /* a3 * a3 */                                   \
            "mad.lo.cc.u32	%6, %11, %11, %6;\n\t"          \
            "madc.hi.cc.u32	%7, %11, %11, %7;\n\t"          \
            "}\n"                                           \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), /*0,  3*/     \
      "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)  /*4,  7*/     \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3));    /*8 ,11*/

// TODO
#define __fp_sqr4x4_32(c0, c1, c2, c3, a0, a1, a2, a3, mu0, mu1, mu2, mu3, p0, p1, p2, p3)    \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7, t0,t1,t2,t3,t4,t5,t6,t7;                    \
    __school_sqr32_4x4(f0,f1,f2,f3,f4,f5,f6,f7, a0,a1,a2,a3)                    \
    __school32_4x4_low(c0,c1,c2,c3, a0,a1,a2,a3, mu0,mu1,mu2,mu3)            \
    __school32_4x4(t0,t1,t2,t3,t4,t5,t6,t7, c0,c1,c2,c3, p0,p1,p2,p3)    \
    __reduce32_sec4x4(c0,c1,c2,c3, f0,f1,f2,f3,f4,f5,f6,f7, p0,p1,p2,p3, t0,t1,t2,t3,t4,t5,t6,t7)


// ja kein plan
#define __school64_1x1_v2(c0, c1, a0, b0)                           \
    asm ("{\n\t"                                                    \
         ".reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\n\t"        \
         "mov.b64         {alo,ahi}, %2;\n\t"                       \
         "mov.b64         {blo,bhi}, %3;\n\t"                       \
         "mul.lo.u32      r0, alo, blo;\n\t"                        \
         "mul.hi.u32      r1, alo, blo; \n\t"                       \
         "mad.lo.cc.u32   r1, alo, bhi, r1;\n\t"                    \
         "madc.hi.u32     r2, alo, bhi, 0;\n\t"                     \
         "mad.lo.cc.u32   r1, ahi, blo, r1;\n\t"                    \
         "madc.hi.cc.u32  r2, ahi, blo, r2;\n\t"                    \
         "madc.hi.u32     r3, ahi, bhi, 0;\n\t"                     \
         "mad.lo.cc.u32   r2, ahi, bhi, r2;\n\t"                    \
         "addc.u32        r3, r3, 0;\n\t"                           \
         "mov.b64         %0, {r0,r1};\n\t"                         \
         "mov.b64         %1, {r2,r3};\n\t"                         \
         "}"                                                        \
         : "=l"(c0), "=l"(c1)                                       \
         : "l"(a0), "l"(b0));


#define __school64_1x1_v3(c0, c1, a0, b0)                           \
    asm ("{\n\t"                                                    \
         ".reg .u32       alo, ahi, blo, bhi, r0, r1, r2, r3;\n\t"  \
         ".reg .u32       s0, s1, s2, s3, t0, t1, t2, t3;\n\t"      \
         ".reg .u16       a0, a1, a2, a3, b0, b1, b2, b3;\n\t"      \
         /* split inputs into 16-bit chunks */                      \
         "mov.b64         {alo,ahi}, %2;\n\t"                       \
         "mov.b64         {blo,bhi}, %3;\n\t"                       \
         "mov.b32         {a0,a1}, alo;\n\t"                        \
         "mov.b32         {a2,a3}, ahi;\n\t"                        \
         "mov.b32         {b0,b1}, blo;\n\t"                        \
         "mov.b32         {b2,b3}, bhi;\n\t"                        \
         /* first partial sum:                          */          \
         /* a3b3.wide  a1b3.wide  a0b2.wide  a0b0.wide  */          \
         /*     0      a2b2.wide  a1b1.wide             */          \
         /*     0      a3b1.wide  a2b0.wide             */          \
         "mul.wide.u16    r0, a0, b0;\n\t"                          \
         "mul.wide.u16    r1, a0, b2;\n\t"                          \
         "mul.wide.u16    r2, a1, b3;\n\t"                          \
         "mul.wide.u16    r3, a3, b3;\n\t"                          \
         "mul.wide.u16    t1, a1, b1;\n\t"                          \
         "mul.wide.u16    t2, a2, b2;\n\t"                          \
         "add.cc.u32      r1, r1, t1;\n\t"                          \
         "addc.cc.u32     r2, r2, t2;\n\t"                          \
         "addc.u32        r3, r3, 0;\n\t"                           \
         "mul.wide.u16    t1, a2, b0;\n\t"                          \
         "mul.wide.u16    t2, a3, b1;\n\t"                          \
         "add.cc.u32      r1, r1, t1;\n\t"                          \
         "addc.cc.u32     r2, r2, t2;\n\t"                          \
         "addc.u32        r3, r3, 0;\n\t"                           \
         /* second partial sum:             */                      \
         /* a2b3.wide  a0b3.wide  a0b1.wide */                      \
         /* a3b2.wide  a1b2.wide  a1b0.wide */                      \
         /*     0      a2b1.wide            */                      \
         /*     0      a3b0.wide            */                      \
         "mul.wide.u16    s0, a0, b1;\n\t"                          \
         "mul.wide.u16    s1, a0, b3;\n\t"                          \
         "mul.wide.u16    s2, a2, b3;\n\t"                          \
         "mul.wide.u16    t1, a2, b1;\n\t"                          \
         "add.cc.u32      s1, s1, t1;\n\t"                          \
         "addc.u32        s2, s2, 0;\n\t"                           \
         "mul.wide.u16    t1, a3, b0;\n\t"                          \
         "add.cc.u32      s1, s1, t1;\n\t"                          \
         "addc.u32        s2, s2, 0;\n\t"                           \
         "mul.wide.u16    t0, a1, b0;\n\t"                          \
         "mul.wide.u16    t1, a1, b2;\n\t"                          \
         "mul.wide.u16    t2, a3, b2;\n\t"                          \
         "add.cc.u32      s0, s0, t0;\n\t"                          \
         "addc.cc.u32     s1, s1, t1;\n\t"                          \
         "addc.cc.u32     s2, s2, t2;\n\t"                          \
         "addc.u32        s3, 0, 0;\n\t"                            \
         /* offset second partial sum by 16 bits to the left*/      \
         "shf.l.clamp.b32 t3, s2, s3, 16;\n\t"                      \
         "shf.l.clamp.b32 t2, s1, s2, 16;\n\t"                      \
         "shf.l.clamp.b32 t1, s0, s1, 16;\n\t"                      \
         "shf.l.clamp.b32 t0,  0, s0, 16;\n\t"                      \
         /* add first sum in r{0,1,2,3} to second sum in t{0,1,2,3} */\
         "add.cc.u32      r0, r0, t0;\n\t"                          \
         "addc.cc.u32     r1, r1, t1;\n\t"                          \
         "addc.cc.u32     r2, r2, t2;\n\t"                          \
         "addc.u32        r3, r3, t3;\n\t"                          \
         /* pack outputs */                                         \
         "mov.b64         %0, {r0,r1};\n\t"                         \
         "mov.b64         %1, {r2,r3};\n\t"                         \
         "}"                                                        \
         : "=l"(c0), "=l"(c1)                                       \
         : "l"(a0), "l"(b0));


// from the internen
#define __school32_4x4_v3(c0, c1, c2, c3, c4, c5, c6, c7, aa0, aa1, aa2, aa3, bb0, bb1, bb2, bb3)    \
    asm ("{\n\t"                                                            \
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"                    \
         ".reg .u32 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"                    \
         "mov.b32         a0,  %8;\n\t"                                     \
         "mov.b32         a1,  %9;\n\t"                                     \
         "mov.b32         a2, %10;\n\t"                                     \
         "mov.b32         a3, %11;\n\t"                                     \
         "mov.b32         b0, %12;\n\t"                                     \
         "mov.b32         b1, %13;\n\t"                                     \
         "mov.b32         b2, %14;\n\t"                                     \
         "mov.b32         b3, %15;\n\t"                                     \
         "mul.lo.u32      r0, a0, b0;\n\t"                                  \
         "mul.hi.u32      r1, a0, b0;\n\t"                                  \
         "mad.lo.cc.u32   r1, a0, b1, r1;\n\t"                              \
         "madc.hi.u32     r2, a0, b1, 0;\n\t"                               \
         "mad.lo.cc.u32   r1, a1, b0, r1;\n\t"                              \
         "madc.hi.cc.u32  r2, a1, b0, r2;\n\t"                              \
         "madc.hi.u32     r3, a0, b2, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a0, b2, r2;\n\t"                              \
         "madc.hi.cc.u32  r3, a1, b1, r3;\n\t"                              \
         "madc.hi.u32     r4, a0, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a1, b1, r2;\n\t"                              \
         "madc.hi.cc.u32  r3, a2, b0, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a1, b2, r4;\n\t"                              \
         "madc.hi.u32     r5, a1, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a2, b0, r2;\n\t"                              \
         "madc.lo.cc.u32  r3, a0, b3, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a2, b1, r4;\n\t"                              \
         "madc.hi.cc.u32  r5, a2, b2, r5;\n\t"                              \
         "madc.hi.u32     r6, a2, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r3, a1, b2, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a3, b0, r4;\n\t"                              \
         "madc.hi.cc.u32  r5, a3, b1, r5;\n\t"                              \
         "madc.hi.cc.u32  r6, a3, b2, r6;\n\t"                              \
         "madc.hi.u32     r7, a3, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r3, a2, b1, r3;\n\t"                              \
         "madc.lo.cc.u32  r4, a1, b3, r4;\n\t"                              \
         "madc.lo.cc.u32  r5, a2, b3, r5;\n\t"                              \
         "madc.lo.cc.u32  r6, a3, b3, r6;\n\t"                              \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mad.lo.cc.u32   r3, a3, b0, r3;\n\t"                              \
         "madc.lo.cc.u32  r4, a2, b2, r4;\n\t"                              \
         "madc.lo.cc.u32  r5, a3, b2, r5;\n\t"                              \
         "addc.cc.u32     r6, r6, 0;\n\t"                                   \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mad.lo.cc.u32   r4, a3, b1, r4;\n\t"                              \
         "addc.cc.u32     r5, r5, 0;\n\t"                                   \
         "addc.cc.u32     r6, r6, 0;\n\t"                                   \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mov.b32         %0, r0;\n\t"                                      \
         "mov.b32         %1, r1;\n\t"                                      \
         "mov.b32         %2, r2;\n\t"                                      \
         "mov.b32         %3, r3;\n\t"                                      \
         "mov.b32         %4, r4;\n\t"                                      \
         "mov.b32         %5, r5;\n\t"                                      \
         "mov.b32         %6, r6;\n\t"                                      \
         "mov.b32         %7, r7;\n\t"                                      \
         "}"                                                                \
         : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3),                          \
           "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)                           \
         : "r"(aa0), "r"(aa1), "r"(aa2), "r"(aa3),                          \
           "r"(bb0), "r"(bb1), "r"(bb2), "r"(bb3));

// from the internet
#define __school32_4x4_v4(c0, c1, c2, c3, c4, c5, c6, c7, d0, d1, d2, d3, e0, e1, e2, e3)    \
    asm ("{\n\t"                                                            \
         ".reg .u32 aa0, aa1, aa2, aa3, bb0, bb1, bb2, bb3;\n\t"            \
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"    \
         ".reg .u32 s0, s1, s2, s3, s4, s5, s6, s7;\n\t"    \
         ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7;\n\t"    \
         ".reg .u16 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"    \
         ".reg .u16 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"    \
         /* unpack source operands */                       \
         "mov.b32         {a0,a1},  %8;\n\t"                \
         "mov.b32         {a2,a3},  %9;\n\t"                \
         "mov.b32         {a4,a5}, %10;\n\t"                \
         "mov.b32         {a6,a7}, %11;\n\t"                \
         "mov.b32         {b0,b1}, %12;\n\t"                \
         "mov.b32         {b2,b3}, %13;\n\t"                \
         "mov.b32         {b4,b5}, %14;\n\t"                \
         "mov.b32         {b6,b7}, %15;\n\t"                \
         /* compute first partial sum */                    \
         "mul.wide.u16    r0, a0, b0;\n\t"                  \
         "mul.wide.u16    r1, a0, b2;\n\t"                  \
         "mul.wide.u16    r2, a0, b4;\n\t"                  \
         "mul.wide.u16    r3, a0, b6;\n\t"                  \
         "mul.wide.u16    r4, a1, b7;\n\t"                  \
         "mul.wide.u16    r5, a3, b7;\n\t"                  \
         "mul.wide.u16    r6, a5, b7;\n\t"                  \
         "mul.wide.u16    r7, a7, b7;\n\t"                  \
         "mul.wide.u16    t3, a1, b5;\n\t"                  \
         "mul.wide.u16    t4, a2, b6;\n\t"                  \
         "add.cc.u32      r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.u32        r5, r5, 0;\n\t"                   \
         "mul.wide.u16    t3, a2, b4;\n\t"                  \
         "mul.wide.u16    t4, a3, b5;\n\t"                  \
         "add.cc.u32      r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.u32        r5, r5, 0;\n\t"                   \
         "mul.wide.u16    t2, a1, b3;\n\t"                  \
         "mul.wide.u16    t3, a3, b3;\n\t"                  \
         "mul.wide.u16    t4, a4, b4;\n\t"                  \
         "mul.wide.u16    t5, a4, b6;\n\t"                  \
         "add.cc.u32      r2, r2, t2;\n\t"                  \
         "addc.cc.u32     r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.cc.u32     r5, r5, t5;\n\t"                  \
         "addc.u32        r6, r6, 0;\n\t"                   \
         "mul.wide.u16    t2, a2, b2;\n\t"                  \
         "mul.wide.u16    t3, a4, b2;\n\t"                  \
         "mul.wide.u16    t4, a5, b3;\n\t"                  \
         "mul.wide.u16    t5, a5, b5;\n\t"                  \
         "add.cc.u32      r2, r2, t2;\n\t"                  \
         "addc.cc.u32     r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.cc.u32     r5, r5, t5;\n\t"                  \
         "addc.u32        r6, r6, 0;\n\t"                   \
         "mul.wide.u16    t1, a1, b1;\n\t"                  \
         "mul.wide.u16    t2, a3, b1;\n\t"                  \
         "mul.wide.u16    t3, a5, b1;\n\t"                  \
         "mul.wide.u16    t4, a6, b2;\n\t"                  \
         "mul.wide.u16    t5, a6, b4;\n\t"                  \
         "mul.wide.u16    t6, a6, b6;\n\t"                  \
         "add.cc.u32      r1, r1, t1;\n\t"                  \
         "addc.cc.u32     r2, r2, t2;\n\t"                  \
         "addc.cc.u32     r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.cc.u32     r5, r5, t5;\n\t"                  \
         "addc.cc.u32     r6, r6, t6;\n\t"                  \
         "addc.u32        r7, r7, 0;\n\t"                   \
         "mul.wide.u16    t1, a2, b0;\n\t"                  \
         "mul.wide.u16    t2, a4, b0;\n\t"                  \
         "mul.wide.u16    t3, a6, b0;\n\t"                  \
         "mul.wide.u16    t4, a7, b1;\n\t"                  \
         "mul.wide.u16    t5, a7, b3;\n\t"                  \
         "mul.wide.u16    t6, a7, b5;\n\t"                  \
         "add.cc.u32      r1, r1, t1;\n\t"                  \
         "addc.cc.u32     r2, r2, t2;\n\t"                  \
         "addc.cc.u32     r3, r3, t3;\n\t"                  \
         "addc.cc.u32     r4, r4, t4;\n\t"                  \
         "addc.cc.u32     r5, r5, t5;\n\t"                  \
         "addc.cc.u32     r6, r6, t6;\n\t"                  \
         "addc.u32        r7, r7, 0;\n\t"                   \
         /* compute second partial sum */                   \
         "mul.wide.u16    t0, a0, b1;\n\t"                  \
         "mul.wide.u16    t1, a0, b3;\n\t"                  \
         "mul.wide.u16    t2, a0, b5;\n\t"                  \
         "mul.wide.u16    t3, a0, b7;\n\t"                  \
         "mul.wide.u16    t4, a2, b7;\n\t"                  \
         "mul.wide.u16    t5, a4, b7;\n\t"                  \
         "mul.wide.u16    t6, a6, b7;\n\t"                  \
         "mul.wide.u16    s3, a1, b6;\n\t"                  \
         "add.cc.u32      t3, t3, s3;\n\t"                  \
         "addc.u32        t4, t4, 0;\n\t"                   \
         "mul.wide.u16    s3, a2, b5;\n\t"                  \
         "add.cc.u32      t3, t3, s3;\n\t"                  \
         "addc.u32        t4, t4, 0;\n\t"                   \
         "mul.wide.u16    s2, a1, b4;\n\t"                  \
         "mul.wide.u16    s3, a3, b4;\n\t"                  \
         "mul.wide.u16    s4, a3, b6;\n\t"                  \
         "add.cc.u32      t2, t2, s2;\n\t"                  \
         "addc.cc.u32     t3, t3, s3;\n\t"                  \
         "addc.cc.u32     t4, t4, s4;\n\t"                  \
         "addc.u32        t5, t5, 0;\n\t"                   \
         "mul.wide.u16    s2, a2, b3;\n\t"                  \
         "mul.wide.u16    s3, a4, b3;\n\t"                  \
         "mul.wide.u16    s4, a4, b5;\n\t"                  \
         "add.cc.u32      t2, t2, s2;\n\t"                  \
         "addc.cc.u32     t3, t3, s3;\n\t"                  \
         "addc.cc.u32     t4, t4, s4;\n\t"                  \
         "addc.u32        t5, t5, 0;\n\t"                   \
         "mul.wide.u16    s1, a1, b2;\n\t"                  \
         "mul.wide.u16    s2, a3, b2;\n\t"                  \
         "mul.wide.u16    s3, a5, b2;\n\t"                  \
         "mul.wide.u16    s4, a5, b4;\n\t"                  \
         "mul.wide.u16    s5, a5, b6;\n\t"                  \
         "add.cc.u32      t1, t1, s1;\n\t"                  \
         "addc.cc.u32     t2, t2, s2;\n\t"                  \
         "addc.cc.u32     t3, t3, s3;\n\t"                  \
         "addc.cc.u32     t4, t4, s4;\n\t"                  \
         "addc.cc.u32     t5, t5, s5;\n\t"                  \
         "addc.u32        t6, t6, 0;\n\t"                   \
         "mul.wide.u16    s1, a2, b1;\n\t"                  \
         "mul.wide.u16    s2, a4, b1;\n\t"                  \
         "mul.wide.u16    s3, a6, b1;\n\t"                  \
         "mul.wide.u16    s4, a6, b3;\n\t"                  \
         "mul.wide.u16    s5, a6, b5;\n\t"                  \
         "add.cc.u32      t1, t1, s1;\n\t"                  \
         "addc.cc.u32     t2, t2, s2;\n\t"                  \
         "addc.cc.u32     t3, t3, s3;\n\t"                  \
         "addc.cc.u32     t4, t4, s4;\n\t"                  \
         "addc.cc.u32     t5, t5, s5;\n\t"                  \
         "addc.u32        t6, t6, 0;\n\t"                   \
         "mul.wide.u16    s0, a1, b0;\n\t"                  \
         "mul.wide.u16    s1, a3, b0;\n\t"                  \
         "mul.wide.u16    s2, a5, b0;\n\t"                  \
         "mul.wide.u16    s3, a7, b0;\n\t"                  \
         "mul.wide.u16    s4, a7, b2;\n\t"                  \
         "mul.wide.u16    s5, a7, b4;\n\t"                  \
         "mul.wide.u16    s6, a7, b6;\n\t"                  \
         "add.cc.u32      t0, t0, s0;\n\t"                  \
         "addc.cc.u32     t1, t1, s1;\n\t"                  \
         "addc.cc.u32     t2, t2, s2;\n\t"                  \
         "addc.cc.u32     t3, t3, s3;\n\t"                  \
         "addc.cc.u32     t4, t4, s4;\n\t"                  \
         "addc.cc.u32     t5, t5, s5;\n\t"                  \
         "addc.cc.u32     t6, t6, s6;\n\t"                  \
         "addc.u32        t7, 0, 0;\n\t"                    \
         /* offset second partial sum by 16 bits */         \
         "shf.l.clamp.b32 s7, t6, t7, 16;\n\t"              \
         "shf.l.clamp.b32 s6, t5, t6, 16;\n\t"              \
         "shf.l.clamp.b32 s5, t4, t5, 16;\n\t"              \
         "shf.l.clamp.b32 s4, t3, t4, 16;\n\t"              \
         "shf.l.clamp.b32 s3, t2, t3, 16;\n\t"              \
         "shf.l.clamp.b32 s2, t1, t2, 16;\n\t"              \
         "shf.l.clamp.b32 s1, t0, t1, 16;\n\t"              \
         "shf.l.clamp.b32 s0,  0, t0, 16;\n\t"              \
         /* add partial sums */                             \
         "add.cc.u32      r0, r0, s0;\n\t"                  \
         "addc.cc.u32     r1, r1, s1;\n\t"                  \
         "addc.cc.u32     r2, r2, s2;\n\t"                  \
         "addc.cc.u32     r3, r3, s3;\n\t"                  \
         "addc.cc.u32     r4, r4, s4;\n\t"                  \
         "addc.cc.u32     r5, r5, s5;\n\t"                  \
         "addc.cc.u32     r6, r6, s6;\n\t"                  \
         "addc.u32        r7, r7, s7;\n\t"                  \
         /* pack up result */                               \
         "mov.b32         %0, r0;\n\t"                      \
         "mov.b32         %1, r1;\n\t"                      \
         "mov.b32         %2, r2;\n\t"                      \
         "mov.b32         %3, r3;\n\t"                      \
         "mov.b32         %4, r4;\n\t"                      \
         "mov.b32         %5, r5;\n\t"                      \
         "mov.b32         %6, r6;\n\t"                      \
         "mov.b32         %7, r7;\n\t"                      \
         "}"                                                \
         : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3),          \
           "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)           \
         : "r"(d0), "r"(d1), "r"(d2), "r"(d3),              \
           "r"(e0), "r"(e1), "r"(e2), "r"(e3));


#define __mul128_32(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, b0, b1, b2, b3)    \
    __school32_4x4_v3(c0,c1,c2,c3,c4,c5,c6,c7,a0,a1,a2,a3,b0,b1,b2,b3)

////////////////////////////////////////////////////////////////////////////////
// 									5x5										  //
////////////////////////////////////////////////////////////////////////////////
// 5x5 32bit limbs school book mutliplication
// uses the expensive `mad` instruction
// optimized for less register usage
#define __school32_5x5(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4)\
    asm volatile ("{\n\t"                                \
            ".reg .u32 t1,t2,t3,t4,tc;\n\t"                \
            /*Multiplication involving a0*/                    \
            "mul.hi.u32     %0, %10, %15;\n\t"            \
            "mad.lo.cc.u32  %0, %10, %16, %0;\n\t"        \
            "mul.hi.u32     t1, %10, %16;\n\t"            \
            "madc.lo.cc.u32 t1, %10, %17, t1;\n\t"        \
            "mul.hi.u32     t2, %10, %17;\n\t"            \
            "madc.lo.cc.u32 t2, %10, %18, t2;\n\t"        \
            "mul.hi.u32     t3, %10, %18;\n\t"            \
            "madc.lo.cc.u32 t3, %10, %19, t3;\n\t"        \
            "madc.hi.cc.u32 t4, %10, %19, 0;\n\t"            \
            "addc.u32       tc, 0, 0;\n\t"                \
            /*Multiplication involving a1 first diagonal*/    \
            "mad.lo.cc.u32  %1, %11, %15, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %11, %15, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %11, %16, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %11, %17, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %11, %18, t4;\n\t"            \
            "madc.hi.u32 	%0, %11, %19, tc;\n\t"            \
            /*Multiplication involving a1 second diagonal*/    \
            "mad.lo.cc.u32  t1, %11, %16, t1;\n\t"        \
            "madc.lo.cc.u32 t2, %11, %17, t2;\n\t"        \
            "madc.lo.cc.u32 t3, %11, %18, t3;\n\t"        \
            "madc.lo.cc.u32 t4, %11, %19, t4;\n\t"        \
            "addc.cc.u32    %0, %0 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a2 first diagonal*/    \
            "mad.lo.cc.u32  %2, %12, %15, t1;\n\t"        \
            "madc.hi.cc.u32 t2, %12, %15, t2;\n\t"        \
            "madc.hi.cc.u32 t3, %12, %16, t3;\n\t"        \
            "madc.hi.cc.u32 t4, %12, %17, t4;\n\t"        \
            "madc.hi.cc.u32 %0, %12, %18, %0;\n\t"        \
            "madc.hi.cc.u32 t1, %12, %19, tc;\n\t"            \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a2 second diagonal*/    \
            "mad.lo.cc.u32  t2, %12, %16, t2;\n\t"            \
            "madc.lo.cc.u32 t3, %12, %17, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %12, %18, t4;\n\t"            \
            "madc.lo.cc.u32 %0, %12, %19, %0;\n\t"            \
            "addc.cc.u32    t1, t1 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a3 first diagonal*/    \
            "mad.lo.cc.u32  %3, %13, %15, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %13, %15, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %13, %16, t4;\n\t"            \
            "madc.hi.cc.u32 %0, %13, %17, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %13, %18, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %13, %19, tc;\n\t"            \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a3 second diagonal*/    \
            "mad.lo.cc.u32  t3, %13, %16, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %13, %17, t4;\n\t"            \
            "madc.lo.cc.u32 %0, %13, %18, %0;\n\t"            \
            "madc.lo.cc.u32 t1, %13, %19, t1;\n\t"            \
            "addc.cc.u32    t2, t2 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a4 first diagonal*/    \
            "mad.lo.cc.u32  %4, %14, %15, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %14, %15, t4;\n\t"            \
            "madc.hi.cc.u32 %0, %14, %16, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %14, %17, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %14, %18, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %14, %19, tc;\n\t"            \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a4 second diagonal*/    \
            "mad.lo.cc.u32  %5, %14, %16, t4;\n\t"            \
            "madc.lo.cc.u32 %6, %14, %17, %0;\n\t"            \
            "madc.lo.cc.u32 %7, %14, %18, t1;\n\t"            \
            "madc.lo.cc.u32 %8, %14, %19, t2;\n\t"            \
            "addc.u32       %9, t3 , tc;\n\t"                \
            "mul.lo.u32     %0, %10, %15;\n\t"            \
            "}\n"                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), /*0,  4*/ \
      "=r"(c5), "=r"(c6), "=r"(c7), "=r"(c8), "=r"(c9)  /*5,  9*/ \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4),      /*10,14*/ \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4));     /*15,19*/

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**160
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// inplace version
#define __school32_inplace_5x5_low(a0, a1, a2, a3, a4, mu0, mu1, mu2, mu3, mu4)\
    asm volatile ("{\n\t"                       \
            ".reg .u32 t10,t20,t30,t40;\n\t"\
            /* t0-t5: m = a * mu mod 2^160 */   \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %10, %5;\n\t"    \
            "mul.hi.u32     %1,  %10, %5;\n\t"  \
            "mul.lo.u32     t10, %10, %6;\n\t"  \
            "mul.hi.u32     %2,  %10, %6;\n\t"  \
            "mul.lo.u32     t20, %10, %7;\n\t"  \
            "mul.hi.u32     %3,  %10, %7;\n\t"  \
            "mul.lo.u32     t30, %10, %8;\n\t"  \
            "mul.hi.u32     %4,  %10, %8;\n\t"  \
            "mul.lo.u32     t40, %10, %9;\n\t"  \
            /* now fixing the carry chains */   \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.u32       %4,  %4,  t40;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/          \
            "mul.lo.u32     t10, %11, %5;\n\t"  \
            "mul.lo.u32     t20, %11, %6;\n\t"  \
            "mul.lo.u32     t30, %11, %7;\n\t"  \
            "mul.lo.u32     t40, %11, %8;\n\t"  \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32      %4, %4, t40;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/        \
            "mul.hi.u32     t20, %11, %5;\n\t"  \
            "mul.hi.u32     t30, %11, %6;\n\t"  \
            "mul.hi.u32     t40, %11, %7;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/          \
            "mul.lo.u32     t20, %12, %5;\n\t"  \
            "mul.lo.u32     t30, %12, %6;\n\t"  \
            "mul.lo.u32     t40, %12, %7;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/        \
            "mul.hi.u32     t30, %12, %5;\n\t"  \
            "mul.hi.u32     t40, %12, %6;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/          \
            "mul.lo.u32     t30, %13, %5;\n\t"  \
            "mul.lo.u32     t40, %13, %6;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.u32    	%4, %4, t40;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/        \
            "mul.hi.u32     t40, %13, %5;\n\t"  \
            "add.u32     	%4, %4, t40;\n\t"   \
            /*Multiplication involving a4 second (low) diagonal*/        \
            "mul.lo.u32     t40, %14, %5;\n\t"  \
            "add.u32         %4, %4, t40;\n\t"  \
            "}\n"                                                        \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4)  /*0 , 4*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4), /*5 , 9*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4)    /*10,14*/    \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**160
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// Non inplace version
#define __school32_5x5_low(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, mu0, mu1, mu2, mu3, mu4)\
    asm volatile ("{\n\t"                       \
            ".reg .u32 t10,t20,t30,t40;\n\t"\
            /* t0-t5: m = a * mu mod 2^160 */   \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %10, %5;\n\t"    \
            "mul.hi.u32     %1,  %10, %5;\n\t"  \
            "mul.lo.u32     t10, %10, %6;\n\t"  \
            "mul.hi.u32     %2,  %10, %6;\n\t"  \
            "mul.lo.u32     t20, %10, %7;\n\t"  \
            "mul.hi.u32     %3,  %10, %7;\n\t"  \
            "mul.lo.u32     t30, %10, %8;\n\t"  \
            "mul.hi.u32     %4,  %10, %8;\n\t"  \
            "mul.lo.u32     t40, %10, %9;\n\t"  \
            /* now fixing the carry chains */   \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.u32       %4,  %4,  t40;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/          \
            "mul.lo.u32     t10, %11, %5;\n\t"  \
            "mul.lo.u32     t20, %11, %6;\n\t"  \
            "mul.lo.u32     t30, %11, %7;\n\t"  \
            "mul.lo.u32     t40, %11, %8;\n\t"  \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/        \
            "mul.hi.u32     t20, %11, %5;\n\t"  \
            "mul.hi.u32     t30, %11, %6;\n\t"  \
            "mul.hi.u32     t40, %11, %7;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/          \
            "mul.lo.u32     t20, %12, %5;\n\t"  \
            "mul.lo.u32     t30, %12, %6;\n\t"  \
            "mul.lo.u32     t40, %12, %7;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/        \
            "mul.hi.u32     t30, %12, %5;\n\t"  \
            "mul.hi.u32     t40, %12, %6;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.u32       %4, %4, t40;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/          \
            "mul.lo.u32     t30, %13, %5;\n\t"  \
            "mul.lo.u32     t40, %13, %6;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.u32    	%4, %4, t40;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/        \
            "mul.hi.u32     t40, %13, %5;\n\t"  \
            "add.u32     	%4, %4, t40;\n\t"   \
            /*Multiplication involving a4 second (low) diagonal*/        \
            "mul.lo.u32     t40, %14, %5;\n\t"  \
            "add.u32         %4, %4, t40;\n\t"  \
            "}\n"                                                        \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4)  /*0 , 4*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4), /*5 , 9*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4)    /*10,14*/    \
        );

// second part of the montgomery reduction written by andre
#define __reduce32_sec5x5(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, p0, p1, p2, p3, p4, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9)        \
    asm volatile ("{\n\t"                                            \
            /*%0-%4: r= (a + u) div 2^(5*64)*/                        \
            ".reg.u32 		t0, t5,t6,t7,t8,t9;\n\t"                    \
            "add.cc.u32     %0, %20, %10;\n\t"                            \
            "addc.cc.u32    %1, %21, %11;\n\t"                            \
            "addc.cc.u32    %2, %22, %12;\n\t"                            \
            "addc.cc.u32    %3, %23, %13;\n\t"                            \
            "addc.cc.u32    %4, %24, %14;\n\t"                        \
            "addc.cc.u32    t5, %25, %15;\n\t"                        \
            "addc.cc.u32    t6, %26, %16;\n\t"                        \
            "addc.cc.u32    t7, %27, %17;\n\t"                        \
            "addc.cc.u32    t8, %28, %18;\n\t"                        \
            "addc.cc.u32    t9, %29, %19;\n\t"                        \
            "addc.u32       t0, t0, 0;\n\t"                            \
            "sub.cc.u32     t5, t5, %5;\n\t"                            \
            "subc.cc.u32    t6, t6, %6;\n\t"                            \
            "subc.cc.u32    t7, t7, %7;\n\t"                            \
            "subc.cc.u32    t8, t8, %8;\n\t"                            \
            "subc.cc.u32    t9, t9, %9;\n\t"                            \
            "subc.u32       t0, 0, 0;\n\t"                                \
            "and.b32        %0, t0, %5;\n\t"                            \
            "and.b32        %1, t0, %6;\n\t"                            \
            "and.b32        %2, t0, %7;\n\t"                            \
            "and.b32        %3, t0, %8;\n\t"                            \
            "and.b32        %4, t0, %9;\n\t"                            \
            "add.cc.u32     %0, %0, t5;\n\t"                            \
            "addc.cc.u32    %1, %1, t6;\n\t"                            \
            "addc.cc.u32    %2, %2, t7;\n\t"                            \
            "addc.cc.u32    %3, %3, t8;\n\t"                            \
            "addc.u32       %4, %4, t9;\n\t"                            \
            "}\n"                                                        \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4)    /*0 , 4*/  \
    : "r"(p0),"r"(p1),"r"(p2),"r"(p3),"r"(p4),        /*5 , 9*/  \
      "r"(i0),"r"(i1),"r"(i2),"r"(i3),"r"(i4),        /*10,14*/  \
      "r"(i5),"r"(i6),"r"(i7),"r"(i8),"r"(i9),        /*15,19*/  \
      "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),        /*20,23*/  \
      "r"(a5),"r"(a6),"r"(a7),"r"(a8),"r"(a9)        /*25,29*/  \
    );

// montgomery reduction for 12 32bit limbs. Written by andre, adapted by floyd
#define __reduce32_5x5(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, mu0, mu1, mu2, mu3, mu4, p0, p1, p2, p3, p4)    \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9;                                                                    \
    __school32_5x5_low(c0,c1,c2,c3,c4, a0,a1,a2,a3,a4, mu0,mu1,mu2,mu3,mu4)                                    \
    __school32_5x5(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9, c0,c1,c2,c3,c4, p0,p1,p2,p3,p4)                            \
    __reduce32_sec5x5(c0,c1,c2,c3,c4, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9, p0,p1,p2,p3,p4, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9)


// TODO not implemented
#define __karatsuba32_5x5(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4)

// TODO not implemented
#define __sqr160_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, a0, a1, a2, a3, a4)

#define __mul160_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4)\
    __school32_5x5(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,a0,a1,a2,a3,a4,b0,b1,b2,b3,b4)

#define __fp_mul5x5_32(c0, c1, c2, c3, c4, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, mu0, mu1, mu2, mu3, mu4, p0, p1, p2, p3, p4)    \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;                                                                    \
    __school32_5x5(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9, a0,a1,a2,a3,a4, b0,b1,b2,b3,b4)                            \
    __reduce32_5x5(c0,c1,c2,c3,c4, f0,f1,f2,f3,f4,f5,f6,f7,f8,f9, mu0,mu1,mu2,mu3,mu4, p0,p1,p2,p3,p4)


////////////////////////////////////////////////////////////////////////////////
// 									6x6										  //
////////////////////////////////////////////////////////////////////////////////
// schoolbook multiplikation written by andre.
// needs 6 tmp registers
#define __school32_6x6(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)\
    asm volatile ("{\n\t"                                \
            ".reg .u32 t1,t2,t3,t4,t5,tc;\n\t"           \
            /*Multiplication involving a0*/                    \
            "mul.hi.u32     %0, %12, %18;\n\t"            \
            "mad.lo.cc.u32  %0, %12, %19, %0;\n\t"        \
            "mul.hi.u32     t1, %12, %19;\n\t"            \
            "madc.lo.cc.u32 t1, %12, %20, t1;\n\t"        \
            "mul.hi.u32     t2, %12, %20;\n\t"            \
            "madc.lo.cc.u32 t2, %12, %21, t2;\n\t"        \
            "mul.hi.u32     t3, %12, %21;\n\t"            \
            "madc.lo.cc.u32 t3, %12, %22, t3;\n\t"        \
            "mul.hi.u32     t4, %12, %22;\n\t"            \
            "madc.lo.cc.u32 t4, %12, %23, t4;\n\t"        \
            "mul.hi.u32     t5, %12, %23;\n\t"            \
            "addc.cc.u32    t5, t5, 0;\n\t"                \
            "addc.u32       tc, 0, 0;\n\t"                \
            /*Multiplication involving a1 first diagonal*/    \
            "mad.lo.cc.u32  %1, %13, %18, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %13, %18, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %13, %19, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %13, %20, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %13, %21, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %13, %22, t5;\n\t"            \
            "mul.hi.u32     %0, %13, %23;\n\t"                \
            "addc.cc.u32    %0, %0 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a1 second diagonal*/    \
            "mad.lo.cc.u32  t1, %13, %19, t1;\n\t"        \
            "madc.lo.cc.u32 t2, %13, %20, t2;\n\t"        \
            "madc.lo.cc.u32 t3, %13, %21, t3;\n\t"        \
            "madc.lo.cc.u32 t4, %13, %22, t4;\n\t"        \
            "madc.lo.cc.u32 t5, %13, %23, t5;\n\t"        \
            "addc.cc.u32    %0, %0 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a2 first diagonal*/    \
            "mad.lo.cc.u32  %2, %14, %18, t1;\n\t"        \
            "madc.hi.cc.u32 t2, %14, %18, t2;\n\t"        \
            "madc.hi.cc.u32 t3, %14, %19, t3;\n\t"        \
            "madc.hi.cc.u32 t4, %14, %20, t4;\n\t"        \
            "madc.hi.cc.u32 t5, %14, %21, t5;\n\t"        \
            "madc.hi.cc.u32 %0, %14, %22, %0;\n\t"        \
            "mul.hi.u32     t1, %14, %23;\n\t"            \
            "addc.cc.u32    t1, t1 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a2 second diagonal*/    \
            "mad.lo.cc.u32  t2, %14, %19, t2;\n\t"            \
            "madc.lo.cc.u32 t3, %14, %20, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %14, %21, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %14, %22, t5;\n\t"            \
            "madc.lo.cc.u32 %0, %14, %23, %0;\n\t"            \
            "addc.cc.u32    t1, t1 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a3 first diagonal*/    \
            "mad.lo.cc.u32  %3, %15, %18, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %15, %18, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %15, %19, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %15, %20, t5;\n\t"            \
            "madc.hi.cc.u32 %0, %15, %21, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %15, %22, t1;\n\t"            \
            "mul.hi.u32     t2, %15, %23;\n\t"                \
            "addc.cc.u32    t2, t2 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                     \
            /*Multiplication involving a3 second diagonal*/    \
            "mad.lo.cc.u32  t3, %15, %19, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %15, %20, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %15, %21, t5;\n\t"            \
            "madc.lo.cc.u32 %0, %15, %22, %0;\n\t"            \
            "madc.lo.cc.u32 t1, %15, %23, t1;\n\t"            \
            "addc.cc.u32    t2, t2 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a4 first diagonal*/    \
            "mad.lo.cc.u32  %4, %16, %18, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %16, %18, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %16, %19, t5;\n\t"            \
            "madc.hi.cc.u32 %0, %16, %20, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %16, %21, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %16, %22, t2;\n\t"            \
            "mul.hi.u32     t3, %16, %23;\n\t"                \
            "addc.cc.u32    t3, t3 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a4 second diagonal*/    \
            "mad.lo.cc.u32  t4, %16, %19, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %16, %20, t5;\n\t"            \
            "madc.lo.cc.u32 %0, %16, %21, %0;\n\t"            \
            "madc.lo.cc.u32 t1, %16, %22, t1;\n\t"            \
            "madc.lo.cc.u32 t2, %16, %23, t2;\n\t"            \
            "addc.cc.u32    t3, t3 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a5 first diagonal*/    \
            "mad.lo.cc.u32  %5, %17, %18, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %17, %18, t5;\n\t"            \
            "madc.hi.cc.u32 %0, %17, %19, %0;\n\t"            \
            "madc.hi.cc.u32 t1, %17, %20, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %17, %21, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %17, %22, t3;\n\t"            \
            "mul.hi.u32     t4, %17, %23;\n\t"                \
            "addc.cc.u32    t4, t4 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a5 second diagonal*/    \
            "mad.lo.cc.u32  %6, %17, %19, t5;\n\t"            \
            "madc.lo.cc.u32 %7, %17, %20, %0;\n\t"            \
            "madc.lo.cc.u32 %8, %17, %21, t1;\n\t"            \
            "madc.lo.cc.u32 %9, %17, %22, t2;\n\t"            \
            "madc.lo.cc.u32 %10, %17, %23, t3;\n\t"        \
            "addc.u32       %11, t4 , tc;\n\t"                \
            "mul.lo.u32     %0, %12, %18;\n\t"            \
            "}\n"                                        \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3), "=r"(c4), "=r"(c5), /*0,  5*/ \
      "=r"(c6), "=r"(c7), "=r"(c8), "=r"(c9), "=r"(c10), "=r"(c11)/*6, 11*/ \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5),       /*12,17*/ \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5));      /*18,23*/

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**196
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
#define __school32_inplace_6x6_low(a0, a1, a2, a3, a4, a5, mu0, mu1, mu2, mu3, mu4, mu5)\
    asm volatile ("{\n\t"                       \
            ".reg .u32 t10,t20,t30,t40,t50;\n\t"\
            /* t0-t5: m = a * mu mod 2^196 */   \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %12, %6;\n\t"    \
            "mul.hi.u32     %1,  %12, %6;\n\t"  \
            "mul.lo.u32     t10, %12, %7;\n\t"  \
            "mul.hi.u32     %2,  %12, %7;\n\t"  \
            "mul.lo.u32     t20, %12, %8;\n\t"  \
            "mul.hi.u32     %3,  %12, %8;\n\t"  \
            "mul.lo.u32     t30, %12, %9;\n\t"  \
            "mul.hi.u32     %4,  %12, %9;\n\t"  \
            "mul.lo.u32     t40, %12, %10;\n\t" \
            "mul.hi.u32     %5,  %12, %10;\n\t" \
            "mul.lo.u32     t50, %12, %11;\n\t" \
            /* now fixing the carry chains */   \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.cc.u32    %4,  %4,  t40;\n\t" \
            "addc.u32       %5,  %5,  t50;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/          \
            "mul.lo.u32     t10, %13, %6;\n\t"  \
            "mul.lo.u32     t20, %13, %7;\n\t"  \
            "mul.lo.u32     t30, %13, %8;\n\t"  \
            "mul.lo.u32     t40, %13, %9;\n\t"  \
            "mul.lo.u32     t50, %13, %10;\n\t" \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/        \
            "mul.hi.u32     t20, %13, %6;\n\t"  \
            "mul.hi.u32     t30, %13, %7;\n\t"  \
            "mul.hi.u32     t40, %13, %8;\n\t"  \
            "mul.hi.u32     t50, %13, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/          \
            "mul.lo.u32     t20, %14, %6;\n\t"  \
            "mul.lo.u32     t30, %14, %7;\n\t"  \
            "mul.lo.u32     t40, %14, %8;\n\t"  \
            "mul.lo.u32     t50, %14, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/        \
            "mul.hi.u32     t30, %14, %6;\n\t"  \
            "mul.hi.u32     t40, %14, %7;\n\t"  \
            "mul.hi.u32     t50, %14, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/          \
            "mul.lo.u32     t30, %15, %6;\n\t"  \
            "mul.lo.u32     t40, %15, %7;\n\t"  \
            "mul.lo.u32     t50, %15, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/        \
            "mul.hi.u32     t40, %15, %6;\n\t"  \
            "mul.hi.u32     t50, %15, %7;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a4 second (low) diagonal*/          \
            "mul.lo.u32     t40, %16, %6;\n\t"  \
            "mul.lo.u32     t50, %16, %7;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a4 first (high) diagonal*/        \
            "mul.hi.u32     t50, %16, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            /*Multiplication involving a5 second (low) diagonal*/        \
            "mul.lo.u32     t50, %17, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            "}\n"                                                    \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),"=r"(a5)  /*0 , 5*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5), /*6 ,11*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5)         /*12,17*/    \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**196
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// needs 5 tmp registers
#define __school32_6x6_low_nonopt(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, mu0, mu1, mu2, mu3, mu4, mu5)\
    asm volatile ("{\n\t"                       \
            ".reg .u32 t10,t20,t30,t40,t50;\n\t"\
            /* t0-t5: m = a * mu mod 2^196 */   \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %12, %6;\n\t"    \
            "mul.hi.u32     %1,  %12, %6;\n\t"  \
            "mul.lo.u32     t10, %12, %7;\n\t"  \
            "mul.hi.u32     %2,  %12, %7;\n\t"  \
            "mul.lo.u32     t20, %12, %8;\n\t"  \
            "mul.hi.u32     %3,  %12, %8;\n\t"  \
            "mul.lo.u32     t30, %12, %9;\n\t"  \
            "mul.hi.u32     %4,  %12, %9;\n\t"  \
            "mul.lo.u32     t40, %12, %10;\n\t" \
            "mul.hi.u32     %5,  %12, %10;\n\t" \
            "mul.lo.u32     t50, %12, %11;\n\t" \
            /* now fixing the carry chains */   \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.cc.u32    %4,  %4,  t40;\n\t" \
            "addc.u32       %5,  %5,  t50;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/          \
            "mul.lo.u32     t10, %13, %6;\n\t"  \
            "mul.lo.u32     t20, %13, %7;\n\t"  \
            "mul.lo.u32     t30, %13, %8;\n\t"  \
            "mul.lo.u32     t40, %13, %9;\n\t"  \
            "mul.lo.u32     t50, %13, %10;\n\t" \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/        \
            "mul.hi.u32     t20, %13, %6;\n\t"  \
            "mul.hi.u32     t30, %13, %7;\n\t"  \
            "mul.hi.u32     t40, %13, %8;\n\t"  \
            "mul.hi.u32     t50, %13, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/          \
            "mul.lo.u32     t20, %14, %6;\n\t"  \
            "mul.lo.u32     t30, %14, %7;\n\t"  \
            "mul.lo.u32     t40, %14, %8;\n\t"  \
            "mul.lo.u32     t50, %14, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/        \
            "mul.hi.u32     t30, %14, %6;\n\t"  \
            "mul.hi.u32     t40, %14, %7;\n\t"  \
            "mul.hi.u32     t50, %14, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/          \
            "mul.lo.u32     t30, %15, %6;\n\t"  \
            "mul.lo.u32     t40, %15, %7;\n\t"  \
            "mul.lo.u32     t50, %15, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/        \
            "mul.hi.u32     t40, %15, %6;\n\t"  \
            "mul.hi.u32     t50, %15, %7;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a4 second (low) diagonal*/          \
            "mul.lo.u32     t40, %16, %6;\n\t"  \
            "mul.lo.u32     t50, %16, %7;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a4 first (high) diagonal*/        \
            "mul.hi.u32     t50, %16, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            /*Multiplication involving a5 second (low) diagonal*/        \
            "mul.lo.u32     t50, %17, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            "}\n"                                                    \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5)  /*0 , 5*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5), /*6 ,11*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5)         /*12,17*/    \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**196
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// needs 5 tmp registers
// speciaal version which uses the fact the multiple limbs in mu are zero
#define __school32_6x6_low(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, mu0, mu1, mu2, mu3, mu4, mu5)\
    asm volatile ("{\n\t"                       \
            ".reg .u32 t10,t20,t30,t40,t50;\n\t"\
            /* t0-t5: m = a * mu mod 2^196 */   \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %12, %6;\n\t"    \
            "mul.hi.u32     %1,  %12, %6;\n\t"  \
            /*"mul.lo.u32     t10, %12, %7;\n\t"*/  \
            /*"mul.hi.u32     %2,  %12, %7;\n\t"*/  \
            "mul.lo.u32     t20, %12, %8;\n\t"  \
            "mul.hi.u32     %3,  %12, %8;\n\t"  \
            "mul.lo.u32     t30, %12, %9;\n\t"  \
            "mul.hi.u32     %4,  %12, %9;\n\t"  \
            "mul.lo.u32     t40, %12, %10;\n\t" \
            "mul.hi.u32     %5,  %12, %10;\n\t" \
            "mul.lo.u32     t50, %12, %11;\n\t" \
            /* now fixing the carry chains */   \
            /*"add.cc.u32     %1,  %1,  t10;\n\t"*/ \
            "add.cc.u32    %2,  0,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.cc.u32    %4,  %4,  t40;\n\t" \
            "addc.u32       %5,  %5,  t50;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/          \
            "mul.lo.u32     t10, %13, %6;\n\t"  \
            /*"mul.lo.u32     t20, %13, %7;\n\t"*/  \
            "mul.lo.u32     t30, %13, %8;\n\t"  \
            "mul.lo.u32     t40, %13, %9;\n\t"  \
            "mul.lo.u32     t50, %13, %10;\n\t" \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, 0;\n\t"     \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/        \
            "mul.hi.u32     t20, %13, %6;\n\t"  \
            /*"mul.hi.u32     t30, %13, %7;\n\t"*/  \
            "mul.hi.u32     t40, %13, %8;\n\t"  \
            "mul.hi.u32     t50, %13, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            /*"addc.cc.u32    %3, %3, t30;\n\t"*/   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/          \
            "mul.lo.u32     t20, %14, %6;\n\t"  \
            /*"mul.lo.u32     t30, %14, %7;\n\t"*/  \
            "mul.lo.u32     t40, %14, %8;\n\t"  \
            "mul.lo.u32     t50, %14, %9;\n\t"  \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, 0;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/        \
            "mul.hi.u32     t30, %14, %6;\n\t"  \
            /*"mul.hi.u32     t40, %14, %7;\n\t"*/  \
            "mul.hi.u32     t50, %14, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            /*"addc.cc.u32    %4, %4, t40;\n\t"*/   \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/          \
            "mul.lo.u32     t30, %15, %6;\n\t"  \
            /*"mul.lo.u32     t40, %15, %7;\n\t"*/  \
            "mul.lo.u32     t50, %15, %8;\n\t"  \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, 0;\n\t"     \
            "addc.u32       %5, %5, t50;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/        \
            "mul.hi.u32     t40, %15, %6;\n\t"  \
            /*"mul.hi.u32     t50, %15, %7;\n\t"*/  \
            "add.u32     %4, %4, t40;\n\t"   \
            /*"addc.u32       %5, %5, t50;\n\t"*/   \
            /*Multiplication involving a4 second (low) diagonal*/          \
            "mul.lo.u32     t40, %16, %6;\n\t"  \
            /*"mul.lo.u32     t50, %16, %7;\n\t"*/  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.u32       %5, %5, 0;\n\t"   \
            /*Multiplication involving a4 first (high) diagonal*/        \
            "mul.hi.u32     t50, %16, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            /*Multiplication involving a5 second (low) diagonal*/        \
            "mul.lo.u32     t50, %17, %6;\n\t"  \
            "add.u32         %5, %5, t50;\n\t"  \
            "}\n"                                                    \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5)  /*0 , 5*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5), /*6 ,11*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5)         /*12,17*/    \
        );


// c2c1c0 = a2a1a0+b2b1b0
#define __karatsuba32_6x6_3add_helper(c0, c1, c2, a0, a1, a2, b0, b1, b2)   \
    asm (    "{\n\t"                                                        \
            "add.cc.u32 	%0, %3, %6;\n\t"                                \
            "addc.cc.u32    %1, %4, %7;\n\t"                                \
            "addc.u32       %2, %5, %8;\n\t"                                \
            "}\n\t"                                                         \
        : "=r"(c0),"=r"(c1),"=r"(c2)    /*0,2*/                             \
        : "r"(a0),"r"(a1),"r"(a2),      /*3,5*/                             \
          "r"(b0),"r"(b1),"r"(b2));     /*6,8*/

// c3c2c1c0 = a2a1a0+b2b1b0
#define __karatsuba32_6x6_3add_helper_v2(c0, c1, c2, c3, a0, a1, a2, b0, b1, b2)\
    asm (    "{\n\t"                                                            \
            "add.cc.u32 	%0, %4, %7;\n\t"                                    \
            "addc.cc.u32    %1, %5, %8;\n\t"                                    \
            "addc.cc.u32    %2, %6, %9;\n\t"                                    \
            "addc.u32       %3,  0,  0;\n\t"                                    \
            "}\n\t"                                                             \
        : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)  /*0,3*/                          \
        : "r"(a0),"r"(a1),"r"(a2),             /*4,6*/                          \
          "r"(b0),"r"(b1),"r"(b2));            /*7,9*/


// karatsuba 6x6 multiplication without reduction
// TODO make a helper version of this
// TODO remove _karatsuba 4x4 with extended karatsuba 3x3
#define __karatsuba32_6x6(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)    \
    uint32_t irt0,irt1,irt2,irt3,irt4,irt5,irt6,irt7;                                       \
    __karatsuba32_6x6_3add_helper_v2(c0,c1,c2,c3, a0,a1,a2, a3,a4,a5)                       \
    __karatsuba32_6x6_3add_helper_v2(c4,c5,c6,c7, b0,b1,b2, b3,b4,b5)                       \
    __school32_4x4(irt0,irt1,irt2,irt3,irt4,irt5,irt6,irt7, c0,c1,c2,c3,c4,c5,c6,c7)        \
    __school32_3x3(c0,c1,c2,c3,c4,c5,  a0,a1,a2, b0,b1,b2)                                  \
    __school32_3x3(c6,c7,c8,c9,c10,c11,a3,a4,a5, b3,b4,b5)                                  \
    /*UsVs -= U1V1*/                                                                        \
    asm (    "{\n\t"                                                                        \
            "sub.cc.u32  %0, %12, %6 ;\n\t"                                                 \
            "subc.cc.u32 %1, %13, %7 ;\n\t"                                                 \
            "subc.cc.u32 %2, %14, %8 ;\n\t"                                                 \
            "subc.cc.u32 %3, %15, %9 ;\n\t"                                                 \
            "subc.cc.u32 %4, %16, %10;\n\t"                                                 \
            "subc.u32    %5, %17, %11;\n\t"                                                 \
            "}\n\t"                                                                         \
        : "=r"(irt0),"=r"(irt1),"=r"(irt2),"=r"(irt3),"=r"(irt4),"=r"(irt5) /*output 0 ,5 */\
        : "r"(c6),"r"(c7),"r"(c8),"r"(c9),"r"(c10),"r"(c11),                /*U1V1   6 ,11*/\
          "r"(irt0),"r"(irt1),"r"(irt2),"r"(irt3),"r"(irt4),"r"(irt5)       /*output 12,17*/\
    );                                                                                      \
    /*UsVs -= U0V0*/                                                                        \
    asm (    "{\n\t"                                                                        \
            "sub.cc.u32  %0, %12, %6 ;\n\t"                                                 \
            "subc.cc.u32 %1, %13, %7 ;\n\t"                                                 \
            "subc.cc.u32 %2, %14, %8 ;\n\t"                                                 \
            "subc.cc.u32 %3, %15, %9 ;\n\t"                                                 \
            "subc.cc.u32 %4, %16, %10;\n\t"                                                 \
            "subc.u32    %5, %17, %11;\n\t"                                                 \
            "}\n\t"                                                                         \
        : "=r"(irt0),"=r"(irt1),"=r"(irt2),"=r"(irt3),"=r"(irt4),"=r"(irt5) /*output 0 ,5 */\
        : "r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(c4),"r"(c5),                  /*U0V0   6 ,11*/\
          "r"(irt0),"r"(irt1),"r"(irt2),"r"(irt3),"r"(irt4),"r"(irt5)       /*output 12,17*/\
    );                                                                                      \
    /*Final addition and result is written */                                               \
    asm volatile("{\n\t"                                                                    \
            "add.cc.u32  %0, %12, %6 ;\n\t"                                                 \
            "addc.cc.u32 %1, %13, %7 ;\n\t"                                                 \
            "addc.cc.u32 %2, %14, %8 ;\n\t"                                                 \
            "addc.cc.u32 %3, %15, %9 ;\n\t"                                                 \
            "addc.cc.u32 %4, %16, %10;\n\t"                                                 \
            "addc.u32    %5, %17, %11;\n\t"                                                 \
            "}\n"                                                                           \
        : "=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7),"=r"(c8)        /*0, 5*/             \
        : "r"(irt0),"r"(irt1),"r"(irt2),"r"(irt3),"r"(irt4),"r"(irt5), /*6, 11*/            \
          "r"(c3),"r"(c4),"r"(c5),"r"(c6),"r"(c7),"r"(c8));            /*12,17*/

#define __mul6x6_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)    \
    __karatsuba32_6x6(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5)


// TODO not implemented
#define __sqr6x6_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, a0, a1, a2, a3, a4, a5)

// second part of the montgomery reduction written by andre
#define __reduce32_inplace_sec6x6(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, p0, p1, p2, p3, p4, p5, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11)        \
    asm volatile ("{\n\t"                                               \
            /*%0-%5: r= (a + u) div 2^(3*64)*/                          \
            ".reg.u32 		t0, t6,t7,t8,t9,t10,t11;\n\t"               \
            "mov.b32        t0, 0;\n\t"                                 \
            "add.cc.u32     %0, %24, %12;\n\t"                          \
            "addc.cc.u32    %1, %25, %13;\n\t"                          \
            "addc.cc.u32    %2, %26, %14;\n\t"                          \
            "addc.cc.u32    %3, %27, %15;\n\t"                          \
            "addc.cc.u32    %4, %28, %16;\n\t"                          \
            "addc.cc.u32    %5, %29, %17;\n\t"                          \
            "addc.cc.u32    t6, %30, %18;\n\t"                          \
            "addc.cc.u32    t7, %31, %19;\n\t"                          \
            "addc.cc.u32    t8, %32, %20;\n\t"                          \
            "addc.cc.u32    t9, %33, %21;\n\t"                          \
            "addc.cc.u32    t10, %34, %22;\n\t"                         \
            "addc.cc.u32    t11, %35, %23;\n\t"                         \
            "addc.u32       t0, t0, 0;\n\t"                             \
            "sub.cc.u32     t6, t6, %6;\n\t"                            \
            "subc.cc.u32    t7, t7, %7;\n\t"                            \
            "subc.cc.u32    t8, t8, %8;\n\t"                            \
            "subc.cc.u32    t9, t9, %9;\n\t"                            \
            "subc.cc.u32    t10, t10, %10;\n\t"                         \
            "subc.cc.u32    t11, t11, %11;\n\t"                         \
            "subc.u32       t0, t0, 0;\n\t"                             \
            "and.b32        %0, t0, %6;\n\t"                            \
            "and.b32        %1, t0, %7;\n\t"                            \
            "and.b32        %2, t0, %8;\n\t"                            \
            "and.b32        %3, t0, %9;\n\t"                            \
            "and.b32        %4, t0, %10;\n\t"                           \
            "and.b32        %5, t0, %11;\n\t"                           \
            "add.cc.u32     %0, %0, t6;\n\t"                            \
            "addc.cc.u32    %1, %1, t7;\n\t"                            \
            "addc.cc.u32    %2, %2, t8;\n\t"                            \
            "addc.cc.u32    %3, %3, t9;\n\t"                            \
            "addc.cc.u32    %4, %4, t10;\n\t"                           \
            "addc.u32       %5, %5, t11;\n\t"                           \
            "}\n"                                                       \
    : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),"=r"(a5)  /*0 , 5*/  \
    : "r"(p0),"r"(p1),"r"(p2),"r"(p3),"r"(p4),"r"(p5),       /*6 ,11*/  \
      "r"(i0),"r"(i1),"r"(i2),"r"(i3),"r"(i4),"r"(i5),       /*12,17*/  \
      "r"(i6),"r"(i7),"r"(i8),"r"(i9),"r"(i10),"r"(i11),     /*18,23*/  \
      "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),       /*24,29*/  \
      "r"(a6),"r"(a7),"r"(a8),"r"(a9),"r"(a10),"r"(a11)      /*30,35*/  \
    );

// second part of the montgomery reduction written by andre
#define __reduce32_sec6x6(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, p0, p1, p2, p3, p4, p5, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11)        \
    asm volatile ("{\n\t"                                               \
            /*%0-%5: r= (a + u) div 2^(6*64)*/                          \
            ".reg.u32 		t0, t6,t7,t8,t9,t10,t11;\n\t"               \
            "add.cc.u32     %0, %24, %12;\n\t"                          \
            "addc.cc.u32    %1, %25, %13;\n\t"                          \
            "addc.cc.u32    %2, %26, %14;\n\t"                          \
            "addc.cc.u32    %3, %27, %15;\n\t"                          \
            "addc.cc.u32    %4, %28, %16;\n\t"                          \
            "addc.cc.u32    %5, %29, %17;\n\t"                          \
            "addc.cc.u32    t6, %30, %18;\n\t"                          \
            "addc.cc.u32    t7, %31, %19;\n\t"                          \
            "addc.cc.u32    t8, %32, %20;\n\t"                          \
            "addc.cc.u32    t9, %33, %21;\n\t"                          \
            "addc.cc.u32    t10, %34, %22;\n\t"                         \
            "addc.cc.u32    t11, %35, %23;\n\t"                         \
            "addc.u32       t0, t0, 0;\n\t"                             \
            "sub.cc.u32     t6, t6, %6;\n\t"                            \
            "subc.cc.u32    t7, t7, %7;\n\t"                            \
            "subc.cc.u32    t8, t8, %8;\n\t"                            \
            "subc.cc.u32    t9, t9, %9;\n\t"                            \
            "subc.cc.u32    t10, t10, %10;\n\t"                         \
            "subc.cc.u32    t11, t11, %11;\n\t"                         \
            "subc.u32       t0, 0, 0;\n\t"                              \
            "and.b32        %0, t0, %6;\n\t"                            \
            "and.b32        %1, t0, %7;\n\t"                            \
            "and.b32        %2, t0, %8;\n\t"                            \
            "and.b32        %3, t0, %9;\n\t"                            \
            "and.b32        %4, t0, %10;\n\t"                           \
            "and.b32        %5, t0, %11;\n\t"                           \
            "add.cc.u32     %0, %0, t6;\n\t"                            \
            "addc.cc.u32    %1, %1, t7;\n\t"                            \
            "addc.cc.u32    %2, %2, t8;\n\t"                            \
            "addc.cc.u32    %3, %3, t9;\n\t"                            \
            "addc.cc.u32    %4, %4, t10;\n\t"                           \
            "addc.u32       %5, %5, t11;\n\t"                           \
            "}\n"                                                       \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5)  /*0 , 5*/  \
    : "r"(p0),"r"(p1),"r"(p2),"r"(p3),"r"(p4),"r"(p5),       /*6 ,11*/  \
      "r"(i0),"r"(i1),"r"(i2),"r"(i3),"r"(i4),"r"(i5),       /*12,17*/  \
      "r"(i6),"r"(i7),"r"(i8),"r"(i9),"r"(i10),"r"(i11),     /*18,23*/  \
      "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),       /*24,29*/  \
      "r"(a6),"r"(a7),"r"(a8),"r"(a9),"r"(a10),"r"(a11)      /*30,35*/  \
    );

// montgomery reduction for 12 32bit limbs. Written by andre, adapted by floyd
#define __reduce32_inplace6x6(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, mu0, mu1, mu2, mu3, mu4, mu5, p0, p1, p2, p3, p4, p5)    \
    uint32_t t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17;                                                                               \
    __school32_inplace_6x6_low(a0,a1,a2,a3,a4,a5,mu0,mu1,mu2,mu3,mu4,mu5)                                                               \
    __school32_6x6(t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,a0,a1,a2,a3,a4,a5,p0,p1,p2,p3,p4,p5)                                     \
    __reduce32_sec6x6(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,p0,p1,p2,p3,p4,p5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17)

// montgomery reduction for 12 32bit limbs. Written by andre, adapted by floyd
#define __reduce32_6x6(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, mu0, mu1, mu2, mu3, mu4, mu5, p0, p1, p2, p3, p4, p5)  \
    uint32_t t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17;                                                                                               \
    __school32_6x6_low(c0,c1,c2,c3,c4,c5, a0,a1,a2,a3,a4,a5, mu0,mu1,mu2,mu3,mu4,mu5)                                                                   \
    __school32_6x6(t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17, c0,c1,c2,c3,c4,c5, p0,p1,p2,p3,p4,p5)                                                   \
    __reduce32_sec6x6(c0,c1,c2,c3,c4,c5, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11, p0,p1,p2,p3,p4,p5, t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17)

// wrapper function. Implements
// c5c4c3c2c1c0 = a5a4a3a2a1a0 * b5b4b3b2b1b0 mod p5p4p3p2p1p0
// using school
#define __fp_mul6x6_32(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, mu0, mu1, mu2, mu3, mu4, mu5, p0, p1, p2, p3, p4, p5)    \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;                        /*mult helper*/                                                              \
    __school32_6x6(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11, a0,a1,a2,a3,a4,a5, b0,b1,b2,b3,b4,b5)                                                         \
    __reduce32_6x6(c0,c1,c2,c3,c4,c5, f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11, mu0,mu1,mu2,mu3,mu4,mu5, p0,p1,p2,p3,p4,p5)

#define __fp_mul6x6_32_v2(c0, c1, c2, c3, c4, c5, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, mu0, mu1, mu2, mu3, mu4, mu5, p0, p1, p2, p3, p4, p5) \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;                        /*mult helper*/                                                              \
    uint32_t t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17;                                                                                               \
    __school32_6x6(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11, a0,a1,a2,a3,a4,a5, b0,b1,b2,b3,b4,b5)                                                         \
    __school32_6x6_low(c0,c1,c2,c3,c4,c5, a0,a1,a2,a3,a4,a5, mu0,mu1,mu2,mu3,mu4,mu5)                                                                   \
    __school32_6x6(t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17, c0,c1,c2,c3,c4,c5, p0,p1,p2,p3,p4,p5)                                                   \
    __reduce32_sec6x6(c0,c1,c2,c3,c4,c5, f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11, p0,p1,p2,p3,p4,p5, t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17)


////////////////////////////////////////////////////////////////////////////////
// 									7x7 									  //
////////////////////////////////////////////////////////////////////////////////
// simple school book multiplication using the expensive `mad` instruction
// uses 8 tmp registers
// 7x7 32bit limbs
#define __school32_7x7(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6)\
    asm volatile ("{\n\t"                                   \
            ".reg .u32 t0,t1,t2,t3,t4,t5,t6,tc;\n\t"        \
            /*Multiplication involving a0*/                 \
            "mul.lo.u32     %0, %14, %21;\n\t"              \
            "mul.hi.u32     t0, %14, %21;\n\t"              \
            "mad.lo.cc.u32  t0, %14, %22, t0;\n\t"          \
            "mul.hi.u32     t1, %14, %22;\n\t"              \
            "madc.lo.cc.u32 t1, %14, %23, t1;\n\t"          \
            "mul.hi.u32     t2, %14, %23;\n\t"              \
            "madc.lo.cc.u32 t2, %14, %24, t2;\n\t"          \
            "mul.hi.u32     t3, %14, %24;\n\t"              \
            "madc.lo.cc.u32 t3, %14, %25, t3;\n\t"          \
            "mul.hi.u32     t4, %14, %25;\n\t"              \
            "madc.lo.cc.u32 t4, %14, %26, t4;\n\t"          \
            "mul.hi.u32     t5, %14, %26;\n\t"              \
            "madc.lo.cc.u32 t5, %14, %27, t5;\n\t"          \
            "mul.hi.u32     t6, %14, %27;\n\t"              \
            "addc.cc.u32    t6, t6, 0;\n\t"                 \
            "addc.u32       tc, 0, 0;\n\t"                  \
            /*Multiplication involving a1 first diagonal*/  \
            "mad.lo.cc.u32  %1, %15, %21, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %15, %21, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %15, %22, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %15, %23, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %15, %24, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %15, %25, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %15, %26, t6;\n\t"          \
            "mul.hi.u32     t0, %15, %27;\n\t"              \
            "addc.cc.u32    t0, t0 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a1 second diagonal*/ \
            "mad.lo.cc.u32  t1, %15, %22, t1;\n\t"          \
            "madc.lo.cc.u32 t2, %15, %23, t2;\n\t"          \
            "madc.lo.cc.u32 t3, %15, %24, t3;\n\t"          \
            "madc.lo.cc.u32 t4, %15, %25, t4;\n\t"          \
            "madc.lo.cc.u32 t5, %15, %26, t5;\n\t"          \
            "madc.lo.cc.u32 t6, %15, %27, t6;\n\t"          \
            "addc.cc.u32    t0, t0 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a2 first diagonal*/  \
            "mad.lo.cc.u32  %2, %16, %21, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %16, %21, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %16, %22, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %16, %23, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %16, %24, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %16, %25, t6;\n\t"          \
            "madc.hi.cc.u32 t0, %16, %26, t0;\n\t"          \
            "mul.hi.u32     t1, %16, %27;\n\t"              \
            "addc.cc.u32    t1, t1 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a2 second diagonal*/ \
            "mad.lo.cc.u32  t2, %16, %22, t2;\n\t"          \
            "madc.lo.cc.u32 t3, %16, %23, t3;\n\t"          \
            "madc.lo.cc.u32 t4, %16, %24, t4;\n\t"          \
            "madc.lo.cc.u32 t5, %16, %25, t5;\n\t"          \
            "madc.lo.cc.u32 t6, %16, %26, t6;\n\t"          \
            "madc.lo.cc.u32 t0, %16, %27, t0;\n\t"          \
            "addc.cc.u32    t1, t1 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a3 first diagonal*/  \
            "mad.lo.cc.u32  %3, %17, %21, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %17, %21, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %17, %22, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %17, %23, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %17, %24, t6;\n\t"          \
            "madc.hi.cc.u32 t0, %17, %25, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %17, %26, t1;\n\t"          \
            "mul.hi.u32     t2, %17, %27;\n\t"              \
            "addc.cc.u32    t2, t2 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a3 second diagonal*/ \
            "mad.lo.cc.u32  t3, %17, %22, t3;\n\t"          \
            "madc.lo.cc.u32 t4, %17, %23, t4;\n\t"          \
            "madc.lo.cc.u32 t5, %17, %24, t5;\n\t"          \
            "madc.lo.cc.u32 t6, %17, %25, t6;\n\t"          \
            "madc.lo.cc.u32 t0, %17, %26, t0;\n\t"          \
            "madc.lo.cc.u32 t1, %17, %27, t1;\n\t"          \
            "addc.cc.u32    t2, t2 , tc;\n\t"               \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a4 first diagonal*/  \
            "mad.lo.cc.u32  %4, %18, %21, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %18, %21, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %18, %22, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %18, %23, t6;\n\t"          \
            "madc.hi.cc.u32 t0, %18, %24, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %18, %25, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %18, %26, t2;\n\t"          \
            "mul.hi.u32     t3, %18, %27;\n\t"              \
            "addc.cc.u32    t3, t3 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a4 second diagonal*/ \
            "mad.lo.cc.u32  t4, %18, %22, t4;\n\t"          \
            "madc.lo.cc.u32 t5, %18, %23, t5;\n\t"          \
            "madc.lo.cc.u32 t6, %18, %24, t6;\n\t"          \
            "madc.lo.cc.u32 t0, %18, %25, t0;\n\t"          \
            "madc.lo.cc.u32 t1, %18, %26, t1;\n\t"          \
            "madc.lo.cc.u32 t2, %18, %27, t2;\n\t"          \
            "addc.cc.u32    t3, t3 , tc;\n\t"               \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a5 first diagonal*/  \
            "mad.lo.cc.u32  %5, %19, %21, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %19, %21, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %19, %22, t6;\n\t"          \
            "madc.hi.cc.u32 t0, %19, %23, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %19, %24, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %19, %25, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %19, %26, t3;\n\t"          \
            "mul.hi.u32     t4, %19, %27;\n\t"              \
            "addc.cc.u32    t4, t4 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a5 second diagonal*/ \
            "mad.lo.cc.u32  t5, %19, %22, t5;\n\t"          \
            "madc.lo.cc.u32 t6, %19, %23, t6;\n\t"          \
            "madc.lo.cc.u32 t0, %19, %24, t0;\n\t"          \
            "madc.lo.cc.u32 t1, %19, %25, t1;\n\t"          \
            "madc.lo.cc.u32 t2, %19, %26, t2;\n\t"          \
            "madc.lo.cc.u32 t3, %19, %27, t3;\n\t"          \
            "addc.cc.u32    t4, t4 , tc;\n\t"               \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a6 first diagonal*/  \
            "mad.lo.cc.u32  %6, %20, %21, t5;\n\t"          \
            "madc.hi.cc.u32 t6, %20, %21, t6;\n\t"          \
            "madc.hi.cc.u32 t0, %20, %22, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %20, %23, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %20, %24, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %20, %25, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %20, %26, t4;\n\t"          \
            "mul.hi.u32     t5, %20, %27;\n\t"              \
            "addc.cc.u32    t5, t5 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a6 second diagonal*/ \
            "mad.lo.cc.u32  %7, %20, %22, t6;\n\t"          \
            "madc.lo.cc.u32 %8, %20, %23, t0;\n\t"          \
            "madc.lo.cc.u32 %9, %20, %24, t1;\n\t"          \
            "madc.lo.cc.u32 %10, %20, %25, t2;\n\t"         \
            "madc.lo.cc.u32 %11, %20, %26, t3;\n\t"         \
            "madc.lo.cc.u32 %12, %20, %27, t4;\n\t"         \
            "addc.u32       %13, t5 , tc;\n\t"              \
            "}\n"                                           \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),   /*0,  6*/ \
      "=r"(c7),"=r"(c8),"=r"(c9),"=r"(c10),"=r"(c11),"=r"(c12),"=r"(c13)/*7, 13*/ \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6),    /*14,20*/ \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6));   /*21,27*/

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**224
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
#define __school32_inplace_7x7_low(a0, a1, a2, a3, a4, a5, a6, mu0, mu1, mu2, mu3, mu4, mu5, mu6)\
    asm volatile ("{\n\t"                        \
            ".reg .u32 t10,t20,t30,t40,t50,t60;\n\t" \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %14, %7;\n\t"    \
            "mul.hi.u32     %1,  %14, %7;\n\t"  \
            "mul.lo.u32     t10, %14, %8;\n\t"  \
            "mul.hi.u32     %2,  %14, %8;\n\t"  \
            "mul.lo.u32     t20, %14, %9;\n\t"  \
            "mul.hi.u32     %3,  %14, %9;\n\t"  \
            "mul.lo.u32     t30, %14, %10;\n\t" \
            "mul.hi.u32     %4,  %14, %10;\n\t" \
            "mul.lo.u32     t40, %14, %11;\n\t" \
            "mul.hi.u32     %5,  %14, %11;\n\t" \
            "mul.lo.u32     t50, %14, %12;\n\t" \
            "mul.hi.u32     %6,  %14, %12;\n\t" \
            "mul.lo.u32     t60, %14, %13;\n\t" \
            /*now fixing the carry chain*/      \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.cc.u32    %4,  %4,  t40;\n\t" \
            "addc.cc.u32    %5,  %5,  t50;\n\t" \
            "addc.u32       %6,  %6,  t60;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mul.lo.u32     t10, %15, %7;\n\t"  \
            "mul.lo.u32     t20, %15, %8;\n\t"  \
            "mul.lo.u32     t30, %15, %9;\n\t"  \
            "mul.lo.u32     t40, %15, %10;\n\t" \
            "mul.lo.u32     t50, %15, %11;\n\t" \
            "mul.lo.u32     t60, %15, %12;\n\t" \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mul.hi.u32     t20, %15, %7;\n\t"  \
            "mul.hi.u32     t30, %15, %8;\n\t"  \
            "mul.hi.u32     t40, %15, %9;\n\t"  \
            "mul.hi.u32     t50, %15, %10;\n\t" \
            "mul.hi.u32     t60, %15, %11;\n\t" \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mul.lo.u32     t20, %16, %7;\n\t"  \
            "mul.lo.u32     t30, %16, %8;\n\t"  \
            "mul.lo.u32     t40, %16, %9;\n\t"  \
            "mul.lo.u32     t50, %16, %10;\n\t" \
            "mul.lo.u32     t60, %16, %11;\n\t" \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/   \
            "mul.hi.u32     t30, %16, %7;\n\t"  \
            "mul.hi.u32     t40, %16, %8;\n\t"  \
            "mul.hi.u32     t50, %16, %9;\n\t"  \
            "mul.hi.u32     t60, %16, %10;\n\t" \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/   \
            "mul.lo.u32     t30, %17, %7;\n\t"  \
            "mul.lo.u32     t40, %17, %8;\n\t"  \
            "mul.lo.u32     t50, %17, %9;\n\t"  \
            "mul.lo.u32     t60, %17, %10;\n\t" \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/   \
            "mul.hi.u32     t40, %17, %7;\n\t"  \
            "mul.hi.u32     t50, %17, %8;\n\t"  \
            "mul.hi.u32     t60, %17, %9;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a4 second (low) diagonal*/   \
            "mul.lo.u32     t40, %18, %7;\n\t"  \
            "mul.lo.u32     t50, %18, %8;\n\t"  \
            "mul.lo.u32     t60, %18, %9;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a4 first (high) diagonal*/   \
            "mul.hi.u32     t40, %18, %7;\n\t"  \
            "mul.hi.u32     t50, %18, %8;\n\t"  \
            "add.cc.u32     %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a5 second (low) diagonal*/   \
            "mul.lo.u32     t50, %19, %7;\n\t"  \
            "mul.lo.u32     t60, %19, %8;\n\t"  \
            "add.cc.u32     %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a5 first (high) diagonal*/   \
            "mul.hi.u32     t60, %19, %7;\n\t"  \
            "add.u32        %6, %6, t60;\n\t"   \
            /*multiplication involving a6 second (low) diagonal*/   \
            "mul.hi.u32     t60, %20, %7;\n\t"                      \
            "add.u32        %6, %6, t60;\n\t"                       \
            "}\n"                                                   \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),"=r"(a5),"=r"(a6)  /*0 , 6*/     \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5),"r"(mu6), /*7 ,13*/     \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6)         /*14,20*/     \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**224
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// non inplace version
// uses  7 tmp registers
#define __school32_7x7_low_nonopt(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, mu0, mu1, mu2, mu3, mu4, mu5, mu6)\
    asm volatile ("{\n\t"                        \
            ".reg .u32 t10,t20,t30,t40,t50,t60;\n\t" \
            /* Multiplication involving a0 */   \
            "mul.lo.u32     %0,  %14, %7;\n\t"  \
            "mul.hi.u32     %1,  %14, %7;\n\t"  \
            "mul.lo.u32     t10, %14, %8;\n\t"  \
            "mul.hi.u32     %2,  %14, %8;\n\t"  \
            "mul.lo.u32     t20, %14, %9;\n\t"  \
            "mul.hi.u32     %3,  %14, %9;\n\t"  \
            "mul.lo.u32     t30, %14, %10;\n\t" \
            "mul.hi.u32     %4,  %14, %10;\n\t" \
            "mul.lo.u32     t40, %14, %11;\n\t" \
            "mul.hi.u32     %5,  %14, %11;\n\t" \
            "mul.lo.u32     t50, %14, %12;\n\t" \
            "mul.hi.u32     %6,  %14, %12;\n\t" \
            "mul.lo.u32     t60, %14, %13;\n\t" \
            /*now fixing the carry chain*/      \
            "add.cc.u32     %1,  %1,  t10;\n\t" \
            "addc.cc.u32    %2,  %2,  t20;\n\t" \
            "addc.cc.u32    %3,  %3,  t30;\n\t" \
            "addc.cc.u32    %4,  %4,  t40;\n\t" \
            "addc.cc.u32    %5,  %5,  t50;\n\t" \
            "addc.u32       %6,  %6,  t60;\n\t" \
            /*Multiplication involving a1 second (low) diagonal*/ \
            "mul.lo.u32     t10, %15, %7;\n\t"  \
            "mul.lo.u32     t20, %15, %8;\n\t"  \
            "mul.lo.u32     t30, %15, %9;\n\t"  \
            "mul.lo.u32     t40, %15, %10;\n\t" \
            "mul.lo.u32     t50, %15, %11;\n\t" \
            "mul.lo.u32     t60, %15, %12;\n\t" \
            "add.cc.u32     %1, %1, t10;\n\t"   \
            "addc.cc.u32    %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a1 first (high) diagonal*/ \
            "mul.hi.u32     t20, %15, %7;\n\t"  \
            "mul.hi.u32     t30, %15, %8;\n\t"  \
            "mul.hi.u32     t40, %15, %9;\n\t"  \
            "mul.hi.u32     t50, %15, %10;\n\t" \
            "mul.hi.u32     t60, %15, %11;\n\t" \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a2 second (low) diagonal*/ \
            "mul.lo.u32     t20, %16, %7;\n\t"  \
            "mul.lo.u32     t30, %16, %8;\n\t"  \
            "mul.lo.u32     t40, %16, %9;\n\t"  \
            "mul.lo.u32     t50, %16, %10;\n\t" \
            "mul.lo.u32     t60, %16, %11;\n\t" \
            "add.cc.u32     %2, %2, t20;\n\t"   \
            "addc.cc.u32    %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a2 first (high) diagonal*/ \
            "mul.hi.u32     t30, %16, %7;\n\t"  \
            "mul.hi.u32     t40, %16, %8;\n\t"  \
            "mul.hi.u32     t50, %16, %9;\n\t"  \
            "mul.hi.u32     t60, %16, %10;\n\t" \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a3 second (low) diagonal*/ \
            "mul.lo.u32     t30, %17, %7;\n\t"  \
            "mul.lo.u32     t40, %17, %8;\n\t"  \
            "mul.lo.u32     t50, %17, %9;\n\t"  \
            "mul.lo.u32     t60, %17, %10;\n\t" \
            "add.cc.u32     %3, %3, t30;\n\t"   \
            "addc.cc.u32    %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*Multiplication involving a3 first (high) diagonal*/ \
            "mul.hi.u32     t40, %17, %7;\n\t"  \
            "mul.hi.u32     t50, %17, %8;\n\t"  \
            "mul.hi.u32     t60, %17, %9;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a4 second (low) diagonal*/ \
            "mul.lo.u32     t40, %18, %7;\n\t"  \
            "mul.lo.u32     t50, %18, %8;\n\t"  \
            "mul.lo.u32     t60, %18, %9;\n\t"  \
            "add.cc.u32     %4, %4, t40;\n\t"   \
            "addc.cc.u32    %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a4 first (high) diagonal*/ \
            "mul.hi.u32     t50, %18, %7;\n\t"  \
            "mul.hi.u32     t60, %18, %8;\n\t"  \
            "add.cc.u32     %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a5 second (low) diagonal*/ \
            "mul.lo.u32     t50, %19, %7;\n\t"  \
            "mul.lo.u32     t60, %19, %8;\n\t"  \
            "add.cc.u32     %5, %5, t50;\n\t"   \
            "addc.u32       %6, %6, t60;\n\t"   \
            /*multiplication involving a5 first (high) diagonal*/ \
            "mul.hi.u32     t60, %19, %7;\n\t"  \
            "add.u32        %6, %6, t60;\n\t"   \
            /*multiplication involving a6 second (low) diagonal*/ \
            "mul.lo.u32     t60, %20, %7;\n\t"  \
            "add.u32        %6, %6, t60;\n\t"   \
            "}\n"                                \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6)  /*0 , 6*/ \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5),"r"(mu6), /*7 ,13*/ \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6)          /*14,20*/    \
        );

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**224
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// non inplace version
// uses  7 tmp registers
#define __school32_7x7_low(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, mu0, mu1, mu2, mu3, mu4, mu5, mu6)\
    asm volatile ("{\n\t"                           \
            ".reg .u32 t10,t20,t30,t40,t50,t60;\n\t"\
            /* Multiplication involving a0 */       \
            "mul.lo.u32     %0,  %14, %7;\n\t"      \
            "mul.hi.u32     %1,  %14, %7;\n\t"      \
            /*"mul.lo.u32     t10, %14, %8;\n\t"*/  \
            /*"mul.hi.u32     %2,  %14, %8;\n\t"*/  \
            /*"mul.lo.u32     t20, %14, %9;\n\t"*/  \
            /*"mul.hi.u32     %3,  %14, %9;\n\t"*/  \
            "mul.lo.u32     t30, %14, %10;\n\t"     \
            "mul.hi.u32     %4,  %14, %10;\n\t"     \
            "mul.lo.u32     t40, %14, %11;\n\t"     \
            "mul.hi.u32     %5,  %14, %11;\n\t"     \
            "mul.lo.u32     t50, %14, %12;\n\t"     \
            "mul.hi.u32     %6,  %14, %12;\n\t"     \
            "mul.lo.u32     t60, %14, %13;\n\t"     \
            /*now fixing the carry chain*/          \
            /*"add.cc.u32     %1,  %1,  t10;\n\t"*/ \
            "addc.cc.u32    %2,  0,    0;\n\t"      \
            "add.cc.u32     %3,  0,  t30;\n\t"      \
            "addc.cc.u32    %4,  %4,  t40;\n\t"     \
            "addc.cc.u32    %5,  %5,  t50;\n\t"     \
            "addc.u32       %6,  %6,  t60;\n\t"     \
            /*Multiplication involving a1 second (low) diagonal*/ \
            "mul.lo.u32     t10, %15, %7;\n\t"      \
            /*"mul.lo.u32     t20, %15, %8;\n\t"*/  \
            /*"mul.lo.u32     t30, %15, %9;\n\t"*/  \
            "mul.lo.u32     t40, %15, %10;\n\t"     \
            "mul.lo.u32     t50, %15, %11;\n\t"     \
            "mul.lo.u32     t60, %15, %12;\n\t"     \
            "add.cc.u32     %1, %1, t10;\n\t"       \
            /*"addc.cc.u32    %2, %2, t20;\n\t"*/   \
            /*"addc.cc.u32    %3, %3, t30;\n\t"*/   \
            "addc.cc.u32    %4, %4, t40;\n\t"       \
            "addc.cc.u32    %5, %5, t50;\n\t"       \
            "addc.u32       %6, %6, t60;\n\t"       \
            /*Multiplication involving a1 first (high) diagonal*/ \
            "mul.hi.u32     t20, %15, %7;\n\t"      \
            /*"mul.hi.u32     t30, %15, %8;\n\t"*/  \
            /*"mul.hi.u32     t40, %15, %9;\n\t"*/  \
            "mul.hi.u32     t50, %15, %10;\n\t"     \
            "mul.hi.u32     t60, %15, %11;\n\t"     \
            "add.cc.u32     %2, %2, t20;\n\t"       \
            /*"addc.cc.u32    %3, %3, 0;\n\t"*/     \
            /*"addc.cc.u32    %4, %4, 0;\n\t"*/     \
            "addc.cc.u32    %5, %5, t50;\n\t"       \
            "addc.u32       %6, %6, t60;\n\t"       \
            /*Multiplication involving a2 second (low) diagonal*/ \
            "mul.lo.u32     t20, %16, %7;\n\t"      \
            /*"mul.lo.u32     t30, %16, %8;\n\t"*/  \
            /*"mul.lo.u32     t40, %16, %9;\n\t"*/  \
            "mul.lo.u32     t50, %16, %10;\n\t"     \
            "mul.lo.u32     t60, %16, %11;\n\t"     \
            "add.cc.u32     %2, %2, t20;\n\t"       \
            "addc.cc.u32    %3, %3, 0;\n\t"         \
            "addc.cc.u32    %4, %4, 0;\n\t"         \
            "addc.cc.u32    %5, %5, t50;\n\t"       \
            "addc.u32       %6, %6, t60;\n\t"       \
            /*Multiplication involving a2 first (high) diagonal*/ \
            "mul.hi.u32     t30, %16, %7;\n\t"      \
            /*"mul.hi.u32     t40, %16, %8;\n\t"*/  \
            /*"mul.hi.u32     t50, %16, %9;\n\t"*/  \
            "mul.hi.u32     t60, %16, %10;\n\t"     \
            "add.cc.u32     %3, %3, t30;\n\t"       \
            /*"addc.cc.u32    %4, %4, t40;\n\t"*/   \
            /*"addc.cc.u32    %5, %5, t50;\n\t"*/   \
            "addc.u32       %6, %6, t60;\n\t"       \
            /*Multiplication involving a3 second (low) diagonal*/ \
            "mul.lo.u32     t30, %17, %7;\n\t"      \
            /*"mul.lo.u32     t40, %17, %8;\n\t"*/  \
            /*"mul.lo.u32     t50, %17, %9;\n\t"*/  \
            "mul.lo.u32     t60, %17, %10;\n\t"     \
            "add.cc.u32     %3, %3, t30;\n\t"       \
            "addc.cc.u32    %4, %4, 0;\n\t"         \
            "addc.cc.u32    %5, %5, 0;\n\t"         \
            "addc.u32       %6, %6, t60;\n\t"       \
            /*Multiplication involving a3 first (high) diagonal*/ \
            "mul.hi.u32     t40, %17, %7;\n\t"      \
            /*"mul.hi.u32     t50, %17, %8;\n\t"*/  \
            /*"mul.hi.u32     t60, %17, %9;\n\t"*/  \
            "add.u32     %4, %4, t40;\n\t"          \
            /*"addc.cc.u32    %5, %5, t50;\n\t"*/   \
            /*"addc.u32       %6, %6, t60;\n\t"*/   \
            /*multiplication involving a4 second (low) diagonal*/ \
            "mul.lo.u32     t40, %18, %7;\n\t"      \
            /*"mul.lo.u32     t50, %18, %8;\n\t"*/  \
            /*"mul.lo.u32     t60, %18, %9;\n\t"*/  \
            "add.cc.u32     %4, %4, t40;\n\t"       \
            "addc.cc.u32    %5, %5, 0;\n\t"         \
            "addc.u32       %6, %6, 0;\n\t"         \
            /*multiplication involving a4 first (high) diagonal*/ \
            "mul.hi.u32     t50, %18, %7;\n\t"      \
            /*"mul.hi.u32     t60, %18, %8;\n\t"*/  \
            "add.u32     %5, %5, t50;\n\t"          \
            /*"addc.u32       %6, %6, t60;\n\t" */  \
            /*multiplication involving a5 second (low) diagonal*/ \
            "mul.lo.u32     t50, %19, %7;\n\t"      \
            /*"mul.lo.u32     t60, %19, %8;\n\t"*/  \
            "add.cc.u32     %5, %5, t50;\n\t"       \
            "addc.u32       %6, %6, 0;\n\t"         \
            /*multiplication involving a5 first (high) diagonal*/ \
            "mul.hi.u32     t60, %19, %7;\n\t"      \
            "add.u32        %6, %6, t60;\n\t"       \
            /*multiplication involving a6 second (low) diagonal*/ \
            "mul.lo.u32     t60, %20, %7;\n\t"      \
            "add.u32        %6, %6, t60;\n\t"       \
            "}\n"                                   \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6)  /*0 , 6*/ \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5),"r"(mu6), /*7 ,13*/ \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6)         /*14,20*/ \
        );

// adds c0c1c2c3 = a0a1a2+b0b1b2b3
// needed for the karatsuba multiplication 7x7
#define __karatsuba32_7x7_4add_helper(c0, c1, c2, c3, a0, a1, a2, b0, b1, b2, b3)\
    asm (    "{\n\t"                                                     \
            "add.cc.u32 	%0, %7, %4;\n\t"                             \
            "addc.cc.u32 	%1, %8, %5;\n\t"                             \
            "addc.cc.u32 	%2, %9, %6;\n\t"                             \
            "addc.u32       %3,%10,  0;\n\t"                             \
        "}\n\t"                                                          \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)/*0,3 */                     \
    : "r"(a0), "r"(a1), "r"(a2),            /*4,6 */                     \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3)    /*7,10*/                     \
    );

// adds c0c1c2c3 = a0a1a2+b0b1b2b3
// needed for the karatsuba multiplication 7x7
#define __karatsuba32_7x7_4add_helper_v2(c0, c1, c2, c3, c4, a0, a1, a2, b0, b1, b2, b3) \
    asm (    "{\n\t"                                                            \
            "add.cc.u32 	%0, %5, %8;\n\t"                                    \
            "addc.cc.u32 	%1, %6, %9;\n\t"                                    \
            "addc.cc.u32 	%2, %7,%10;\n\t"                                    \
            "addc.cc.u32    %3, 0, %11;\n\t"                                    \
            "addc.u32       %4, 0,   0;\n\t"                                    \
        "}\n\t"                                                                 \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4)  /*0,4 */                    \
    : "r"(a0), "r"(a1), "r"(a2),                    /*5,7 */                    \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3)            /*8,11*/                    \
    );

// karatsuba 7x7 32bit limb multiplication without reduction
// rt0-rt7 helper registers
// c13-c0 = a6-a0 * b6-b0
#define __karatsuba32_helper_7x7(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6, rt0, rt1, rt2, rt3, rt4, rt5, rt6, rt7) \
    __karatsuba32_7x7_4add_helper(c0,c1,c2,c3, a0,a1,a2, a3,a4,a5,a6)           \
    __karatsuba32_7x7_4add_helper(c4,c5,c6,c7, b0,b1,b2, b3,b4,b5,b6)           \
    __school32_4x4(rt0,rt1,rt2,rt3,rt4,rt5,rt6,rt7, c0,c1,c2,c3, c4,c5,c6,c7)   \
    __school32_3x3(c0,c1,c2,c3,c4,c5, a0,a1,a2, b0,b1,b2)                       \
    __school32_4x4(c6,c7,c8,c9,c10,c11,c12,c13, a3,a4,a5,a6, b3,b4,b5,b6)       \
    /*UsVs -= U1V1*/                                                            \
    asm volatile (    "{\n\t"                                                   \
            "sub.cc.u32  %0, %16, %8;\n\t"                                      \
            "subc.cc.u32 %1, %17, %9;\n\t"                                      \
            "subc.cc.u32 %2, %18, %10;\n\t"                                     \
            "subc.cc.u32 %3, %19, %11;\n\t"                                     \
            "subc.cc.u32 %4, %20, %12;\n\t"                                     \
            "subc.cc.u32 %5, %21, %13;\n\t"                                     \
            "subc.cc.u32 %6, %22, %14;\n\t"                                     \
            "subc.u32    %7, %23, %15;\n\t"                                     \
            "}\n\t"                                                             \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3),"=r"(rt4),"=r"(rt5),"=r"(rt6),"=r"(rt7) /*output 0 , 7*/\
        : "r"(c6),"r"(c7),"r"(c8),"r"(c9),"r"(c10),"r"(c11),"r"(c12),"r"(c13)              /*U1V1   8 ,15*/\
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),"r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7)          /*output 16,23*/\
    );                                                                          \
    /*UsVs -= U0V0*/                                                            \
    asm volatile (    "{\n\t"                                                   \
            "sub.cc.u32  %0, %14, %8;\n\t"                                      \
            "subc.cc.u32 %1, %15, %9;\n\t"                                      \
            "subc.cc.u32 %2, %16, %10;\n\t"                                     \
            "subc.cc.u32 %3, %17, %11;\n\t"                                     \
            "subc.cc.u32 %4, %18, %12;\n\t"                                     \
            "subc.cc.u32 %5, %19, %13;\n\t"                                     \
            "subc.cc.u32 %6, %20,   0;\n\t"                                     \
            "subc.u32    %7, %21,   0;\n\t"                                     \
            "}\n\t"                                                             \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3),"=r"(rt4),"=r"(rt5),"=r"(rt6),"=r"(rt7)   /*output 0 , 7*/    \
        : "r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(c4),"r"(c5),                                  /*U0V0   8, 13*/    \
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),"r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7)           /*output 14,21*/    \
    );                                                                      \
    /*Final addition and result is written */                               \
    asm volatile("{\n\t"                                                    \
            "add.cc.u32  %0, %8,  %16;\n\t"                                 \
            "addc.cc.u32 %1, %9,  %17;\n\t"                                 \
            "addc.cc.u32 %2, %10, %18;\n\t"                                 \
            "addc.cc.u32 %3, %11, %19;\n\t"                                 \
            "addc.cc.u32 %4, %12, %20;\n\t"                                 \
            "addc.cc.u32 %5, %13, %21;\n\t"                                 \
            "addc.cc.u32 %6, %14, %22;\n\t"                                 \
            "addc.u32    %7, %15, %23;\n\t"                                 \
            "}\n"                                                           \
        : "=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7),"=r"(c8),"=r"(c9),"=r"(c10)    /*0 , 7*/                    \
        : "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),"r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7),    /*8 ,15*/                    \
          "r"(c3),"r"(c4),"r"(c5),"r"(c6),"r"(c7),"r"(c8),"r"(c9),"r"(c10));          /*16,23*/

// 7x7 Karatsuba implementation. Result is a 14 limb value
// No mod reduction calculated
// c13c12c11c10c9c8c7c6c5c4c3c2c1c0 = a6a5a4a3a2a1a0 * b6b5b4b3b2b1b0
#define __karatsuba32_7x7(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6)    \
    uint32_t frt0,frt1,frt2,frt3,frt4,frt5,frt6,frt7;                    \
    __karatsuba32_helper_7x7(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, a0,a1,a2,a3,a4,a5,a6, b0,b1,b2,b3,b4,b5,b6,  frt0,frt1,frt2,frt3,frt4,frt5,frt6,frt7)

// wrapper around the mutliplication, so one can easily switch between schoolbook and karatsuba mult
#define __mul7x7_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6) \
    __karatsuba32_7x7(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, a0,a1,a2,a3,a4,a5,a6, b0,b1,b2,b3,b4,b5,b6)

// TODO not implemented
#define __sqr7x7_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0, a1, a2, a3, a4, a5, a6)
// TODO describe
#define __reduce32_sec7x7(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, p0, p1, p2, p3, p4, p5, p6, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13) \
        asm volatile ("{\n\t"                                   \
            /* %0-%6 = r = (a + u) div 2^(7*32)*/               \
             ".reg .u32		 t0,t7,t8,t9,t10,t11,t12,t13;\n\t"  \
             "add.cc.u32     %0, %28, %14;\n\t"                 \
             "addc.cc.u32    %1, %29, %15;\n\t"                 \
             "addc.cc.u32    %2, %30, %16;\n\t"                 \
             "addc.cc.u32    %3, %31, %17;\n\t"                 \
             "addc.cc.u32    %4, %32, %18;\n\t"                 \
             "addc.cc.u32    %5, %33, %19;\n\t"                 \
             "addc.cc.u32    %6, %34, %20;\n\t"                 \
             "addc.cc.u32    t7, %35, %21;\n\t"                 \
             "addc.cc.u32    t8, %36, %22;\n\t"                 \
             "addc.cc.u32    t9, %37, %23;\n\t"                 \
             "addc.cc.u32    t10,%38, %24;\n\t"                 \
             "addc.cc.u32    t11,%39, %25;\n\t"                 \
             "addc.cc.u32    t12,%40, %26;\n\t"                 \
             "addc.cc.u32    t13,%41, %27;\n\t"                 \
             "addc.u32       t0,   0,   0;\n\t"                 \
             "sub.cc.u32     t7, t7, %7;\n\t"                   \
             "subc.cc.u32    t8, t8, %8;\n\t"                   \
             "subc.cc.u32    t9, t9, %9;\n\t"                   \
             "subc.cc.u32    t10, t10, %10;\n\t"                \
             "subc.cc.u32    t11, t11, %11;\n\t"                \
             "subc.cc.u32    t12, t12, %12;\n\t"                \
             "subc.cc.u32    t13, t13, %13;\n\t"                \
             "subc.u32       t0, t0,  0;\n\t"                   \
             "and.b32        %0, t0, %7;\n\t"                   \
             "and.b32        %1, t0, %8;\n\t"                   \
             "and.b32        %2, t0, %9;\n\t"                   \
             "and.b32        %3, t0, %10;\n\t"                  \
             "and.b32        %4, t0, %11;\n\t"                  \
             "and.b32        %5, t0, %12;\n\t"                  \
             "and.b32        %6, t0, %13;\n\t"                  \
             "add.cc.u32     %0, %0, t7;\n\t"                   \
             "addc.cc.u32    %1, %1, t8;\n\t"                   \
             "addc.cc.u32    %2, %2, t9;\n\t"                   \
             "addc.cc.u32    %3, %3, t10;\n\t"                  \
             "addc.cc.u32    %4, %4, t11;\n\t"                  \
             "addc.cc.u32    %5, %5, t12;\n\t"                  \
             "addc.u32       %6, %6, t13;\n\t"                  \
        "}\n\t"                                                 \
        : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6)     /*0 - 6*/\
        : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5), "r"(p6),     /*7 -13*/\
          "r"(i0), "r"(i1), "r"(i2), "r"(i3), "r"(i4), "r"(i5), "r"(i6),     /*14-20*/\
          "r"(i7), "r"(i8), "r"(i9), "r"(i10), "r"(i11), "r"(i12), "r"(i13), /*21-27*/\
          "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6),     /*28-34*/\
          "r"(a7), "r"(a8), "r"(a9), "r"(a10), "r"(a11), "r"(a12),"r"(a13)     /*35-41*/\
        )

// TODO describe
#define __reduce32_inplace7x7(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, mu0, mu1, mu2, mu3, mu4, mu5, mu6, p0, p1, p2, p3, p4, p5, p6)    \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13;                                                                        \
    __school32_inplace_7x7_low_v2(a0,a1,a2,a3,a4,a5,a6, mu0,mu1,mu2,mu3,mu4,mu5,mu6)                                            \
    __school32_7x7(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13, a0,a1,a2,a3,a4,a5,a6, p0,p1,p2,p3,p4,p5,p6)                    \
    __reduce32_sec7x7(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13, p0,p1,p2,p3,p4,p5,t6, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13)

// TODO describe
#define __reduce32_7x7(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, mu0, mu1, mu2, mu3, mu4, mu5, mu6, p0, p1, p2, p3, p4, p5, p6)    \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13;                                                                                        \
    uint32_t m0,m1,m2,m3,m4,m5,m6;                                                                                                                \
    __school32_7x7_low(m0,m1,m2,m3,m4,m5,m6, a0,a1,a2,a3,a4,a5,a6, mu0,mu1,mu2,mu3,mu4,mu5,mu6)                                                \
    __school32_7x7(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13, m0,m1,m2,m3,m4,m5,m6, p0,p1,p2,p3,p4,p5,p6)                                    \
    __reduce32_sec7x7(c0,c1,c2,c3,c4,c5,c6, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13, p0,p1,p2,p3,p4,p5,p6, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13)

#define __fp_mul7x7_32(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6, mu0, mu1, mu2, mu3, mu4, mu5, mu6, p0, p1, p2, p3, p4, p5, p6)    \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13;                                                                                        \
    __karatsuba32_7x7(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13, a0,a1,a2,a3,a4,a5,a6, b0,b1,b2,b3,b4,b5,b6)                                \
    __reduce32_7x7(c0,c1,c2,c3,c4,c5,c6, f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13, mu0,mu1,mu2,mu3,mu4,mu5,mu6, p0,p1,p2,p3,p4,p5,p6)


#define __fp_mul7x7_32_v2(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6, mu0, mu1, mu2, mu3, mu4, mu5, mu6, p0, p1, p2, p3, p4, p5, p6)    \
    uint32_t f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13;                                                                                        \
    uint32_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13;                                                                                        \
    __school32_7x7(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13, a0,a1,a2,a3,a4,a5,a6, b0,b1,b2,b3,b4,b5,b6)                                    \
    __school32_7x7_low(c0,c1,c2,c3,c4,c5,c6, a0,a1,a2,a3,a4,a5,a6, mu0,mu1,mu2,mu3,mu4,mu5,mu6)                                                \
    __school32_7x7(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13, c0,c1,c2,c3,c4,c5,c6, p0,p1,p2,p3,p4,p5,p6)                                    \
    __reduce32_sec7x7(c0,c1,c2,c3,c4,c5,c6, f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13, p0,p1,p2,p3,p4,p5,p6, t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13)

/// not implemented
#define __fp_sqr7x7_32(c0, c1, c2, c3, c4, c5, c6, a0, a1, a2, a3, a4, a5, a6, mu0, mu1, mu2, mu3, mu4, mu5, mu6, p0, p1, p2, p3, p4, p5, p6)
////////////////////////////////////////////////////////////////////////////////
// 									8x8										  //
////////////////////////////////////////////////////////////////////////////////
// Simple 8x8 32bit limb school book Multiplication
// uses the expensive `mad` instruction
#define __school32_8x8(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)\
    asm volatile ("{\n\t"                                \
            ".reg .u32 t0;" ".reg .u32 t1;\n\t"            \
            ".reg .u32 t2;" ".reg .u32 t3;\n\t"            \
            ".reg .u32 t4;" ".reg .u32 t5;\n\t"            \
            ".reg .u32 t6;" ".reg .u32 t7;\n\t"            \
            ".reg .u32 tc;\n\t"                            \
            /*Multiplication involving a0*/                    \
            "mul.lo.u32     %0, %16, %24;\n\t"            \
            "mul.hi.u32     t0, %16, %24;\n\t"            \
            "mad.lo.cc.u32  t0, %16, %25, t0;\n\t"        \
            "mul.hi.u32     t1, %16, %25;\n\t"            \
            "madc.lo.cc.u32 t1, %16, %26, t1;\n\t"        \
            "mul.hi.u32     t2, %16, %26;\n\t"            \
            "madc.lo.cc.u32 t2, %16, %27, t2;\n\t"        \
            "mul.hi.u32     t3, %16, %27;\n\t"            \
            "madc.lo.cc.u32 t3, %16, %28, t3;\n\t"        \
            "mul.hi.u32     t4, %16, %28;\n\t"            \
            "madc.lo.cc.u32 t4, %16, %29, t4;\n\t"        \
            "mul.hi.u32     t5, %16, %29;\n\t"            \
            "madc.lo.cc.u32 t5, %16, %30, t5;\n\t"        \
            "mul.hi.u32     t6, %16, %30;\n\t"            \
            "madc.lo.cc.u32 t6, %16, %31, t6;\n\t"        \
            "mul.hi.u32     t7, %16, %31;\n\t"            \
            "addc.cc.u32    t7, t7, 0;\n\t"                \
            "addc.u32       tc, 0, 0;\n\t"                \
            /*Multiplication involving a1 first diagonal*/  \
            "mad.lo.cc.u32  %1, %17, %24, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %17, %24, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %17, %25, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %17, %26, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %17, %27, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %17, %28, t5;\n\t"            \
            "madc.hi.cc.u32 t6, %17, %29, t6;\n\t"            \
            "madc.hi.cc.u32 t7, %17, %30, t7;\n\t"            \
            "mul.hi.u32     t0, %17, %31;\n\t"                \
            "addc.cc.u32    t0, t0 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a1 second diagonal*/ \
            "mad.lo.cc.u32  t1, %17, %25, t1;\n\t"        \
            "madc.lo.cc.u32 t2, %17, %26, t2;\n\t"        \
            "madc.lo.cc.u32 t3, %17, %27, t3;\n\t"        \
            "madc.lo.cc.u32 t4, %17, %28, t4;\n\t"        \
            "madc.lo.cc.u32 t5, %17, %29, t5;\n\t"        \
            "madc.lo.cc.u32 t6, %17, %30, t6;\n\t"        \
            "madc.lo.cc.u32 t7, %17, %31, t7;\n\t"        \
            "addc.cc.u32    t0, t0 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a2 first diagonal*/  \
            "mad.lo.cc.u32  %2, %18, %24, t1;\n\t"        \
            "madc.hi.cc.u32 t2, %18, %24, t2;\n\t"        \
            "madc.hi.cc.u32 t3, %18, %25, t3;\n\t"        \
            "madc.hi.cc.u32 t4, %18, %26, t4;\n\t"        \
            "madc.hi.cc.u32 t5, %18, %27, t5;\n\t"        \
            "madc.hi.cc.u32 t6, %18, %28, t6;\n\t"        \
            "madc.hi.cc.u32 t7, %18, %29, t7;\n\t"        \
            "madc.hi.cc.u32 t0, %18, %30, t0;\n\t"        \
            "mul.hi.u32     t1, %18, %31;\n\t"            \
            "addc.cc.u32    t1, t1 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a2 second diagonal*/ \
            "mad.lo.cc.u32  t2, %18, %25, t2;\n\t"            \
            "madc.lo.cc.u32 t3, %18, %26, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %18, %27, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %18, %28, t5;\n\t"            \
            "madc.lo.cc.u32 t6, %18, %29, t6;\n\t"            \
            "madc.lo.cc.u32 t7, %18, %30, t7;\n\t"            \
            "madc.lo.cc.u32 t0, %18, %31, t0;\n\t"            \
            "addc.cc.u32    t1, t1 , 0;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a3 first diagonal*/  \
            "mad.lo.cc.u32  %3, %19, %24, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %19, %24, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %19, %25, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %19, %26, t5;\n\t"            \
            "madc.hi.cc.u32 t6, %19, %27, t6;\n\t"            \
            "madc.hi.cc.u32 t7, %19, %28, t7;\n\t"            \
            "madc.hi.cc.u32 t0, %19, %29, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %19, %30, t1;\n\t"            \
            "mul.hi.u32     t2, %19, %31;\n\t"                \
            "addc.cc.u32    t2, t2 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a3 second diagonal*/ \
            "mad.lo.cc.u32  t3, %19, %25, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %19, %26, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %19, %27, t5;\n\t"            \
            "madc.lo.cc.u32 t6, %19, %28, t6;\n\t"            \
            "madc.lo.cc.u32 t7, %19, %29, t7;\n\t"            \
            "madc.lo.cc.u32 t0, %19, %30, t0;\n\t"            \
            "madc.lo.cc.u32 t1, %19, %31, t1;\n\t"            \
            "addc.cc.u32    t2, t2 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a4 first diagonal*/  \
            "mad.lo.cc.u32  %4, %20, %24, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %20, %24, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %20, %25, t5;\n\t"            \
            "madc.hi.cc.u32 t6, %20, %26, t6;\n\t"            \
            "madc.hi.cc.u32 t7, %20, %27, t7;\n\t"            \
            "madc.hi.cc.u32 t0, %20, %28, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %20, %29, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %20, %30, t2;\n\t"            \
            "mul.hi.u32     t3, %20, %31;\n\t"                \
            "addc.cc.u32    t3, t3 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a4 second diagonal*/ \
            "mad.lo.cc.u32  t4, %20, %25, t4;\n\t"            \
            "madc.lo.cc.u32 t5, %20, %26, t5;\n\t"            \
            "madc.lo.cc.u32 t6, %20, %27, t6;\n\t"            \
            "madc.lo.cc.u32 t7, %20, %28, t7;\n\t"            \
            "madc.lo.cc.u32 t0, %20, %29, t0;\n\t"            \
            "madc.lo.cc.u32 t1, %20, %30, t1;\n\t"            \
            "madc.lo.cc.u32 t2, %20, %31, t2;\n\t"            \
            "addc.cc.u32    t3, t3 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a5 first diagonal*/  \
            "mad.lo.cc.u32  %5, %21, %24, t4;\n\t"            \
            "madc.hi.cc.u32 t5, %21, %24, t5;\n\t"            \
            "madc.hi.cc.u32 t6, %21, %25, t6;\n\t"            \
            "madc.hi.cc.u32 t7, %21, %26, t7;\n\t"            \
            "madc.hi.cc.u32 t0, %21, %27, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %21, %28, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %21, %29, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %21, %30, t3;\n\t"            \
            "mul.hi.u32     t4, %21, %31;\n\t"                \
            "addc.cc.u32    t4, t4 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a5 second diagonal*/ \
            "mad.lo.cc.u32  t5, %21, %25, t5;\n\t"            \
            "madc.lo.cc.u32 t6, %21, %26, t6;\n\t"            \
            "madc.lo.cc.u32 t7, %21, %27, t7;\n\t"            \
            "madc.lo.cc.u32 t0, %21, %28, t0;\n\t"            \
            "madc.lo.cc.u32 t1, %21, %29, t1;\n\t"            \
            "madc.lo.cc.u32 t2, %21, %30, t2;\n\t"            \
            "madc.lo.cc.u32 t3, %21, %31, t3;\n\t"            \
            "addc.cc.u32    t4, t4 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a6 first diagonal*/  \
            "mad.lo.cc.u32  %6, %22, %24, t5;\n\t"            \
            "madc.hi.cc.u32 t6, %22, %24, t6;\n\t"            \
            "madc.hi.cc.u32 t7, %22, %25, t7;\n\t"            \
            "madc.hi.cc.u32 t0, %22, %26, t0;\n\t"            \
            "madc.hi.cc.u32 t1, %22, %27, t1;\n\t"            \
            "madc.hi.cc.u32 t2, %22, %28, t2;\n\t"            \
            "madc.hi.cc.u32 t3, %22, %29, t3;\n\t"            \
            "madc.hi.cc.u32 t4, %22, %30, t4;\n\t"            \
            "mul.hi.u32     t5, %22, %31;\n\t"                \
            "addc.cc.u32    t5, t5 , tc;\n\t"                \
            "addc.u32       tc, 0 , 0;\n\t"                \
            /*Multiplication involving a6 second diagonal*/ \
            "mad.lo.cc.u32  t6, %22, %25, t6;\n\t"            \
            "madc.lo.cc.u32 t7, %22, %26, t7;\n\t"            \
            "madc.lo.cc.u32 t0, %22, %27, t0;\n\t"            \
            "madc.lo.cc.u32 t1, %22, %28, t1;\n\t"            \
            "madc.lo.cc.u32 t2, %22, %29, t2;\n\t"            \
            "madc.lo.cc.u32 t3, %22, %30, t3;\n\t"            \
            "madc.lo.cc.u32 t4, %22, %31, t4;\n\t"            \
            "addc.cc.u32    t5, t5 , tc;\n\t"                \
            "addc.u32       tc, tc , 0;\n\t"                \
            /*Multiplication involving a7 first diagonal*/  \
            "mad.lo.cc.u32  %7, %23, %24, t6;\n\t"          \
            "madc.hi.cc.u32 t7, %23, %24, t7;\n\t"          \
            "madc.hi.cc.u32 t0, %23, %25, t0;\n\t"          \
            "madc.hi.cc.u32 t1, %23, %26, t1;\n\t"          \
            "madc.hi.cc.u32 t2, %23, %27, t2;\n\t"          \
            "madc.hi.cc.u32 t3, %23, %28, t3;\n\t"          \
            "madc.hi.cc.u32 t4, %23, %29, t4;\n\t"          \
            "madc.hi.cc.u32 t5, %23, %30, t5;\n\t"          \
            "mul.hi.u32     t6, %23, %31;\n\t"              \
            "addc.cc.u32    t6, t6 , tc;\n\t"               \
            "addc.u32       tc, 0 , 0;\n\t"                 \
            /*Multiplication involving a7 second diagonal*/ \
            "mad.lo.cc.u32  %8,  %23, %25, t7;\n\t"         \
            "madc.lo.cc.u32 %9,  %23, %26, t0;\n\t"         \
            "madc.lo.cc.u32 %10, %23, %27, t1;\n\t"         \
            "madc.lo.cc.u32 %11, %23, %28, t2;\n\t"         \
            "madc.lo.cc.u32 %12, %23, %29, t3;\n\t"         \
            "madc.lo.cc.u32 %13, %23, %30, t4;\n\t"         \
            "madc.lo.cc.u32 %14, %23, %31, t5;\n\t"         \
            "addc.u32       %15, t6 , tc;\n\t"              \
            "}\n"                                           \
    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7)           /*0,  7*/ \
      "=r"(c8),"=r"(c9),"=r"(c10),"=r"(c11),"=r"(c12),"=r"(c13),"=r"(c14),"=r"(c15)     /*8, 15*/ \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),           /*16,23*/ \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7));          /*24,31*/

#define __mul256_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)\
    __school32_8x8(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7)

// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**256
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// inplace version
#define __school32_inplace_8x8_low(a0, a1, a2, a3, a4, a5, a6, a7, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7)\
    asm volatile ("{\n\t"                                           \
            ".reg .u32 t10,t20,t30,t40,t50,t60,t70;\n\t"            \
            /* t0-t5: m = a * mu mod 2^256 */                       \
            /* Multiplication involving a0 */                       \
            "mul.lo.u32     %0,  %16, %8;\n\t"                      \
            "mul.hi.u32     %1,  %16, %8;\n\t"                      \
            "mul.lo.u32     t10, %16, %9;\n\t"                      \
            "mul.hi.u32     %2,  %16, %9;\n\t"                      \
            "mul.lo.u32     t20, %16, %10;\n\t"                     \
            "mul.hi.u32     %3,  %16, %10;\n\t"                     \
            "mul.lo.u32     t30, %16, %11;\n\t"                     \
            "mul.hi.u32     %4,  %16, %11;\n\t"                     \
            "mul.lo.u32     t40, %16, %12;\n\t"                     \
            "mul.hi.u32     %5,  %16, %12;\n\t"                     \
            "mul.lo.u32     t50, %16, %13;\n\t"                     \
            "mul.hi.u32     %6,  %16, %13;\n\t"                     \
            "mul.lo.u32     t60, %16, %14;\n\t"                     \
            "mul.hi.u32     %7,  %16, %14;\n\t"                     \
            "mul.lo.u32     t70, %16, %15;\n\t"                     \
            /*now fixing the carry chain*/                          \
            "add.cc.u32     %1,  %1,  t10;\n\t"                     \
            "addc.cc.u32    %2,  %2,  t20;\n\t"                     \
            "addc.cc.u32    %3,  %3,  t30;\n\t"                     \
            "addc.cc.u32    %4,  %4,  t40;\n\t"                     \
            "addc.cc.u32    %5,  %5,  t50;\n\t"                     \
            "addc.cc.u32    %6,  %6,  t60;\n\t"                     \
            "addc.u32       %7,  %7,  t70;\n\t"                     \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mul.lo.u32     t10, %17, %8;\n\t"                      \
            "mul.lo.u32     t20, %17, %9;\n\t"                      \
            "mul.lo.u32     t30, %17, %10;\n\t"                     \
            "mul.lo.u32     t40, %17, %11;\n\t"                     \
            "mul.lo.u32     t50, %17, %12;\n\t"                     \
            "mul.lo.u32     t60, %17, %13;\n\t"                     \
            "mul.lo.u32     t70, %17, %14;\n\t"                     \
            "add.cc.u32     %1, %1, t10;\n\t"                       \
            "addc.cc.u32    %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mul.hi.u32     t20, %17, %8;\n\t"                      \
            "mul.hi.u32     t30, %17, %9;\n\t"                      \
            "mul.hi.u32     t40, %17, %10;\n\t"                     \
            "mul.hi.u32     t50, %17, %11;\n\t"                     \
            "mul.hi.u32     t60, %17, %12;\n\t"                     \
            "mul.hi.u32     t70, %17, %13;\n\t"                     \
            "add.cc.u32     %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mul.lo.u32     t20, %18, %8;\n\t"                      \
            "mul.lo.u32     t30, %18, %9;\n\t"                      \
            "mul.lo.u32     t40, %18, %10;\n\t"                     \
            "mul.lo.u32     t50, %18, %11;\n\t"                     \
            "mul.lo.u32     t60, %18, %12;\n\t"                     \
            "mul.lo.u32     t70, %18, %13;\n\t"                     \
            "add.cc.u32     %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a2 first (high) diagonal*/   \
            "mul.hi.u32     t30, %18, %8;\n\t"                      \
            "mul.hi.u32     t40, %18, %9;\n\t"                      \
            "mul.hi.u32     t50, %18, %10;\n\t"                     \
            "mul.hi.u32     t60, %18, %11;\n\t"                     \
            "mul.hi.u32     t70, %18, %12;\n\t"                     \
            "add.cc.u32     %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a3 second (low) diagonal*/   \
            "mul.lo.u32     t30, %19, %8;\n\t"                      \
            "mul.lo.u32     t40, %19, %9;\n\t"                      \
            "mul.lo.u32     t50, %19, %10;\n\t"                     \
            "mul.lo.u32     t60, %19, %11;\n\t"                     \
            "mul.lo.u32     t70, %19, %12;\n\t"                     \
            "add.cc.u32     %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a3 first (high) diagonal*/   \
            "mul.hi.u32     t40, %19, %8;\n\t"                      \
            "mul.hi.u32     t50, %19, %9;\n\t"                      \
            "mul.hi.u32     t60, %19, %10;\n\t"                     \
            "mul.hi.u32     t70, %19, %11;\n\t"                     \
            "add.cc.u32     %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a4 second (low) diagonal*/   \
            "mul.lo.u32     t40, %20, %8;\n\t"                      \
            "mul.lo.u32     t50, %20, %9;\n\t"                      \
            "mul.lo.u32     t60, %20, %10;\n\t"                     \
            "mul.lo.u32     t70, %20, %11;\n\t"                     \
            "add.cc.u32     %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a4 first (high) diagonal*/   \
            "mul.hi.u32     t40, %20, %8;\n\t"                      \
            "mul.hi.u32     t50, %20, %9;\n\t"                      \
            "mul.hi.u32     t60, %20, %10;\n\t"                     \
            "add.cc.u32     %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a5 second (low) diagonal*/   \
            "mul.lo.u32     t50, %21, %8;\n\t"                      \
            "mul.lo.u32     t60, %21, %9;\n\t"                      \
            "mul.lo.u32     t70, %21, %10;\n\t"                     \
            "add.cc.u32     %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a5 first (high) diagonal*/   \
            "mul.hi.u32     t60, %21, %8;\n\t"                      \
            "mul.hi.u32     t70, %21, %9;\n\t"                      \
            "add.cc.u32     %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a6 second (low) diagonal*/   \
            "mul.lo.u32     t60, %22, %8;\n\t"                      \
            "mul.lo.u32     t70, %22, %9;\n\t"                      \
            "add.cc.u32     %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a6 first (high) diagonal*/   \
            "mul.hi.u32     t70, %22, %8;\n\t"                      \
            "add.u32        %7, %7, t70;\n\t"                       \
            /*multiplication involving a7 second (low) diagonal*/   \
            "mul.lo.u32     t70, %23, %8;\n\t"                      \
            "add.u32       	%7, %7, t70;\n\t"                       \
            "}\n"                                                   \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),"=r"(a5),"=r"(a6),"=r"(a7)  /*0 , 7*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5),"r"(mu6),"r"(mu7), /*8 ,15*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6),"r"(a7)          /*16,23*/    \
        );


// first part of montgomery reduction
// essential its a multiplication a*mu mod 2**256
// This version does not use the 'mad' instructions, and does it save each
// 'addition' lane in seperate tmp registers.
// Non inplace version
#define __school32_8x8_low(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7)\
    asm volatile ("{\n\t"                                           \
            ".reg .u32 t10,t20,t30,t40,t50,t60,t70;\n\t"            \
            /* t0-t7: m = a * mu mod 2^256 */                       \
            /* Multiplication involving a0 */                       \
            "mul.lo.u32     %0,  %16, %8;\n\t"                      \
            "mul.hi.u32     %1,  %16, %8;\n\t"                      \
            "mul.lo.u32     t10, %16, %9;\n\t"                      \
            "mul.hi.u32     %2,  %16, %9;\n\t"                      \
            "mul.lo.u32     t20, %16, %10;\n\t"                     \
            "mul.hi.u32     %3,  %16, %10;\n\t"                     \
            "mul.lo.u32     t30, %16, %11;\n\t"                     \
            "mul.hi.u32     %4,  %16, %11;\n\t"                     \
            "mul.lo.u32     t40, %16, %12;\n\t"                     \
            "mul.hi.u32     %5,  %16, %12;\n\t"                     \
            "mul.lo.u32     t50, %16, %13;\n\t"                     \
            "mul.hi.u32     %6,  %16, %13;\n\t"                     \
            "mul.lo.u32     t60, %16, %14;\n\t"                     \
            "mul.hi.u32     %7,  %16, %14;\n\t"                     \
            "mul.lo.u32     t70, %16, %15;\n\t"                     \
            /*now fixing the carry chain*/                          \
            "add.cc.u32     %1,  %1,  t10;\n\t"                     \
            "addc.cc.u32    %2,  %2,  t20;\n\t"                     \
            "addc.cc.u32    %3,  %3,  t30;\n\t"                     \
            "addc.cc.u32    %4,  %4,  t40;\n\t"                     \
            "addc.cc.u32    %5,  %5,  t50;\n\t"                     \
            "addc.cc.u32    %6,  %6,  t60;\n\t"                     \
            "addc.u32       %7,  %7,  t70;\n\t"                     \
            /*Multiplication involving a1 second (low) diagonal*/   \
            "mul.lo.u32     t10, %17, %8;\n\t"                      \
            "mul.lo.u32     t20, %17, %9;\n\t"                      \
            "mul.lo.u32     t30, %17, %10;\n\t"                     \
            "mul.lo.u32     t40, %17, %11;\n\t"                     \
            "mul.lo.u32     t50, %17, %12;\n\t"                     \
            "mul.lo.u32     t60, %17, %13;\n\t"                     \
            "mul.lo.u32     t70, %17, %14;\n\t"                     \
            "add.cc.u32     %1, %1, t10;\n\t"                       \
            "addc.cc.u32    %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a1 first (high) diagonal*/   \
            "mul.hi.u32     t20, %17, %8;\n\t"                      \
            "mul.hi.u32     t30, %17, %9;\n\t"                      \
            "mul.hi.u32     t40, %17, %10;\n\t"                     \
            "mul.hi.u32     t50, %17, %11;\n\t"                     \
            "mul.hi.u32     t60, %17, %12;\n\t"                     \
            "mul.hi.u32     t70, %17, %13;\n\t"                     \
            "add.cc.u32     %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a2 second (low) diagonal*/   \
            "mul.lo.u32     t20, %18, %8;\n\t"                      \
            "mul.lo.u32     t30, %18, %9;\n\t"                      \
            "mul.lo.u32     t40, %18, %10;\n\t"                     \
            "mul.lo.u32     t50, %18, %11;\n\t"                     \
            "mul.lo.u32     t60, %18, %12;\n\t"                     \
            "mul.lo.u32     t70, %18, %13;\n\t"                     \
            "add.cc.u32     %2, %2, t20;\n\t"                       \
            "addc.cc.u32    %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a2 first (high) diagonal*/   \
            "mul.hi.u32     t30, %18, %8;\n\t"                      \
            "mul.hi.u32     t40, %18, %9;\n\t"                      \
            "mul.hi.u32     t50, %18, %10;\n\t"                     \
            "mul.hi.u32     t60, %18, %11;\n\t"                     \
            "mul.hi.u32     t70, %18, %12;\n\t"                     \
            "add.cc.u32     %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a3 second (low) diagonal*/   \
            "mul.lo.u32     t30, %19, %8;\n\t"                      \
            "mul.lo.u32     t40, %19, %9;\n\t"                      \
            "mul.lo.u32     t50, %19, %10;\n\t"                     \
            "mul.lo.u32     t60, %19, %11;\n\t"                     \
            "mul.lo.u32     t70, %19, %12;\n\t"                     \
            "add.cc.u32     %3, %3, t30;\n\t"                       \
            "addc.cc.u32    %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*Multiplication involving a3 first (high) diagonal*/ \
            "mul.hi.u32     t40, %19, %8;\n\t"                      \
            "mul.hi.u32     t50, %19, %9;\n\t"                      \
            "mul.hi.u32     t60, %19, %10;\n\t"                     \
            "mul.hi.u32     t70, %19, %11;\n\t"                     \
            "add.cc.u32     %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a4 second (low) diagonal*/   \
            "mul.lo.u32     t40, %20, %8;\n\t"                      \
            "mul.lo.u32     t50, %20, %9;\n\t"                      \
            "mul.lo.u32     t60, %20, %10;\n\t"                     \
            "mul.lo.u32     t70, %20, %11;\n\t"                     \
            "add.cc.u32     %4, %4, t40;\n\t"                       \
            "addc.cc.u32    %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a4 first (high) diagonal*/   \
            "mul.hi.u32     t50, %20, %8;\n\t"                      \
            "mul.hi.u32     t60, %20, %9;\n\t"                      \
            "mul.hi.u32     t70, %20, %10;\n\t"                     \
            "add.cc.u32     %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a5 second (low) diagonal*/   \
            "mul.lo.u32     t50, %21, %8;\n\t"                      \
            "mul.lo.u32     t60, %21, %9;\n\t"                      \
            "mul.lo.u32     t70, %21, %10;\n\t"                     \
            "add.cc.u32     %5, %5, t50;\n\t"                       \
            "addc.cc.u32    %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a5 first (high) diagonal*/   \
            "mul.hi.u32     t60, %21, %8;\n\t"                      \
            "mul.hi.u32     t70, %21, %9;\n\t"                      \
            "add.cc.u32     %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a6 second (low) diagonal*/   \
            "mul.lo.u32     t60, %22, %8;\n\t"                      \
            "mul.lo.u32     t70, %22, %9;\n\t"                      \
            "add.cc.u32     %6, %6, t60;\n\t"                       \
            "addc.u32       %7, %7, t70;\n\t"                       \
            /*multiplication involving a6 first (high) diagonal*/   \
            "mul.hi.u32     t70, %22, %8;\n\t"                      \
            "add.u32        %7, %7, t70;\n\t"                       \
            /*multiplication involving a7 second (low) diagonal*/   \
            "mul.lo.u32     t70, %23, %8;\n\t"                      \
            "add.u32       	%7, %7, t70;\n\t"                       \
            "}\n"                                                   \
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7)  /*0 , 7*/    \
            : "r"(mu0),"r"(mu1),"r"(mu2),"r"(mu3),"r"(mu4),"r"(mu5),"r"(mu6),"r"(mu7), /*8 ,15*/    \
              "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(a4),"r"(a5),"r"(a6),"r"(a7)          /*16,23*/    \
        );

// c = a+b, where a and b are four limbs wide
#define __karatsuba32_8x8_4add_helper(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3)   \
    asm volatile("{\n\t"                                                                \
            "add.cc.u32  %0, %4, %8 ;\n\t"                                              \
            "addc.cc.u32 %1, %5, %9 ;\n\t"                                              \
            "addc.cc.u32 %2, %6, %10;\n\t"                                              \
            "addc.u32 %3, %7, %11;\n\t"                                                 \
        "}\n"                                                                           \
    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)    /*0,3*/                                 \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),        /*4,7*/                                \
      "r"(b0), "r"(b1), "r"(b2), "r"(b3));        /*8,11*/

// karatsuba 8x8 multiplication without reduction
// c15-c0 = a7-a0 * b7-b0
#define __karatsuba32_8x8(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)\
    uint32_t rt0,rt1,rt2,rt3,rt4,rt5,rt6,rt7;                               \
    __karatsuba32_8x8_4add_helper(c0,c1,c2,c3, a0,a1,a2,a3, a4,a5,a6,a7)    \
    __karatsuba32_8x8_4add_helper(c4,c5,c6,c7, b0,b1,b2,b3, b4,b5,b6,b7)    \
    __school32_4x4(rt0,rt1,rt2,rt3,rt4,rt5,rt6,rt7, c0,c1,c2,c3,c4,c5,c6,c7)\
    __school32_4x4(c0,c1,c2,c3,c4,c5,c6,c7,       a0,a1,a2,a3, b0,b1,b2,b3) \
    __school32_4x4(c8,c9,c10,c11,c12,c13,c14,c15, a4,a5,a6,a7, b4,b5,b6,b7) \
    asm volatile("{\n\t"                                                    \
            /*UsVs - U1V1 - U0V0*/                                          \
            "sub.cc.u32  %0, %24, %8 ;\n\t"                                 \
            "subc.cc.u32 %1, %25, %9 ;\n\t"                                 \
            "subc.cc.u32 %2, %26, %10;\n\t"                                 \
            "subc.cc.u32 %3, %27, %11;\n\t"                                 \
            "subc.cc.u32 %4, %28, %12;\n\t"                                 \
            "subc.cc.u32 %5, %29, %13;\n\t"                                 \
            "subc.cc.u32 %6, %30, %14;\n\t"                                 \
            "subc.u32    %7, %31, %15;\n\t"                                 \
            "sub.cc.u32  %0, %0, %16;\n\t"                                  \
            "subc.cc.u32 %1, %1, %17;\n\t"                                  \
            "subc.cc.u32 %2, %2, %18;\n\t"                                  \
            "subc.cc.u32 %3, %3, %19;\n\t"                                  \
            "subc.cc.u32 %4, %4, %20;\n\t"                                  \
            "subc.cc.u32 %5, %5, %21;\n\t"                                  \
            "subc.cc.u32 %6, %6, %22;\n\t"                                  \
            "subc.u32    %7, %7, %23;\n\t"                                  \
            "}\n"                                                           \
        : "=r"(rt0),"=r"(rt1),"=r"(rt2),"=r"(rt3),"=r"(rt4),"=r"(rt5),"=r"(rt6),"=r"(rt7) /*output 0, 7 */    \
        : "r"(c8),"r"(c9),"r"(c10),"r"(c11),"r"(c12),"r"(c13),"r"(c14),"r"(c15),          /*U1V1   8, 15*/    \
          "r"(c0),"r"(c1),"r"(c2),"r"(c3),"r"(c4),"r"(c5),"r"(c6),"r"(c7),                /*U0V0   16,23*/    \
          "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),"r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7)          /*output 24,31*/   \
    );                                                                 \
    /*Final addition and result is written */                          \
    asm volatile("{\n\t"                                               \
            "add.cc.u32  %0, %16, %8 ;\n\t"                            \
            "addc.cc.u32 %1, %17, %9 ;\n\t"                            \
            "addc.cc.u32 %2, %18, %10;\n\t"                            \
            "addc.cc.u32 %3, %19, %11;\n\t"                            \
            "addc.cc.u32 %4, %20, %12;\n\t"                            \
            "addc.cc.u32 %5, %21, %13;\n\t"                            \
            "addc.cc.u32 %6, %22, %14;\n\t"                            \
            "addc.u32    %7, %23, %15 ;\n\t"                           \
        "}\n"                                                          \
    : "=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7),"=r"(c8),"=r"(c9),"=r"(c10),"=r"(c11) /*0, 7 */ \
    : "r"(rt0),"r"(rt1),"r"(rt2),"r"(rt3),"r"(rt4),"r"(rt5),"r"(rt6),"r"(rt7)   /*8 ,15*/ \
      "r"(c4),"r"(c5),"r"(c6),"r"(c7),"r"(c8),"r"(c9),"r"(c10),"r"(c11));       /*16,23*/

#define __mul8x8_32(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)\
    __karatsuba32_8x8(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7)

// TODO not implemented
#define __fp_mul8x8_32(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, p0, p1, p2, p3, p4, p5, p6, p7)

#define __fp_mul8x8_32_v2(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, p0, p1, p2, p3, p4, p5, p6, p7)


#define __school64_2x2(c0, c1, c2, c3, a0, a1, b0, b1)      \
    asm volatile ("{\n\t"                                   \
            ".reg .u64 t0,t1,tc;\n\t"                       \
            /*Multiplication involving a0*/                 \
            "mul.lo.u64     %0, %4, %6;\n\t"                \
            "mul.hi.u64     t0, %4, %6;\n\t"                \
            "mul.lo.u64  	%1,%4, %7;\n\t"                 \
            "mul.hi.u64     t1, %4, %7;\n\t"                \
            "add.cc.u64		t0, t0, %1;\n\t"                \
            "addc.cc.u64    t1, t1, 0;\n\t"                 \
            "addc.u64       tc, 0, 0;\n\t"                  \
            /*Multiplication involving a1 first diagonal*/  \
            "mul.lo.u64  	%1, %5, %6;\n\t"                \
            "mul.hi.u64 	%2, %5, %6;\n\t"                \
            "mul.hi.u64     %3, %5, %7;\n\t"                \
            "add.cc.u64		%1, %1, t0;\n\t"                \
            "addc.cc.u64	t1, t1, %2;\n\t"                \
            "addc.u64    	t0, %3, tc;\n\t"                \
            /*Multiplication involving a1 second diagonal*/ \
            "mul.lo.u64		%3, %5, %7;\n\t"                \
            "add.cc.u64		%2, %3, t1;\n\t"                \
            "addc.u64       %3, t0 , 0;\n\t"                \
            "}\n"                                           \
    : "=l"(c0), "=l"(c1),"=l"(c2), "=l"(c3)    /*0,  3*/    \
    : "l"(a0), "l"(a1),                        /*4 , 5*/    \
      "l"(b0), "l"(b1));                       /*6 , 7*/

#define __school64_2x2_v2(c0, c1, c2, c3, a0, a1, b0, b1)                   \
    asm ("{\n\t"                                                            \
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"                    \
         ".reg .u32 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"                    \
         "mov.b64         {a0,a1}, %4;\n\t"                                 \
         "mov.b64         {a2,a3}, %5;\n\t"                                 \
         "mov.b64         {b0,b1}, %6;\n\t"                                 \
         "mov.b64         {b2,b3}, %7;\n\t"                                 \
         "mul.lo.u32      r0, a0, b0;\n\t"                                  \
         "mul.hi.u32      r1, a0, b0;\n\t"                                  \
         "mad.lo.cc.u32   r1, a0, b1, r1;\n\t"                              \
         "madc.hi.u32     r2, a0, b1, 0;\n\t"                               \
         "mad.lo.cc.u32   r1, a1, b0, r1;\n\t"                              \
         "madc.hi.cc.u32  r2, a1, b0, r2;\n\t"                              \
         "madc.hi.u32     r3, a0, b2, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a0, b2, r2;\n\t"                              \
         "madc.hi.cc.u32  r3, a1, b1, r3;\n\t"                              \
         "madc.hi.u32     r4, a0, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a1, b1, r2;\n\t"                              \
         "madc.hi.cc.u32  r3, a2, b0, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a1, b2, r4;\n\t"                              \
         "madc.hi.u32     r5, a1, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r2, a2, b0, r2;\n\t"                              \
         "madc.lo.cc.u32  r3, a0, b3, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a2, b1, r4;\n\t"                              \
         "madc.hi.cc.u32  r5, a2, b2, r5;\n\t"                              \
         "madc.hi.u32     r6, a2, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r3, a1, b2, r3;\n\t"                              \
         "madc.hi.cc.u32  r4, a3, b0, r4;\n\t"                              \
         "madc.hi.cc.u32  r5, a3, b1, r5;\n\t"                              \
         "madc.hi.cc.u32  r6, a3, b2, r6;\n\t"                              \
         "madc.hi.u32     r7, a3, b3, 0;\n\t"                               \
         "mad.lo.cc.u32   r3, a2, b1, r3;\n\t"                              \
         "madc.lo.cc.u32  r4, a1, b3, r4;\n\t"                              \
         "madc.lo.cc.u32  r5, a2, b3, r5;\n\t"                              \
         "madc.lo.cc.u32  r6, a3, b3, r6;\n\t"                              \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mad.lo.cc.u32   r3, a3, b0, r3;\n\t"                              \
         "madc.lo.cc.u32  r4, a2, b2, r4;\n\t"                              \
         "madc.lo.cc.u32  r5, a3, b2, r5;\n\t"                              \
         "addc.cc.u32     r6, r6, 0;\n\t"                                   \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mad.lo.cc.u32   r4, a3, b1, r4;\n\t"                              \
         "addc.cc.u32     r5, r5, 0;\n\t"                                   \
         "addc.cc.u32     r6, r6, 0;\n\t"                                   \
         "addc.u32        r7, r7, 0;\n\t"                                   \
         "mov.b64         %0, {r0,r1};\n\t"                                 \
         "mov.b64         %1, {r2,r3};\n\t"                                 \
         "mov.b64         %2, {r4,r5};\n\t"                                 \
         "mov.b64         %3, {r6,r7};\n\t"                                 \
         "}"                                                                \
         : "=l"(c0), "=l"(c1), "=l"(c2), "=l"(c3)                           \
         : "l"(b0), "l"(b1),                                                \
           "l"(c0), "l"(c1));
