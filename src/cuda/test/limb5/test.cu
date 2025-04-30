#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>


#include "test_vecs.cuh"
#include "../../src/arith.cuh"


static inline int compare(const limb_t *a, 
		const limb_t *b, 
		const uint32_t len=LIMBS) {     
	for (uint32_t i = 0; i < len; i++) {
		if (a[i] != b[i]) {
			return 1; 
		}	
	}

	return 0;
}

__global__ void add_fp(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[LIMBS] = {0};

    __fp_add160_32(c[0], c[1], c[2], c[3], c[4],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  b[i][0], b[i][1], b[i][2], b[i][3], b[i][4],
            	  __p[0], __p[1], __p[2], __p[3], __p[4]);

    for (uint32_t j = 0; j < LIMBS; j++) {
		out[i*LIMBS + j] = c[j];
	}
}

__global__ void sub_fp(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[LIMBS] = {0};

    __fp_sub160_32(c[0], c[1], c[2], c[3], c[4],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  b[i][0], b[i][1], b[i][2], b[i][3], b[i][4],
            	  __p[0], __p[1], __p[2], __p[3], __p[4]);

    for (uint32_t j = 0; j < LIMBS; j++) {
		out[i*LIMBS + j] = c[j];
	}
}

__global__ void schoolbookmul(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[2*LIMBS] = {0};

    __school32_5x5(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  b[i][0], b[i][1], b[i][2], b[i][3], b[i][4]);

    for (uint32_t j = 0; j < 2*LIMBS; j++) {
		out[2*i*LIMBS + j] = c[j];
	}
}

__global__ void mul(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[2*LIMBS] = {0};

    __mul160_32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  b[i][0], b[i][1], b[i][2], b[i][3], b[i][4]);

    for (uint32_t j = 0; j < 2*LIMBS; j++) {
		out[2*i*LIMBS + j] = c[j];
	}
}

__global__ void schoolbookmullow(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[LIMBS] = {0};

    __school32_5x5_low(c[0], c[1], c[2], c[3], c[4],
				  	   a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  	   b[i][0], b[i][1], b[i][2], b[i][3], b[i][4]);

    for (uint32_t j = 0; j < LIMBS; j++) {
		out[i*LIMBS + j] = c[j];
	}
}

__global__ void mul_fp(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[LIMBS] = {0};

    __fp_mul5x5_32(c[0], c[1], c[2], c[3], c[4],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4],
				  b[i][0], b[i][1], b[i][2], b[i][3], b[i][4],
				  __mu[0], __mu[1], __mu[2], __mu[3], __mu[4],
            	  __p[0], __p[1], __p[2], __p[3], __p[4]);

    for (uint32_t j = 0; j < LIMBS; j++) {
		out[i*LIMBS + j] = c[j];
	}
}


__global__ void sqr_fp(limb_t *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    limb_t c[2*LIMBS] = {0};
   
	__sqr160_32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
				  a[i][0], a[i][1], a[i][2], a[i][3], a[i][4])
    
	for (uint32_t j = 0; j < 2*LIMBS; j++) {
		out[2*i*LIMBS + j] = c[j];
	}

}


void test_add_fp() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing Addition in Fp: ");
    cudaMallocManaged(&c, TESTS * LIMBS * sizeof(limb_t));

    add_fp<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[i*LIMBS], add_fp_ok[i])) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[i*LIMBS + 4], c[i*LIMBS + 3], c[i*LIMBS + 2], c[i*LIMBS + 1], c[i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", add_fp_ok[i][4], add_fp_ok[i][3], add_fp_ok[i][2], add_fp_ok[i][1], add_fp_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_sub_fp() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing Subtraction in Fp: ");
    cudaMallocManaged(&c, TESTS * LIMBS * sizeof(limb_t));

    sub_fp<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[i*LIMBS], sub_fp_ok[i])) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[i*LIMBS + 4], c[i*LIMBS + 3], c[i*LIMBS + 2], c[i*LIMBS + 1], c[i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", sub_fp_ok[i][4], sub_fp_ok[i][3], sub_fp_ok[i][2], sub_fp_ok[i][1], sub_fp_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_mul() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing Multiplication: ");
    cudaMallocManaged(&c, 2 * TESTS * LIMBS * sizeof(limb_t));

    mul<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[2*i*LIMBS], mul_ok[i], 2*LIMBS)) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[2*i*LIMBS + 9], c[2*i*LIMBS + 8], c[2*i*LIMBS + 7], c[2*i*LIMBS + 6], c[2*i*LIMBS + 5], c[2*i*LIMBS + 4], c[2*i*LIMBS + 3], c[2*i*LIMBS + 2], c[2*i*LIMBS + 1], c[2*i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX\n", mul_ok[i][9], mul_ok[i][8], mul_ok[i][7], mul_ok[i][6], mul_ok[i][5], mul_ok[i][4], mul_ok[i][3], mul_ok[i][2], mul_ok[i][1], mul_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_schoolbookmullow() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing SchoolBook Multiplication Low: ");
    cudaMallocManaged(&c, TESTS * LIMBS * sizeof(limb_t));

    schoolbookmullow<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[i*LIMBS], mul_ok[i])) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[i*LIMBS + 4],  c[i*LIMBS + 3], c[i*LIMBS + 2], c[i*LIMBS + 1], c[i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", mul_ok[i][4], mul_ok[i][3], mul_ok[i][2], mul_ok[i][1], mul_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_schoolbookmul() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing SchoolBook Multiplication: ");
    cudaMallocManaged(&c, 2 * TESTS * LIMBS * sizeof(limb_t));

    schoolbookmul<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[2*i*LIMBS], mul_ok[i], 2*LIMBS)) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[2*i*LIMBS + 9], c[2*i*LIMBS + 8], c[2*i*LIMBS + 7], c[2*i*LIMBS + 6], c[2*i*LIMBS + 5], c[2*i*LIMBS + 4], c[2*i*LIMBS + 3], c[2*i*LIMBS + 2], c[2*i*LIMBS + 1], c[2*i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX %.8lX\n", mul_ok[i][9], mul_ok[i][8], mul_ok[i][7], mul_ok[i][6], mul_ok[i][5], mul_ok[i][4], mul_ok[i][3], mul_ok[i][2], mul_ok[i][1], mul_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_mul_fp() {
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing Multiplication in Fp: ");
    cudaMallocManaged(&c, TESTS * LIMBS * sizeof(limb_t));

    mul_fp<<<blocks, threads>>>(c);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < TESTS; i++) {
        if (compare(&c[i*LIMBS], mul_fp_ok[i])) {
            printf("(%i) FAIL\n", i);
            printf("(%d) 0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", i, c[i*LIMBS + 4], c[i*LIMBS + 3], c[i*LIMBS + 2], c[i*LIMBS + 1], c[i*LIMBS + 0]);
            printf("ok  0x%.8lX %.8lX %.8lX %.8lX %.8lX\n", mul_fp_ok[i][4], mul_fp_ok[i][3], mul_fp_ok[i][2], mul_fp_ok[i][1], mul_fp_ok[i][0]);
            return;
        }
    }
    printf("OK\n");

    cudaFree(c);
}

void test_sqr_fp() {
    int i;
    uint32_t blocks = 1;
    uint32_t threads = TESTS;
    limb_t *c;

    if (threads > 1024) {
        while (threads <= 1024) {
            threads >>= 1;
            blocks += 1;
        }
    }

    printf("Testing Square in Fp: ");
    cudaMallocManaged(&c, TESTS * sizeof(limb_t));

    sqr_fp<<<blocks, threads>>>(c);

    cudaDeviceSynchronize();

    for (i = 0; i < TESTS; i++) {
        if (compare(&c[i], sqr_fp_ok[i])) {
            printf("(%i) FAIL\n", i);
            /*
            printf("(%d) 0x%.16lX%.16lX%.16lX\n", i, c[2],c[1],c[0]);
            printf("ok 0x%.16lX%.16lX%.16lX\n",
                    mul_fp_ok[i][2],mul_fp_ok[i][1],mul_fp_ok[i][0]);
            */
            return;
        }
    }

    printf("OK\n");
    cudaFree(c);
}

void test_arith() {
    test_add_fp();
    test_sub_fp();
	test_mul();
	test_schoolbookmullow();
	test_schoolbookmul();
    test_mul_fp();
    //test_sqr_fp();
}


int main() {
	test_arith();
}
