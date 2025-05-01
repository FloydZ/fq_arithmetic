#!/usr/bin/env python3
""" generator script for finite field matrix and vector arithmetic """

from math import log2, ceil

class Vector:
    """
    Vector generation class
    Assumes the existance of the following functions:
        - z = gf256_add_u256(x, y)
        - z = gf256_mul_u256(x, y)
        - TODO gf_q extensions to gf_{q^mu}
    """

    # bit size of a basic register/limb
    BASE_RADIX = 64

    def __init__(self, q: int,
                 mu: int,
                 n: int,
                 simd_width: int = 32,
                 padding=False) -> None:
        """
        :param q: base prime
        :param mu: extension degree
        :param n: length of the vector
        :param simd_width: TODO not sure whats correct here. Either:
            - u8x32 as a enum (u8x16 for neon)
            - 256: simply as a number (128 for  neon)
        :param padding: if true the script assumes that the allocated length of
            the vector is a multiple (in terms of gf elements) of `simd_width`.
        """
        self.q, self.mu, self.n, self.simd_width, self.padding =\
            q, mu, n, simd_width, padding
        self.tail = not padding

        # TODO: well, actually thatsnot 100% correct. It could be that only 
        # 7 bits are used, so technically we need to round here
        self.bits_q = ceil(log2(q)) 
        self.bits_q_mu = ceil(log2(q**mu))
        self.q_per_limb = Vector.BASE_RADIX/self.bits_q_mu
        self.q_per_simd = self.simd_width/self.bits_q_mu

        tmp = (((n*self.bits_q_mu + self.simd_width - 1) // self.simd_width)) * self.q_per_simd
        self.internal_n = n if not padding else tmp
        #self.bits = self.
        #self.limbs = 

    def gen_add(self) -> str:
        """
        """
        return "TODO"


class Matrix(Vector):
    def __init__(self, q: int,
                 mu: int, 
                 n: int,
                 m: int,
                 k: int,
                 simd_width: int = 32,
                 col_major=True,
                 padding=False) -> None:
        """
             k           m           k    
         ┌───────┐   ┌───────┐   ┌───────┐
         │       │   │       │   │       │
        n│   C   │ = │   A   │n m│   B   │
         │       │   │       │   │       │
         └───────┘   └───────┘   └───────┘

        :param q: base prime
        :param mu: extension degree
        :param n: number of rows in A
        :param m: number of cols in A
        :param k: number of cols in B
        :param simd_width: TODO 
        :param col_major: if true, the matrices are column major else row major
        :param padding: if true the script assumes that the allocated length of
        each col (or row if row major) is a multiple (in terms of gf elements)
        of `simd_width`.
        """
        pass


if __name__ == "__main__":
    v = Vector(2, 1, 100, 32)
    print(v.gen_add())
