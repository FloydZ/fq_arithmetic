#!/usr/bin/env sage
""" script generating the 2**12 multiplication lookup table """


F = GF(2**12, 'x')
F.inject_variables()
f = F.modulus()

def toBin(t):
    return "0b"+''.join([str(a) for a in t.list()][::-1])
