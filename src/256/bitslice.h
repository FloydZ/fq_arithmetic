#define XOR(a, b) ((a) ^ (b))
#define AND(a, b) ((a) & (b))

static uint64_t mask_l[6] = {
    0xaaaaaaaaaaaaaaaaUL,
    0xccccccccccccccccUL,
    0xf0f0f0f0f0f0f0f0UL,
    0xff00ff00ff00ff00UL,
    0xffff0000ffff0000UL,
    0xffffffff00000000UL
};

static uint64_t mask_r[6] = {
    0x5555555555555555UL,
    0x3333333333333333UL,
    0x0f0f0f0f0f0f0f0fUL,
    0x00ff00ff00ff00ffUL,
    0x0000ffff0000ffffUL,
    0x00000000ffffffffUL
};


void real_ortho(uint64_t data[]) {
    for (uint64_t i = 0; i < 6; i ++) {
        uint64_t n = (1UL << i);
        for (uint64_t j = 0; j < 64; j += (2 * n)) {
            for (uint64_t k = 0; k < n; k++) {
                uint64_t u = data[j + k] & mask_l[i];
                uint64_t v = data[j + k] & mask_r[i];
                uint64_t x = data[j + n + k] & mask_l[i];
                uint64_t y = data[j + n + k] & mask_r[i];
                data[j + k] = u | (x >> n);
                data[j + n + k] = (v << n) | y;
            }
        }
    }
}

void orthogonalize(uint64_t* data, uint64_t* out) {
  for (uint64_t i = 0; i < 64; i++)
    out[i] = data[i];
  real_ortho(out);
}

void unorthogonalize(uint64_t *in, uint64_t* data) {
  for (uint64_t i = 0; i < 64; i++)
    data[i] = in[i];
  real_ortho(data);
}

#define T uint64_t
// #define MUL(c__, a__, b__)
void MUL(uint64_t *c__, const uint64_t *a__, const uint64_t *b__) {
    T tmp_62464 = XOR(a__[15], a__[11]);
    T tmp_63744 = XOR(a__[14], a__[10]);
    T tmp_64000 = XOR(tmp_62464, a__[10]);
    T tmp_64576 = XOR(a__[13], a__[9]);
    T tmp_64640 = XOR(tmp_63744, a__[9]);
    T tmp_64768 = XOR(tmp_64000, a__[9]);
    T tmp_65040 = XOR(a__[12], a__[8]);
    T tmp_65056 = XOR(tmp_64576, a__[8]);
    T tmp_65088 = XOR(tmp_64640, a__[8]);
    T _tmp130817_ = AND(a__[8], a__[7]);
    T _tmp130945_ = AND(a__[9], a__[6]);
    T tmp_65408 = XOR(_tmp130817_, _tmp130945_);
    T _tmp131009_ = AND(a__[10], a__[5]);
    T tmp_65472 = XOR(tmp_65408, _tmp131009_);
    T _tmp131041_ = AND(a__[11], a__[4]);
    T tmp_65504 = XOR(tmp_65472, _tmp131041_);
    T _tmp131057_ = AND(tmp_65040, a__[3]);
    T tmp_65520 = XOR(tmp_65504, _tmp131057_);
    T _tmp131065_ = AND(tmp_65056, a__[2]);
    T tmp_65528 = XOR(tmp_65520, _tmp131065_);
    T _tmp131069_ = AND(tmp_65088, a__[1]);
    T tmp_65532 = XOR(tmp_65528, _tmp131069_);
    T _tmp131071_ = AND(tmp_64768, a__[0]);
    c__[0] = XOR(tmp_65532, _tmp131071_);
    T tmp_128000 = XOR(a__[12], a__[11]);
    T tmp_129536 = XOR(tmp_128000, a__[10]);
    T _tmp261888_ = AND(a__[9], a__[7]);
    T _tmp262016_ = AND(a__[10], a__[6]);
    T tmp_130944 = XOR(_tmp261888_, _tmp262016_);
    T _tmp262080_ = AND(a__[11], a__[5]);
    T tmp_131008 = XOR(tmp_130944, _tmp262080_);
    T _tmp262112_ = AND(tmp_65040, a__[4]);
    T tmp_131040 = XOR(tmp_131008, _tmp262112_);
    T _tmp262128_ = AND(tmp_65056, a__[3]);
    T tmp_131056 = XOR(tmp_131040, _tmp262128_);
    T _tmp262136_ = AND(tmp_65088, a__[2]);
    T tmp_131064 = XOR(tmp_131056, _tmp262136_);
    T _tmp262140_ = AND(tmp_64768, a__[1]);
    T tmp_131068 = XOR(tmp_131064, _tmp262140_);
    T _tmp262142_ = AND(tmp_129536, a__[0]);
    c__[1] = XOR(tmp_131068, _tmp262142_);
    T tmp_190464 = XOR(a__[13], a__[12]);
    T tmp_193536 = XOR(tmp_190464, a__[11]);
    T _tmp392959_ = AND(a__[10], a__[7]);
    T _tmp393087_ = AND(a__[11], a__[6]);
    T tmp_196480 = XOR(_tmp392959_, _tmp393087_);
    T _tmp393151_ = AND(tmp_65040, a__[5]);
    T tmp_196544 = XOR(tmp_196480, _tmp393151_);
    T _tmp393183_ = AND(tmp_65056, a__[4]);
    T tmp_196576 = XOR(tmp_196544, _tmp393183_);
    T _tmp393199_ = AND(tmp_65088, a__[3]);
    T tmp_196592 = XOR(tmp_196576, _tmp393199_);
    T _tmp393207_ = AND(tmp_64768, a__[2]);
    T tmp_196600 = XOR(tmp_196592, _tmp393207_);
    T _tmp393211_ = AND(tmp_129536, a__[1]);
    T tmp_196604 = XOR(tmp_196600, _tmp393211_);
    T _tmp393213_ = AND(tmp_193536, a__[0]);
    c__[2] = XOR(tmp_196604, _tmp393213_);
    T tmp_249856 = XOR(a__[14], a__[13]);
    T tmp_256000 = XOR(tmp_249856, a__[12]);
    T tmp_261760 = XOR(tmp_256000, a__[8]);
    T _tmp524030_ = AND(a__[11], a__[7]);
    T _tmp524158_ = AND(tmp_65040, a__[6]);
    T tmp_262016 = XOR(_tmp524030_, _tmp524158_);
    T _tmp524222_ = AND(tmp_65056, a__[5]);
    T tmp_262080 = XOR(tmp_262016, _tmp524222_);
    T _tmp524254_ = AND(tmp_65088, a__[4]);
    T tmp_262112 = XOR(tmp_262080, _tmp524254_);
    T _tmp524270_ = AND(tmp_64768, a__[3]);
    T tmp_262128 = XOR(tmp_262112, _tmp524270_);
    T _tmp524278_ = AND(tmp_129536, a__[2]);
    T tmp_262136 = XOR(tmp_262128, _tmp524278_);
    T _tmp524282_ = AND(tmp_193536, a__[1]);
    T tmp_262140 = XOR(tmp_262136, _tmp524282_);
    T _tmp524284_ = AND(tmp_261760, a__[0]);
    c__[3] = XOR(tmp_262140, _tmp524284_);
    T tmp_324608 = XOR(tmp_249856, a__[11]);
    T tmp_325664 = XOR(a__[15], a__[10]);
    T tmp_325696 = XOR(a__[11], a__[10]);
    T tmp_325888 = XOR(tmp_190464, a__[10]);
    T tmp_326144 = XOR(tmp_324608, a__[10]);
    T tmp_326664 = XOR(a__[14], a__[9]);
    T tmp_326672 = XOR(tmp_325664, a__[9]);
    T tmp_326720 = XOR(tmp_128000, a__[9]);
    T tmp_326784 = XOR(tmp_325888, a__[9]);
    T tmp_327170 = XOR(a__[13], a__[8]);
    T tmp_327172 = XOR(tmp_326664, a__[8]);
    T tmp_327184 = XOR(tmp_325696, a__[8]);
    T tmp_327200 = XOR(tmp_326720, a__[8]);
    T _tmp655101_ = AND(a__[12], a__[7]);
    T _tmp655229_ = AND(tmp_327170, a__[6]);
    T tmp_327552 = XOR(_tmp655101_, _tmp655229_);
    T _tmp655293_ = AND(tmp_327172, a__[5]);
    T tmp_327616 = XOR(tmp_327552, _tmp655293_);
    T _tmp655325_ = AND(tmp_326672, a__[4]);
    T tmp_327648 = XOR(tmp_327616, _tmp655325_);
    T _tmp655341_ = AND(tmp_327184, a__[3]);
    T tmp_327664 = XOR(tmp_327648, _tmp655341_);
    T _tmp655349_ = AND(tmp_327200, a__[2]);
    T tmp_327672 = XOR(tmp_327664, _tmp655349_);
    T _tmp655353_ = AND(tmp_326784, a__[1]);
    T tmp_327676 = XOR(tmp_327672, _tmp655353_);
    T _tmp655355_ = AND(tmp_326144, a__[0]);
    c__[4] = XOR(tmp_327676, _tmp655355_);
    T tmp_387072 = XOR(a__[14], a__[12]);
    T tmp_389632 = XOR(a__[13], a__[11]);
    T tmp_391296 = XOR(a__[12], a__[10]);
    T tmp_391680 = XOR(tmp_387072, a__[10]);
    T tmp_392200 = XOR(a__[15], a__[9]);
    T tmp_392224 = XOR(a__[11], a__[9]);
    T tmp_392320 = XOR(tmp_389632, a__[9]);
    T tmp_392448 = XOR(tmp_391680, a__[9]);
    T tmp_392706 = XOR(a__[14], a__[8]);
    T tmp_392712 = XOR(a__[10], a__[8]);
    T tmp_392736 = XOR(tmp_391296, a__[8]);
    T tmp_392768 = XOR(tmp_392320, a__[8]);
    T _tmp786172_ = AND(a__[13], a__[7]);
    T _tmp786300_ = AND(tmp_392706, a__[6]);
    T tmp_393088 = XOR(_tmp786172_, _tmp786300_);
    T _tmp786364_ = AND(tmp_392200, a__[5]);
    T tmp_393152 = XOR(tmp_393088, _tmp786364_);
    T _tmp786396_ = AND(tmp_392712, a__[4]);
    T tmp_393184 = XOR(tmp_393152, _tmp786396_);
    T _tmp786412_ = AND(tmp_392224, a__[3]);
    T tmp_393200 = XOR(tmp_393184, _tmp786412_);
    T _tmp786420_ = AND(tmp_392736, a__[2]);
    T tmp_393208 = XOR(tmp_393200, _tmp786420_);
    T _tmp786424_ = AND(tmp_392768, a__[1]);
    T tmp_393212 = XOR(tmp_393208, _tmp786424_);
    T _tmp786426_ = AND(tmp_392448, a__[0]);
    c__[5] = XOR(tmp_393212, _tmp786426_);
    T _tmp917243_ = AND(a__[14], a__[7]);
    T _tmp917371_ = AND(a__[15], a__[6]);
    T tmp_458624 = XOR(_tmp917243_, _tmp917371_);
    T _tmp917435_ = AND(a__[8], a__[5]);
    T tmp_458688 = XOR(tmp_458624, _tmp917435_);
    T _tmp917467_ = AND(a__[9], a__[4]);
    T tmp_458720 = XOR(tmp_458688, _tmp917467_);
    T _tmp917483_ = AND(a__[10], a__[3]);
    T tmp_458736 = XOR(tmp_458720, _tmp917483_);
    T _tmp917491_ = AND(a__[11], a__[2]);
    T tmp_458744 = XOR(tmp_458736, _tmp917491_);
    T _tmp917495_ = AND(tmp_65040, a__[1]);
    T tmp_458748 = XOR(tmp_458744, _tmp917495_);
    T _tmp917497_ = AND(tmp_65056, a__[0]);
    c__[6] = XOR(tmp_458748, _tmp917497_);
    T _tmp1048314_ = AND(a__[15], a__[7]);
    T _tmp1048442_ = AND(a__[8], a__[6]);
    T tmp_524160 = XOR(_tmp1048314_, _tmp1048442_);
    T _tmp1048506_ = AND(a__[9], a__[5]);
    T tmp_524224 = XOR(tmp_524160, _tmp1048506_);
    T _tmp1048538_ = AND(a__[10], a__[4]);
    T tmp_524256 = XOR(tmp_524224, _tmp1048538_);
    T _tmp1048554_ = AND(a__[11], a__[3]);
    T tmp_524272 = XOR(tmp_524256, _tmp1048554_);
    T _tmp1048562_ = AND(tmp_65040, a__[2]);
    T tmp_524280 = XOR(tmp_524272, _tmp1048562_);
    T _tmp1048566_ = AND(tmp_65056, a__[1]);
    T tmp_524284 = XOR(tmp_524280, _tmp1048566_);
    T _tmp1048568_ = AND(tmp_65088, a__[0]);
    c__[7] = XOR(tmp_524284, _tmp1048568_);
}
// TODO
//#undef AND
//#undef XOR
