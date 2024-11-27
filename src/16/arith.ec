require import AllCore IntDiv CoreMap List Distr.
from Jasmin require import JModel_x86.


require import Array16 Array96 Array256.
require import WArray16 WArray96 WArray256.

abbrev gf16_inverse_tab = Array16.of_list witness [W8.of_int 0; W8.of_int 1;
W8.of_int 9; W8.of_int 14; W8.of_int 13; W8.of_int 11; W8.of_int 7;
W8.of_int 6; W8.of_int 15; W8.of_int 2; W8.of_int 12; W8.of_int 5;
W8.of_int 10; W8.of_int 4; W8.of_int 3; W8.of_int 8].


abbrev __gf16_mulbase = Array96.of_list witness [W8.of_int 0; W8.of_int 2;
W8.of_int 4; W8.of_int 6; W8.of_int 8; W8.of_int 10; W8.of_int 12;
W8.of_int 14; W8.of_int 3; W8.of_int 1; W8.of_int 7; W8.of_int 5;
W8.of_int 11; W8.of_int 9; W8.of_int 15; W8.of_int 13; W8.of_int 0;
W8.of_int 2; W8.of_int 4; W8.of_int 6; W8.of_int 8; W8.of_int 10;
W8.of_int 12; W8.of_int 14; W8.of_int 3; W8.of_int 1; W8.of_int 7;
W8.of_int 5; W8.of_int 11; W8.of_int 9; W8.of_int 15; W8.of_int 13;
W8.of_int 0; W8.of_int 4; W8.of_int 8; W8.of_int 12; W8.of_int 3;
W8.of_int 7; W8.of_int 11; W8.of_int 15; W8.of_int 6; W8.of_int 2;
W8.of_int 14; W8.of_int 10; W8.of_int 5; W8.of_int 1; W8.of_int 13;
W8.of_int 9; W8.of_int 0; W8.of_int 4; W8.of_int 8; W8.of_int 12;
W8.of_int 3; W8.of_int 7; W8.of_int 11; W8.of_int 15; W8.of_int 6;
W8.of_int 2; W8.of_int 14; W8.of_int 10; W8.of_int 5; W8.of_int 1;
W8.of_int 13; W8.of_int 9; W8.of_int 0; W8.of_int 8; W8.of_int 3;
W8.of_int 11; W8.of_int 6; W8.of_int 14; W8.of_int 5; W8.of_int 13;
W8.of_int 12; W8.of_int 4; W8.of_int 15; W8.of_int 7; W8.of_int 10;
W8.of_int 2; W8.of_int 9; W8.of_int 1; W8.of_int 0; W8.of_int 8; W8.of_int 3;
W8.of_int 11; W8.of_int 6; W8.of_int 14; W8.of_int 5; W8.of_int 13;
W8.of_int 12; W8.of_int 4; W8.of_int 15; W8.of_int 7; W8.of_int 10;
W8.of_int 2; W8.of_int 9; W8.of_int 1].


abbrev gf16_mult_table = Array256.of_list witness [W8.of_int 0; W8.of_int 0;
W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0;
W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 0;
W8.of_int 0; W8.of_int 0; W8.of_int 0; W8.of_int 1; W8.of_int 2; W8.of_int 3;
W8.of_int 4; W8.of_int 5; W8.of_int 6; W8.of_int 7; W8.of_int 8; W8.of_int 9;
W8.of_int 10; W8.of_int 11; W8.of_int 12; W8.of_int 13; W8.of_int 14;
W8.of_int 15; W8.of_int 0; W8.of_int 2; W8.of_int 4; W8.of_int 6;
W8.of_int 8; W8.of_int 10; W8.of_int 12; W8.of_int 14; W8.of_int 3;
W8.of_int 1; W8.of_int 7; W8.of_int 5; W8.of_int 11; W8.of_int 9;
W8.of_int 15; W8.of_int 13; W8.of_int 0; W8.of_int 3; W8.of_int 6;
W8.of_int 5; W8.of_int 12; W8.of_int 15; W8.of_int 10; W8.of_int 9;
W8.of_int 11; W8.of_int 8; W8.of_int 13; W8.of_int 14; W8.of_int 7;
W8.of_int 4; W8.of_int 1; W8.of_int 2; W8.of_int 0; W8.of_int 4; W8.of_int 8;
W8.of_int 12; W8.of_int 3; W8.of_int 7; W8.of_int 11; W8.of_int 15;
W8.of_int 6; W8.of_int 2; W8.of_int 14; W8.of_int 10; W8.of_int 5;
W8.of_int 1; W8.of_int 13; W8.of_int 9; W8.of_int 0; W8.of_int 5;
W8.of_int 10; W8.of_int 15; W8.of_int 7; W8.of_int 2; W8.of_int 13;
W8.of_int 8; W8.of_int 14; W8.of_int 11; W8.of_int 4; W8.of_int 1;
W8.of_int 9; W8.of_int 12; W8.of_int 3; W8.of_int 6; W8.of_int 0;
W8.of_int 6; W8.of_int 12; W8.of_int 10; W8.of_int 11; W8.of_int 13;
W8.of_int 7; W8.of_int 1; W8.of_int 5; W8.of_int 3; W8.of_int 9;
W8.of_int 15; W8.of_int 14; W8.of_int 8; W8.of_int 2; W8.of_int 4;
W8.of_int 0; W8.of_int 7; W8.of_int 14; W8.of_int 9; W8.of_int 15;
W8.of_int 8; W8.of_int 1; W8.of_int 6; W8.of_int 13; W8.of_int 10;
W8.of_int 3; W8.of_int 4; W8.of_int 2; W8.of_int 5; W8.of_int 12;
W8.of_int 11; W8.of_int 0; W8.of_int 8; W8.of_int 3; W8.of_int 11;
W8.of_int 6; W8.of_int 14; W8.of_int 5; W8.of_int 13; W8.of_int 12;
W8.of_int 4; W8.of_int 15; W8.of_int 7; W8.of_int 10; W8.of_int 2;
W8.of_int 9; W8.of_int 1; W8.of_int 0; W8.of_int 9; W8.of_int 1; W8.of_int 8;
W8.of_int 2; W8.of_int 11; W8.of_int 3; W8.of_int 10; W8.of_int 4;
W8.of_int 13; W8.of_int 5; W8.of_int 12; W8.of_int 6; W8.of_int 15;
W8.of_int 7; W8.of_int 14; W8.of_int 0; W8.of_int 10; W8.of_int 7;
W8.of_int 13; W8.of_int 14; W8.of_int 4; W8.of_int 9; W8.of_int 3;
W8.of_int 15; W8.of_int 5; W8.of_int 8; W8.of_int 2; W8.of_int 1;
W8.of_int 11; W8.of_int 6; W8.of_int 12; W8.of_int 0; W8.of_int 11;
W8.of_int 5; W8.of_int 14; W8.of_int 10; W8.of_int 1; W8.of_int 15;
W8.of_int 4; W8.of_int 7; W8.of_int 12; W8.of_int 2; W8.of_int 9;
W8.of_int 13; W8.of_int 6; W8.of_int 8; W8.of_int 3; W8.of_int 0;
W8.of_int 12; W8.of_int 11; W8.of_int 7; W8.of_int 5; W8.of_int 9;
W8.of_int 14; W8.of_int 2; W8.of_int 10; W8.of_int 6; W8.of_int 1;
W8.of_int 13; W8.of_int 15; W8.of_int 3; W8.of_int 4; W8.of_int 8;
W8.of_int 0; W8.of_int 13; W8.of_int 9; W8.of_int 4; W8.of_int 1;
W8.of_int 12; W8.of_int 8; W8.of_int 5; W8.of_int 2; W8.of_int 15;
W8.of_int 11; W8.of_int 6; W8.of_int 3; W8.of_int 14; W8.of_int 10;
W8.of_int 7; W8.of_int 0; W8.of_int 14; W8.of_int 15; W8.of_int 1;
W8.of_int 13; W8.of_int 3; W8.of_int 2; W8.of_int 12; W8.of_int 9;
W8.of_int 7; W8.of_int 6; W8.of_int 8; W8.of_int 4; W8.of_int 10;
W8.of_int 11; W8.of_int 5; W8.of_int 0; W8.of_int 15; W8.of_int 13;
W8.of_int 2; W8.of_int 9; W8.of_int 6; W8.of_int 4; W8.of_int 11;
W8.of_int 1; W8.of_int 14; W8.of_int 12; W8.of_int 3; W8.of_int 8;
W8.of_int 7; W8.of_int 5; W8.of_int 10].


module M = {
  proc gf16_mul (a:W8.t, b:W8.t) : W8.t = {
    
    var r:W8.t;
    var b1:W8.t;
    var r1:W8.t;
    var r2:W8.t;
    
    b1 <- b;
    b1 <- (b1 `>>` (W8.of_int 3));
    b1 <- (- b1);
    r <- (b1 `&` a);
    b1 <- b;
    b1 <- (b1 `>>` (W8.of_int 2));
    b1 <- (b1 `&` (W8.of_int 1));
    b1 <- (- b1);
    b1 <- (b1 `&` a);
    r1 <- r;
    r1 <- (r1 `>>` (W8.of_int 3));
    r1 <- (- r1);
    r1 <- (r1 `&` (W8.of_int 3));
    r2 <- r;
    r2 <- (r2 + r2);
    r2 <- (r2 `&` (W8.of_int 15));
    r <- (r `^` r2);
    r <- (r `^` r1);
    b1 <- b;
    b1 <- (b1 `>>` (W8.of_int 1));
    b1 <- (b1 `&` (W8.of_int 1));
    b1 <- (- b1);
    b1 <- (b1 `&` a);
    r1 <- r;
    r1 <- (r1 `>>` (W8.of_int 3));
    r1 <- (- r1);
    r1 <- (r1 `&` (W8.of_int 3));
    r2 <- r;
    r2 <- (r2 + r2);
    r2 <- (r2 `&` (W8.of_int 15));
    r <- (r `^` r2);
    r <- (r `^` r1);
    b1 <- b;
    b1 <- (b1 `&` (W8.of_int 1));
    b1 <- (- b1);
    b1 <- (b1 `&` a);
    r1 <- r;
    r1 <- (r1 `>>` (W8.of_int 3));
    r1 <- (- r1);
    r1 <- (r1 `&` (W8.of_int 3));
    r2 <- r;
    r2 <- (r2 + r2);
    r2 <- (r2 `&` (W8.of_int 15));
    r <- (r `^` r2);
    r <- (r `^` r1);
    return (r);
  }
}.

