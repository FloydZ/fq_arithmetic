let
  pkgs = (import <nixpkgs> {}).pkgsCross.aarch64-multiplatform;
in
{ 
}:
pkgs.gccStdenv.mkDerivation {
  name = "fq_arithmetic";
  src = ./.;
  nativeBuildInputs = with pkgs; [ 
    gcc 
    clang
  ];
}
