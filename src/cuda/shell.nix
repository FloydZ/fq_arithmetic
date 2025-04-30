with import <nixpkgs> {};
let
  my-python = pkgs.python3;
  python-with-my-packages = my-python.withPackages (p: with p; [
  ]);
in
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "cuda_fp_arithmetic";
  src = ./.;

   buildInputs = with pkgs; [
     python-with-my-packages
     git
	 gitRepo
	 gnupg
	 autoconf
	 curl
     procps
	 gnumake
	 cmake
	 clang
	 util-linux
	 m4
	 gperf
	 unzip
     cudatoolkit
	 linuxPackages.nvidia_x11
     libGLU
	 libGL
     xorg.libXi
	 xorg.libXmu 
	 freeglut
     xorg.libXext
	 xorg.libX11
	 xorg.libXv
	 xorg.libXrandr
	 zlib 
     ncurses5
	 stdenv.cc
	 binutils
   ];

   shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
      export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"

	  # python hook
      PYTHONPATH=${python-with-my-packages}/${python-with-my-packages.sitePackages}
   ''; 
}
