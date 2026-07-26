{ pkgs }:

let
  runtimeLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    portaudio
    alsa-lib
  ];
in
pkgs.mkShell {
  packages = with pkgs; [
    python313
    uv
    just
    git
    pkg-config
    portaudio
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

  shellHook = ''
    echo "neuroacoustic-resonator dev shell"
    python --version
    uv --version
  '';
}
