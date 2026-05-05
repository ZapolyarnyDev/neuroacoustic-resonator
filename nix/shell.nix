{ pkgs }:

let
  runtimeLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    libGL
    libxkbcommon
    wayland
    portaudio
    alsa-lib
    xorg.libX11
    xorg.libXext
    xorg.libxcb
    xorg.libXrender
    xorg.libXrandr
    xorg.libXi
    xorg.libXcursor
    xorg.libXfixes
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
