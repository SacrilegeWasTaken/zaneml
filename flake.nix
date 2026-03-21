{
  description = "zaneml";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
      zon = builtins.readFile ./build.zig.zon;
      version = builtins.head (builtins.match ''.*\.version\s*=\s*"([^"]+)".*'' zon);
    in {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        pname = "zaneml";
        inherit version;
        src = ./.;

        nativeBuildInputs = with pkgs; [
          zig
          uv
        ];
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          zig
          zls
          uv
          macdylibbundler
        ];
      };
    };
}
