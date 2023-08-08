{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # mach-nix.url = "github:DavHau/mach-nix/master";
    # mach-nix.inputs.nixpkgs.follows = "nixpkgs";
    # mach-nix.inputs.pypi-deps-db.follows = "pypi-deps-db";
    # pypi-deps-db.url = "github:DavHau/pypi-deps-db";
    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = {self, nixpkgs, nixgl, ... }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;
        overlays=[nixgl.overlay]; config.allowUnfree=true; config.cudaSupport =
          true; config.cudaCapabilities = [ "8.6" ];}));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let
            pybox2d = pkgs.python3.pkgs.buildPythonPackage rec {
                pname = "Box2D";
                version = "2.3.10";
              
                src = pkgs.fetchFromGitHub {
                    owner = "pybox2d";
                    repo = "pybox2d";
                    rev = "master";
                    sha256 = "a4JjUrsSbAv9SjqZLwuqXhz2x2YhRzZZTytu4X5YWX8=";
                };
                nativeBuildInputs = [ pkgs.pkgconfig pkgs.swig ];
                doCheck = false;
                format="setuptools";
              };
            python = pkgs.python3.withPackages (p: with p;[numpy pygame pybullet
              matplotlib gymnasium tensorflow tqdm keras pybox2d ]);
            anchored-rl = pkgs.python3.pkgs.buildPythonPackage rec {
                pname = "anchored_rl";
                version = "0.1.0";
              
                src = ./.;
                doCheck = false;
                #format = "setuputils";
              
                propagatedBuildInputs = [
                  python
                ];
              };
            
        in pkgs.mkShell {
            buildInputs = [
                pkgs.nixgl.auto.nixGLDefault
                (pkgs.python3.withPackages (p: with p;[numpy pygame pybullet
                matplotlib gymnasium tensorflow keras tqdm anchored-rl pybox2d mypy]))
            ];
          }
        );
    };
}
