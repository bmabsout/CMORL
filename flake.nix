{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
    mach-nix.url = "github:DavHau/mach-nix/master";
    mach-nix.inputs.nixpkgs.follows = "nixpkgs";
    # mach-nix.inputs.pypi-deps-db.follows = "pypi-deps-db";
    # pypi-deps-db.url = "github:DavHau/pypi-deps-db";
    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {self, nixpkgs, mach-nix, nixgl,... }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system; overlays=[nixgl.overlay];}));
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs: pkgs.mkShell {
        buildInputs = [
            pkgs.nixgl.auto.nixGLDefault
            (mach-nix.lib."${system}".mkPython {
                packagesExtra = [];
                requirements = ''
                  tensorflow
                  gymnasium
                  matplotlib
                  pybullet
                  scipy
                  pygame
                  numpy
                '';
                providers.pygame = "nixpkgs";
                providers.pybullet = "nixpkgs";
                providers.tensorflow = "nixpkgs";
            })
        ];
      });
    };
}
