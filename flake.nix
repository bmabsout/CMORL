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
        (system: f system (import nixpkgs {
          inherit system;
          overlays=[nixgl.overlay];
          config.allowUnfree=true;
          # config.cudaSupport = true;
          # config.cudaCapabilities = [ "8.6" ];
        }));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let cmorl = pkgs.python3Packages.callPackage ./nix/cmorl.nix {};
            python = pkgs.python3.withPackages (p: cmorl.propagatedBuildInputs);
        in pkgs.mkShell {
            buildInputs = [
                pkgs.nixgl.auto.nixGLDefault
                python
            ];
            shellHook = ''
              export PYTHONPATH=$PYTHONPATH:$(pwd) # to allow cmorl to be imported as editable
            '';
          }
        );
    };
}
