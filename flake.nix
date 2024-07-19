{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # cuda stuff requires compilation in nixos-unstable
    # nixpkgs-unfree.url = github:SomeoneSerge/nixpkgs-unfree;
    # nixpkgs-unfree.inputs.nixpkgs.follows = "nixpkgs";
    nixgl.url = "github:kenranunderscore/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = {self, nixpkgs, nixgl, ... }@inp:
    let
      nixpkgs_configs = {
        default={};
        with_cuda={
          cudaCapabilities = ["8.6"];
          cudaSupport = true;
          allowUnfree = true;
        };
      };
      system = "x86_64-linux";
    in
    {
      # enter this python environment by executing `nix shell .`
      devShells."${system}" = nixpkgs.lib.attrsets.mapAttrs (name: config:
          let pkgs = import nixpkgs { overlays=[nixgl.overlay]; inherit system; config=config;};
              python = pkgs.python311.override {
                packageOverrides = import ./nix/python-overrides.nix;
              };
              cmorl = python.pkgs.callPackage ./nix/cmorl.nix {python3Packages = python.pkgs;};
              pythonWithCMORL = python.withPackages (p: cmorl.propagatedBuildInputs);
          in pkgs.mkShell {
              buildInputs = [
                  pkgs.nixgl.nixGLIntel
                  pythonWithCMORL
              ];
              shellHook = ''
                export PYTHONPATH=$PYTHONPATH:$(pwd) # to allow cmorl to be imported as editable
                export LD_LIBRARY_PATH=${pkgs.wayland}/lib:$LD_LIBRARY_PATH:/run/opengl-driver/lib
              '';
            }
        ) nixpkgs_configs;
    };
}
