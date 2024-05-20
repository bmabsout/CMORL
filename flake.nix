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
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {
          inherit system;
          overlays=[nixgl.overlay];
         # config.cudaSupport = true;
          config.allowUnfree = true;
        #  config.cudaCapabilities = [ "8.6" ];
        }));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let python = pkgs.python311.override {
              packageOverrides = (self: super: {
                # torch = throw "lesh";
                # tensorflow = super.tensorflow-bin;
                #torch = super.torch-bin;
              });
            };
            cmorl = python.pkgs.callPackage ./nix/cmorl.nix {python3Packages = python.pkgs;};
            pythonWithCMORL = python.withPackages (p: cmorl.propagatedBuildInputs);
        in pkgs.mkShell {
            buildInputs = [
                # pkgs.nixgl.auto.nixGLDefault
                pkgs.nixgl.nixGLIntel
                pythonWithCMORL
            ];
            shellHook = ''
              export PYTHONPATH=$PYTHONPATH:$(pwd) # to allow cmorl to be imported as editable
            '';
          }
        );
    };
}
