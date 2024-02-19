{ pkgs }:

pkgs.python3.pkgs.buildPythonPackage rec {
  pname = "Box2D";
  version = "2.3.10";

  src = pkgs.fetchFromGitHub {
      owner = "pybox2d";
      repo = "pybox2d";
      rev = "master";
      sha256 = "a4JjUrsSbAv9SjqZLwuqXhz2x2YhRzZZTytu4X5YWX8=";
  };
  nativeBuildInputs = [ pkgs.pkg-config pkgs.swig ];
  doCheck = false;
  format="setuptools";
}
