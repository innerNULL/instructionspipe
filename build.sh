set -x

export NIX_CONF_DIR="$(pwd)"
export NIX_PATH="$(pwd)/configuration.nix${NIX_PATH:+:}$NIX_PATH"
NIXPKGS_ALLOW_UNFREE=1 nix-shell \
  --impure shell.nix \
  --extra-nix-path nixos-config=$(pwd)/configuration.nix
