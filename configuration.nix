{ config, pkgs, ... }:
{
  # Allow unfree (proprietary) drivers
  nixpkgs.config.allowUnfree = true;

  hardware.opengl.enable = true;
  hardware.nvidia.package = config.boot.kernelPackages.nvidiaPackages.stable;
  hardware.nvidia.nvidiaSettings = true;  # Includes nvidia-smi :contentReference[oaicite:2]{index=2}
  nixpkgs.config.nvidia.acceptLicense = true;
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.nvidia.modesetting.enable = true;
}

