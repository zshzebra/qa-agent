{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  services.caddy = {
    enable = true;
    config = ''
      http://localhost:8020 {
        handle_path /api/* {
          reverse_proxy localhost:${toString config.processes.backend.ports.backend.value}
        }
        reverse_proxy localhost:${toString config.processes.frontend.ports.frontend.value}
      }
    '';
  };
}
