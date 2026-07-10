{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  languages.python = {
    enable = true;
    directory = "${config.git.root}/backend";
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  services.postgres = {
    enable = true;
    # TODO: Configure postgres initial databases, auth, etc
  };

  processes.backend = {
    ports.backend.allocate = 8000;
    exec = "fastapi dev --port ${toString config.processes.backend.ports.backend.value}";
    cwd = "${config.git.root}/backend";
  };
}
