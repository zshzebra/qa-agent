{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  languages.javascript = {
    enable = true;
    directory = "${config.git.root}/frontend";
    pnpm.enable = true;
    pnpm.install.enable = true;
  };

  processes.frontend = {
    ports.frontend.allocate = 5173;
    exec = "pnpm run dev --port ${toString config.processes.frontend.ports.frontend.value}";
    cwd = "${config.git.root}/frontend";
  };
}
