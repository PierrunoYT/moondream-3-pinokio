module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Install torch first: accelerate depends on torch, so installing the
    // requirements first would pull a default PyPI build that torch.js then
    // replaces with the platform-specific one.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          path: "app",
          venv: "env",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
  ]
}
