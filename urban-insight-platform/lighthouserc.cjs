module.exports = {
  ci: {
    collect: {
      startServerCommand: "python3 -u -m http.server 4181",
      startServerReadyPattern: "Serving HTTP",
      url: ["http://localhost:4181/"],
      numberOfRuns: 3,
      puppeteerScript: "scripts/lighthouse-mock.cjs",
      puppeteerLaunchOptions: { args: ["--no-sandbox"] },
      settings: { disableStorageReset: true },
    },
    assert: {
      aggregationMethod: "median",
      assertions: {
        "categories:performance": ["error", { minScore: 0.95 }],
        "categories:accessibility": ["error", { minScore: 0.95 }],
        "categories:best-practices": ["error", { minScore: 0.95 }],
        "categories:seo": ["error", { minScore: 0.95 }],
      },
    },
    upload: { target: "filesystem", outputDir: ".lighthouseci" },
  },
};
