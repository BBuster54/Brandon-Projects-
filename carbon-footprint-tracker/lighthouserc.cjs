module.exports = {
  ci: {
    collect: {
      startServerCommand: "python3 -u -m http.server 4173",
      startServerReadyPattern: "Serving HTTP",
      url: ["http://localhost:4173/"],
      numberOfRuns: 1,
      settings: { chromeFlags: "--no-sandbox" },
    },
    assert: {
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
