import js from "@eslint/js";
import globals from "globals";
export default [
  js.configs.recommended,
  {
    files: ["js/**/*.js", "scripts/**/*.{mjs,cjs}", "tests/**/*.{js,mjs}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
    },
    rules: { "no-unused-vars": ["error", { argsIgnorePattern: "^_" }] },
  },
  {
    ignores: [
      "vendor/**",
      "js/atlas.js",
      "node_modules/**",
      "artifacts/**",
      ".lighthouseci/**",
    ],
  },
];
