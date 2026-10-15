import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // This dashboard is nested inside a larger projects repository. Pinning the
  // Turbopack root prevents it from adopting an unrelated lockfile higher in
  // the filesystem during local development or production builds.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
