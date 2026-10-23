# Source verification backlog

- Legacy illustrative congestion, recycling and EV counts have no verified measurement dates or authoritative sources. Keep them collapsed and out of rankings, comparison, and headline views. Replace only after source verification, otherwise remove.
- CAMS grid estimates are not local monitor readings. Do not describe them as EPA observations.
- Census derived percentage MOEs are not computed; preserve published component MOEs and state this limitation.
- Review 14 current npm development-tool advisories in Lighthouse CI/Puppeteer transitive dependencies (down from 17 after dependency resolution). npm's suggested forced change downgrades LHCI to 0.1.0; do not apply that destructive compatibility change blindly. No npm dependency runtime bundle is deployed. Keep test input/provider URLs controlled and test a supported toolchain upgrade separately.
