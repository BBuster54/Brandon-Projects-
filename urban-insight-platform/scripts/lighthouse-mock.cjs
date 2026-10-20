const fs = require("node:fs");
const air = fs.readFileSync("tests/fixtures/open-meteo-recorded.json");
const tile = Buffer.from(
  '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><path fill="#e2e6e9" d="M0 0h256v256H0z"/></svg>',
);
module.exports = async (browser) => {
  if (browser.__urbanMock) return;
  browser.__urbanMock = true;
  async function intercept(target) {
    try {
      const session = await target.createCDPSession();
      session.on("Fetch.requestPaused", (event) => {
        const isAir = event.request.url.includes("air-quality-api");
        session
          .send("Fetch.fulfillRequest", {
            requestId: event.requestId,
            responseCode: 200,
            responseHeaders: [
              {
                name: "Content-Type",
                value: isAir ? "application/json" : "image/svg+xml",
              },
              { name: "Access-Control-Allow-Origin", value: "*" },
            ],
            body: (isAir ? air : tile).toString("base64"),
          })
          .catch(() => {});
      });
      await session.send("Fetch.enable", {
        patterns: [
          { urlPattern: "https://air-quality-api.open-meteo.com/*" },
          { urlPattern: "https://tile.openstreetmap.org/*" },
        ],
      });
    } catch {
      /* Non-page targets do not support interception. */
    }
  }
  browser.on("targetcreated", intercept);
  for (const target of browser.targets()) await intercept(target);
};
