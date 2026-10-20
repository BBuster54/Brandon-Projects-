export async function loadACS({ fetchImpl = globalThis.fetch } = {}) {
  try {
    const response = await fetchImpl("data/acs.json", {
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error("Official snapshot unavailable");
    const data = await response.json();
    return data && typeof data.cities === "object" ? data : { cities: {} };
  } catch {
    return { cities: {}, status: "not-loaded" };
  }
}
