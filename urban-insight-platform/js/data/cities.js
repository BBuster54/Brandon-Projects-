export async function loadCities() {
  const res = await fetch("data/cities.json", {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error("City metadata unavailable");
  return (await res.json()).cities;
}
