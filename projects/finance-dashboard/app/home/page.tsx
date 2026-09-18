import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Home",
  description: "Main finance dashboard.",
};

export default function Home() {
  return (
    <div className="bg-background text-foreground min-h-screen font-sans">
      <div className="flex flex-col lg:flex-row w-full h-screen gap-6 p-6">
        {/* Placeholder dashboard content — full dashboard will be built out here */}
        <div className="flex-1 bg-card border border-border-muted rounded-lg p-6">
          <h1 className="text-3xl font-semibold">Dashboard</h1>
          <p className="text-text-muted text-sm mt-1">
            Your main dashboard will appear here.
          </p>
        </div>
      </div>
    </div>
  );
}