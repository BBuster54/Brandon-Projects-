import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Testimonials",
  description: "Setup and onboarding.",
};

export default function Testimonials() {
  return (
    <div className="bg-background text-foreground min-h-screen font-sans">
      <div className="flex flex-col gap-6 p-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-semibold">Testimonials</h1>
          <p className="text-text-muted text-sm mt-1">
            What the brightest minds in finance are saying.
          </p>
        </div>

        {/* Setup Placeholder */}
        <div className="flex-1 bg-card border border-border-muted rounded-lg p-6">
          <p className="text-text-muted text-sm">
            Setup steps will appear here.
          </p>
        </div>
      </div>
    </div>
  );
}