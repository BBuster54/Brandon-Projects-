import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Commands",
  description: "Keyboard commands and shortcuts.",
};

export default function Commands() {
  return (
    <div className="bg-background text-foreground min-h-screen font-sans">
      <div className="flex flex-col gap-6 p-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-semibold">Keyboard commands</h1>
          <p className="text-text-muted text-sm mt-1">
            Speed up your workflow with shortcuts.
          </p>
        </div>

        {/* Commands List Placeholder */}
        <div className="flex-1 bg-card border border-border-muted rounded-lg p-6">
          <p className="text-text-muted text-sm">
            Keyboard shortcuts will be listed here.
          </p>
        </div>
      </div>
    </div>
  );
}