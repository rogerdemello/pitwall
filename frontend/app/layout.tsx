import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import ThemeToggle, { themeInitScript } from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "PIT WALL — The Silent Co-Driver",
  description:
    "Driver stress from F1 team radio, time-synced to real lap telemetry.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Stamps the saved theme before first paint. In an effect this would
            flash the default theme on every load. */}
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body>
        <div className="shell">
          <header className="topbar">
            <div className="brand">
              <span>PIT<span className="accent">·</span>WALL</span>
              <span className="sub">The Silent Co-Driver</span>
            </div>
            <Nav />
            <ThemeToggle />
          </header>
          <main className="page">{children}</main>
        </div>
      </body>
    </html>
  );
}
