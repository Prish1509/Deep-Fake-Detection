import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DualForensics — Deepfake Video Detection",
  description:
    "Upload a video to analyze with DualForensics: verdict, GradCAM heatmaps, temporal cues, and facial region scores.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  );
}
