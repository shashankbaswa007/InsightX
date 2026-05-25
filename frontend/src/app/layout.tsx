import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "InsightX AI — Explainable AI Platform",
  description:
    "Make AI decisions interpretable, trustworthy, and actionable. Upload datasets, train models, and visualize predictions with SHAP, LIME, and fairness analysis.",
  keywords: [
    "Explainable AI",
    "XAI",
    "SHAP",
    "LIME",
    "Machine Learning",
    "Model Interpretation",
    "Bias Detection",
    "Feature Importance",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark`}
    >
      <body className="min-h-screen bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
