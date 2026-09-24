// app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "NLP Text Classifier | TensorFlow Sentiment Analysis",
  description:
    "Live binary sentiment classifier trained on IMDb movie reviews using TensorFlow and TensorFlow Hub. Built with FastAPI + Next.js.",
  keywords: ["NLP", "sentiment analysis", "TensorFlow", "machine learning", "text classification", "IMDb"],
  openGraph: {
    title: "NLP Text Classifier",
    description: "Live ML demo — classify movie reviews as positive or negative using TensorFlow.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
