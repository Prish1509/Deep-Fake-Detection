"use client";

import type { PredictResponse } from "@/lib/api";
import { GradCAMGrid } from "./GradCAMGrid";
import { RegionChart } from "./RegionChart";
import { TemporalChart } from "./TemporalChart";

type Props = {
  data: PredictResponse;
};

export function ResultsSection({ data }: Props) {
  const isFake = data.prediction === "DEEPFAKE";
  const pct = Math.round(data.confidence * 1000) / 10;

  const frameTitles = data.heatmap_frames.map((_, i) => `Frame slot ${i + 1}`);

  return (
    <section className="mt-12 space-y-10">
      <div
        className={`rounded-2xl border px-6 py-8 text-center shadow-lg ${
          isFake
            ? "border-red-900/60 bg-gradient-to-br from-red-950/80 to-zinc-950"
            : "border-emerald-900/60 bg-gradient-to-br from-emerald-950/80 to-zinc-950"
        }`}
      >
        <p className="text-sm uppercase tracking-[0.2em] text-zinc-400">Verdict</p>
        <p className={`mt-2 text-4xl font-black sm:text-5xl ${isFake ? "text-red-400" : "text-emerald-400"}`}>
          {data.prediction}
        </p>
        <p className="mt-2 text-2xl font-semibold text-zinc-100">{pct}% confidence</p>
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
        <h3 className="mb-3 text-sm font-semibold text-zinc-200">GradCAM dashboard</h3>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${data.gradcam_image}`}
          alt="GradCAM dashboard: faces and heatmap overlays"
          className="w-full rounded-lg border border-zinc-800"
        />
      </div>

      <div>
        <h3 className="mb-3 text-lg font-semibold text-zinc-100">Face frames with heatmap overlays</h3>
        <GradCAMGrid heatmapFrames={data.heatmap_frames} titles={frameTitles} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <TemporalChart values={data.temporal_importance} />
        <RegionChart scores={data.region_scores} />
      </div>

      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
        <h3 className="text-sm font-semibold text-zinc-300">Explanation</h3>
        <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-zinc-200">{data.explanation}</p>
      </div>
    </section>
  );
}
