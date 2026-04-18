"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ResultsSection } from "@/components/ResultsSection";
import { UploadSection } from "@/components/UploadSection";
import { apiBase, fetchDemo, fetchHealth, predictVideo, type PredictResponse } from "@/lib/api";

const LOADING_STEPS = [
  "Extracting frames…",
  "Detecting and aligning faces…",
  "Analyzing with DualForensics…",
  "Running GradCAM and region analysis…",
  "Generating explanations…",
];

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelOk, setModelOk] = useState<boolean | null>(null);

  const api = useMemo(() => apiBase(), []);

  useEffect(() => {
    fetchHealth()
      .then((h) => setModelOk(h.model_loaded))
      .catch(() => setModelOk(false));
  }, []);

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => {
      setStepIdx((i) => (i + 1) % LOADING_STEPS.length);
    }, 2200);
    return () => clearInterval(t);
  }, [loading]);

  const run = useCallback(async (fn: () => Promise<PredictResponse>) => {
    setError(null);
    setResult(null);
    setLoading(true);
    setStepIdx(0);
    try {
      const data = await fn();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const onAnalyze = useCallback(() => {
    if (!file) return;
    run(() => predictVideo(file));
  }, [file, run]);

  const onDemo = useCallback(() => run(fetchDemo), [run]);

  return (
    <main className="mx-auto max-w-6xl px-4 pb-20 pt-10 sm:px-6 lg:px-8">
      <header className="mb-10 text-center sm:text-left">
        <p className="text-sm font-medium uppercase tracking-widest text-sky-400">Research preview</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
          DualForensics: Deepfake Video Detection
        </h1>
        <p className="mx-auto mt-4 max-w-3xl text-base text-zinc-400 sm:mx-0">
          Uniformly sample 16 frames from your clip, crop faces to 224×224, and run the DualForensics
          spatiotemporal model. You get a REAL / DEEPFAKE verdict, GradCAM overlays on the EfficientNet
          feature map, temporal self-attention weights per frame, and region-wise heatmap scores across
          forehead, eyes, nose, mouth, and jaw.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
          <span>
            API: <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-300">{api}</code>
          </span>
          {modelOk === false ? (
            <span className="rounded-full bg-amber-500/10 px-2 py-1 text-amber-400">
              Model not loaded on server — place weights and restart the API
            </span>
          ) : modelOk === true ? (
            <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-emerald-400">Model ready</span>
          ) : null}
        </div>
      </header>

      <UploadSection
        disabled={loading}
        selectedName={file?.name ?? null}
        onFileChange={(f) => {
          setFile(f);
          setResult(null);
        }}
        onAnalyze={onAnalyze}
        onDemo={onDemo}
      />

      {loading ? (
        <div className="mt-10 flex flex-col items-center justify-center gap-4 rounded-2xl border border-zinc-800 bg-zinc-900/50 py-16">
          <div
            className="h-12 w-12 animate-spin rounded-full border-2 border-zinc-600 border-t-sky-500"
            aria-hidden
          />
          <p className="max-w-md text-center text-sm text-zinc-300">{LOADING_STEPS[stepIdx]}</p>
          <p className="text-xs text-zinc-500">CPU inference may take 30–60 seconds.</p>
        </div>
      ) : null}

      {error ? (
        <div className="mt-8 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      {result ? <ResultsSection data={result} /> : null}

      <footer className="mt-16 border-t border-zinc-800 pt-8 text-center text-xs text-zinc-500">
        FaceForensics++ (c23) style pipeline · Not legal evidence · For research use only
      </footer>
    </main>
  );
}
