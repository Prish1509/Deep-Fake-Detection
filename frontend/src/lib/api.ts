export type PredictResponse = {
  prediction: "DEEPFAKE" | "REAL";
  confidence: number;
  explanation: string;
  gradcam_image: string;
  face_frames: string[];
  heatmap_frames: string[];
  temporal_importance: number[];
  region_scores: Record<string, number>;
};

export function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";
}

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const r = await fetch(`${apiBase()}/api/health`, { cache: "no-store" });
  if (!r.ok) throw new Error("Health check failed");
  return r.json();
}

export async function predictVideo(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${apiBase()}/api/predict`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

export async function fetchDemo(): Promise<PredictResponse> {
  const r = await fetch(`${apiBase()}/api/demo`, { cache: "no-store" });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}
