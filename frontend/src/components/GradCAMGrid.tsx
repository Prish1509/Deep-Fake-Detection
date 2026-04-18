"use client";

type Props = {
  heatmapFrames: string[];
  titles?: string[];
};

export function GradCAMGrid({ heatmapFrames, titles }: Props) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {heatmapFrames.map((b64, i) => (
        <figure
          key={i}
          className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-inner"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${b64}`}
            alt={titles?.[i] ?? `Frame ${i + 1} with GradCAM overlay`}
            className="aspect-square w-full object-cover"
          />
          <figcaption className="px-2 py-1 text-center text-xs text-zinc-500">
            {titles?.[i] ?? `Sample ${i + 1}`}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
