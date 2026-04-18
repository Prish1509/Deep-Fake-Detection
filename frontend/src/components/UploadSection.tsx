"use client";

import { useCallback, useRef, useState } from "react";

type Props = {
  disabled?: boolean;
  selectedName: string | null;
  onFileChange: (file: File) => void;
  onAnalyze: () => void;
  onDemo: () => void;
};

export function UploadSection({ disabled, selectedName, onFileChange, onAnalyze, onDemo }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const onPick = useCallback(() => inputRef.current?.click(), []);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      const f = list?.[0];
      if (f) onFileChange(f);
    },
    [onFileChange],
  );

  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 shadow-xl backdrop-blur">
      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") onPick();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 transition ${
          dragOver ? "border-sky-500 bg-sky-500/5" : "border-zinc-700 hover:border-zinc-500"
        }`}
        onClick={onPick}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/x-matroska"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="rounded-full bg-zinc-800 p-4 text-sky-400">
          <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-lg font-medium text-zinc-100">Drop an MP4 here, or click to browse</p>
          <p className="mt-1 text-sm text-zinc-400">Max ~50 MB · 16 uniformly sampled face frames</p>
          {selectedName ? (
            <p className="mt-3 text-sm font-medium text-sky-300">Selected: {selectedName}</p>
          ) : null}
        </div>
      </div>
      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="button"
          disabled={disabled || !selectedName}
          onClick={(e) => {
            e.stopPropagation();
            onAnalyze();
          }}
          className="inline-flex items-center justify-center rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Analyze Video
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={(e) => {
            e.stopPropagation();
            onDemo();
          }}
          className="inline-flex items-center justify-center rounded-lg border border-zinc-600 bg-zinc-800 px-5 py-2.5 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Try demo (no upload)
        </button>
      </div>
    </section>
  );
}
