"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Props = {
  values: number[];
};

export function TemporalChart({ values }: Props) {
  const data = values.map((v, i) => ({ frame: String(i + 1), score: v }));
  return (
    <div className="h-72 w-full rounded-xl border border-zinc-800 bg-zinc-900/50 p-3">
      <h3 className="mb-2 text-sm font-semibold text-zinc-200">Temporal importance</h3>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis dataKey="frame" tick={{ fill: "#a1a1aa", fontSize: 11 }} interval={1} />
          <YAxis tick={{ fill: "#a1a1aa", fontSize: 11 }} domain={[0, 1]} width={32} />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
            labelStyle={{ color: "#e4e4e7" }}
          />
          <Bar dataKey="score" fill="#38bdf8" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
