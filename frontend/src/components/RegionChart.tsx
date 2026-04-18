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
  scores: Record<string, number>;
};

export function RegionChart({ scores }: Props) {
  const data = Object.entries(scores).map(([region, score]) => ({ region, score }));
  return (
    <div className="h-72 w-full rounded-xl border border-zinc-800 bg-zinc-900/50 p-3">
      <h3 className="mb-2 text-sm font-semibold text-zinc-200">Face region analysis</h3>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis type="number" domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="region"
            width={88}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
            labelStyle={{ color: "#e4e4e7" }}
          />
          <Bar dataKey="score" fill="#a78bfa" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
