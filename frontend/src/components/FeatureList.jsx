import React from "react";

export default function FeatureList({ features }) {
  const list = features ?? [
    {
      title: "High Accuracy",
      desc: "XGBoost-powered predictions with proven performance.",
    },
    {
      title: "Match Analytics",
      desc: "In-depth stats and situational insights for every fixture.",
    },
    {
      title: "Easy to Use",
      desc: "Generate predictions quickly with a simple interface.",
    },
  ];

  return (
    <div className="">
      <section className="max-w-7xl mx-auto mb-10">
        <h2 className="text-5xl font-bold text-center mb-16">Main Features</h2>
        <div className="flex justify-between sm:flex-col lg:flex-row gap-6">
          {list.map((f, i) => (
            <div
              key={i}
              className="backdrop-blur-md bg-white/5 border border-white/10 rounded-xl p-4 hover:scale-[1.02] transition-transform"
            >
              <div className="text-3xl font-semibold mb-8">{f.title}</div>
              <p className="text-lg text-gray-300">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
