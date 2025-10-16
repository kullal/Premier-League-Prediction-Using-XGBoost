import React from "react";
import { Link } from "react-router-dom";

export default function FeatureList({ features }) {
  const list = features ?? [
    {
      title: "High Accuracy",
      description: "XGBoost-powered predictions with proven performance.",
      label: "Accuracy",
    },
    {
      title: "Match Analytics",
      description: "In-depth stats and situational insights for every fixture.",
      label: "Analytics",
    },
    {
      title: "Easy to Use",
      description: "Generate predictions quickly with a simple interface.",
      label: "Usability",
    },
  ];

  return (
    <div className="py-12">
      <section className="max-w-7xl mx-auto mb-8 px-4">
        <h2 className="text-4xl md:text-5xl font-extrabold text-center mb-4">
          Main Features
        </h2>
        <p className="text-center text-gray-300 max-w-3xl mx-auto mb-8">
          Discover the key features that make our EPL Predictor a powerful tool
          for football enthusiasts and analysts alike.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
          {/* Konten kiri */}
          <div className="bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-3xl p-6 flex flex-col justify-between shadow-2xl overflow-hidden transform transition-transform duration-200 hover:-translate-y-1 cursor-pointer relative z-10">
            <div className="flex items-start gap-4">
              <div>
                <h3 className="text-2xl font-bold">EPL Match Predictor</h3>
                <p className="text-gray-300 mt-2">
                  Accurate, fast, and insightful predictions powered by XGBoost
                  and historical data.
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-end justify-between">
              <img
                src="/src/assets/Premier-League-Logo-White.png"
                alt="Logo EPL"
                className="w-24 h-auto opacity-70 hover:opacity-100 transition-opacity duration-300"
              />
              <Link
                to="/predict"
                className="inline-flex items-center gap-2 bg-white/6 hover:bg-white/15 text-white px-4 py-2 rounded-xl border border-white/10 shadow-sm transition"
              >
                Try it now!
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 12h14M12 5l7 7-7 7"
                  />
                </svg>
              </Link>
            </div>
          </div>

          {/* Bento cards grid */}
          <div className="col-span-1 md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6 items-stretch">
            {list.map((f, idx) => {
              const isFull = idx > 1;
              const articleClass = `relative group bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-2xl p-6 hover:shadow-xl transform transition-transform duration-300 hover:-translate-y-1 overflow-hidden flex flex-col cursor-pointer z-0 h-full ${
                isFull
                  ? "sm:col-span-2 sm:flex-col sm:items-start sm:justify-between"
                  : "sm:col-span-1 sm:flex-col sm:items-start sm:justify-between"
              }`;

              return (
                <a href="/predict" key={idx} className={articleClass}>
                  <div className="flex items-start justify-between w-full">
                    <div className="flex items-start gap-4">
                      <div>
                        <h4 className="text-lg font-semibold">{f.title}</h4>
                        <p className="text-sm text-gray-300 mt-2">
                          {f.description}
                        </p>
                      </div>
                    </div>

                    <span className="text-xs font-medium bg-white/6 text-white px-3 py-1 rounded-full border border-white/8">
                      {f.label}
                    </span>
                  </div>

                  <div className="mt-24 flex items-start justify-between w-full">
                    <div className="text-sm text-gray-400">Learn more</div>
                    <div className="text-xs text-gray-400">→</div>
                  </div>
                </a>
              );
            })}
          </div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto mb-10 text-center px-4">
        <p className="text-gray-400">
          Note: Features may vary based on the specific implementation and data
          sources used.
        </p>
      </section>
    </div>
  );
}
