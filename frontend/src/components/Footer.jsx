import React from "react";
import { Link } from "react-router-dom";

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="mx-auto w-[calc(100%-2rem)] max-w-7xl mt-12 z-40">
      <div className="backdrop-blur-md bg-gradient-to-r from-white/6 via-white/4 to-white/6 border border-white/10 rounded-t-3xl shadow-2xl p-6 flex flex-col md:flex-row items-start justify-between gap-4 text-white">
        {/* Brand */}
        <div className="flex items-start gap-4">
          <div className="p-2 items-center justify-center rounded-xl bg-white/8 border border-white/8">
            <img
              src="/assets/Premier-League-Logo-White.png"
              alt=""
              className="w-8 h-auto"
            />
          </div>
          <div>
            <div className="text-lg font-semibold">EPL Predictor</div>
            <div className="text-sm text-white/70">
              Prediksi pertandingan EPL dengan XGBoost
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex gap-12">
          <div className="flex flex-col gap-2 mb-4">
            <h4 className="text-lg font-bold">Navigation</h4>
            <Link
              to="/"
              className="text-sm text-white/50 hover:text-white/90 transition-colors"
            >
              Home
            </Link>
            <Link
              to="/predict"
              className="text-sm text-white/50 hover:text-white/90 transition-colors"
            >
              Predict
            </Link>
            <Link
              to="/about"
              className="text-sm text-white/50 hover:text-white/90 transition-colors"
            >
              About
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h4 className="text-lg font-bold">Resources</h4>

            {/* External links should use <a> not react-router <Link> */}
            <a
              href="https://github.com/kullal/Premier-League-Prediction-Using-XGBoost"
              className="text-sm text-white/50 hover:text-white/90 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Repositories
            </a>

            <a
              href="https://github.com/kullal/Premier-League-Prediction-Using-XGBoost/issues"
              className="text-sm text-white/50 hover:text-white/90 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Report Issues
            </a>
          </div>
        </nav>

        {/* Right */}
        <div className="flex items-center gap-4">
          <div className="flex gap-3 items-center">
            <a
              href="https://github.com/kullal/Premier-League-Prediction-Using-XGBoost"
              aria-label="GitHub"
              className="p-2 rounded-md bg-white/6 hover:bg-white/8 transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 2C6.48 2 2 6.48 2 12c0 4.42 2.87 8.17 6.84 9.49.5.09.68-.22.68-.48 0-.24-.01-.87-.01-1.71-2.78.6-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.89 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.55-1.11-4.55-4.95 0-1.09.39-1.98 1.03-2.67-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.56 9.56 0 0112 6.8c.85.004 1.71.116 2.51.34 1.9-1.3 2.74-1.02 2.74-1.02.55 1.37.2 2.39.1 2.64.64.69 1.03 1.58 1.03 2.67 0 3.85-2.34 4.7-4.57 4.95.36.31.68.92.68 1.86 0 1.34-.01 2.42-.01 2.75 0 .26.18.58.69.48C19.14 20.17 22 16.42 22 12c0-5.52-4.48-10-10-10z"
                  fill="white"
                  fillOpacity="0.9"
                />
              </svg>
            </a>
            <div className="text-xs text-white/60">
              © {year} EPL Predictor · Built with React & Flask
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
