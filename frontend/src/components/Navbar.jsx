import React from "react";
import { Link } from "react-router-dom";

export default function Navbar() {
  const [open, setOpen] = React.useState(false);

  return (
    <nav className="fixed left-1/2 transform -translate-x-1/2 top-6 w-[calc(100%-2rem)] max-w-7xl z-50">
      <div className="flex items-center justify-between px-6 py-4 rounded-3xl bg-white/2 backdrop-blur-md border border-white/10 shadow-2xl transition-transform duration-300 ease-out hover:-translate-y-1">
        <Link
          to={"/"}
          className="flex items-center gap-3 font-bold text-white drop-shadow"
        >
          <div className="flex items-center">
            <img
              src="/assets/Premier-League-Logo-White.png"
              alt="Hero"
              className="w-12 h-auto pr-4"
            />
          </div>
          <h2 className="md:text-2xl sm:text-xl">EPL Predictor</h2>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex gap-6 font-semibold">
          <Link to={"/"} className="text-white hover:text-gray-400">
            Home
          </Link>
          <Link to={"/predict"} className="text-white hover:text-gray-400">
            Predict
          </Link>
          <Link to={"/about"} className="text-white hover:text-gray-400">
            About
          </Link>
        </div>

        <Link
          to={
            "https://github.com/kullal/Premier-League-Prediction-Using-XGBoost/tree/main"
          }
          className="hidden md:inline-block py-2 px-6 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-lg font-semibold text-white rounded-full hover:bg-white/15 transition-colors duration-300"
          style={{ zIndex: 2 }}
        >
          <svg
            className="inline-block w-5 h-5 mr-2 -mt-1"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              fillRule="evenodd"
              d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              clipRule="evenodd"
            />
          </svg>
          Docs
        </Link>

        {/* Mobile Toggle */}
        <div className="md:hidden">
          <button
            onClick={() => setOpen((s) => !s)}
            aria-label="Toggle menu"
            aria-expanded={open}
            className="p-2 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-white/30"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d={open ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile Menu (floating glass) */}
      {open && (
        <div
          className="mt-3 mx-2 md:hidden rounded-2xl bg-white/6 backdrop-blur-md border border-white/10
                                            shadow-xl py-3 px-4 flex flex-col gap-2 text-white transition-opacity duration-200"
        >
          <Link
            to={"/"}
            onClick={() => setOpen(false)}
            className="py-2 px-3 rounded hover:bg-white/5"
          >
            Home
          </Link>
          <Link
            to={"/predict"}
            onClick={() => setOpen(false)}
            className="py-2 px-3 rounded hover:bg-white/5"
          >
            Predict
          </Link>
          <Link
            to={"/about"}
            onClick={() => setOpen(false)}
            className="py-2 px-3 rounded hover:bg-white/5"
          >
            About
          </Link>
          <Link
            to={"https://github.com/kullal/Premier-League-Prediction-Using-XGBoost/tree/main"
            }
            onClick={() => setOpen(false)}
            className="py-2 px-3 rounded hover:bg-white/5"
          >
            <svg
            className="inline-block w-5 h-5 mr-2 -mt-1"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              fillRule="evenodd"
              d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              clipRule="evenodd"
            />
          </svg>
            Docs
          </Link>
        </div>
      )}
    </nav>
  );
}
