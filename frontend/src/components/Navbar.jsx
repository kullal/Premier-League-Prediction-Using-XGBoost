import React from "react";
import { Link } from "react-router-dom";

export default function Navbar() {
  const [open, setOpen] = React.useState(false);

  return (
    <nav className="fixed left-1/2 transform -translate-x-1/2 top-6 w-[calc(100%-2rem)] max-w-7xl z-50">
      <div className="flex items-center justify-between px-6 py-4 rounded-3xl bg-white/2 backdrop-blur-md border border-white/10 shadow-2xl transition-transform duration-300 ease-out hover:-translate-y-1">
        <Link to={"/"} className="flex items-center gap-3 font-bold text-white drop-shadow">
          <div className="flex items-center">
            <img src="/assets/Premier-League-Logo-White.png" alt="Hero" className="w-12 h-auto pr-4" />
          </div>
          <h2 className="text-2xl">EPL Predictor</h2>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex gap-6 font-semibold">
          <Link to={"/"} className="text-white hover:text-red-500">
            Home
          </Link>
          <Link to={"/predict"} className="text-white hover:text-red-500">
            Predict
          </Link>
          <Link to={"/about"} className="text-white hover:text-red-500">
            About
          </Link>
        </div>

        <Link
            to={
              "https://github.com/kullal/Premier-League-Prediction-Using-XGBoost"
            }
            className="text-white hover:text-black-200 hover:bg-white/15 px-6 py-2 rounded-3xl bg-white/2 backdrop-blur-md border border-white/10 shadow-2xl"
          >
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
        </div>
      )}
    </nav>
  );
}
