import React from "react";
import { Link } from "react-router-dom";
import Background from "../components/HomeBackground";
import FeatureList from "../components/FeatureList";
import CTAHomepage from "../components/CTAHomepage";

function Home() {
  const base = import.meta.env.BASE_URL || "/";
  
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      <div className="absolute w-full h-full">
        <Background />
      </div>
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-8">
        <div className="max-w-7xl flex mx-auto">
          <div className="w-5/8 space-y-8">
            {/* Hero Section */}
            <h1 className="text-6xl font-bold mb-16 drop-shadow-lg">
              Ready to Predict Next EPL Season!! 🔥
            </h1>
            <p className="text-xl text-justify text-indent-40 text-gray-300 mb-16 drop-shadow-md">
              Dive into the world of football analytics with our cutting-edge
              Premier League Match Predictor. Leveraging historical data and
              advanced machine learning algorithms, our tool provides accurate
              predictions for upcoming matches. Whether you're a die-hard fan,
              a fantasy league enthusiast, or just curious about the beautiful
              game, our predictor offers insights that can enhance your viewing
              experience. Get ready to make informed predictions and elevate
              your football knowledge to the next level!
            </p>
            <a
              href="/predict"
              className="py-3 px-10 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-xl font-semibold text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg inline-block"
            >
              Predict Now!
            </a>
          </div>
          <div className="w-3/8 flex justify-end items-center">
            <img src={`${base}assets/Premier-League-Logo-White.png`} alt="Hero" className="w-96 h-auto pr-4" />
          </div>
        </div>
      </div>
      <FeatureList />
      <CTAHomepage />
    </div>
  );
}

export default Home;
