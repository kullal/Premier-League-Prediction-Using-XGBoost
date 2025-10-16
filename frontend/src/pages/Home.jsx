import React from "react";
import { Link } from "react-router-dom";
import Background from "../components/HomeBackground";
import heroImage from "../assets/Premier-League-Logo-White.png";
import FeatureList from "../components/FeatureList";

function Home() {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      <div className="absolute w-full h-full">
        <Background
          colorStops={["#0F0520", "#3A0CA3", "#6A4BC7", "#A57BFF", "#FF94B4"]}
          blend={0.9}
          amplitude={1.6}
          speed={0.32}
        />
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-8">
        <div className="max-w-7xl flex mx-auto">
          <div className="w-5/8 space-y-8">
            {/* Hero Section */}
            <h1 className="text-6xl font-bold mb-16 drop-shadow-lg">
              Ready to Predict Next EPL Season!! 🔥
            </h1>
            <p className="text-xl text-justify text-gray-300 mb-16 drop-shadow-md">
              <strong>Prediksi Match EPL Next Season 2025/2026</strong> dengan
              akurasi tinggi dan analisis mendalam. Dapatkan prediksi
              pertandingan yang akurat dan analisis mendalam untuk setiap
              pertandingan. Prediksi pertandingan EPL Next Season 2025/2026
              dengan akurasi tinggi dan analisis mendalam. Dapatkan prediksi
              pertandingan yang akurat dan analisis mendalam untuk setiap
              pertandingan.{" "}
            </p>
            <a
              href="/predict"
              className="py-3 px-10 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-xl font-semibold text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg inline-block"
            >
              Predict Now!
            </a>
          </div>
          <div className="w-3/8 flex justify-end items-center">
            <img src={heroImage} alt="Hero" className="w-96 h-auto pr-4" />
          </div>
        </div>
      </div>
      <FeatureList />
    </div>
  );
}

export default Home;
