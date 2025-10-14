import React from 'react';

function About() {
  return (
    <div className="bg-gray-900 text-white min-h-screen p-8">
      <h1 className="text-4xl font-bold text-center mb-8">About Us</h1>
      <p className="text-center text-lg">This app uses XGBoost to predict Premier League match outcomes based on historical data.</p>
      <p className="text-center mt-4">Built with React, Flask, and Tailwind CSS.</p>
    </div>
  );
}

export default About;