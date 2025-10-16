import React, { useState, useEffect } from "react";
import { Menu } from "@headlessui/react";

const API_URL = "http://localhost:5000/api";

function Predict() {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState("Predict History");
  const BASE = import.meta.env.BASE_URL || "/"; // public asset base

  const teamNameToLogoMap = {
    Tottenham: "Tottenham_Hotspur",
    Leicester: "Leicester_City",
    "West Ham": "West_Ham_United",
    Newcastle: "Newcastle_United",
    Brighton: "Brighton__Hove_Albion",
    "Brighton & Hove Albion": "Brighton__Hove_Albion",
    Wolves: "Wolverhampton_Wanderers",
    Wolverhampton: "Wolverhampton_Wanderers",
    Leeds: "Leeds_United",
    Norwich: "Norwich_City",
    "Norwich City": "Norwich_City",
    "West Brom": "West_Bromwich_Albion",
    "West Bromwich": "West_Bromwich_Albion",
    "West Bromwich Albion": "West_Bromwich_Albion",
    "Man City": "Manchester_City",
    "Manchester City": "Manchester_City",
    "Man United": "Manchester_United",
    "Man Utd": "Manchester_United",
    "Manchester United": "Manchester_United",
    "Sheffield Utd": "Sheffield_United",
    "Sheffield United": "Sheffield_United",
    "Nott'm Forest": "Nottingham_Forest",
    "Nottm Forest": "Nottingham_Forest",
    "Nottingham Forest": "Nottingham_Forest",
  };

  const getTeamLogo = (teamName) => {
    const fallback = `${BASE}assets/Premier-League-Logo-White.png`;
    if (!teamName) return fallback;

    let filename = teamNameToLogoMap[teamName] || teamName;
    filename = filename
      .trim()
      .replace(/\s+/g, "_")
      .replace(/[^\w\-_.]/g, "_");

    return `${BASE}assets/team-logos/${encodeURIComponent(filename)}.png`;
  };

  useEffect(() => {
    fetch(`${API_URL}/teams`)
      .then((res) => res.json())
      .then((data) => {
        if (data.teams) setTeams(data.teams);
      })
      .catch((err) => console.error("Error fetching teams:", err));
  }, []);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam) {
      alert("Please select both teams.");
      return;
    }
    setLoading(true);
    setPrediction(null);
    try {
      const response = await fetch(
        `${API_URL}/predict/history?home_team=${encodeURIComponent(
          homeTeam
        )}&away_team=${encodeURIComponent(awayTeam)}`
      );
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Prediction failed:", error);
      setPrediction({
        error: "Prediction failed. Is the backend server running?",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-black text-white overflow-visible">
      <div className="absolute w-full h-full">
        <img
          src={`${BASE}assets/nighttime-serenity-a-3d-rendering_13271352.jpg`}
          alt=""
          className="h-screen w-screen object-cover opacity-50"
        />
      </div>
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-8">
        <div className="max-w-4xl mx-auto item-center">
          <h1 className="text-6xl text-center font-bold mb-12 drop-shadow-lg">
            EPL Match Predictor 2025/2026 !
          </h1>
          <p className="text-xl text-center justify-center text-gray-300 mb-16 drop-shadow-md">
            <strong>Welcome to the EPL Match Predictor!</strong> Predict the
            outcome of upcoming Premier League matches using historical data and
            advanced machine learning algorithms. Select your teams below and
            click "Predict Now!" to see the results.
          </p>

          <div className="flex justify-center">
            <button
              type="button"
              onClick={() =>
                document
                  .getElementById("prediction-top")
                  .scrollIntoView({ behavior: "smooth", block: "start" })
              }
              className="py-3 px-10 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-xl font-semibold text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg"
            >
              Predict Now!
            </button>
          </div>
        </div>
      </div>

      {/* Prediction */}
      <div
        id="prediction-top"
        className="relative my-8 mx-auto w-full flex lg:flex-col sm:flex-row items-center justify-center gap-4"
      >
        <h2 className="text-6xl font-bold text-center">Prediction </h2>
        <p className="text-xl mb-16 text-center">
          Select your best team to predict the match outcome
        </p>

        <div className="w-full flex gap-2 mx-auto justify-center items-center">
          <h4 className="text-xl font-bold">Select your prediction method:</h4>
          <Menu as="div" className="relative inline-block text-left">
            <Menu.Button className="py-3 px-8 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-xl text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg flex items-center gap-2">
              {method}
              <span className="text-sm opacity-70">▾</span>
            </Menu.Button>

            <Menu.Items className="absolute right-0 mt-2 w-56 bg-white/5 backdrop-blur-md border border-white/10 shadow text-xl font-semibold rounded-xl text-white z-50 focus:outline-none">
              {["Predict History", "Predict Future"].map((m) => (
                <Menu.Item key={m}>
                  {({ active }) => (
                    <button
                      onClick={() => setMethod(m)}
                      className={`block w-full text-left px-4 py-2 text-lg ${
                        active ? "bg-gray-700" : "bg-transparent"
                      }`}
                    >
                      {m}
                    </button>
                  )}
                </Menu.Item>
              ))}
            </Menu.Items>
          </Menu>
        </div>

        <section className="w-full max-w-7xl mx-auto bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-xl text-white rounded-xl mt-8 p-8">
          <div className="flex flex-col items-center justify-center text-center w-full">
            <div className="max-w-3xl">
              <h2 className="text-3xl md:text-4xl font-bold mb-1">
                Choose Your Team
              </h2>
              <p className="text-sm text-gray-300 mb-4">
                Please select your teams from the dropdown menus.
              </p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* Home card */}
            <div className="bg-black/30 border border-white/6 rounded-xl p-6 flex flex-col min-h-[24rem]">
              <label className="block text-sm text-gray-300 mb-3">
                Choose Home Team:
              </label>
              <Menu
                as="div"
                className="relative inline-block w-full text-left mb-4"
              >
                <Menu.Button className="w-full py-3 px-4 bg-white/5 border border-white/8 rounded-lg text-left flex justify-between items-center">
                  <span className="truncate">
                    {homeTeam || "Select Home Team"}
                  </span>
                  <span className="opacity-60">▾</span>
                </Menu.Button>
                <Menu.Items className="absolute left-0 mt-2 w-full max-h-56 overflow-y-auto bg-white/10 backdrop-blur-[1000px] border border-white/10 rounded-lg z-50">
                  {teams.map((team) => (
                    <Menu.Item key={team}>
                      {({ active }) => (
                        <button
                          onClick={() => setHomeTeam(team)}
                          className={`w-full text-left px-4 py-2 ${
                            active ? "bg-white/10" : ""
                          }`}
                        >
                          {team}
                        </button>
                      )}
                    </Menu.Item>
                  ))}
                </Menu.Items>
              </Menu>

              <div className="flex-1 flex items-center justify-center bg-black/40 border border-white/6 rounded-lg p-4">
                <img
                  src={getTeamLogo(homeTeam)}
                  alt={homeTeam || "Home"}
                  onError={(e) => {
                    e.currentTarget.onerror = null;
                    e.currentTarget.src = `${BASE}assets/Premier-League-Logo-White.png`;
                  }}
                  className="max-h-72 object-contain transition-all"
                />
              </div>

              <div className="mt-4 text-center text-gray-300 text-sm">
                {homeTeam ? `${homeTeam} (Home)` : "Home"}
              </div>
            </div>

            {/* VS center */}
            <div className="flex flex-col items-center justify-center h-full">
              <div className="text-5xl font-bold">VS</div>
            </div>

            {/* Away card */}
            <div className="bg-black/30 border border-white/6 rounded-xl p-6 flex flex-col min-h-[24rem]">
              <label className="block text-sm text-gray-300 mb-3">
                Choose Away Team:
              </label>
              <Menu
                as="div"
                className="relative inline-block w-full text-left mb-4"
              >
                <Menu.Button className="w-full py-3 px-4 bg-white/5 border border-white/8 rounded-lg text-left flex justify-between items-center">
                  <span className="truncate">
                    {awayTeam || "Select Away Team"}
                  </span>
                  <span className="opacity-60">▾</span>
                </Menu.Button>
                <Menu.Items className="absolute left-0 mt-2 w-full max-h-56 overflow-y-auto bg-white/10 backdrop-blur-[1000px] border border-white/10 rounded-lg z-50">
                  {teams.map((team) => (
                    <Menu.Item key={team}>
                      {({ active }) => (
                        <button
                          onClick={() => setAwayTeam(team)}
                          className={`w-full text-left px-4 py-2 ${
                            active ? "bg-white/10" : ""
                          }`}
                        >
                          {team}
                        </button>
                      )}
                    </Menu.Item>
                  ))}
                </Menu.Items>
              </Menu>

              <div className="flex-1 flex items-center justify-center bg-black/40 border border-white/6 rounded-lg p-4">
                <img
                  src={getTeamLogo(awayTeam)}
                  alt={awayTeam || "Away"}
                  onError={(e) => {
                    e.currentTarget.onerror = null;
                    e.currentTarget.src = `${BASE}assets/Premier-League-Logo-White.png`;
                  }}
                  className="max-h-72 object-contain transition-all"
                />
              </div>

              <div className="mt-4 text-center text-gray-300 text-sm">
                {awayTeam ? `${awayTeam} (Away)` : "Away"}
              </div>
            </div>
          </div>
          <div className="flex justify-center mt-8">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="py-3 px-10 backdrop-blur-md border border-white/5  shadow-2xl text-xl font-semibold text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg disabled:opacity-50"
            >
              {loading ? "Loading..." : "Get Prediction"}
            </button>
          </div>
        </section>

        {/* Prediction Result */}
        <section>
          <div className="w-72 bg-white/6 border border-white/8 rounded-xl p-4 text-center">
            {prediction ? (
              prediction.error ? (
                <div className="text-red-400">{prediction.error}</div>
              ) : (
                <>
                  <div className="font-bold text-lg">
                    {prediction.predicted_outcome || "Prediksi"}
                  </div>
                  <div className="text-sm text-gray-300 mt-2">
                    {prediction.home_win_prob !== undefined ? (
                      <>
                        Home: {(prediction.home_win_prob * 100).toFixed(1)}% •
                        Draw: {(prediction.draw_prob * 100).toFixed(1)}% • Away:{" "}
                        {(prediction.away_win_prob * 100).toFixed(1)}%
                      </>
                    ) : (
                      <span className="text-gray-400">
                        Pilih tim lalu klik Dapatkan Prediksi
                      </span>
                    )}
                  </div>
                </>
              )
            ) : (
              <div className="text-gray-300">
                Pilih tim lalu klik Dapatkan Prediksi
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default Predict;
