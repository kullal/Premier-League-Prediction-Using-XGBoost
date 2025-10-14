import React, { useState, useEffect } from "react";
 
import './index.css';

const API_URL = 'http://localhost:5000/api';

function App() {
    const [teams, setTeams] = useState([]);
    const [homeTeam, setHomeTeam] = useState('');
    const [awayTeam, setAwayTeam] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetch(`${API_URL}/teams`)
            .then(res => res.json())
            .then(data => {
                if (data.teams) {
                    setTeams(data.teams);
                }
            })
            .catch(err => console.error("Error fetching teams:", err));
    }, []);

    const handlePredict = async () => {
        if (!homeTeam || !awayTeam) {
            alert("Please select both teams.");
            return;
        }
        setLoading(true);
        setPrediction(null);
        try {
            const response = await fetch(`${API_URL}/predict/history?home_team=${homeTeam}&away_team=${awayTeam}`);
            const data = await response.json();
            setPrediction(data);
        } catch (error) {
            console.error("Prediction failed:", error);
            setPrediction({ error: "Prediction failed. Is the backend server running?" });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen p-8 font-sans">
            <h1 className="text-4xl font-bold text-center mb-8">EPL Match Prediction</h1>
            
            <div className="max-w-2xl mx-auto bg-gray-800 p-6 rounded-lg shadow-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div>
                        <label htmlFor="home-team" className="block mb-2 text-sm font-medium">Home Team</label>
                        <select id="home-team" value={homeTeam} onChange={e => setHomeTeam(e.target.value)} className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5">
                            <option value="">Select Team</option>
                            {teams.map(team => <option key={team} value={team}>{team}</option>)}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="away-team" className="block mb-2 text-sm font-medium">Away Team</label>
                        <select id="away-team" value={awayTeam} onChange={e => setAwayTeam(e.target.value)} className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5">
                            <option value="">Select Team</option>
                            {teams.map(team => <option key={team} value={team}>{team}</option>)}
                        </select>
                    </div>
                </div>

                <button onClick={handlePredict} disabled={loading} className="w-full bg-blue-600 hover:bg-blue-700 focus:ring-4 focus:outline-none focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center disabled:bg-gray-500">
                    {loading ? 'Predicting...' : 'Get Prediction'}
                </button>

                {prediction && (
                    <div className="mt-8 p-4 bg-gray-700 rounded-lg">
                        {prediction.error ? (
                            <p className="text-red-400">{prediction.error}</p>
                        ) : (
                            <div>
                                <h2 className="text-2xl font-semibold mb-2">Prediction Result</h2>
                                <p><strong>Predicted Outcome:</strong> {prediction.predicted_outcome}</p>
                                <p><strong>Home Win Probability:</strong> {(prediction.home_win_prob * 100).toFixed(2)}%</p>
                                <p><strong>Draw Probability:</strong> {(prediction.draw_prob * 100).toFixed(2)}%</p>
                                <p><strong>Away Win Probability:</strong> {(prediction.away_win_prob * 100).toFixed(2)}%</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;