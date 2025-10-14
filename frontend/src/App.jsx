import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Predict from './pages/Predict';
import About from './pages/About';
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
        <Router>
            <div className="min-h-screen bg-gray-900 text-white">
                <nav className="bg-gray-800 p-4">
                    <div className="max-w-7xl mx-auto flex justify-between items-center">
                        <Link to="/" className="text-xl font-bold">EPL Predictor</Link>
                        <div className="space-x-4">
                            <Link to="/" className="hover:text-blue-400">Home</Link>
                            <Link to="/predict" className="hover:text-blue-400">Predict</Link>
                            <Link to="/about" className="hover:text-blue-400">About</Link>
                        </div>
                    </div>
                </nav>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/predict" element={<Predict />} />
                    <Route path="/about" element={<About />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;