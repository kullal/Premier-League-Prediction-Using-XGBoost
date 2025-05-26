import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const PredictionContainer = styled.div`
  background: linear-gradient(to bottom, #0a192f, #112240);
  border-radius: 10px;
  padding: 2rem;
  margin: 2rem 0;
  color: white;
`;

const SectionTitle = styled.h2`
  text-align: center;
  margin-bottom: 2rem;
`;

const TeamsContainer = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 2rem;
`;

const TeamCard = styled.div`
  background: rgba(10, 25, 47, 0.7);
  border-radius: 10px;
  padding: 1.5rem;
  width: 45%;
  text-align: center;
`;

const TeamLogo = styled.img`
  width: 120px;
  height: 120px;
  margin-bottom: 1rem;
`;

const TeamName = styled.h3`
  margin-bottom: 0.5rem;
`;

const VsContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  font-weight: bold;
`;

const ResultsContainer = styled.div`
  margin-top: 2rem;
`;

const WinnerCard = styled.div`
  background: rgba(10, 25, 47, 0.7);
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  text-align: center;
`;

const WinnerTitle = styled.h3`
  margin-bottom: 1rem;
`;

const WinnerContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const MatchPrediction = () => {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch teams from API
    const fetchTeams = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/teams');
        setTeams(response.data);
        if (response.data.length > 0) {
          setHomeTeam(response.data[0]);
          setAwayTeam(response.data[1]);
        }
      } catch (error) {
        console.error('Error fetching teams:', error);
      }
    };

    fetchTeams();
  }, []);

  const handlePredict = async () => {
    if (homeTeam === awayTeam) {
      alert('Please select different teams');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/api/predict', {
        home_team: homeTeam,
        away_team: awayTeam,
        match_date: new Date().toISOString().split('T')[0]
      });
      setPrediction(response.data);
    } catch (error) {
      console.error('Error making prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  const getWinnerTeam = () => {
    if (!prediction) return null;
    
    const homeWinProb = parseFloat(prediction.match_result['Home Team Win'].replace('%', ''));
    const awayWinProb = parseFloat(prediction.match_result['Away Team Win'].replace('%', ''));
    const drawProb = parseFloat(prediction.match_result['Draw'].replace('%', ''));
    
    if (drawProb > homeWinProb && drawProb > awayWinProb) {
      return 'Draw';
    }
    return homeWinProb > awayWinProb ? homeTeam : awayTeam;
  };

  const resultChartData = {
    labels: ['Home Win', 'Draw', 'Away Win'],
    datasets: [
      {
        label: 'Probability (%)',
        data: prediction ? [
          parseFloat(prediction.match_result['Home Team Win'].replace('%', '')),
          parseFloat(prediction.match_result['Draw'].replace('%', '')),
          parseFloat(prediction.match_result['Away Team Win'].replace('%', ''))
        ] : [0, 0, 0],
        backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(153, 102, 255, 0.6)', 'rgba(255, 159, 64, 0.6)'],
      },
    ],
  };

  const goalTimingChartData = {
    labels: ['First Half', 'Second Half'],
    datasets: [
      {
        label: 'Probability (%)',
        data: prediction ? [
          parseFloat(prediction.goal_timing.first_half_goals.replace('%', '')),
          parseFloat(prediction.goal_timing.second_half_goals.replace('%', ''))
        ] : [0, 0],
        backgroundColor: ['rgba(54, 162, 235, 0.6)', 'rgba(255, 99, 132, 0.6)'],
      },
    ],
  };

  return (
    <PredictionContainer>
      <SectionTitle>Prediction Match</SectionTitle>
      
      <TeamsContainer>
        <TeamCard>
          <TeamLogo src={`/team-logos/${homeTeam.replace(/ /g, '-').toLowerCase()}.png`} alt={homeTeam} />
          <TeamName>{homeTeam}</TeamName>
          <select value={homeTeam} onChange={(e) => setHomeTeam(e.target.value)}>
            {teams.map(team => (
              <option key={`home-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </TeamCard>
        
        <VsContainer>VS</VsContainer>
        
        <TeamCard>
          <TeamLogo src={`/team-logos/${awayTeam.replace(/ /g, '-').toLowerCase()}.png`} alt={awayTeam} />
          <TeamName>{awayTeam}</TeamName>
          <select value={awayTeam} onChange={(e) => setAwayTeam(e.target.value)}>
            {teams.map(team => (
              <option key={`away-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </TeamCard>
      </TeamsContainer>
      
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Match'}
      </button>
      
      {prediction && (
        <ResultsContainer>
          <WinnerCard>
            <WinnerTitle>The Winner</WinnerTitle>
            <WinnerContent>
              <TeamLogo 
                src={`/team-logos/${getWinnerTeam().replace(/ /g, '-').toLowerCase()}.png`} 
                alt={getWinnerTeam()} 
              />
              <TeamName>{getWinnerTeam()}</TeamName>
            </WinnerContent>
          </WinnerCard>
          
          <div>
            <h3>Winner Prediction</h3>
            <Bar data={resultChartData} />
          </div>
          
          <div>
            <h3>Goal Timing</h3>
            <Bar data={goalTimingChartData} />
          </div>
        </ResultsContainer>
      )}
    </PredictionContainer>
  );
};

export default MatchPrediction;