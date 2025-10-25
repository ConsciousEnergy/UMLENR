import React, { useState } from 'react';
import './App.css';
import SimulationForm from './components/SimulationForm';
import ResultsDisplay from './components/ResultsDisplay';
import Visualization3D from './components/Visualization3D';
import { runSimulation, getParameterRanges } from './api/simulation';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSimulation = async (parameters) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await runSimulation(parameters);
      setResults(result);
    } catch (err) {
      setError(err.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>LENR Mathematical Simulation Framework</h1>
        <p>Based on "Theoretical and Mathematical Framework for Low-Energy Nuclear Reactions"</p>
      </header>
      
      <div className="container">
        <div className="left-panel">
          <SimulationForm onSubmit={handleSimulation} loading={loading} />
          {error && <div className="error-message">{error}</div>}
        </div>
        
        <div className="center-panel">
          <h2>3D Visualization</h2>
          <Visualization3D results={results} />
        </div>
        
        <div className="right-panel">
          <ResultsDisplay results={results} />
        </div>
      </div>
    </div>
  );
}

export default App;
