import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ResultsDisplay.css';

const ResultsDisplay = ({ results }) => {
  if (!results) {
    return (
      <div className="results-display">
        <h2>Results</h2>
        <p>Run a simulation to see results...</p>
      </div>
    );
  }

  const formatNumber = (num) => {
    if (!num) return 'N/A';
    if (num > 1e6) return num.toExponential(2);
    return num.toFixed(2);
  };

  const validationClass = (isValid) => isValid ? 'valid' : 'invalid';

  // Prepare chart data
  const chartData = [
    { name: 'Tunneling', value: Math.log10(results.results?.tunneling_probability || 1) },
    { name: 'Screening', value: Math.log10((results.results?.screening_energy || 1) * 100) },
    { name: 'Lattice', value: Math.log10((results.results?.lattice_enhancement || 1) * 1e6) },
    { name: 'Interface', value: Math.log10(results.results?.interface_enhancement || 1) },
    { name: 'Total', value: Math.log10(results.results?.total_enhancement || 1) }
  ];

  return (
    <div className="results-display">
      <h2>Simulation Results</h2>
      
      <div className="result-section">
        <h3>Enhancement Factors</h3>
        <div className="result-item">
          <span>Total Enhancement:</span>
          <strong className="highlight">{formatNumber(results.results?.total_enhancement)}x</strong>
        </div>
        <div className="result-item">
          <span>Tunneling Enhancement:</span>
          <strong>{formatNumber(results.results?.tunneling_probability)}x</strong>
        </div>
        <div className="result-item">
          <span>Lattice Enhancement:</span>
          <strong>{formatNumber(results.results?.lattice_enhancement)}x</strong>
        </div>
        <div className="result-item">
          <span>Interface Enhancement:</span>
          <strong>{formatNumber(results.results?.interface_enhancement)}x</strong>
        </div>
      </div>

      <div className="result-section">
        <h3>Physical Quantities</h3>
        <div className="result-item">
          <span>Screening Energy:</span>
          <strong>{results.results?.screening_energy?.toFixed(1) || 'N/A'} eV</strong>
        </div>
        <div className="result-item">
          <span>Energy Concentration:</span>
          <strong>{results.results?.energy_concentration?.toFixed(1) || 'N/A'} eV/atom</strong>
        </div>
        <div className="result-item">
          <span>Max Interface Field:</span>
          <strong>{formatNumber(results.results?.max_interface_field)} V/m</strong>
        </div>
        <div className="result-item">
          <span>Coherent Energy:</span>
          <strong>{results.results?.coherent_energy?.toFixed(3) || 'N/A'} eV/atom</strong>
        </div>
      </div>

      {results.results?.reaction_rate && (
        <div className="result-section">
          <h3>Reaction Rates</h3>
          <div className="result-item">
            <span>Reaction Rate:</span>
            <strong>{formatNumber(results.results.reaction_rate)} /s</strong>
          </div>
          <div className="result-item">
            <span>Power Density:</span>
            <strong>{formatNumber(results.results.power_density)} W/m³</strong>
          </div>
        </div>
      )}

      {results.results?.validation && (
        <div className="result-section">
          <h3>Validation Against Paper</h3>
          <div className={`result-item ${validationClass(results.results.validation.enhancement_in_range)}`}>
            <span>Enhancement (10³-10⁵):</span>
            <strong>{results.results.validation.enhancement_in_range ? '✓' : '✗'}</strong>
          </div>
          <div className={`result-item ${validationClass(results.results.validation.screening_in_range)}`}>
            <span>Screening (10-100 eV):</span>
            <strong>{results.results.validation.screening_in_range ? '✓' : '✗'}</strong>
          </div>
          <div className={`result-item ${validationClass(results.results.validation.field_in_range)}`}>
            <span>Field (10⁹-10¹¹ V/m):</span>
            <strong>{results.results.validation.field_in_range ? '✓' : '✗'}</strong>
          </div>
          <div className={`result-item ${validationClass(results.results.validation.energy_concentration_in_range)}`}>
            <span>Energy (10-100 eV):</span>
            <strong>{results.results.validation.energy_concentration_in_range ? '✓' : '✗'}</strong>
          </div>
        </div>
      )}

      <div className="result-section">
        <h3>Enhancement Components (log₁₀)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" name="log₁₀(Enhancement)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="timestamp">
        Simulation ID: {results.simulation_id}<br />
        Completed: {new Date(results.completed_at || results.created_at).toLocaleString()}
      </div>
    </div>
  );
};

export default ResultsDisplay;
