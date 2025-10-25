import React, { useState } from 'react';
import './SimulationForm.css';

const SimulationForm = ({ onSubmit, loading }) => {
  const [parameters, setParameters] = useState({
    material: 'Pd',
    temperature: 300,
    loading_ratio: 0.95,
    electric_field: 1e10,
    surface_potential: 0.5,
    defect_density: 1e21,
    coherence_domain_size: 1000,
    energy: 10.0
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParameters(prev => ({
      ...prev,
      [name]: name.includes('field') || name.includes('density') 
        ? parseFloat(value) 
        : parseFloat(value) || value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(parameters);
  };

  return (
    <form className="simulation-form" onSubmit={handleSubmit}>
      <h2>Simulation Parameters</h2>
      
      <div className="form-group">
        <label>Material</label>
        <select name="material" value={parameters.material} onChange={handleChange}>
          <option value="Pd">Palladium (Pd)</option>
          <option value="Ni">Nickel (Ni)</option>
          <option value="Ti">Titanium (Ti)</option>
        </select>
      </div>

      <div className="form-group">
        <label>Temperature (K)</label>
        <input
          type="number"
          name="temperature"
          value={parameters.temperature}
          onChange={handleChange}
          min="200"
          max="500"
          step="10"
        />
      </div>

      <div className="form-group">
        <label>Loading Ratio</label>
        <input
          type="number"
          name="loading_ratio"
          value={parameters.loading_ratio}
          onChange={handleChange}
          min="0"
          max="1"
          step="0.01"
        />
        <small>Critical: &gt;0.85</small>
      </div>

      <div className="form-group">
        <label>Electric Field (V/m)</label>
        <input
          type="number"
          name="electric_field"
          value={parameters.electric_field}
          onChange={handleChange}
          min="1e6"
          max="1e12"
          step="1e9"
        />
      </div>

      <div className="form-group">
        <label>Surface Potential (V)</label>
        <input
          type="number"
          name="surface_potential"
          value={parameters.surface_potential}
          onChange={handleChange}
          min="0"
          max="5"
          step="0.1"
        />
      </div>

      <div className="form-group">
        <label>Defect Density (/m³)</label>
        <input
          type="number"
          name="defect_density"
          value={parameters.defect_density}
          onChange={handleChange}
          min="1e18"
          max="1e23"
          step="1e20"
        />
      </div>

      <div className="form-group">
        <label>Coherence Domain (atoms)</label>
        <input
          type="number"
          name="coherence_domain_size"
          value={parameters.coherence_domain_size}
          onChange={handleChange}
          min="10"
          max="100000"
          step="100"
        />
      </div>

      <div className="form-group">
        <label>Incident Energy (eV)</label>
        <input
          type="number"
          name="energy"
          value={parameters.energy}
          onChange={handleChange}
          min="0.1"
          max="10000"
          step="1"
        />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? 'Running...' : 'Run Simulation'}
      </button>

      <div className="preset-buttons">
        <h3>Presets</h3>
        <button 
          type="button"
          onClick={() => setParameters({
            material: 'Pd',
            temperature: 300,
            loading_ratio: 0.95,
            electric_field: 1e10,
            surface_potential: 0.5,
            defect_density: 1e21,
            coherence_domain_size: 1000,
            energy: 10.0
          })}
        >
          High Enhancement
        </button>
        <button 
          type="button"
          onClick={() => setParameters({
            material: 'Pd',
            temperature: 300,
            loading_ratio: 0.90,
            electric_field: 5e9,
            surface_potential: 0.4,
            defect_density: 5e20,
            coherence_domain_size: 500,
            energy: 10.0
          })}
        >
          Stable Operation
        </button>
      </div>
    </form>
  );
};

export default SimulationForm;
