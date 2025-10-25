import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const runSimulation = async (parameters) => {
  try {
    // Create simulation request
    const response = await api.post('/api/v1/simulations/', {
      parameters: {
        material: parameters.material,
        temperature: parameters.temperature,
        loading_ratio: parameters.loading_ratio,
        electric_field: parameters.electric_field,
        surface_potential: parameters.surface_potential,
        defect_density: parameters.defect_density,
        coherence_domain_size: parameters.coherence_domain_size
      },
      energy: parameters.energy || 10.0,
      calculate_rate: true,
      include_validation: true
    });

    const simulationId = response.data.simulation_id;

    // Poll for results
    let attempts = 0;
    const maxAttempts = 20;
    
    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 500)); // Wait 500ms
      
      const resultResponse = await api.get(`/api/v1/simulations/${simulationId}`);
      
      if (resultResponse.data.status === 'completed') {
        return resultResponse.data;
      } else if (resultResponse.data.status === 'failed') {
        throw new Error(resultResponse.data.error || 'Simulation failed');
      }
      
      attempts++;
    }
    
    throw new Error('Simulation timeout');
  } catch (error) {
    console.error('Simulation error:', error);
    throw error;
  }
};

export const getParameterRanges = async () => {
  try {
    const response = await api.get('/api/v1/parameters/ranges');
    return response.data;
  } catch (error) {
    console.error('Failed to get parameter ranges:', error);
    return null;
  }
};

export const getOptimalParameters = async () => {
  try {
    const response = await api.get('/api/v1/parameters/optimal');
    return response.data;
  } catch (error) {
    console.error('Failed to get optimal parameters:', error);
    return null;
  }
};

export const runParameterScan = async (parameterName, values, baseParameters = null) => {
  try {
    const response = await api.post('/api/v1/parameters/scan', {
      parameter_name: parameterName,
      values: values,
      base_parameters: baseParameters,
      energy: 10.0
    });
    return response.data;
  } catch (error) {
    console.error('Parameter scan error:', error);
    throw error;
  }
};

export const suggestNextParameters = async (explorationWeight = 1.0) => {
  try {
    const response = await api.post('/api/v1/ml/optimize/suggest', null, {
      params: { exploration_weight: explorationWeight }
    });
    return response.data;
  } catch (error) {
    console.error('ML suggestion error:', error);
    return null;
  }
};
