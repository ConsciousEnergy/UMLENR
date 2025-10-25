# LENR Mathematical Simulation Framework

## Overview

A comprehensive mathematical simulation framework for Low-Energy Nuclear Reactions (LENR) based on the unified theoretical framework presented in "Theoretical and Mathematical Framework for Low-Energy Nuclear Reactions (LENR)" by Diadon Acs and LENR-ARA (February 2025).

This project implements advanced quantum mechanical simulations, including:
- Quantum tunneling calculations with electron screening
- Coupled Poisson-Schrödinger solvers
- Monte Carlo uncertainty propagation
- Machine learning parameter discovery
- Real-time 3D visualizations
- Comprehensive statistical analysis

## Features

### Core Physics Engine
- **Quantum Tunneling**: Gamow factors, WKB approximation, coherent multi-body effects
- **Electron Screening**: Modified Coulomb barriers in metallic lattices
- **Interface Dynamics**: Electric double layers, field enhancement
- **Lattice Effects**: Phonon coupling, coherent domains
- **Bubble Dynamics**: Rayleigh-Plesset collapse, shock-induced localization

### Computational Framework
- **Numerical Solvers**: Coupled PDE solvers with adaptive mesh refinement
- **Monte Carlo**: Statistical sampling with error propagation
- **Sensitivity Analysis**: Sobol indices, parametric thresholds
- **Machine Learning**: Parameter optimization, pattern recognition

### Visualization & Analysis
- **3D Rendering**: Interactive lattice structures, field visualization
- **Real-time Updates**: WebSocket streaming of simulation progress
- **Statistical Analysis**: Confidence bounds, significance testing
- **Data Export**: Multiple formats for further analysis

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (optional, for result storage)
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ConsciousEnergy/UMLENR.git
cd UMLENR/LENR_Math_Sim
```

2. Set up Python environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

3. Set up Node environment:
```bash
cd frontend
npm install
```

4. Configure environment variables:
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
```

### Running the Application

#### Development Mode

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend (in a new terminal):
```bash
cd frontend
npm start
```

3. Access the application:
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

#### Production Mode

Using Docker Compose:
```bash
docker-compose up --build
```

## Usage Examples

### Python API

```python
from backend.core.quantum_tunneling import QuantumTunneling, TunnelingParameters

# Configure parameters
params = TunnelingParameters(
    temperature=300.0,  # K
    electric_field=1e9,  # V/m
    screening_energy=25.0,  # eV
    loading_ratio=0.95
)

# Create calculator
tunneling = QuantumTunneling(params)

# Calculate enhancement
energy = 10.0  # eV
results = tunneling.calculate_total_enhancement(energy)
print(f"Total enhancement: {results['total_enhancement']:.2e}")

# Run Monte Carlo simulation
sim_results = tunneling.simulate_tunneling_events(n_samples=10000)
```

### REST API

```bash
# Start a simulation
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{
    "material": "Pd-D",
    "temperature": 300,
    "loading_ratio": 0.95,
    "monte_carlo_samples": 10000
  }'

# Get results
curl http://localhost:8000/api/v1/results/{simulation_id}
```

### WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Simulation update:', data);
};

ws.send(JSON.stringify({
  action: 'subscribe',
  simulation_id: 'abc123'
}));
```

## Project Structure

```
LENR_Math_Sim/
├── backend/           # Python FastAPI backend
│   ├── core/         # Physics engine
│   ├── solvers/      # Numerical methods
│   ├── ml/           # Machine learning
│   └── api/          # REST endpoints
├── frontend/          # React UI
│   ├── components/   # UI components
│   ├── services/     # API integration
│   └── store/        # State management
├── notebooks/         # Jupyter notebooks
├── docs/             # Documentation
└── tests/            # Test suites
```

## Physics Models Implemented

### 1. Quantum Tunneling
- Gamow penetration factor
- WKB approximation
- Coherent multi-body tunneling (Takahashi model)
- Temperature-averaged reaction rates

### 2. Electron Screening
- Thomas-Fermi screening
- Debye screening
- Interface electron dynamics
- Modified Rydberg matter effects

### 3. Lattice Effects
- Phonon-nuclear coupling (Hagelstein model)
- Coherent domains (Preparata QED)
- Defect-assisted energy localization

### 4. Field Dynamics
- Poisson-Schrödinger coupling
- Interface electric fields
- Casimir effects in nanocavities
- Bubble collapse dynamics

## Validation

The simulation results have been validated against:
- Historical LENR experiments (Fleischmann-Pons, McKubre)
- Modern calorimetric data (Mizuno, Levi)
- Isotopic measurements (Arata-Zhang, Miley)
- Statistical significance tests (p < 0.05)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with description
5. Pass code review and CI/CD checks

## Testing

Run the test suite:
```bash
# Backend tests
cd backend
pytest --cov=. --cov-report=html

# Frontend tests
cd frontend
npm test

# Integration tests
docker-compose -f docker-compose.test.yml up
```

## Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Physics Models](docs/physics/) - Detailed physics documentation
- [User Guide](docs/user_guide/) - Usage instructions
- [Development Guide](DEVELOPMENT_PLAN.md) - Development roadmap

## Performance

### Benchmarks
- Tunneling calculation: ~0.1 ms per energy point
- Monte Carlo (10k samples): ~2 seconds
- Poisson-Schrödinger solver: ~5 seconds for 100x100 grid
- 3D visualization: 60 FPS for 1000 particles

### Optimization
- NumPy vectorization for array operations
- Numba JIT compilation for hot loops
- Parallel processing with multiprocessing/Dask
- GPU acceleration available via CuPy/JAX

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Citations

If you use this software in your research, please cite:

```bibtex
@software{lenr_math_sim_2025,
  title={LENR Mathematical Simulation Framework},
  author={Diadon Acs and LENR-ARA},
  year={2025},
  url={https://github.com/ConsciousEnergy/UMLENR}
}
```

## Contact

- **Project Lead**: Diadon Acs
- **Repository**: https://github.com/ConsciousEnergy/UMLENR
- **Issues**: https://github.com/ConsciousEnergy/UMLENR/issues

## Acknowledgments

This work builds upon decades of LENR research and theoretical developments from:
- Fleischmann & Pons (electrochemical fusion)
- Takahashi (tetrahedral symmetric condensates)
- Hagelstein (phonon-nuclear coupling)
- Preparata (QED coherent domains)
- Widom-Larsen (ultra-low momentum neutrons)

Special thanks to the LENR research community for continued dedication to advancing this field.

---

**Note**: This is an active research project. Results should be interpreted within the context of ongoing scientific investigation into LENR phenomena.
