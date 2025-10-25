# LENR Mathematical Simulation - Project Structure

## Directory Structure

```
LENR_Math_Sim/
│
├── backend/                    # FastAPI backend
│   ├── api/                   # API endpoints
│   │   ├── __init__.py
│   │   ├── simulation.py      # Simulation control endpoints
│   │   ├── parameters.py      # Parameter management
│   │   ├── results.py         # Results retrieval
│   │   └── websocket.py       # Real-time updates
│   │
│   ├── core/                  # Core physics engine
│   │   ├── __init__.py
│   │   ├── quantum_tunneling.py
│   │   ├── electron_screening.py
│   │   ├── lattice_effects.py
│   │   ├── interface_dynamics.py
│   │   ├── bubble_dynamics.py
│   │   └── enb_coupling.py
│   │
│   ├── solvers/               # Numerical solvers
│   │   ├── __init__.py
│   │   ├── poisson_schrodinger.py
│   │   ├── monte_carlo.py
│   │   ├── sensitivity_analysis.py
│   │   └── numerical_methods.py
│   │
│   ├── ml/                    # Machine learning components
│   │   ├── __init__.py
│   │   ├── parameter_discovery.py
│   │   ├── optimization.py
│   │   ├── bayesian_inference.py
│   │   └── models/
│   │
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── materials.py       # Material properties
│   │   ├── simulation.py      # Simulation parameters
│   │   ├── results.py         # Result structures
│   │   └── database.py        # Database models
│   │
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── constants.py       # Physical constants
│   │   ├── validators.py      # Input validation
│   │   ├── logging.py         # Logging configuration
│   │   └── cache.py           # Caching utilities
│   │
│   ├── tests/                 # Backend tests
│   │   ├── __init__.py
│   │   ├── test_core/
│   │   ├── test_solvers/
│   │   ├── test_api/
│   │   └── test_integration/
│   │
│   ├── config/                # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py        # Environment settings
│   │   └── database.py        # Database configuration
│   │
│   ├── main.py               # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile           # Backend container
│
├── frontend/                  # React frontend
│   ├── public/               # Static assets
│   │   └── index.html
│   │
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── Simulation/
│   │   │   │   ├── SimulationControl.tsx
│   │   │   │   ├── ParameterInput.tsx
│   │   │   │   └── ResultsDisplay.tsx
│   │   │   │
│   │   │   ├── Visualization/
│   │   │   │   ├── LatticeView3D.tsx
│   │   │   │   ├── EnergyDensityMap.tsx
│   │   │   │   ├── FieldVisualization.tsx
│   │   │   │   └── ParticleDynamics.tsx
│   │   │   │
│   │   │   ├── Charts/
│   │   │   │   ├── HeatOutput.tsx
│   │   │   │   ├── IsotopeProduction.tsx
│   │   │   │   └── StatisticalAnalysis.tsx
│   │   │   │
│   │   │   └── Layout/
│   │   │       ├── Header.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── Footer.tsx
│   │   │
│   │   ├── services/          # API services
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── types.ts
│   │   │
│   │   ├── store/             # State management
│   │   │   ├── index.ts
│   │   │   ├── simulationSlice.ts
│   │   │   └── resultsSlice.ts
│   │   │
│   │   ├── utils/             # Frontend utilities
│   │   │   ├── constants.ts
│   │   │   ├── helpers.ts
│   │   │   └── validators.ts
│   │   │
│   │   ├── App.tsx            # Main application
│   │   ├── index.tsx          # Entry point
│   │   └── styles/            # CSS/SCSS files
│   │
│   ├── package.json           # Node dependencies
│   ├── tsconfig.json         # TypeScript configuration
│   └── Dockerfile            # Frontend container
│
├── notebooks/                 # Jupyter notebooks
│   ├── theory_validation.ipynb
│   ├── experimental_correlation.ipynb
│   └── parameter_exploration.ipynb
│
├── data/                      # Data directory
│   ├── experimental/          # Experimental data
│   ├── simulations/           # Simulation results
│   └── models/               # Trained ML models
│
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── physics/              # Physics models documentation
│   ├── user_guide/           # User guides
│   └── validation/           # Validation reports
│
├── scripts/                   # Utility scripts
│   ├── setup.py              # Setup script
│   ├── validate_physics.py   # Physics validation
│   └── benchmark.py          # Performance benchmarks
│
├── .github/                   # GitHub configuration
│   └── workflows/            # CI/CD workflows
│       ├── test.yml
│       ├── build.yml
│       └── deploy.yml
│
├── docker-compose.yml         # Multi-container setup
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore file
├── README.md                # Project documentation
├── LICENSE                  # License file
└── CONTRIBUTING.md          # Contribution guidelines
```

## Key Files to Create

### Backend Core Files
1. `backend/core/quantum_tunneling.py` - Gamow factor and tunneling calculations
2. `backend/solvers/poisson_schrodinger.py` - Coupled solver implementation
3. `backend/ml/parameter_discovery.py` - ML-based parameter optimization

### Frontend Core Files
1. `frontend/src/components/Visualization/LatticeView3D.tsx` - 3D lattice visualization
2. `frontend/src/services/api.ts` - API integration layer
3. `frontend/src/store/simulationSlice.ts` - Simulation state management

### Configuration Files
1. `docker-compose.yml` - Multi-container orchestration
2. `.env.example` - Environment variables template
3. `backend/requirements.txt` - Python dependencies
4. `frontend/package.json` - Node dependencies

## Module Responsibilities

### Backend Modules

#### Core Physics (`backend/core/`)
- Implement fundamental physics models
- Calculate tunneling probabilities
- Model electron screening effects
- Simulate lattice coherence
- Handle interface dynamics

#### Solvers (`backend/solvers/`)
- Numerical PDE solvers
- Monte Carlo simulations
- Statistical analysis
- Uncertainty propagation

#### Machine Learning (`backend/ml/`)
- Parameter space exploration
- Pattern recognition
- Predictive modeling
- Optimization algorithms

#### API (`backend/api/`)
- RESTful endpoints
- WebSocket connections
- Input validation
- Result streaming

### Frontend Modules

#### Visualization (`frontend/src/components/Visualization/`)
- 3D rendering with Three.js
- Real-time field visualization
- Energy density mapping
- Interactive controls

#### Charts (`frontend/src/components/Charts/`)
- Data plotting with D3.js
- Statistical displays
- Time series analysis
- Comparative views

#### Services (`frontend/src/services/`)
- API communication
- WebSocket management
- Data transformation
- Type definitions
