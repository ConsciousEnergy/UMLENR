# LENR Simulation API Documentation

## 🚀 Quick Start

### Starting the API Server

```bash
# Navigate to backend directory
cd backend

# Install dependencies (if not already done)
pip install fastapi uvicorn

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Testing the API

```bash
# In a new terminal, from the LENR_Math_Sim directory
python test_api.py
```

## 📚 API Endpoints

### Health & Status

#### GET `/`
Root endpoint with API information.

#### GET `/health`
Health check with module status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "physics_modules": {
    "quantum_tunneling": true,
    "electron_screening": true,
    "lattice_effects": true,
    "interface_dynamics": true,
    "bubble_dynamics": true
  },
  "solvers": {
    "poisson_schrodinger": true,
    "monte_carlo": true
  }
}
```

### Simulations

#### POST `/api/v1/simulations/`
Create and run a new LENR simulation.

**Request Body:**
```json
{
  "parameters": {
    "material": "Pd",
    "temperature": 300.0,
    "loading_ratio": 0.95,
    "electric_field": 1e10,
    "surface_potential": 0.5,
    "defect_density": 1e21,
    "coherence_domain_size": 1000
  },
  "energy": 10.0,
  "calculate_rate": true,
  "include_validation": true
}
```

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "status": "pending",
  "parameters": {...},
  "energy": 10.0,
  "created_at": "2024-01-01T12:00:00"
}
```

#### GET `/api/v1/simulations/{simulation_id}`
Retrieve simulation results.

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "status": "completed",
  "parameters": {...},
  "results": {
    "total_enhancement": 2.5e7,
    "tunneling_probability": 8.6e7,
    "energy_concentration": 42.3,
    "screening_energy": 36.5,
    "reaction_rate": 1.2e20,
    "power_density": 5.8e15,
    "validation": {
      "enhancement_in_range": false,
      "screening_in_range": true,
      "field_in_range": true,
      "energy_concentration_in_range": true,
      "all_checks_passed": false
    }
  }
}
```

#### GET `/api/v1/simulations/`
List all simulations with optional filtering.

**Query Parameters:**
- `status`: Filter by status (pending, running, completed, failed)
- `limit`: Maximum number of results (default: 100)
- `offset`: Pagination offset (default: 0)

#### POST `/api/v1/simulations/batch`
Create multiple simulations in batch (max 100).

#### POST `/api/v1/simulations/{simulation_id}/validate`
Validate simulation results against paper predictions.

### Parameters

#### GET `/api/v1/parameters/defaults`
Get default simulation parameters.

#### GET `/api/v1/parameters/ranges`
Get valid parameter ranges with descriptions.

**Response:**
```json
{
  "temperature": {
    "min": 200.0,
    "max": 500.0,
    "default": 300.0,
    "unit": "K",
    "description": "System temperature"
  },
  "loading_ratio": {
    "min": 0.0,
    "max": 1.0,
    "default": 0.9,
    "critical": 0.85,
    "unit": "dimensionless",
    "description": "D/Pd or H/Ni loading ratio"
  }
  // ... more parameters
}
```

#### POST `/api/v1/parameters/scan`
Perform a parameter scan.

**Request Body:**
```json
{
  "parameter_name": "loading_ratio",
  "values": [0.80, 0.85, 0.90, 0.95, 0.99],
  "energy": 10.0,
  "base_parameters": null
}
```

**Response:**
```json
{
  "scan_id": "scan_xyz789",
  "parameter_name": "loading_ratio",
  "parameter_values": [0.80, 0.85, 0.90, 0.95, 0.99],
  "total_enhancement": [1.2e5, 5.6e5, 2.3e6, 1.8e7, 4.2e7],
  "tunneling_probability": [...],
  "energy_concentration": [...]
}
```

#### GET `/api/v1/parameters/optimal`
Get optimal parameter combinations.

**Response:**
```json
{
  "high_enhancement": {
    "description": "Parameters for maximum enhancement",
    "parameters": {...},
    "expected_enhancement": "10^7 - 10^8"
  },
  "stable_operation": {
    "description": "Parameters for stable, reproducible results",
    "parameters": {...},
    "expected_enhancement": "10^5 - 10^6"
  }
}
```

#### POST `/api/v1/parameters/validate`
Validate a set of parameters.

### WebSocket

#### WS `/ws`
Real-time simulation updates via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Performance
MAX_WORKERS=4
BATCH_SIZE=1000
```

## 📊 Response Models

### SimulationResult
```typescript
interface SimulationResult {
  simulation_id: string;
  status: "pending" | "running" | "completed" | "failed";
  parameters: SimulationParameters;
  energy: number;
  results?: {
    total_enhancement: number;
    tunneling_probability: number;
    energy_concentration: number;
    screening_energy: number;
    // ... more fields
  };
  error?: string;
  created_at: string;
  completed_at?: string;
}
```

### ValidationResult
```typescript
interface ValidationResult {
  enhancement_in_range: boolean;
  screening_in_range: boolean;
  field_in_range: boolean;
  energy_concentration_in_range: boolean;
  all_checks_passed: boolean;
  details: object;
}
```

## 🔍 Example Usage

### Python Client Example

```python
import requests
import time

# Create simulation
response = requests.post(
    "http://localhost:8000/api/v1/simulations/",
    json={
        "parameters": {
            "material": "Pd",
            "temperature": 300.0,
            "loading_ratio": 0.95,
            "electric_field": 1e10
        },
        "energy": 10.0
    }
)
sim = response.json()
sim_id = sim['simulation_id']

# Poll for results
while True:
    response = requests.get(f"http://localhost:8000/api/v1/simulations/{sim_id}")
    result = response.json()
    
    if result['status'] == 'completed':
        print(f"Enhancement: {result['results']['total_enhancement']:.2e}")
        break
    elif result['status'] == 'failed':
        print(f"Error: {result['error']}")
        break
    
    time.sleep(0.5)
```

### JavaScript/TypeScript Example

```javascript
// Create simulation
const response = await fetch('http://localhost:8000/api/v1/simulations/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    parameters: {
      material: 'Pd',
      temperature: 300.0,
      loading_ratio: 0.95,
      electric_field: 1e10
    },
    energy: 10.0
  })
});

const sim = await response.json();

// Get results
const resultResponse = await fetch(
  `http://localhost:8000/api/v1/simulations/${sim.simulation_id}`
);
const result = await resultResponse.json();
```

## 🔐 Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include details:
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## 📈 Performance Considerations

- Simulations run asynchronously in the background
- Batch endpoints limited to 100 simulations
- Parameter scans execute sequentially
- WebSocket connections for real-time updates
- Results stored in memory (use database for production)

## 🚧 Production Deployment

For production deployment:

1. Use a production ASGI server:
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. Add a reverse proxy (nginx)
3. Configure SSL/TLS certificates
4. Use a persistent database (PostgreSQL)
5. Implement authentication/authorization
6. Add rate limiting
7. Set up monitoring and logging

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Interactive API Docs](http://localhost:8000/docs)
- [LENR Theory Paper](../README.md)
- [Physics Module Documentation](../backend/core/)
