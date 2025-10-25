# LENR Simulation Frontend

React-based frontend with 3D visualizations for the LENR Mathematical Simulation Framework.

## Features

- **Interactive Parameter Input**: Adjust simulation parameters with real-time validation
- **3D Visualization**: Three.js powered visualization of enhancement fields
- **Results Display**: Comprehensive display of simulation results with charts
- **API Integration**: Seamless connection to the FastAPI backend
- **Preset Configurations**: Quick access to optimal parameter sets

## Installation

```bash
# Install dependencies
npm install

# Or using yarn
yarn install
```

## Running the Frontend

```bash
# Development mode
npm start

# The app will open at http://localhost:3000
```

## Building for Production

```bash
npm run build
# Creates optimized build in ./build directory
```

## Configuration

Set the API URL in your environment:
```bash
# .env file
REACT_APP_API_URL=http://localhost:8000
```

## Prerequisites

- Node.js 16+
- Backend API running on port 8000
- Modern web browser with WebGL support

## Technology Stack

- **React 18**: UI framework
- **Three.js/React Three Fiber**: 3D graphics
- **Recharts**: Data visualization
- **Axios**: API communication
- **CSS Grid/Flexbox**: Responsive layout

## Components

1. **SimulationForm**: Parameter input and validation
2. **Visualization3D**: 3D enhancement field visualization  
3. **ResultsDisplay**: Results presentation with charts
4. **API Module**: Backend communication layer

## Usage

1. Start the backend API first:
```bash
cd ../backend
uvicorn main:app --reload
```

2. Start the frontend:
```bash
npm start
```

3. Access the application at `http://localhost:3000`

4. Enter simulation parameters or use presets

5. View real-time 3D visualization and results

## Screenshots

- Parameter input panel (left)
- 3D visualization (center)  
- Results display (right)

## Contributing

See main project README for contribution guidelines.
