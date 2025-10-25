import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Text } from '@react-three/drei';
import * as THREE from 'three';

const EnhancementField = ({ enhancement }) => {
  const meshRef = useRef();
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.2;
    }
  });

  // Map enhancement to color intensity
  const intensity = Math.log10(enhancement || 1) / 8; // Normalize to 0-1
  const color = new THREE.Color(intensity, 0.2, 1 - intensity);

  return (
    <group ref={meshRef}>
      {/* Nucleus representation */}
      <Sphere args={[0.3, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial color="red" emissive="red" emissiveIntensity={0.5} />
      </Sphere>
      
      {/* Electron cloud */}
      <Sphere args={[1, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color={color} 
          transparent 
          opacity={0.3} 
          wireframe 
        />
      </Sphere>
      
      {/* Enhancement field visualization */}
      <Box args={[3, 3, 0.1]} position={[0, 0, -1.5]}>
        <meshStandardMaterial 
          color={color}
          transparent 
          opacity={0.5}
          emissive={color}
          emissiveIntensity={intensity}
        />
      </Box>
      
      {/* Field lines */}
      {[0, 45, 90, 135].map(angle => (
        <group key={angle} rotation={[0, 0, THREE.MathUtils.degToRad(angle)]}>
          <Box args={[4, 0.05, 0.05]} position={[0, 0, 0]}>
            <meshStandardMaterial color="cyan" emissive="cyan" emissiveIntensity={0.3} />
          </Box>
        </group>
      ))}
    </group>
  );
};

const Visualization3D = ({ results }) => {
  const enhancement = results?.results?.total_enhancement || 1;
  const screeningEnergy = results?.results?.screening_energy || 0;
  const fieldStrength = results?.results?.max_interface_field || 0;

  return (
    <div className="visualization-3d">
      <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        
        <EnhancementField enhancement={enhancement} />
        
        {/* Labels */}
        <Text
          position={[0, 3, 0]}
          fontSize={0.3}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {`Enhancement: ${enhancement?.toExponential(2) || 'N/A'}`}
        </Text>
        
        <Text
          position={[0, -3, 0]}
          fontSize={0.2}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {`Screening: ${screeningEnergy?.toFixed(1) || '0'} eV`}
        </Text>
        
        {/* Grid helper */}
        <gridHelper args={[10, 10, 0x444444, 0x222222]} />
      </Canvas>
      
      <div className="viz-legend">
        <h4>Visualization Legend</h4>
        <p><span className="red-dot"></span> Nucleus</p>
        <p><span className="blue-dot"></span> Electron Screening</p>
        <p><span className="cyan-line"></span> Electric Field Lines</p>
        <p>Color intensity = Enhancement magnitude</p>
      </div>
    </div>
  );
};

export default Visualization3D;
