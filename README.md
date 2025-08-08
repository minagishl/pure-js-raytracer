# Pure JavaScript Ray Tracer

A high-performance real-time ray tracing engine built entirely with vanilla JavaScript and HTML Canvas, featuring reflections, refractions, dynamic lighting, and interactive controls.

## Features

### Core Ray Tracing

- **Real-time rendering** with Canvas API (no WebGL)
- **Physically-based reflections** and refractions
- **Dynamic lighting** with multiple light sources
- **Anti-aliasing** support (1x to 4x samples)
- **Configurable bounce depth** (1-8 bounces)

### Performance Optimizations

- **Multi-threaded rendering** using Web Workers
- **Adaptive resolution scaling** (quarter/half/full resolution)
- **Optimized ray-object intersection algorithms**
- **Real-time performance monitoring** (FPS, render time, ray count)

### Interactive Features

- **Camera controls**: Click and drag to rotate, mouse wheel to zoom
- **Dynamic object manipulation**: Add/remove spheres, adjust materials
- **Light source control**: Add/remove lights, adjust intensity
- **Material properties**: Metallic, transparency, roughness controls
- **Real-time animation** with speed control
- **Environment settings**: Background color, ambient lighting

### Supported Objects

- **Spheres** with configurable radius and materials
- **Planes** for ground surfaces
- **Extensible architecture** for additional primitive types

### Material System

- **Albedo** (base color)
- **Metallic** surfaces with accurate reflections
- **Transparency** with realistic refraction
- **Roughness** for surface scattering
- **Refractive index** control
- **Emission** for light-emitting materials

## Getting Started

1. Clone or download this repository
2. Open `index.html` in a modern web browser
3. Start exploring the ray-traced scene!

### Controls

#### Mouse Controls

- **Click & Drag**: Rotate the camera
- **Mouse Wheel**: Zoom in/out

#### Keyboard Controls

**Movement (WASD + QE):**

- **W/S**: Move forward/backward
- **A/D**: Move left/right
- **Q/E**: Move down/up

**Camera (Arrow Keys):**

- **←/→**: Rotate left/right
- **↑/↓**: Rotate up/down
- **R**: Reset camera to default position

**Features:**

- **Space**: Toggle animation on/off
- **I**: Toggle importance sampling
- **H**: Toggle HDR environment mapping
- **T**: Toggle denoising
- **K**: Toggle keyframe animation

**Quick Add Objects:**

- **M**: Add random sphere
- **V**: Add random volume
- **L**: Add random light

- **Control Panel**: Adjust all rendering and scene parameters

## Technical Implementation

### Architecture

- **Modular class-based design** with separate components:
  - `Vector3`: 3D vector mathematics
  - `Ray`: Ray representation and operations
  - `Material`: Material properties and behavior
  - `Sphere`/`Plane`: Geometric primitives
  - `Light`: Light source implementation
  - `Camera`: View transformation and ray generation
  - `RayTracer`: Main rendering engine

### Ray Tracing Algorithm

1. **Ray Generation**: Cast rays from camera through each pixel
2. **Intersection Testing**: Find closest object intersection
3. **Lighting Calculation**: Compute direct and indirect lighting
4. **Reflection/Refraction**: Recursively trace secondary rays
5. **Anti-aliasing**: Sample multiple rays per pixel
6. **Tone Mapping**: Apply gamma correction and exposure

### Performance Features

- **Web Workers**: Parallel processing across CPU cores
- **Adaptive Quality**: Dynamic resolution scaling
- **Spatial Optimization**: Efficient intersection algorithms
- **Memory Management**: Optimized object pooling

## Browser Compatibility

- **Chrome/Edge**: Full support with Web Workers
- **Firefox**: Full support with Web Workers
- **Safari**: Full support with Web Workers
- **Mobile**: Supported but may have reduced performance

## Performance Tips

1. **Reduce resolution** for real-time interaction
2. **Lower bounce depth** for faster rendering
3. **Disable anti-aliasing** for maximum speed
4. **Use fewer objects** in complex scenes
5. **Enable animation** only when needed

## File Structure

```
pure-js-raytracer/
├── index.html          # Main HTML page with UI
├── style.css           # Custom CSS styling
├── raytracer.js        # Main ray tracing engine
├── worker.js           # Web Worker for parallel processing
├── README.md           # This documentation
└── LICENSE             # MIT License file
```

## Customization

### Adding New Object Types

Extend the ray tracer by implementing new primitive types:

```javascript
class Triangle {
  constructor(v0, v1, v2, material) {
    this.v0 = v0;
    this.v1 = v1;
    this.v2 = v2;
    this.material = material;
    this.type = "triangle";
  }

  intersect(ray) {
    // Implement ray-triangle intersection
    // Return intersection data or null
  }
}
```

### Custom Materials

Create specialized materials with unique properties:

```javascript
const glassMaterial = new Material({
  albedo: new Vector3(0.9, 0.9, 0.9),
  transparency: 0.95,
  refractiveIndex: 1.52,
  roughness: 0.01,
});

const mirrorMaterial = new Material({
  albedo: new Vector3(0.9, 0.9, 0.9),
  metallic: 1.0,
  roughness: 0.0,
});
```

## Known Limitations

- **CPU-only rendering** (no GPU acceleration)
- **Limited to geometric primitives** (no mesh loading)
- **Simplified lighting model** (no area lights, HDR environments)
- **No volumetric effects** (fog, subsurface scattering)

## Future Enhancements

- [x] Mesh loading (OBJ/GLTF support)
- [x] Volumetric rendering
- [x] Importance sampling
- [x] Denoising algorithms
- [x] HDR environment mapping
- [x] Animation keyframes
- [x] Scene serialization/loading

## Contributing

This is an educational project demonstrating ray tracing concepts. Feel free to:

1. Fork the repository
2. Add new features or optimizations
3. Submit pull requests
4. Report issues or suggestions

## Credits

Developed as a demonstration of real-time ray tracing techniques using only web technologies. No external libraries or frameworks were used in the core implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project was created for educational purposes as part of a school contest.
