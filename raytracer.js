class Vector3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  add(v) {
    return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
  }

  subtract(v) {
    return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
  }

  multiply(scalar) {
    return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  divide(scalar) {
    return new Vector3(this.x / scalar, this.y / scalar, this.z / scalar);
  }

  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v) {
    return new Vector3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  lengthSquared() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  normalize() {
    const len = this.length();
    if (len === 0) return new Vector3();
    return this.divide(len);
  }

  reflect(normal) {
    return this.subtract(normal.multiply(2 * this.dot(normal)));
  }

  refract(normal, eta) {
    const cosI = -this.dot(normal);
    const sinT2 = eta * eta * (1.0 - cosI * cosI);
    if (sinT2 >= 1.0) return null; // Total internal reflection
    const cosT = Math.sqrt(1.0 - sinT2);
    return this.multiply(eta).add(normal.multiply(eta * cosI - cosT));
  }

  lerp(v, t) {
    return this.multiply(1 - t).add(v.multiply(t));
  }

  clone() {
    return new Vector3(this.x, this.y, this.z);
  }
}

class Ray {
  constructor(origin, direction) {
    this.origin = origin;
    this.direction = direction.normalize();
  }

  at(t) {
    return this.origin.add(this.direction.multiply(t));
  }
}

class Material {
  constructor(options = {}) {
    this.albedo = options.albedo || new Vector3(0.5, 0.5, 0.5);
    this.metallic = options.metallic || 0.0;
    this.roughness = options.roughness || 0.5;
    this.transparency = options.transparency || 0.0;
    this.refractiveIndex = options.refractiveIndex || 1.0;
    this.emission = options.emission || new Vector3(0, 0, 0);
  }
}

class Sphere {
  constructor(center, radius, material) {
    this.center = center;
    this.radius = radius;
    this.material = material;
    this.type = "sphere";
  }

  intersect(ray) {
    const oc = ray.origin.subtract(this.center);
    const a = ray.direction.lengthSquared();
    const b = 2.0 * oc.dot(ray.direction);
    const c = oc.lengthSquared() - this.radius * this.radius;
    const discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return null;

    const sqrt_discriminant = Math.sqrt(discriminant);
    const t1 = (-b - sqrt_discriminant) / (2.0 * a);
    const t2 = (-b + sqrt_discriminant) / (2.0 * a);

    const epsilon = 1e-6;
    const t = t1 > epsilon ? t1 : t2 > epsilon ? t2 : null;
    if (t === null) return null;

    const point = ray.at(t);
    const normal = point.subtract(this.center).normalize();

    return {
      t: t,
      point: point,
      normal: normal,
      material: this.material,
    };
  }
}

class Triangle {
  constructor(v0, v1, v2, material) {
    this.v0 = v0;
    this.v1 = v1;
    this.v2 = v2;
    this.material = material;
    this.type = "triangle";
    
    // Pre-calculate normal
    const edge1 = v1.subtract(v0);
    const edge2 = v2.subtract(v0);
    this.normal = edge1.cross(edge2).normalize();
  }

  intersect(ray) {
    const epsilon = 1e-6;
    const edge1 = this.v1.subtract(this.v0);
    const edge2 = this.v2.subtract(this.v0);
    const h = ray.direction.cross(edge2);
    const a = edge1.dot(h);
    
    if (a > -epsilon && a < epsilon) return null;
    
    const f = 1.0 / a;
    const s = ray.origin.subtract(this.v0);
    const u = f * s.dot(h);
    
    if (u < 0.0 || u > 1.0) return null;
    
    const q = s.cross(edge1);
    const v = f * ray.direction.dot(q);
    
    if (v < 0.0 || u + v > 1.0) return null;
    
    const t = f * edge2.dot(q);
    
    if (t > epsilon) {
      const point = ray.at(t);
      return {
        t: t,
        point: point,
        normal: this.normal,
        material: this.material,
      };
    }
    
    return null;
  }
}

class Mesh {
  constructor(vertices, faces, material) {
    this.vertices = vertices;
    this.faces = faces;
    this.material = material;
    this.type = "mesh";
    this.triangles = [];
    
    // Convert faces to triangles
    for (const face of faces) {
      if (face.length >= 3) {
        for (let i = 1; i < face.length - 1; i++) {
          const triangle = new Triangle(
            vertices[face[0]],
            vertices[face[i]],
            vertices[face[i + 1]],
            material
          );
          this.triangles.push(triangle);
        }
      }
    }
  }

  intersect(ray) {
    let closest = null;
    let minDistance = Infinity;

    for (const triangle of this.triangles) {
      const intersection = triangle.intersect(ray);
      if (intersection && intersection.t < minDistance) {
        minDistance = intersection.t;
        closest = intersection;
      }
    }

    return closest;
  }
}

class OBJLoader {
  static async loadFromFile(file) {
    const text = await file.text();
    return this.parseOBJ(text);
  }

  static parseOBJ(objText) {
    const vertices = [];
    const faces = [];
    const lines = objText.split('\n');

    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'v') {
        // Vertex
        vertices.push(new Vector3(
          parseFloat(parts[1]),
          parseFloat(parts[2]),
          parseFloat(parts[3])
        ));
      } else if (parts[0] === 'f') {
        // Face (convert to 0-based indices)
        const face = parts.slice(1).map(part => {
          const vertexIndex = parseInt(part.split('/')[0]) - 1;
          return vertexIndex;
        });
        faces.push(face);
      }
    }

    return { vertices, faces };
  }
}

class Plane {
  constructor(point, normal, material) {
    this.point = point;
    this.normal = normal.normalize();
    this.material = material;
    this.type = "plane";
  }

  intersect(ray) {
    const denom = this.normal.dot(ray.direction);
    if (Math.abs(denom) < 0.0001) return null;

    const t = this.point.subtract(ray.origin).dot(this.normal) / denom;
    const epsilon = 1e-6;
    if (t < epsilon) return null;

    const point = ray.at(t);
    return {
      t: t,
      point: point,
      normal: this.normal,
      material: this.material,
    };
  }
}

class Light {
  constructor(position, color, intensity = 1.0) {
    this.position = position;
    this.color = color;
    this.intensity = intensity;
  }
}

class Camera {
  constructor(position, target, up, fov, aspect) {
    this.position = position;
    this.target = target;
    this.up = up;
    this.fov = fov;
    this.aspect = aspect;
    this.updateVectors();
  }

  updateVectors() {
    this.forward = this.target.subtract(this.position).normalize();
    this.right = this.forward.cross(this.up).normalize();
    this.up = this.right.cross(this.forward);

    const theta = (this.fov * Math.PI) / 180;
    const half_height = Math.tan(theta / 2);
    const half_width = this.aspect * half_height;

    this.horizontal = this.right.multiply(2 * half_width);
    this.vertical = this.up.multiply(2 * half_height);
    this.lower_left_corner = this.position
      .subtract(this.horizontal.divide(2))
      .subtract(this.vertical.divide(2))
      .add(this.forward);
  }

  getRay(u, v) {
    const direction = this.lower_left_corner
      .add(this.horizontal.multiply(u))
      .add(this.vertical.multiply(v))
      .subtract(this.position);

    return new Ray(this.position, direction);
  }

  rotate(deltaX, deltaY) {
    const sensitivity = 0.005;
    const right = this.right.multiply(deltaX * sensitivity);
    const up = this.up.multiply(deltaY * sensitivity);

    this.target = this.target.add(right).add(up);
    this.updateVectors();
  }

  zoom(delta) {
    const forward = this.forward.multiply(delta * 0.1);
    this.position = this.position.add(forward);
    this.updateVectors();
  }
}

class RayTracer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = canvas.width;
    this.height = canvas.height;
    this.imageData = this.ctx.createImageData(this.width, this.height);

    // Rendering settings
    this.maxBounces = 3;
    this.antiAliasingSamples = 1;
    this.backgroundColor = new Vector3(0.05, 0.05, 0.1);
    this.ambientLight = 0.1;
    this.useWorkers = navigator.hardwareConcurrency > 1;
    this.numWorkers = Math.min(navigator.hardwareConcurrency || 4, 8);

    // Scene
    this.objects = [];
    this.lights = [];
    this.camera = new Camera(
      new Vector3(0, 0, 5),
      new Vector3(0, 0, 0),
      new Vector3(0, 1, 0),
      45,
      this.width / this.height
    );

    // Performance tracking
    this.rayCount = 0;
    this.renderTime = 0;
    this.fps = 0;
    this.lastFrameTime = 0;
    this.frameCount = 0;
    this.lastFpsUpdate = 0;
    this.renderErrors = 0;
    this.isRendering = false;

    // Animation
    this.isAnimating = false;
    this.animationTime = 0;
    this.animationSpeed = 1.0;

    // Workers
    this.workers = [];
    this.workersBusy = [];
    this.initWorkers();

    // Start continuous performance monitoring
    this.startPerformanceMonitoring();

    this.setupScene();
    this.setupEventListeners();
  }

  initWorkers() {
    if (!this.useWorkers) return;

    try {
      for (let i = 0; i < this.numWorkers; i++) {
        const worker = new Worker("worker.js");
        this.workers.push(worker);
        this.workersBusy.push(false);
      }
    } catch (error) {
      console.warn(
        "Web Workers not available, falling back to single-threaded rendering:",
        error
      );
      this.useWorkers = false;
      this.workers = [];
      this.workersBusy = [];
    }
  }

  setupScene() {
    // Add some default objects
    this.objects.push(
      new Sphere(
        new Vector3(0, 0, 0),
        1,
        new Material({
          albedo: new Vector3(0.8, 0.3, 0.3),
          metallic: 0.8,
          roughness: 0.2,
        })
      )
    );

    this.objects.push(
      new Sphere(
        new Vector3(-2, 0, -1),
        0.8,
        new Material({
          albedo: new Vector3(0.3, 0.8, 0.3),
          transparency: 0.9,
          refractiveIndex: 1.5,
        })
      )
    );

    this.objects.push(
      new Sphere(
        new Vector3(2, 0, -1),
        0.8,
        new Material({
          albedo: new Vector3(0.3, 0.3, 0.8),
          metallic: 0.1,
          roughness: 0.8,
        })
      )
    );

    this.objects.push(
      new Plane(
        new Vector3(0, -1, 0),
        new Vector3(0, 1, 0),
        new Material({
          albedo: new Vector3(0.8, 0.8, 0.8),
          roughness: 0.9,
        })
      )
    );

    // Add lights
    this.lights.push(
      new Light(new Vector3(4, 4, 4), new Vector3(1, 1, 1), 1.0)
    );

    this.lights.push(
      new Light(new Vector3(-4, 4, 4), new Vector3(0.8, 0.8, 1.0), 0.7)
    );
  }

  setupEventListeners() {
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    // Mouse events
    this.canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      e.preventDefault();
    });

    this.canvas.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      const deltaX = e.clientX - lastX;
      const deltaY = e.clientY - lastY;

      this.camera.rotate(deltaX, deltaY);
      lastX = e.clientX;
      lastY = e.clientY;

      if (!this.isAnimating) this.render();
    });

    this.canvas.addEventListener("mouseup", () => {
      isDragging = false;
    });

    // Touch events for mobile/trackpad
    this.canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;
        e.preventDefault();
      }
    });

    this.canvas.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && isDragging) {
        const deltaX = e.touches[0].clientX - lastX;
        const deltaY = e.touches[0].clientY - lastY;

        this.camera.rotate(deltaX, deltaY);
        lastX = e.touches[0].clientX;
        lastY = e.touches[0].clientY;

        if (!this.isAnimating) this.render();
        e.preventDefault();
      }
    });

    this.canvas.addEventListener("touchend", () => {
      isDragging = false;
    });

    // Wheel and trackpad zoom
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.camera.zoom(e.deltaY > 0 ? 1 : -1);
      if (!this.isAnimating) this.render();
    });

    // Keyboard controls
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        this.toggleAnimation();
      }
    });
  }

  intersectScene(ray) {
    let closest = null;
    let minDistance = Infinity;

    for (const object of this.objects) {
      const intersection = object.intersect(ray);
      if (intersection && intersection.t < minDistance) {
        minDistance = intersection.t;
        closest = intersection;
      }
    }

    return closest;
  }

  calculateLighting(point, normal, material, viewDir) {
    let color = new Vector3(0, 0, 0);

    // Ambient lighting
    color = color.add(material.albedo.multiply(this.ambientLight));

    // Direct lighting from each light source
    for (const light of this.lights) {
      const lightDir = light.position.subtract(point).normalize();
      const lightDistance = light.position.subtract(point).length();

      // Shadow ray
      const epsilon = 1e-6;
      const shadowRay = new Ray(point.add(normal.multiply(epsilon)), lightDir);
      const shadowIntersection = this.intersectScene(shadowRay);

      if (shadowIntersection && shadowIntersection.t < lightDistance) {
        continue; // In shadow
      }

      // Diffuse lighting
      const diffuse = Math.max(0, normal.dot(lightDir));
      const attenuation = 1.0 / (1.0 + lightDistance * lightDistance * 0.01);

      // Specular lighting (Blinn-Phong)
      const halfVector = lightDir.add(viewDir.multiply(-1)).normalize();
      const specular = Math.pow(Math.max(0, normal.dot(halfVector)), 64);

      const lightContribution = light.color
        .multiply(light.intensity * attenuation)
        .multiply(diffuse + specular * material.metallic);

      color = color
        .add(material.albedo.multiply(lightContribution.x))
        .add(new Vector3(lightContribution.y, lightContribution.z, 0));
    }

    // Add emission
    color = color.add(material.emission);

    return color;
  }

  trace(ray, depth = 0) {
    this.rayCount++;

    if (depth >= this.maxBounces) {
      return this.backgroundColor;
    }

    const intersection = this.intersectScene(ray);
    if (!intersection) {
      return this.backgroundColor;
    }

    const { point, normal, material } = intersection;

    // Calculate direct lighting
    let color = this.calculateLighting(point, normal, material, ray.direction);

    // Handle reflections
    if (material.metallic > 0 && depth < this.maxBounces - 1) {
      const reflectionDir = ray.direction.reflect(normal);
      const epsilon = 1e-6;
      const reflectionRay = new Ray(
        point.add(normal.multiply(epsilon)),
        reflectionDir
      );
      const reflectionColor = this.trace(reflectionRay, depth + 1);
      color = color.lerp(reflectionColor, material.metallic);
    }

    // Handle transparency and refraction
    if (material.transparency > 0 && depth < this.maxBounces - 1) {
      const eta =
        ray.direction.dot(normal) < 0
          ? 1.0 / material.refractiveIndex
          : material.refractiveIndex;
      const refractionDir = ray.direction.refract(normal, eta);

      if (refractionDir) {
        const epsilon = 1e-6;
        const refractionRay = new Ray(
          point.subtract(normal.multiply(epsilon)),
          refractionDir
        );
        const refractionColor = this.trace(refractionRay, depth + 1);
        color = color.lerp(refractionColor, material.transparency);
      }
    }

    return color;
  }

  render() {
    if (this.isRendering) {
      console.warn("Render already in progress, queuing next render...");
      this.shouldRenderAgain = true;
      return;
    }
    this.shouldRenderAgain = false;

    this.isRendering = true;
    const startTime = performance.now();
    this.rayCount = 0;

    try {
      if (this.useWorkers && this.workers.length > 0) {
        this.renderWithWorkers(startTime);
      } else {
        this.renderSingleThreaded(startTime);
      }
    } catch (error) {
      console.error("Render error:", error);
      this.isRendering = false;
      this.renderErrors++;

      // Try fallback to single-threaded rendering
      if (this.useWorkers) {
        console.log("Falling back to single-threaded rendering...");
        this.useWorkers = false;
        try {
          this.renderSingleThreaded(startTime);
        } catch (fallbackError) {
          console.error("Fallback render also failed:", fallbackError);
        }
      }
    }
  }

  renderSingleThreaded(startTime) {
    try {
      const data = this.imageData.data;

      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          let color = new Vector3(0, 0, 0);

          // Anti-aliasing
          for (let sample = 0; sample < this.antiAliasingSamples; sample++) {
            const u = (x + (sample > 0 ? Math.random() : 0.5)) / this.width;
            const v =
              1.0 - (y + (sample > 0 ? Math.random() : 0.5)) / this.height;

            const ray = this.camera.getRay(u, v);
            color = color.add(this.trace(ray));
          }

          color = color.divide(this.antiAliasingSamples);

          // Gamma correction and tone mapping with smoother clamping
          const clamp = (value, min, max) => {
            if (value < min) return min;
            if (value > max) return max;
            return value;
          };
          color = new Vector3(
            Math.sqrt(clamp(color.x, 0, 1)),
            Math.sqrt(clamp(color.y, 0, 1)),
            Math.sqrt(clamp(color.z, 0, 1))
          );

          const index = (y * this.width + x) * 4;
          data[index] = Math.floor(color.x * 255);
          data[index + 1] = Math.floor(color.y * 255);
          data[index + 2] = Math.floor(color.z * 255);
          data[index + 3] = 255;
        }
      }

      this.ctx.putImageData(this.imageData, 0, 0);
      this.renderTime = performance.now() - startTime;
      this.updatePerformanceStats();
    } catch (error) {
      console.error("Single-threaded render error:", error);
      this.isRendering = false;
      throw error;
    }
  }

  renderWithWorkers(startTime) {
    const rowsPerWorker = Math.ceil(this.height / this.numWorkers);
    let completedWorkers = 0;
    const data = this.imageData.data;

    // Set a timeout to prevent infinite hangs
    const renderTimeout = setTimeout(() => {
      if (completedWorkers < this.numWorkers) {
        console.warn(
          "Worker rendering timed out, falling back to single-threaded"
        );
        this.useWorkers = false;
        this.isRendering = false;
        this.render();
      }
    }, 60000); // 60 second timeout to prevent premature cuts

    for (let i = 0; i < this.numWorkers; i++) {
      const startY = i * rowsPerWorker;
      const endY = Math.min(startY + rowsPerWorker, this.height);

      if (startY >= this.height) break;

      this.workersBusy[i] = true;

      const workerData = {
        startY: startY,
        endY: endY,
        width: this.width,
        height: this.height,
        objects: this.objects.map((obj) => this.serializeObject(obj)),
        lights: this.lights,
        camera: this.serializeCamera(),
        backgroundColor: this.backgroundColor,
        ambientLight: this.ambientLight,
        maxBounces: this.maxBounces,
        antiAliasingSamples: this.antiAliasingSamples,
      };

      this.workers[i].onmessage = (e) => {
        try {
          const { imageData, startY, endY, rayCount } = e.data;

          // Copy worker result to main image data
          for (let y = startY; y < endY; y++) {
            for (let x = 0; x < this.width; x++) {
              const srcIndex = ((y - startY) * this.width + x) * 4;
              const dstIndex = (y * this.width + x) * 4;

              data[dstIndex] = imageData[srcIndex];
              data[dstIndex + 1] = imageData[srcIndex + 1];
              data[dstIndex + 2] = imageData[srcIndex + 2];
              data[dstIndex + 3] = imageData[srcIndex + 3];
            }
          }

          this.rayCount += rayCount;
          this.workersBusy[i] = false;
          completedWorkers++;

          if (
            completedWorkers === this.numWorkers ||
            completedWorkers === Math.ceil(this.height / rowsPerWorker)
          ) {
            clearTimeout(renderTimeout);
            this.ctx.putImageData(this.imageData, 0, 0);
            this.renderTime = performance.now() - startTime;
            this.updatePerformanceStats();
          }
        } catch (error) {
          console.error("Worker message processing error:", error);
          this.workersBusy[i] = false;
          completedWorkers++;
          this.renderErrors++;

          // If all workers completed (even with errors), finish the render
          if (completedWorkers >= this.numWorkers) {
            clearTimeout(renderTimeout);
            this.isRendering = false;
            this.updatePerformanceStats();
          }
        }
      };

      this.workers[i].onerror = (error) => {
        console.error("Worker error:", error);
        this.workersBusy[i] = false;
        completedWorkers++;
        this.renderErrors++;

        if (completedWorkers >= this.numWorkers) {
          clearTimeout(renderTimeout);
          this.isRendering = false;
          this.updatePerformanceStats();
        }
      };

      this.workers[i].postMessage(workerData);
    }
  }

  serializeObject(obj) {
    if (obj.type === "sphere") {
      return {
        type: "sphere",
        center: { x: obj.center.x, y: obj.center.y, z: obj.center.z },
        radius: obj.radius,
        material: {
          albedo: {
            x: obj.material.albedo.x,
            y: obj.material.albedo.y,
            z: obj.material.albedo.z,
          },
          metallic: obj.material.metallic,
          roughness: obj.material.roughness,
          transparency: obj.material.transparency,
          refractiveIndex: obj.material.refractiveIndex,
          emission: obj.material.emission
            ? {
                x: obj.material.emission.x,
                y: obj.material.emission.y,
                z: obj.material.emission.z,
              }
            : null,
        },
      };
    } else if (obj.type === "plane") {
      return {
        type: "plane",
        point: { x: obj.point.x, y: obj.point.y, z: obj.point.z },
        normal: { x: obj.normal.x, y: obj.normal.y, z: obj.normal.z },
        material: {
          albedo: {
            x: obj.material.albedo.x,
            y: obj.material.albedo.y,
            z: obj.material.albedo.z,
          },
          metallic: obj.material.metallic,
          roughness: obj.material.roughness,
          transparency: obj.material.transparency,
          refractiveIndex: obj.material.refractiveIndex,
          emission: obj.material.emission
            ? {
                x: obj.material.emission.x,
                y: obj.material.emission.y,
                z: obj.material.emission.z,
              }
            : null,
        },
      };
    } else if (obj.type === "mesh") {
      return {
        type: "mesh",
        vertices: obj.vertices.map(v => ({ x: v.x, y: v.y, z: v.z })),
        faces: obj.faces,
        material: {
          albedo: {
            x: obj.material.albedo.x,
            y: obj.material.albedo.y,
            z: obj.material.albedo.z,
          },
          metallic: obj.material.metallic,
          roughness: obj.material.roughness,
          transparency: obj.material.transparency,
          refractiveIndex: obj.material.refractiveIndex,
          emission: obj.material.emission
            ? {
                x: obj.material.emission.x,
                y: obj.material.emission.y,
                z: obj.material.emission.z,
              }
            : null,
        },
      };
    }
  }

  serializeCamera() {
    return {
      position: {
        x: this.camera.position.x,
        y: this.camera.position.y,
        z: this.camera.position.z,
      },
      horizontal: {
        x: this.camera.horizontal.x,
        y: this.camera.horizontal.y,
        z: this.camera.horizontal.z,
      },
      vertical: {
        x: this.camera.vertical.x,
        y: this.camera.vertical.y,
        z: this.camera.vertical.z,
      },
      lower_left_corner: {
        x: this.camera.lower_left_corner.x,
        y: this.camera.lower_left_corner.y,
        z: this.camera.lower_left_corner.z,
      },
    };
  }

  animate() {
    if (!this.isAnimating) return;

    const currentTime = performance.now();
    const deltaTime =
      this.lastFrameTime > 0
        ? (currentTime - this.lastFrameTime) / 1000
        : 0.016;
    this.animationTime += deltaTime * this.animationSpeed;

    // Animate objects
    if (this.objects.length > 0) {
      this.objects[0].center.y = Math.sin(this.animationTime) * 0.5;
      this.objects[0].center.x = Math.cos(this.animationTime * 0.7) * 0.3;
    }

    // Animate lights
    if (this.lights.length > 0) {
      this.lights[0].position.x = Math.cos(this.animationTime * 0.5) * 5;
      this.lights[0].position.z = Math.sin(this.animationTime * 0.5) * 5;
    }

    this.render();
    requestAnimationFrame(() => this.animate());
  }

  toggleAnimation() {
    this.isAnimating = !this.isAnimating;
    const button = document.getElementById("toggle-animation");
    button.textContent = this.isAnimating
      ? "Stop Animation"
      : "Start Animation";
    button.className = this.isAnimating
      ? "w-full bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded"
      : "w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded";

    if (this.isAnimating) {
      this.animate();
    }
  }

  startPerformanceMonitoring() {
    const updateStats = () => {
      const currentTime = performance.now();

      // Update FPS counter every second
      if (currentTime - this.lastFpsUpdate > 1000) {
        if (this.frameCount > 0) {
          this.fps = Math.round(
            (this.frameCount * 1000) / (currentTime - this.lastFpsUpdate)
          );
        } else {
          this.fps = 0;
        }
        this.frameCount = 0;
        this.lastFpsUpdate = currentTime;
      }

      // Update UI
      document.getElementById("fps-counter").textContent = this.fps;
      document.getElementById("render-time").textContent =
        Math.round(this.renderTime) + "ms";
      document.getElementById("ray-count").textContent =
        this.rayCount.toLocaleString();

      // Check if rendering has stalled
      if (this.isRendering && currentTime - this.lastFrameTime > 10000) {
        console.warn(
          "Rendering appears to have stalled, attempting recovery..."
        );
        this.isRendering = false;
        this.renderErrors++;

        // Try to recover by doing a simple render
        try {
          this.render();
        } catch (error) {
          console.error("Render recovery failed:", error);
        }
      }

      requestAnimationFrame(updateStats);
    };

    this.lastFpsUpdate = performance.now();
    updateStats();
  }

  updatePerformanceStats() {
    this.frameCount++;
    this.lastFrameTime = performance.now();
    this.isRendering = false;

    // Render again if queued
    if (this.shouldRenderAgain) {
      setTimeout(() => this.render(), 10);
    }
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.imageData = this.ctx.createImageData(width, height);
    this.camera.aspect = width / height;
    this.camera.updateVectors();
  }

  addSphere(center, radius, material) {
    this.objects.push(new Sphere(center, radius, material));
  }

  addMesh(vertices, faces, material) {
    this.objects.push(new Mesh(vertices, faces, material));
  }

  addLight(position, color, intensity) {
    this.lights.push(new Light(position, color, intensity));
  }

  removeObject(index) {
    if (index >= 0 && index < this.objects.length) {
      this.objects.splice(index, 1);
    }
  }

  removeLight(index) {
    if (index >= 0 && index < this.lights.length) {
      this.lights.splice(index, 1);
    }
  }
}

// UI Controller
class UIController {
  constructor(raytracer) {
    this.raytracer = raytracer;
    this.setupControls();
    this.updateObjectsList();
    this.updateLightsList();
  }

  setupControls() {
    // Resolution control
    document.getElementById("resolution").addEventListener("change", (e) => {
      const scale = parseFloat(e.target.value);
      const canvas = this.raytracer.canvas;
      const rect = canvas.parentElement.getBoundingClientRect();
      this.raytracer.resize(
        Math.floor(rect.width * scale),
        Math.floor(rect.height * scale)
      );
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      this.raytracer.render();
    });

    // Max bounces
    const bouncesSlider = document.getElementById("max-bounces");
    const bouncesValue = document.getElementById("bounces-value");
    bouncesSlider.addEventListener("input", (e) => {
      this.raytracer.maxBounces = parseInt(e.target.value);
      bouncesValue.textContent = e.target.value;
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Anti-aliasing
    const aaSlider = document.getElementById("anti-aliasing");
    const aaValue = document.getElementById("aa-value");
    aaSlider.addEventListener("input", (e) => {
      this.raytracer.antiAliasingSamples = parseInt(e.target.value);
      aaValue.textContent = e.target.value;
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Camera FOV
    const fovSlider = document.getElementById("camera-fov");
    const fovValue = document.getElementById("fov-value");
    fovSlider.addEventListener("input", (e) => {
      this.raytracer.camera.fov = parseInt(e.target.value);
      fovValue.textContent = e.target.value + "°";
      this.raytracer.camera.updateVectors();
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Background color
    document.getElementById("bg-color").addEventListener("change", (e) => {
      const hex = e.target.value;
      const r = parseInt(hex.substr(1, 2), 16) / 255;
      const g = parseInt(hex.substr(3, 2), 16) / 255;
      const b = parseInt(hex.substr(5, 2), 16) / 255;
      this.raytracer.backgroundColor = new Vector3(r, g, b);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Ambient light
    const ambientSlider = document.getElementById("ambient-light");
    const ambientValue = document.getElementById("ambient-value");
    ambientSlider.addEventListener("input", (e) => {
      this.raytracer.ambientLight = parseFloat(e.target.value);
      ambientValue.textContent = e.target.value;
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Animation controls
    document
      .getElementById("toggle-animation")
      .addEventListener("click", () => {
        this.raytracer.toggleAnimation();
      });

    const speedSlider = document.getElementById("animation-speed");
    const speedValue = document.getElementById("speed-value");
    speedSlider.addEventListener("input", (e) => {
      this.raytracer.animationSpeed = parseFloat(e.target.value);
      speedValue.textContent = e.target.value + "x";
    });

    // Reset camera
    document.getElementById("reset-camera").addEventListener("click", () => {
      this.raytracer.camera = new Camera(
        new Vector3(0, 0, 5),
        new Vector3(0, 0, 0),
        new Vector3(0, 1, 0),
        45,
        this.raytracer.width / this.raytracer.height
      );
      if (!this.raytracer.isAnimating) this.raytracer.render();
    });

    // Add object/light buttons
    document.getElementById("add-sphere").addEventListener("click", () => {
      this.addSphere();
    });

    document.getElementById("load-mesh").addEventListener("click", () => {
      document.getElementById("mesh-file-input").click();
    });

    document.getElementById("mesh-file-input").addEventListener("change", (e) => {
      this.handleMeshUpload(e);
    });

    document.getElementById("add-light").addEventListener("click", () => {
      this.addLight();
    });
  }

  addSphere() {
    const material = new Material({
      albedo: new Vector3(Math.random(), Math.random(), Math.random()),
      metallic: Math.random(),
      roughness: Math.random(),
    });

    const center = new Vector3(
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4
    );

    this.raytracer.addSphere(center, 0.5 + Math.random() * 0.5, material);
    this.updateObjectsList();
    if (!this.raytracer.isAnimating) this.raytracer.render();
  }

  async handleMeshUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.obj')) {
      alert('Please select an OBJ file');
      return;
    }

    try {
      const { vertices, faces } = await OBJLoader.loadFromFile(file);
      
      if (vertices.length === 0 || faces.length === 0) {
        alert('Invalid OBJ file or no geometry found');
        return;
      }

      // Scale down mesh if too large
      const scale = 2.0;
      const scaledVertices = vertices.map(v => v.multiply(scale));

      const material = new Material({
        albedo: new Vector3(0.7, 0.7, 0.8),
        metallic: 0.3,
        roughness: 0.4,
      });

      this.raytracer.addMesh(scaledVertices, faces, material);
      this.updateObjectsList();
      if (!this.raytracer.isAnimating) this.raytracer.render();
    } catch (error) {
      console.error('Error loading mesh:', error);
      alert('Error loading mesh file: ' + error.message);
    }
  }

  addLight() {
    const position = new Vector3(
      (Math.random() - 0.5) * 8,
      2 + Math.random() * 4,
      (Math.random() - 0.5) * 8
    );

    const color = new Vector3(Math.random(), Math.random(), Math.random());
    this.raytracer.addLight(position, color, 0.5 + Math.random() * 0.5);
    this.updateLightsList();
    if (!this.raytracer.isAnimating) this.raytracer.render();
  }

  updateObjectsList() {
    const container = document.getElementById("objects-list");
    container.innerHTML = "";

    this.raytracer.objects.forEach((object, index) => {
      const panel = document.createElement("div");
      panel.className = "object-panel";

      if (object.type === "sphere") {
        panel.innerHTML = `
                    <div class="text-sm font-medium mb-2">Sphere ${
                      index + 1
                    }</div>
                    <div class="control-group">
                        <label>Radius:</label>
                        <input type="range" min="0.1" max="2" step="0.1" value="${
                          object.radius
                        }"
                               onchange="ui.updateSphereRadius(${index}, this.value)">
                    </div>
                    <div class="control-group">
                        <label>Metallic:</label>
                        <input type="range" min="0" max="1" step="0.1" value="${
                          object.material.metallic
                        }"
                               onchange="ui.updateSphereMetallic(${index}, this.value)">
                    </div>
                    <div class="control-group">
                        <label>Transparency:</label>
                        <input type="range" min="0" max="1" step="0.1" value="${
                          object.material.transparency
                        }"
                               onchange="ui.updateSphereTransparency(${index}, this.value)">
                    </div>
                    <button class="remove-btn" onclick="ui.removeObject(${index})">Remove</button>
                `;
      } else if (object.type === "mesh") {
        panel.innerHTML = `
                    <div class="text-sm font-medium mb-2">Mesh ${
                      index + 1
                    } (${object.triangles.length} triangles)</div>
                    <div class="control-group">
                        <label>Metallic:</label>
                        <input type="range" min="0" max="1" step="0.1" value="${
                          object.material.metallic
                        }" 
                               onchange="ui.updateObjectMetallic(${index}, this.value)">
                    </div>
                    <div class="control-group">
                        <label>Transparency:</label>
                        <input type="range" min="0" max="1" step="0.1" value="${
                          object.material.transparency
                        }" 
                               onchange="ui.updateObjectTransparency(${index}, this.value)">
                    </div>
                    <button class="remove-btn" onclick="ui.removeObject(${index})">Remove</button>
                `;
      }

      container.appendChild(panel);
    });
  }

  updateLightsList() {
    const container = document.getElementById("lights-list");
    container.innerHTML = "";

    this.raytracer.lights.forEach((light, index) => {
      const panel = document.createElement("div");
      panel.className = "light-panel";

      panel.innerHTML = `
                <div class="text-sm font-medium mb-2">Light ${index + 1}</div>
                <div class="control-group">
                    <label>Intensity:</label>
                    <input type="range" min="0" max="2" step="0.1" value="${
                      light.intensity
                    }"
                           onchange="ui.updateLightIntensity(${index}, this.value)">
                </div>
                <button class="remove-btn" onclick="ui.removeLight(${index})">Remove</button>
            `;

      container.appendChild(panel);
    });
  }

  updateSphereRadius(index, value) {
    if (
      this.raytracer.objects[index] &&
      this.raytracer.objects[index].type === "sphere"
    ) {
      this.raytracer.objects[index].radius = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  updateSphereMetallic(index, value) {
    if (this.raytracer.objects[index]) {
      this.raytracer.objects[index].material.metallic = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  updateSphereTransparency(index, value) {
    if (this.raytracer.objects[index]) {
      this.raytracer.objects[index].material.transparency = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  updateObjectMetallic(index, value) {
    if (this.raytracer.objects[index]) {
      this.raytracer.objects[index].material.metallic = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  updateObjectTransparency(index, value) {
    if (this.raytracer.objects[index]) {
      this.raytracer.objects[index].material.transparency = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  updateLightIntensity(index, value) {
    if (this.raytracer.lights[index]) {
      this.raytracer.lights[index].intensity = parseFloat(value);
      if (!this.raytracer.isAnimating) this.raytracer.render();
    }
  }

  removeObject(index) {
    this.raytracer.removeObject(index);
    this.updateObjectsList();
    if (!this.raytracer.isAnimating) this.raytracer.render();
  }

  removeLight(index) {
    this.raytracer.removeLight(index);
    this.updateLightsList();
    if (!this.raytracer.isAnimating) this.raytracer.render();
  }
}

// Initialize the application
let raytracer, ui; // eslint-disable-line no-unused-vars

window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("raytracer-canvas");
  const container = canvas.parentElement;
  const rect = container.getBoundingClientRect();

  // Set initial canvas size based on resolution setting
  const initialScale = 0.5;
  canvas.width = Math.floor(rect.width * initialScale);
  canvas.height = Math.floor(rect.height * initialScale);
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";

  raytracer = new RayTracer(canvas);
  ui = new UIController(raytracer);

  // Initial render with proper FPS initialization
  setTimeout(() => {
    raytracer.render();
    // Trigger a second render to get accurate FPS calculation
    setTimeout(() => {
      if (!raytracer.isAnimating) raytracer.render();
    }, 100);
  }, 50);

  // Handle window resize
  window.addEventListener("resize", () => {
    const newRect = container.getBoundingClientRect();
    const scale = parseFloat(document.getElementById("resolution").value);
    raytracer.resize(
      Math.floor(newRect.width * scale),
      Math.floor(newRect.height * scale)
    );
    canvas.style.width = newRect.width + "px";
    canvas.style.height = newRect.height + "px";
    if (!raytracer.isAnimating) raytracer.render();
  });
});
