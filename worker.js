// Ray tracing worker for parallel processing
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
    if (sinT2 >= 1.0) return null;
    const cosT = Math.sqrt(1.0 - sinT2);
    return this.multiply(eta).add(normal.multiply(eta * cosI - cosT));
  }

  lerp(v, t) {
    return this.multiply(1 - t).add(v.multiply(t));
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

function intersectSphere(ray, sphere) {
  const center = new Vector3(sphere.center.x, sphere.center.y, sphere.center.z);
  const oc = ray.origin.subtract(center);
  const a = ray.direction.lengthSquared();
  const b = 2.0 * oc.dot(ray.direction);
  const c = oc.lengthSquared() - sphere.radius * sphere.radius;
  const discriminant = b * b - 4 * a * c;

  if (discriminant < 0) return null;

  const sqrt_discriminant = Math.sqrt(discriminant);
  const t1 = (-b - sqrt_discriminant) / (2.0 * a);
  const t2 = (-b + sqrt_discriminant) / (2.0 * a);

  const t = t1 > 0.001 ? t1 : t2 > 0.001 ? t2 : null;
  if (t === null) return null;

  const point = ray.at(t);
  const normal = point.subtract(center).normalize();

  return {
    t: t,
    point: point,
    normal: normal,
    material: sphere.material,
  };
}

function intersectPlane(ray, plane) {
  const point = new Vector3(plane.point.x, plane.point.y, plane.point.z);
  const normal = new Vector3(
    plane.normal.x,
    plane.normal.y,
    plane.normal.z
  ).normalize();

  const denom = normal.dot(ray.direction);
  if (Math.abs(denom) < 0.0001) return null;

  const t = point.subtract(ray.origin).dot(normal) / denom;
  if (t < 0.001) return null;

  const hitPoint = ray.at(t);
  return {
    t: t,
    point: hitPoint,
    normal: normal,
    material: plane.material,
  };
}

function intersectScene(ray, objects) {
  let closest = null;
  let minDistance = Infinity;

  for (const object of objects) {
    let intersection = null;

    if (object.type === "sphere") {
      intersection = intersectSphere(ray, object);
    } else if (object.type === "plane") {
      intersection = intersectPlane(ray, object);
    }

    if (intersection && intersection.t < minDistance) {
      minDistance = intersection.t;
      closest = intersection;
    }
  }

  return closest;
}

function calculateLighting(
  point,
  normal,
  material,
  viewDir,
  lights,
  objects,
  ambientLight
) {
  let color = new Vector3(0, 0, 0);
  const albedo = new Vector3(
    material.albedo.x,
    material.albedo.y,
    material.albedo.z
  );

  // Ambient lighting
  color = color.add(albedo.multiply(ambientLight));

  // Direct lighting from each light source
  for (const light of lights) {
    const lightPos = new Vector3(
      light.position.x,
      light.position.y,
      light.position.z
    );
    const lightColor = new Vector3(light.color.x, light.color.y, light.color.z);
    const lightDir = lightPos.subtract(point).normalize();
    const lightDistance = lightPos.subtract(point).length();

    // Shadow ray
    const shadowRay = new Ray(point.add(normal.multiply(0.001)), lightDir);
    const shadowIntersection = intersectScene(shadowRay, objects);

    if (shadowIntersection && shadowIntersection.t < lightDistance) {
      continue; // In shadow
    }

    // Diffuse lighting
    const diffuse = Math.max(0, normal.dot(lightDir));
    const attenuation = 1.0 / (1.0 + lightDistance * lightDistance * 0.01);

    // Specular lighting
    const halfVector = lightDir.add(viewDir.multiply(-1)).normalize();
    const specular = Math.pow(Math.max(0, normal.dot(halfVector)), 64);

    const lightContribution = lightColor
      .multiply(light.intensity * attenuation)
      .multiply(diffuse + specular * material.metallic);

    color = color
      .add(albedo.multiply(lightContribution.x))
      .add(new Vector3(lightContribution.y, lightContribution.z, 0));
  }

  // Add emission
  if (material.emission) {
    const emission = new Vector3(
      material.emission.x,
      material.emission.y,
      material.emission.z
    );
    color = color.add(emission);
  }

  return color;
}

function trace(
  ray,
  objects,
  lights,
  backgroundColor,
  ambientLight,
  maxBounces,
  depth = 0
) {
  if (depth >= maxBounces) {
    return backgroundColor;
  }

  const intersection = intersectScene(ray, objects);
  if (!intersection) {
    return backgroundColor;
  }

  const { point, normal, material } = intersection;

  // Calculate direct lighting
  let color = calculateLighting(
    point,
    normal,
    material,
    ray.direction,
    lights,
    objects,
    ambientLight
  );

  // Handle reflections
  if (material.metallic > 0 && depth < maxBounces - 1) {
    const reflectionDir = ray.direction.reflect(normal);
    const reflectionRay = new Ray(
      point.add(normal.multiply(0.001)),
      reflectionDir
    );
    const reflectionColor = trace(
      reflectionRay,
      objects,
      lights,
      backgroundColor,
      ambientLight,
      maxBounces,
      depth + 1
    );
    color = color.lerp(reflectionColor, material.metallic);
  }

  // Handle transparency and refraction
  if (material.transparency > 0 && depth < maxBounces - 1) {
    const eta =
      ray.direction.dot(normal) < 0
        ? 1.0 / material.refractiveIndex
        : material.refractiveIndex;
    const refractionDir = ray.direction.refract(normal, eta);

    if (refractionDir) {
      const refractionRay = new Ray(
        point.subtract(normal.multiply(0.001)),
        refractionDir
      );
      const refractionColor = trace(
        refractionRay,
        objects,
        lights,
        backgroundColor,
        ambientLight,
        maxBounces,
        depth + 1
      );
      color = color.lerp(refractionColor, material.transparency);
    }
  }

  return color;
}

function getRay(x, y, width, height, camera) {
  const u = x / width;
  const v = 1.0 - y / height;

  const horizontal = new Vector3(
    camera.horizontal.x,
    camera.horizontal.y,
    camera.horizontal.z
  );
  const vertical = new Vector3(
    camera.vertical.x,
    camera.vertical.y,
    camera.vertical.z
  );
  const lowerLeft = new Vector3(
    camera.lower_left_corner.x,
    camera.lower_left_corner.y,
    camera.lower_left_corner.z
  );
  const position = new Vector3(
    camera.position.x,
    camera.position.y,
    camera.position.z
  );

  const direction = lowerLeft
    .add(horizontal.multiply(u))
    .add(vertical.multiply(v))
    .subtract(position);

  return new Ray(position, direction);
}

self.onmessage = function (e) {
  const {
    startY,
    endY,
    width,
    height,
    objects,
    lights,
    camera,
    backgroundColor,
    ambientLight,
    maxBounces,
    antiAliasingSamples,
  } = e.data;

  const bgColor = new Vector3(
    backgroundColor.x,
    backgroundColor.y,
    backgroundColor.z
  );
  const imageData = new Uint8ClampedArray((endY - startY) * width * 4);
  let rayCount = 0;

  for (let y = startY; y < endY; y++) {
    for (let x = 0; x < width; x++) {
      let color = new Vector3(0, 0, 0);

      // Anti-aliasing
      for (let sample = 0; sample < antiAliasingSamples; sample++) {
        const pixelX = x + (sample > 0 ? Math.random() : 0.5);
        const pixelY = y + (sample > 0 ? Math.random() : 0.5);

        const ray = getRay(pixelX, pixelY, width, height, camera);
        color = color.add(
          trace(ray, objects, lights, bgColor, ambientLight, maxBounces)
        );
        rayCount++;
      }

      color = color.divide(antiAliasingSamples);

      // Gamma correction and tone mapping
      color = new Vector3(
        Math.sqrt(Math.min(1, color.x)),
        Math.sqrt(Math.min(1, color.y)),
        Math.sqrt(Math.min(1, color.z))
      );

      const index = ((y - startY) * width + x) * 4;
      imageData[index] = Math.floor(color.x * 255);
      imageData[index + 1] = Math.floor(color.y * 255);
      imageData[index + 2] = Math.floor(color.z * 255);
      imageData[index + 3] = 255;
    }
  }

  self.postMessage({
    imageData: imageData,
    startY: startY,
    endY: endY,
    rayCount: rayCount,
  });
};
