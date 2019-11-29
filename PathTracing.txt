#define SOLUTION_LIGHT
#define SOLUTION_BOUNCE
#define SOLUTION_THROUGHPUT
#define SOLUTION_HALTON
#define SOLUTION_AA

precision highp float;

#define M_PI 3.1415

struct Material {
#ifdef SOLUTION_LIGHT
    // Put your code here
    vec3 emission;
#endif
  vec3 diffuse;
  vec3 specular;
  float glossiness;
};

struct Sphere {
  vec3 position;
  float radius;
  Material material;
};

struct Plane {
  vec3 normal;
  float d;
  Material material;
};

const int sphereCount = 4;
const int planeCount = 4;
const int emittingSphereCount = 2;
#ifdef SOLUTION_BOUNCE
    // Put your code here
    // As defined by the course Spec
    const int maxPathLength = 3;

#else
const int maxPathLength = 1;
#endif

struct Scene {
  Sphere[sphereCount] spheres;
  Plane[planeCount] planes;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
  bool hit;
  float t;
  vec3 position;
  vec3 normal;
  Material material;
};

HitInfo getEmptyHit() {
  Material emptyMaterial;
#ifdef SOLUTION_LIGHT
    // Put your code here
    emptyMaterial.emission = vec3(0.0);
#endif
  emptyMaterial.diffuse = vec3(0.0);
  emptyMaterial.specular = vec3(0.0);
  emptyMaterial.glossiness = 0.0;
  return HitInfo(false, 0.0, vec3(0.0), vec3(0.0), emptyMaterial);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
  // Make t1 the smaller t
  if(t2 < t1)  {
    float temp = t1;
    t1 = t2;
    t2 = temp;
  }
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
  return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {

  sortT(t0, t1);

  // As t0 is smaller, test this first
  if(isTInInterval(t0, tMin, tMax)) {
  	smallestTInInterval = t0;
    return true;
  }

  // If t0 was not in the interval, still t1 could be
  if(isTInInterval(t1, tMin, tMax)) {
  	smallestTInInterval = t1;
    return true;
  }

  // None was
  return false;
}

HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {

    vec3 to_sphere = ray.origin - sphere.position;

    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

      	float smallestTInInterval;
      	if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
          return getEmptyHit();
        }

      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

      	vec3 normal =
          	length(ray.origin - sphere.position) < sphere.radius + 0.001?
          	-normalize(hitPosition - sphere.position) :
      		normalize(hitPosition - sphere.position);

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(Ray ray, Plane plane) {
  float t = -(dot(ray.origin, plane.normal) + plane.d) / dot(ray.direction, plane.normal);
  vec3 hitPosition = ray.origin + t * ray.direction;
  return HitInfo(
	true,
	t,
	hitPosition,
	normalize(plane.normal),
	plane.material);
    return getEmptyHit();
}

float lengthSquared(const vec3 x) {
  return dot(x, x);
}

HitInfo intersectScene(const Scene scene, Ray ray, const float tMin, const float tMax)
{
    HitInfo best_hit_info;
    best_hit_info.t = tMax;
  	best_hit_info.hit = false;

    for (int i = 0; i < sphereCount; ++i) {
        Sphere sphere = scene.spheres[i];
        HitInfo hit_info = intersectSphere(ray, sphere, tMin, tMax);

        if(	hit_info.hit &&
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

    for (int i = 0; i < planeCount; ++i) {
        Plane plane = scene.planes[i];
        HitInfo hit_info = intersectPlane(ray, plane);

        if(	hit_info.hit &&
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

  return best_hit_info;
}

// Converts a random integer in 15 bits to a float in (0, 1)
float randomInetegerToRandomFloat(int i) {
	return float(i) / 32768.0;
}

// Returns a random integer for every pixel and dimension that remains the same in all iterations
int pixelIntegerSeed(const int dimensionIndex) {
  vec3 p = vec3(gl_FragCoord.xy, dimensionIndex);
  vec3 r = vec3(23.14069263277926, 2.665144142690225,7.358926345 );
  return int(32768.0 * fract(cos(dot(p,r)) * 123456.0));
}

// Returns a random float for every pixel that remains the same in all iterations
float pixelSeed(const int dimensionIndex) {
  	return randomInetegerToRandomFloat(pixelIntegerSeed(dimensionIndex));
}

// The global random seed of this iteration
// It will be set to a new random value in each step
uniform int globalSeed;
int randomSeed;
void initRandomSequence() {
  randomSeed = globalSeed + pixelIntegerSeed(0);
}

// Computesinteger  x modulo y not available in most WEBGL SL implementations
int mod(const int x, const int y) {
  return int(float(x) - floor(float(x) / float(y)) * float(y));
}

// Returns the next integer in a pseudo-random sequence
int rand() {
  	randomSeed = randomSeed * 1103515245 + 12345;
	return mod(randomSeed / 65536, 32768);
}

// Returns the next float in this pixels pseudo-random sequence
float uniformRandom() {
	return randomInetegerToRandomFloat(rand());
}

// Returns the ith prime number for the first 20
const int maxDimensionCount = 10;
int prime(const int index) {
  if(index == 0) return 2;
  if(index == 1) return 3;
  if(index == 2) return 5;
  if(index == 3) return 7;
  if(index == 4) return 11;
  if(index == 5) return 13;
  if(index == 6) return 17;
  if(index == 7) return 19;
  if(index == 8) return 23;
  if(index == 9) return 29;
  if(index == 10) return 31;
  if(index == 11) return 37;
  if(index == 12) return 41;
  if(index == 13) return 43;
  if(index == 14) return 47;
  if(index == 15) return 53;
  return 2;
}

#ifdef SOLUTION_HALTON
    // Put your code here
#endif

float halton(const int sampleIndex, const int dimensionIndex) {
#ifdef SOLUTION_HALTON
    // Put your code here
    // The following implementation is inspired by Halton sequence wiki.
    int index = sampleIndex;
    // return the prime of dimindex here.
    int base = prime(dimensionIndex);
    float result = 0.;
    float f = 1.;

    for (int i = 0; i>=0;i++){
      f = f/float(base);
      result +=  f * float(mod(index,base));
      if (index<=0){break;}

      index = index/base;



    }
    return result;
#else
  // Put your imeplemntation of halton here
  return 0.0;
  #endif
}

// This is the index of the sample controlled by the framework.
// It increments by one in every call of this shader
uniform int baseSampleIndex;

// Returns a well-distributed number in (0,1) for the dimension dimensionIndex
float sample(const int dimensionIndex) {
#ifdef SOLUTION_HALTON
    // Put your code here
    // If we using halton() solely, we would end up with a sequence which is an artifacutually structured pattern. Although it ensures uniformity, still objectionable as all pixels have the same value.
    // To avoid this, we need a uniform random value: using Cranely-Petterson rotation(pixelSeed()). We combine these two by taking the fractional part of the sum between halton and uniform random value.

      return fract(halton(baseSampleIndex, dimensionIndex) + pixelSeed(prime(dimensionIndex)));
#else
  // Replace the line below to use the Halton sequence for variance reduction
  return uniformRandom();
#endif
}

// This is a helper function to sample two-dimensionaly in dimension dimensionIndex
vec2 sample2(const int dimensionIndex) {
  return vec2(sample(dimensionIndex + 0), sample(dimensionIndex + 1));
}

vec3 sample3(const int dimensionIndex) {
  return vec3(sample(dimensionIndex + 0), sample(dimensionIndex + 1), sample(dimensionIndex + 2));
}

// This is a register of all dimensions that we will want to sample.
// Thanks to Iliyan Georgiev from Solid Angle for explaining proper housekeeping of sample dimensions in ranomdized Quasi-Monte Carlo
//
// So if we want to use lens sampling, we call sample(LENS_SAMPLE_DIMENSION).
//
// There are infinitely many path sampling dimensions.
// These start at PATH_SAMPLE_DIMENSION.
// The 2D sample pair for vertex i is at PATH_SAMPLE_DIMENSION + PATH_SAMPLE_DIMENSION_MULTIPLIER * i + 0
#define ANTI_ALIAS_SAMPLE_DIMENSION 0
#define LENS_SAMPLE_DIMENSION 2
#define PATH_SAMPLE_DIMENSION 4

// This is 2 for two dimensions and 2 as we use it for two purposese: NEE and path connection
#define PATH_SAMPLE_DIMENSION_MULTIPLIER (2 * 2)

vec3 randomDirection(const int dimensionIndex) {
#ifdef SOLUTION_BOUNCE
    // Put your code here
    // Using formulae provided by the coursework spec.
    vec2 xi = sample2(dimensionIndex);
    float theta = acos(2. * xi[0] - 1.);
    float phi   = xi[1] * 2. * M_PI;
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
	return vec3(x,y,z);
#else
  // Put yout code to compute a random direction in 3D here
  return vec3(0);
#endif
}

vec3 getEmission(const Material material, const vec3 normal) {
#ifdef SOLUTION_LIGHT
    // Put your code here
    return material.emission;
#else
  	// This is wrong. It just returns the diffuse color so that you see something to be sure it is working.
  	return material.diffuse;
#endif
}

vec3 getReflectance(
  const Material material,
  const vec3 normal,
  const vec3 inDirection,
  const vec3 outDirection)
{
#ifdef SOLUTION_THROUGHPUT
    // Put your code here
    // Implementing the formulae provided by coursework specs.
    // In this part We compute the Physically-correct Phong, which is a variant of Phongthat uses a normalization factor to preserves energy for all values of glossiness.
    // We are assuming the case of Perfectly diffuse, so diffuse = albedo/pi
    // First we need the angle between reflected ray and outDirection, if the angle is greater than 90 there is no specular reflection. To get refrected ray direction(r) we can call reflect(), then we do the dot() between r and outDirection to determine the angle between them.
    vec3 r = reflect(inDirection,normal);
    // Now we can compute the specular refletion based on the formula:
    // specular = ks(specular constant) * normalizationFactor * pow(max(dot(outDirection,r),0.),n) where n is the glossiness.
    // Finally, we sum up the specular and diffuse:
    vec3 f = material.specular * ((material.glossiness + 2.)/(2.* M_PI)) * pow(max(dot(outDirection,r),0.),material.glossiness)+material.diffuse;
    return f;
#else
  return vec3(1.0);
#endif
}

vec3 getGeometricTerm(
  const Material material,
  const vec3 normal,
  const vec3 inDirection,
  const vec3 outDirection)
{
#ifdef SOLUTION_THROUGHPUT
    // Put your code here
    // The geometric term is cos(theta), where the theta is the angle between surface normal and outgoing ray.
    //cosine theta can be obtained by simlpy doing the dot product between normal and the ray.
    //angle greater than 90 gives no reflectance thus set them to 0;
    float theta = max(0.,dot(outDirection,normal));
    return vec3(theta);

#else
  return vec3(1.0);
#endif
}

mat4 rotationMatrixFromAngleAxis(float angle, vec3 axis)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,
                oc * axis.x * axis.y - axis.z * s,
                oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,
                oc * axis.y * axis.y + c,
                oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,
                oc * axis.y * axis.z + axis.x * s,
                oc * axis.z * axis.z + c,           0.0,
                0.0,
                0.0,
                0.0,
                1.0);
}

vec3 getEmitterPosition(const vec3 position, const Sphere sphere, const int dimensionIndex) {
  // This is a simplified version: Just report the sphere center. Will not do well with visibility.
  //return sphere.position;

  // This is the wrong simplified version: Take a random surface point.
  //return sphere.position + randomDirection(dimensionIndex) * sphere.radius;

  // Well we stick our fingers in
  // The ground, heave and
  // Turn the world around

  // This has three main steps:
  //   1) Make a direction
  //   2) Orient it so it points to the sphere
  //   3) Find a point on the sphere along this direction

  // Step 1) Make a random direction in a cone orientedd along th up-pointing z direction
  // .. the opening angle of a sphere in a certain distance
  float apexAngle = asin(sphere.radius / length(position - sphere.position));

  // The rotation around the z axis
  float phi = sample(dimensionIndex + 1) * 2.0 * M_PI;

  // z is the cosine of the angle.
  // We need a random cosine of the angle (which is notthe same as the cosine of a random angle!)
  float z = mix(1.0, cos(apexAngle), sample(dimensionIndex + 0));
  vec3 alignedDirection = vec3(sqrt(1.0-z*z) * cos(phi), sqrt(1.0-z*z) * sin(phi), z);

  // Step 2) Rotate the z axis-aligned dirction to point into the direction of the sphere
  vec3 direction = normalize(sphere.position - position);
  float rotationAngle = acos(dot(direction, vec3(0,0,1)));
  vec3 rotationAxis = cross(direction, vec3(0,0,1));
  mat4 rotationMatrix = rotationMatrixFromAngleAxis(rotationAngle, rotationAxis);
  vec3 worldDirection = (rotationMatrix * vec4(alignedDirection, 0)).xyz;

  // Step 3) Send a ray. it feels this should be easier, but Tobias does not see it.
  Ray emitterRay;
  emitterRay.origin = position;
  emitterRay.direction = worldDirection;
  return intersectSphere(emitterRay, sphere, 0.01, 100000.0).position;
}

vec3 samplePath(const Scene scene, const Ray initialRay) {

  // Initial result is black
  vec3 result = vec3(0);

  Ray incomingRay = initialRay;
  vec3 throughput = vec3(1.0);
  for(int i = 0; i < maxPathLength; i++) {
    HitInfo hitInfo = intersectScene(scene, incomingRay, 0.001, 10000.0);

    if(!hitInfo.hit) return result;

    result += throughput * getEmission(hitInfo.material, hitInfo. normal);

    Ray outgoingRay;
#ifdef SOLUTION_BOUNCE
    // Put your code here
    // Setting the next ray be the ray origin from the intersection point.
    //   with random direction.
    outgoingRay.origin = hitInfo.position;
    outgoingRay.direction = randomDirection(PATH_SAMPLE_DIMENSION+ 2 * i);
#else
   // Put your code to compute the next ray here
#endif

#ifdef SOLUTION_THROUGHPUT
    // Put your code here
    // Throughput is computed here.
    // Up to this point, we are able to find the reflectance and the geometic term. We need to combine these two terms based on the provided formula.
    // Both reflectance and geometric term are computed based on the current intersect point as well as the direction of incoming and outgoing rays.
    vec3 ref = getReflectance(hitInfo.material, hitInfo.normal, incomingRay.direction,outgoingRay.direction);
    vec3 geo = getGeometricTerm(hitInfo.material, hitInfo.normal, incomingRay.direction,outgoingRay.direction);
    // Since we have already include the emission term from the second part(getEmission()), what left is to multiply ref and geo together which represents the weight contribution from reflectance and geometric term respectively.
    throughput *= ref*geo;
#else
    // Replace the line below for proper throughput computation
    throughput *= 0.1;
#endif

    // With importance sampling, this value woudl change
    float probability = 1.0;
    throughput /= probability;

#ifdef SOLUTION_BOUNCE
    // Put your code here
    // Setting the current outgoingRay to the next incomingRay.
    incomingRay = outgoingRay;
#else
    // Put some handling of the next and the current ray here
#endif
  }
  return result;
}

uniform ivec2 resolution;
Ray getFragCoordRay(const vec2 fragCoord) {

  	float sensorDistance = 1.0;
  	vec3 origin = vec3(0, 0, sensorDistance);
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * fragCoord, -sensorDistance));

  	float apertureSize = 0.0;
  	float focalPlane = 100.0;
  	vec3 sensorPosition = origin + focalPlane * direction;
  	origin.xy += apertureSize * (sample2(LENS_SAMPLE_DIMENSION) - vec2(0.5));
  	direction = normalize(sensorPosition - origin);

  	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
  	initRandomSequence();

#ifdef SOLUTION_AA
    // Put your code here
    // To address the anti-aliasing, I use the box filter approach.
    // The box filter is a 3x3 filter which will sum up all pixels underneath and then average them.
    // I can using double-for-loop to get the sum of all neighbouring pixels(including the center pixel) and then average them.
    // The following one is a more general case of using box filter which can be extended to any size of box(odd number).
	// Note, I changed the structure of the function getFragCoordRay() by commenting the line right below the #endif to make my code runing correctly.
    const int boxSize = 3;
    const int lowerBound = -(boxSize-1)/2;
    const int upperBound = (boxSize-1)/2+1;
    vec3 result = vec3(0.);
	float step = 0.3;// Step controls the area of the box.
    //vec2 sampleCoord;
    for(int i = lowerBound; i < upperBound; i++) {
      for(int j = lowerBound; j < upperBound; j++){
        result += samplePath(scene, getFragCoordRay(fragCoord+vec2(i,j)*step));
      }
    }
    return result/vec3(boxSize*boxSize);
	// An alternative approach is to randomly get some offset then adding to the fragCoord. So the fragCoord is referencing it's neighbouring pixels' value.
	// This approach does not require changing the function's structure. See the commented code below:
	/*vec2 sampleCoord = fragCoord + vec2(uniformRandom()-0.5,uniformRandom()-0.5);
  return samplePath(scene, getFragCoordRay(sampleCoord));*/
#else
  	// CHnage this to get anti-aliasing
	vec2 sampleCoord = fragCoord;
  return samplePath(scene, getFragCoordRay(sampleCoord));
#endif
    //return samplePath(scene, getFragCoordRay(sampleCoord));
}


void loadScene1(inout Scene scene) {

  scene.spheres[0].position = vec3( 6, 2, -12);
  scene.spheres[0].radius = 2.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.spheres[0].material.emission = 20. * vec3(0.9,0.5,0.3);
#endif
  scene.spheres[0].material.diffuse = vec3(0.0);
  scene.spheres[0].material.specular = vec3(0.0);
  scene.spheres[0].material.glossiness = 10.0;

  scene.spheres[1].position = vec3(-7, -2, -11);
  scene.spheres[1].radius = 1.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.spheres[1].material.emission = 20. * vec3(0.3,0.9,0.8);

#endif
  scene.spheres[1].material.diffuse = vec3(0.0);
  scene.spheres[1].material.specular = vec3(0.0);
  scene.spheres[1].material.glossiness = 10.0;

  scene.spheres[2].position = vec3(-2, -2, -15);
  scene.spheres[2].radius = 3.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.spheres[2].material.emission = vec3(0.0);

#endif
  scene.spheres[2].material.diffuse = vec3(0.2, 0.3, 0.8);
  scene.spheres[2].material.specular = vec3(0.2);
  scene.spheres[2].material.glossiness = 4.0;

  scene.spheres[3].position = vec3(3, -3.5, -11);
  scene.spheres[3].radius = 1.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.spheres[3].material.emission = vec3(0.0);

#endif
  scene.spheres[3].material.diffuse = vec3(0.0, 0.0, 0.0);
  scene.spheres[3].material.specular = vec3(1.0);
  scene.spheres[3].material.glossiness = 100.0;

  scene.planes[0].normal = vec3(0, 1, 0);
  scene.planes[0].d = 4.5;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.planes[0].material.emission = vec3(0.0);

#endif
  scene.planes[0].material.diffuse = vec3(0.8);
  scene.planes[0].material.specular = vec3(0);
  scene.planes[0].material.glossiness = 50.0;

  scene.planes[1].normal = vec3(0, 0, 1);
  scene.planes[1].d = 18.5;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.planes[1].material.emission = vec3(0.0);

#endif
  scene.planes[1].material.diffuse = vec3(0.9, 0.6, 0.3);
  scene.planes[1].material.specular = vec3(0.02);
  scene.planes[1].material.glossiness = 3000.0;

  scene.planes[2].normal = vec3(1, -0.3, 0);
  scene.planes[2].d = 10.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.planes[2].material.emission = vec3(0.0);

#endif
  scene.planes[2].material.diffuse = vec3(0.2);
  scene.planes[2].material.specular = vec3(0.1);
  scene.planes[2].material.glossiness = 100.0;

  scene.planes[3].normal = vec3(-1, -0.3,0);
  scene.planes[3].d = 10.0;
#ifdef SOLUTION_LIGHT
    // Put your code here
    scene.planes[3].material.emission = vec3(0.0);

#endif
  scene.planes[3].material.diffuse = vec3(0.2);
  scene.planes[3].material.specular = vec3(0.1);
  scene.planes[3].material.glossiness = 1000.0;
}

void main() {
  // Setup scene
  Scene scene;
  loadScene1(scene);

  // compute color for fragment
  gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
  gl_FragColor.a = 1.0;
}
