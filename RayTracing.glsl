#define SOLUTION_CYLINDER_AND_PLANE
#define SOLUTION_MATERIAL
#define SOLUTION_SHADOW
#define SOLUTION_REFLECTION_REFRACTION
#define SOLUTION_FRESNEL

precision highp float;
uniform float time;

struct PointLight {
    vec3 position;
    vec3 color;
};

struct Material {
    vec3  diffuse;
    vec3  specular;
    float glossiness;
#ifdef SOLUTION_MATERIAL
    float  reflectedIndex; //This variable holds the weight of reflection.
    float  refractedIndex; //This variable holds the weight of refraction.
    float  IOR;            //This variable holds the value of IOR(Index of Refraction).
	// Put the variables for reflection and refraction here
#endif
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

struct Cylinder {
    vec3 position;
    vec3 direction;
    float radius;
    Material material;
};

const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;

struct Scene {
    vec3 ambient;
    PointLight[lightCount] lights;
    Sphere[sphereCount] spheres;
    Plane[planeCount] planes;
    Cylinder[cylinderCount] cylinders;
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
	return HitInfo(
      	false,
      	0.0,
      	vec3(0.0),
      	vec3(0.0),
#ifdef SOLUTION_MATERIAL
		// Update the constructor call
		Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0)
#else
		Material(vec3(0.0), vec3(0.0), 0.0)
#endif
	);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
  	// Make t1 the smaller t
    if (t2 < t1) {
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
	if (isTInInterval(t0, tMin, tMax)) {
		smallestTInInterval = t0;
        return true;
	}

	// If t0 was not in the interval, still t1 could be
	if (isTInInterval(t1, tMin, tMax)) {
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
      	if (!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
        }

      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

      	vec3 normal =
			length(ray.origin - sphere.position) < sphere.radius + 0.001 ?
          	-normalize(hitPosition - sphere.position) :
      		normalize(hitPosition - sphere.position);

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material
        );
    }
    return getEmptyHit();
}
/*
 The explanation of Ray-Plane intersection are based on the study of tutorial from
 "https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm"

  The equation of Rays   : P = P0 + t * V
  The equation of Planes : P dot N + d = 0
  Where P : point of intersection
        P0: Origin of the ray
        t : Time of movement
        V : Direction of the ray
        N : Normal of the plane
        d : Displacement of the plane
  By substitution: (P0 + t*V) dot N + d = 0;
  Therefore we can solve the following equation to get t:
  t = -(P0 dot N + d) / (V dot N)

*/
HitInfo intersectPlane(const Ray ray,const Plane plane, const float tMin, const float tMax) {
	// Add your plane intersection code here
#ifdef SOLUTION_CYLINDER_AND_PLANE
  float D = dot(ray.direction, plane.normal);
  float t;
  vec3 hitPosition; //The value of P
	// Add your plane intersection code here
  if(D < 0.0){ //Check the plane is at the bottom side;
    t = -(dot(ray.origin,plane.normal) + plane.d) / dot(ray.direction,plane.normal);
    hitPosition = ray.origin + t * ray.direction;
	return HitInfo(
          true,
          t,
          hitPosition,
          plane.normal,
          plane.material
     );
   }
#endif
	return getEmptyHit();
}


float lengthSquared(vec3 x) {
	return dot(x, x);
}
/**
The explanation of Ray-Cylinder intersection are based on the study of tutorial from
 "https://mrl.nyu.edu/~dzorin/rend05/lecture2.pdf"

  The equation of Rays      : P   = P0 + t * V
  The equation of cylinders : r^2 = (P - Pc - (Vc dot P - Pc) * Vc)^2
  Where P : Point of intersection
        P0: Origin 0f the ray
        t : Time of movement
        V : Direction of the ray
        r : The radius of the cylinder
        Pc: The center point located at the cylinder's caps
        Vc: The direction of Pc/Cylinder
  Put Pc, Vc, t_c=(Vc dot P - Pc) together we have a line equation:
      Pc + Vc * t_c
  Therefore, The equation of the cylinders can be interpreted as a cylinder
   with radius of r rotated along the previously defined line : Pc + t_cVc

  By substitution, we get:
    (P - Pc + V * t - (Vc dot P - Pc + V * t) * Vc)^2 = r^2
  Re-arrange the above term to form a quadratic equation: A * t^2 + B * t + C = 0;
  Where a = (V - (V dot Vc) * Vc)^2
        b = 2 * ((V - (V dot Vc) * Va) dot (D - (D dot Vc) * Vc))
        c = (D - (D dot Vc) * Vc)^2 - r^2
        D = P - Pc


*/
HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
	// Add your cylinder intersection code here

	vec3 to_cylinder = ray.origin - cylinder.position; // the value of D = P - Pa
  float a,b,c,Discriminant,t0,t1;
  vec3 V = ray.direction;
  vec3 Vc= cylinder.direction;
   a = dot((V - dot(V,Vc) * Vc),(V - dot(V,Vc) * Vc));
   b = 2.0 * dot((V - dot(V,Vc) * Vc),(to_cylinder - dot(to_cylinder,Vc) * Vc));
   c = dot((to_cylinder - dot(to_cylinder, Vc) * Vc),(to_cylinder - dot(to_cylinder, Vc) * Vc)) - cylinder.radius * cylinder.radius;
	 Discriminant = b * b - 4.0 * a * c;
	//return getEmptyHit();
  //The rest part of this function works similar to ray_sphere intersection.
	if (Discriminant > 0.0)//If have roots
    { // calculate 2 roots for quadratic equation.
		 t0 = (-b - sqrt(Discriminant)) / (2.0 * a);
		 t1 = (-b + sqrt(Discriminant)) / (2.0 * a);

      	float smallestTInInterval;
        /* The intersection always gives to hit position one is towards the
            object, another is outwards the object. We need the point that is
            closer to the ray's origin therefore we need tMin.
        */
      	if (!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
        }

      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;
        vec3 vector_1 =hitPosition - cylinder.position;
        vec3 normal = normalize(vector_1 - dot(vector_1,cylinder.direction)*cylinder.direction); // The direction term should always be normalized as how it is defined.

    return HitInfo(
            true,
            smallestTInInterval,
            hitPosition,
            normal,
            cylinder.material
      );
    }


#endif
    return getEmptyHit();
}

HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
	if(newHitInfo.hit)
  		if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
          return newHitInfo;
  	return oldHitInfo;
}

HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
	HitInfo bestHitInfo;
	bestHitInfo.t = tMax;
	bestHitInfo.hit = false;
	for (int i = 0; i < cylinderCount; ++i) {
    	bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
	}
	for (int i = 0; i < sphereCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
	}
	for (int i = 0; i < planeCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
	}

	return bestHitInfo;
}

/*
During the ray tracing procedure, after the ray is reflected when it hits the surface(mirror), the shadow will
  appear if there is an object occludes the ray reaching the light source.
Mathematically, a light ray is defined by the following equation:

  Ir = ka * Ia + v * Ii * (kd * max(0,(n dot l)) + ks * (max(0,(h dot n))^m)

Where Ii : normalized intensity of light
      ka : light reflection propotion from ambient
      Ia : scene ambient light
      n  : surface normal
      l  : light Direction
      kd : light reflection propotion from diffuse
      e  : directoin to eye
      h  : the bisection between e and l
      m  : power of light(glossiness);
      v  : visibility of light
      ks : light reflection propotion from specular

To achieve the effect of shadow, we can perform a shadow test. The test is done by creating a new ray named
  shadowRay in my case, do the same ray-object intersection job but always starts from the hit point position.
  Once meet the next intersection, flag the visibility v of the light ray to zero.
*/

vec3 shadeFromLight(
	const Scene scene,
	const Ray ray,
	const HitInfo hit_info,
	const PointLight light)
{
	vec3 hitToLight = light.position - hit_info.position; // this gives the distance from hitpoint to light

	vec3 lightDirection = normalize(hitToLight); // l
	vec3 viewDirection = normalize(hit_info.position - ray.origin); // e
	vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
	float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal)); // kd
	float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness); // ks

  #ifdef SOLUTION_SHADOW
  Ray shadowRay; // Create the shadow ray
  shadowRay.direction = lightDirection; // Set the shadow ray direction equals to light direction.
  shadowRay.origin    = hit_info.position; // Set the shadow ray origin always equals to the hit position.
	float visibility = 1.0; //v
  HitInfo testHit;

  /* Fortunately, the provided function intersectSecne provide a way to detect ray intersection
      for all presented objects by a single function call.
    Calling intersectScene with setting Ray @param to shadowRay to accomplish shadow test. During the test,
      any intersection will cause visibility of light become 0.0;
  */

  /* One noticible error during the shadow test is the inappropriate value of @param tMax when calling
      intersectScene.
     The value of tMax should be the distance from hit point to light, otherwise
        there will be a shadow caused by a intersection point which is the outward point.
        length(hitToLight) caps the intersection point always be the one that is pointing inward to the object.
  */
  testHit = intersectScene(scene,shadowRay , 0.001, length(hitToLight));
  if(testHit.hit){ //Shadow ray intersection detection.
      visibility = 0.0; // Set v = 0.0 creates the shadow
	}
#else
  	float visibility = 1.0;
#endif

	Ray mirrorRay;
	mirrorRay.origin = hit_info.position;
	mirrorRay.direction = reflect(lightDirection, hit_info.normal);
	HitInfo mirrorHitInfo = intersectScene(scene, mirrorRay, 0.001, 100000.0);

  return visibility *
		 light.color * (
		 specular_term * hit_info.material.specular +
		 diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
	// A simple implicit sky that can be used for the background
	return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {

  	if(!hitInfo.hit) {
		return background(ray);
  	}

    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
		shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]);
    }
    return shading;
}


Ray getFragCoordRay(const vec2 frag_coord) {
	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(800, 400);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));

  	return Ray(origin, direction);
}
/*
  Part of this explanation is based on the study of the following tutorial:
    "https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf"
  In reality, a transparent medium is both reflective and refractive. So introducing Fresnel equation to measure how
    much light should be reflacted and how much light should be refracted.
  The fresnel splits light into parallel and perpendicular polarised light.
  The fresnel equation is given as follows:

  F_para   = (eta2 * cos(Theta1) - eta1 * cos(Theta2) / (eta2 * cos(Theta1) + eta1 * cos(Theta2)))^2
  F_perpen = (eta1 * cos(Theta2) - eta2 * cos(Theta1) / (eta1 * cos(Theta2) + eta2 * cos(Theta1)))^2

  F_reflection = 1/2 * (F_para + F_perpen)

  Where Theta1 : angle of incedent
        THeta2 : angle of reflect
  The F_reflection tells how much light should reflected.
  Therefore, F_refraction can be computed by 1 - F_reflection.

  My approach:
  Another approach is to adpot Schlick's approximation which frees the computation of pow() a little bit.
  The approximation equation:
    Rs = R0 + (1 - R0) * (1 - cos(theta_incident))^exponent
    R0 = ((eta1 - eta2) / (eta1 + eta2))^2
  Where theta_incident : the angle of incident
        exponent controls the powers of Schlick's output
  One demerit of using Schlick is that the approximation does not functioning when eta1 > eta2(happens when incident comes from lower IOR medium), therefore, certain condition needs to be established to handle such situation.
*/

/* I modified the fresnel() function by adding one extra @param const float IOR to it. So the
    function takes the input of refracted medium's IOR */
float fresnel(const vec3 viewDirection, const vec3 normal, const float IOR) {
#ifdef SOLUTION_FRESNEL
  float sinTran; // This defines sin(angle of transmitted(refracted))
	// Put your code to compute the Fresnel effect here
  float cosIncident = dot(viewDirection, normal); // This defines cos(theta_incident)

  //Caps the cos() value between -1.0 and 1.0
  if(cosIncident >= 1.0){cosIncident = 1.0;}
  else if(cosIncident <= -1.0){cosIncident = -1.0;}
  else{cosIncident = cosIncident;}

  float eta1 = 1.0 , eta2 = IOR;
  float tempTrans = 1.0 - cosIncident * cosIncident;
  /*
  We know cos(x)^2 + sin(x)^2 = 1, combines with Snell's law,
    we can get sin(Theta_refracted) -- the critical angle of TIR by:
      sin(Theta_refracted) = (eta1 / eta2) * sin(Incident)
      and cos(Theta_refracted) = sqrt(1 - sin(sin(Theta_refracted)))
  */
  if(tempTrans >= 0.0){tempTrans = tempTrans;}
  else{tempTrans = 0.0;}
  sinTran = (eta1 / eta2) * sqrt(tempTrans);


  float schlickR;
  float R0 = (eta1 - eta2) / (eta1 + eta2);
  R0 = R0 * R0;
  float tempCosTran = 1.0 - sinTran * sinTran;
  if(tempCosTran >= 0.0){tempCosTran = tempCosTran;}
  else{tempCosTran = 0.0;}

  float cosTran = sqrt(tempCosTran); // This defines cos(Theta_refracted)
  cosIncident = abs(cosIncident);

  float baseTermI = (1.0 - cosIncident);
  float baseTermT = (1.0 - cosTran);

  /*
    If eta1 > eta2 happens, we can simply swap eta1 and eta2, or replace cos(theta_incident) to
      cos(theta_refracted)
  */
  if(eta1 <= eta2){
    schlickR = R0 + (1.0 - R0) * pow(baseTermI,2.0);
  } // Still need to check the condition whether we are getting into a TIR situation( in that case return 1).
  else if((eta1>eta2)&&((sinTran*sinTran) <= 1.0)){schlickR = R0 + (1.0 - R0)* pow(baseTermT,2.0);}
  else if((eta1>eta2)&&((sinTran*sinTran) > 1.0)){ schlickR = 1.0;}

  return schlickR;
	//return 1.0;
#else
	return 1.0;
#endif
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {

    Ray initialRay = getFragCoordRay(fragCoord);
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);
  	vec3 result = shade(scene, initialRay, initialHitInfo);

  	Ray currentRay;
    Ray reflectedRay;
  	HitInfo currentHitInfo;

  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;

  	// The initial strength of the reflection
  	float reflectionWeight = 1.0;

  	const int maxReflectionStepCount = 2;
  	for (int i = 0; i < maxReflectionStepCount; i++) {
		if (!currentHitInfo.hit) break;

#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your reflection weighting code here
    // Make sure every reflection is a lossy action
    reflectionWeight *= currentHitInfo.material.reflectedIndex;
#endif

#ifdef SOLUTION_FRESNEL
		// Add Fresnel contribution
    // Applying fresnel, which returns the weight of reflection.
    reflectionWeight *= fresnel(currentRay.direction, currentHitInfo.normal, currentHitInfo.material.IOR);
#else
		reflectionWeight *= 0.5;
#endif

		Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your code to compute the reflection ray here
    /*
    Reflection is a light hits the surface with angle theta and reflected with the angle theta along the surface normal.

    The reflection is based on the follwing equation:
      r = -e + 2 * (n dot e) * n  (from lecture slides)
    Where r : the reflected light direction
          e : incident light direction(need normalized)
          n : surface normal
    */
    vec3 IncidentDirection =  currentHitInfo.position - currentRay.origin; // This defines e
    reflectedRay.direction =  normalize(IncidentDirection - 2.0 * dot(currentHitInfo.normal, IncidentDirection) * currentHitInfo.normal); // Using equation to calculate r.
    reflectedRay.origin = currentHitInfo.position;
    nextRay = reflectedRay;

#endif
		currentRay = nextRay;
		currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
		result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }

	// Compute the refraction
	currentRay = initialRay;
	currentHitInfo = initialHitInfo;

  	// The initial medium is air
  	float currentIOR = 1.0;

  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;

  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {

#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your refraction weighting code here
    // Make sure every refraction is a lossy action
    refractionWeight *= currentHitInfo.material.refractedIndex;
		//reflectionWeight *= 0.5;
#else
		refractionWeight *= 0.5;
#endif

#ifdef SOLUTION_FRESNEL
		// Add Fresnel contribution
    // Due to the conservation of energy, fresnel_refracted = 1.0 - fresnel_reflect
    refractionWeight *= (1.0 - fresnel(currentRay.direction, currentHitInfo.normal, currentHitInfo.material.IOR));

#endif

		Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your code to compute the reflection ray and track the IOR

    /*
    This explanation is based on the study of the following tutorial: 'https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel'

    Refraction is a phenomenon that light travels through a transparent medium with tilted direction.
      This tilted direction are determined by the Snell's law.
    The Snell's law states: the ratio of sin of incedence angle and refracted angle is
      an inverse ratio of indices of medium's refraction(IOR). Mathematically, snell's law can be expressed as:

      sin(Theta1) /sin(Theta2) = eta2/eta1
      Where Theta1 is the angle of incidence and Theta2 is the angle of refraction. Eta2
        is the IOR of medium for refracted light and eta1 is the IOR for incendent light.

    The refracted light r can be expressed by:
      eta = eta1 / eta2
      c1  = cos(thetaI) = N dot I // cos of Incident angle
      c2  = sqrt(1 - eta^2) * sin(thetaI)^2
      r = eta * I + (eta * c1 - c2) * N
      where I : direction of incident light
            N : hit point surface normal

    Total internal reflection(TIR) is a phenomenon when incedent angle is greater than certain angle values. Such value often called critical angle. In that case all light is being reflected and no refraction at all. This usually happens when the incident light comes from a medium with lower IOR. The TIR can be detected by the following condition :
      sin(Theta_refracted) > 1.

    */
    vec3 currentSurfaceNormal = currentHitInfo.normal; // This defines N
    float cosTheta = dot(currentSurfaceNormal, currentRay.direction);  // This defines c1;

    // Caps the value of cosTheta so it does not goes beyound 1 and -1
    if(cosTheta >= 1.0){cosTheta = 1.0;}
    else if(cosTheta <= -1.0){cosTheta = -1.0;}
    else{cosTheta = cosTheta;}

    float eta1 = currentIOR;
    float eta2 = currentHitInfo.material.IOR;

    if(cosTheta < 0.0){
      //outside surface
      cosTheta = -cosTheta;
    }
    else{
      //inside the surface
      currentSurfaceNormal = -currentSurfaceNormal;
    }

    float eta = eta1 / eta2;
    float k = 1.0 - (eta * eta * (1.0 - cosTheta * cosTheta));
    vec3 refractedRayDirection; // This defines r
    if(k<=0.0){ //This detects the TIR situation.
      k = 0.0;
    }
    else{
      refractedRayDirection = eta * currentRay.direction + (eta * cosTheta - sqrt(k)) * currentSurfaceNormal; //Compute r.
    }
    nextRay.direction = refractedRayDirection;
    nextRay.origin   = currentHitInfo.position;
    currentRay = nextRay;
    currentIOR = currentHitInfo.material.IOR;
#endif

		currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
		result += refractionWeight * shade(scene, currentRay, currentHitInfo);

		if (!currentHitInfo.hit) break;
	}
	return result;
}

/*The Material struct contains 6 elements:
  Diffuse            -diffuse          : Controls the roughness of the surfaces
  Specular           -specular         : Controls the smoothness of the surfaces
  Glossiness         -glossiness       : Controls the power of light(area of highlight)
  Reflection weight  -reflectedIndex   : Controls how reflection light decays after each reflection
  Refraction weight  -refractedIndex   : Controls how refraction light decays after each refraction
  IOR                -IOR              : Controls the angle of refracted light

  The term diffuse/specular is a vec3 type which means each element represent a single color channel in RGB space. So by assigning different value to diffuse, we can get different color of objects/highlights.

Note:  The IOR is set to 1.0(default value) if there is no refraction for
    that material(It is programmed to tackle such situation). Though its value coincides the value of vacuum.
*/
Material getDefaultMaterial() {
#ifdef SOLUTION_MATERIAL
	// Update the default material call to match the new parameters of Material
	return Material(vec3(0.3), vec3(0), 1.0, 0.0, 1.0, 1.0);
#else
	return Material(vec3(0.3), vec3(0), 1.0);
#endif
}
/*
Paper should have a rough surface(high diffuse vec3(.999)---> this gives white apperance)
  therefore low specular value(vec3(.0)). Setting specular to 0.0 and Glossiness
  to 1.0 can remove any specular reflection(highlight) on the paper surface.
By real world observation, paper does not reflect nor refract any light. therefore
  reflection , refraction weight are set to .0. The value .
Note:  The IOR is set to 1.0(default value) if there is no refraction for
  that material as programmed. Though its value coincides the value of vacuum.
*/
Material getPaperMaterial() {
#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a paper material
	return Material(vec3(.999), vec3(.0), 1., .0, .0,1.);
  	//return getDefaultMaterial();
#else
    return getDefaultMaterial();
#endif
}
/*
Compare to paper, plastic material has a smoother surface but still maintains roughness a bit.
Therefore I set vec3(.75, .25, .1) for diffuse which gives white to yellowish apperance.
Assigning vec3( .9) to specular and 8.5 to glossiness gives the spcular reflection that looks
similar to the demo's.
By real world observation, plastic material should reflect some lights with 30% decay but still no refraction.
Therefore, I assign .7 to reflaction weight and .0, 1.0 to refracted weight and IOR respectively.
*/
Material getPlasticMaterial() {
#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a plastic material
	return Material(vec3(.75, .25, .1), vec3( .9), 8.5, .7, .0, 1.);

#else
  	return getDefaultMaterial();
#endif
}

/*
The glass is trasparent and a flat surface no roughness(no diffuse). Due to the glass from demo gives no highlight, there fore .0 specular and 1. for glossiness.
With the transparent property, the light hitting the glass surface should be reflected or refracted with no lost.
The IOR of glass is close to Vacum so I set 1.1.
*/
Material getGlassMaterial() {
#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a glass material
  	return Material(vec3(.0), vec3(.0), 1., 1., 1., 1.1);
#else
	return getDefaultMaterial();
#endif
}
/*
The mirror has a flat surface so .07 to diffuse(the mirror surface is slightly rougher than glass). Due to the demo's mirror seems to have no highlight on it, I set minimum glossiness and minimum specular to mirror
  to diminish the highlights on mirror.
The mirror should reflect all lights with almost no cost (.8) and no refraction.
*/
Material getSteelMirrorMaterial() {
#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a steel mirror material
  	return Material(vec3(.07), vec3(.0), 1., .8, .0, 1.);

#else
	return getDefaultMaterial();
#endif
}

vec3 tonemap(const vec3 radiance) {
	const float monitorGamma = 2.0;
	return pow(radiance, vec3(1.0 / monitorGamma));
}

void main()
{
    // Setup scene
	Scene scene;
  	scene.ambient = vec3(0.12, 0.15, 0.2);

    // Lights
    scene.lights[0].position = vec3(5, 15, -5);
    scene.lights[0].color    = 0.5 * vec3(0.9, 0.5, 0.1);

  	scene.lights[1].position = vec3(-15, 5, 2);
    scene.lights[1].color    = 0.5 * vec3(0.1, 0.3, 1.0);

    // Primitives
    scene.spheres[0].position            	= vec3(10, -5, -16);
    scene.spheres[0].radius              	= 6.0;
    scene.spheres[0].material 				= getPaperMaterial();

  	scene.spheres[1].position            	= vec3(-7, -1, -13);
    scene.spheres[1].radius             	= 4.0;
    scene.spheres[1].material				= getPlasticMaterial();

    scene.spheres[2].position            	= vec3(0, 0.5, -5);
    scene.spheres[2].radius              	= 2.0;
    scene.spheres[2].material   			= getGlassMaterial();

  	scene.planes[0].normal            		= vec3(0, 1, 0);
  	scene.planes[0].d              			= 4.5;
    scene.planes[0].material				= getSteelMirrorMaterial();

  	scene.cylinders[0].position            	= vec3(-1, 1, -18);
  	scene.cylinders[0].direction            = normalize(vec3(-1, 2, -1));
  	scene.cylinders[0].radius         		= 1.5;
    scene.cylinders[0].material				= getPaperMaterial();

  	scene.cylinders[1].position            	= vec3(4, 1, -5);
  	scene.cylinders[1].direction            = normalize(vec3(1, 4, 1));
  	scene.cylinders[1].radius         		= 0.4;
    scene.cylinders[1].material				= getPlasticMaterial();

	// compute color for fragment
	gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
	gl_FragColor.a = 1.0;

}
