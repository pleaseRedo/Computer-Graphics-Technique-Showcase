#define PROJECTION
#define RASTERIZATION
#define CLIPPING
#define INTERPOLATION
#define ZBUFFERING
//#define ANIMATION

precision highp float;
uniform float time;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 viewport;

struct Vertex {
    vec3 position;
    vec3 color;
};

struct Polygon {
    // Numbers of vertices, i.e., points in the polygon
    int vertexCount;
    // The vertices themselves
    Vertex vertices[MAX_VERTEX_COUNT];
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) {
            polygon.vertices[i] = element;
        }
    }
    polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        destination.vertices[i] = source.vertices[i];
    }
    destination.vertexCount = source.vertexCount;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
    if (index >= polygon.vertexCount) index -= polygon.vertexCount;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == index) return polygon.vertices[i];
    }
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
  polygon.vertexCount = 0;
}

// Clipping part

#define ENTERING 0
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {
#ifdef CLIPPING
    // Put your code here
	// This fuctions takes two line and return the four states: Enter,leave,Outside,Inside
	//Same as what we have done in rasterization part, write out the line equation, find the intersection

	float intersection, slope,dy,dx;
  dy = (wind2.position[1] - wind1.position[1]);
  dx = (wind2.position[0] -wind1.position[0]);
  slope = dy / dx;
  intersection = wind1.position[1] - (slope * wind1.position[0]);

	// We can use the line to test the positional relationship between line poli and wind
	float p1,p2,direction;
	p1 = (slope*poli1.position[0] + intersection - poli1.position[1]);
	p2 = (slope*poli2.position[0] + intersection - poli2.position[1]);
	direction = sign((wind2.position[0] - wind1.position[0]));
	p1 = p1*direction;p2 = p2*direction;
	if(p1*p2 ==0.){return INSIDE;} //when hovering the line
	//This gives the direction of two points, if they have different direction then its the case of leave of enter; if same direction then its in or out.
	// In this case, greater than 0.0 is inside.
	bool checkp1 = p1>0.0; bool checkp2 = p2>0.0;
	if(!checkp1&&checkp2){return ENTERING;} // Is entering because second point is inside and these two points have different direction
	else if(checkp1&&!checkp2){return LEAVING;} 	// Is leaving because second point is outside and these two points have different direction
	else if(!(checkp1&&checkp2)){return OUTSIDE;}
	else if((checkp1&&checkp2)){return INSIDE;}



#else
    return INSIDE;
#endif
}

// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef CLIPPING
    // Put your code here
	// When these two lined intersect with each other, F1 = F2 ,solve  for x and y

	// We konw line can be expressed as L = slope * x + b - y
	Vertex POI;
  float slope1, slope2, intersection1, intersection2;
	slope1 = (b.position[1] - a.position[1]) / (b.position[0] - a.position[0]);
	slope2 = (d.position[1] - c.position[1]) / (d.position[0] - c.position[0]);
	intersection1 =  slope1 * b.position[0] - b.position[1];
	intersection2 =  slope2 * d.position[0] - d.position[1];

	// If two lines intersect, their L value must be the same, so we can have:
		// s1*x1+b1 = s2*x2+b2
		// x = (b1-b2) / (a1-a2)
		// y = a1 * ((b1-b2)/(a1-a2) - x) + b1
	POI.position[0] = (intersection1 - intersection2)/ (slope1-slope2);
	POI.position[1] = (POI.position[0]-b.position[0])*slope1 + b.position[1];

	//According to the slide of interpolation: u_s = u_1 + s(u_2 -u_1) this can applied directly to POI to get s:
	// s = (u_s-u_1)/(u_2-u_1)
	float s = (POI.position[0] - a.position[0])/(b.position[0] - a.position[0]);
	// According to the equation(12) from zBuffer lec slides, we can get Z_t by: 1/z_t = 1/z_1 + s*(1/z_2 - 1/z_1)

	POI.position[2] = (1. / a.position[2] + s * (1./b.position[2] - 1./a.position[2]));
	POI.position[2] = 1. /POI.position[2];
	// Still from the lec slides, we can compute the t: t = z_1s /z_1s + z_2(1-s)
	float t =  a.position[2] * s / (a.position[2] * s + b.position[2]* (1. - s));
	// Color update: do the interpolation as above:
	POI.color = a.color + t * (b.color - a.color);
	return POI;
#else
    return a;
#endif
}


void sutherlandHodgmanClip(Polygon unclipped, Polygon clipWindow, out Polygon result) {
    Polygon clipped;
    copyPolygon(clipped, unclipped);

    // Loop over the clip window
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i >= clipWindow.vertexCount) break;

        // Make a temporary copy of the current clipped polygon
        Polygon oldClipped;
        copyPolygon(oldClipped, clipped);

        // Set the clipped polygon to be empty
        makeEmptyPolygon(clipped);

        // Loop over the current clipped polygon
        for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
            if (j >= oldClipped.vertexCount) break;

            // Handle the j-th vertex of the clipped polygon. This should make use of the function
            // intersect() to be implemented above.
#ifdef CLIPPING
    // Put your code here
      Vertex cWin1,cWin2,p1,p2;
      cWin1 = getWrappedPolygonVertex(clipWindow,i);cWin2 = getWrappedPolygonVertex(clipWindow,i+1);
      p1 = getWrappedPolygonVertex(oldClipped,j);p2 = getWrappedPolygonVertex(oldClipped,j+1);

		Vertex POI = intersect2D(p1,p2,cWin1,cWin2);
      if(getCrossType(p1,p2,cWin1,cWin2) == ENTERING){ //Entering
        //According to the pdf, append both point
        appendVertexToPolygon(clipped,POI);
        appendVertexToPolygon(clipped,p2);
      }
      if(getCrossType(p1,p2,cWin1,cWin2) == LEAVING){ //Leaving
        //According to the pdf, append Interscetion point only
        appendVertexToPolygon(clipped,POI);
      }
      if(getCrossType(p1,p2,cWin1,cWin2) == OUTSIDE){ //Outside
        //According to the pdf, do nothing
      }
      if(getCrossType(p1,p2,cWin1,cWin2) == INSIDE){ //Inside
        //According to the pdf, append p2.
        appendVertexToPolygon(clipped,p2);
      }
#else
            appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
        }
    }

    // Copy the last version to the output
    copyPolygon(result, clipped);
}

// Rasterization and culling part

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point
// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {
#ifdef RASTERIZATION
    // Put your code here
    /*The idea of this test is to compute a line equation:
      L_i(x_i,y_i) = a_ix + b_iy + c_i
      if L_i > 0, then point is at positive half-space
      if L_i < 0, then point is at negative half-space
      if L   >=0, then all points are within the triangle space.
      The above test is also called half-space test.(According the slides)
    */

    /* The line equation can be written in the slope/intersection form:
        y = slope * x + intersection
        where, a is the slope
               b is the intersection
    */
    float slope, intersection;
    // slope = dy/dx = (y2-y1) / (x2-x1); intersection = y - slope * x
    float dy = (a.position[1] - b.position[1]);
    float dx = (a.position[0] - b.position[0]);
    slope = dy / dx;
    intersection = a.position[1] - (slope * a.position[0]);

    //Now we complete the line equation, so we can do the half-space test
    // using L = (slope_line * x_point + intersection_line - y_point)
    float L = slope * point[0] + intersection - point[1];
    //bool checkClockwise = (b.position[0] - a.position[0]) > 0.;
	  // As defined later in the isPointInPolygon check, vertex b  comes after vertex a,
    // vector at RHS is defined as the inner side.
	   // so we need the sign of b - a to maintain the clockwise direction.
   	bool verdict = (L * sign(b.position[0] - a.position[0])) >0.;
	  if (verdict ){return INNER_SIDE;}

#endif
    return OUTER_SIDE;
}

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
    // Don't evaluate empty polygons
    if (polygon.vertexCount == 0) return false;
    // Check against each edge of the polygon
    bool rasterise = true;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#ifdef RASTERIZATION
    // Put your code here


		// calling getWrappedPolygonVertex() will return vetex on demand.
		Vertex a = getWrappedPolygonVertex(polygon,i);
    Vertex b = getWrappedPolygonVertex(polygon,i+1);

		//Perform the half-test, if the vertex is outside, discard it(continue at for-loop)
  	bool verdict = edge(point, a, b) == INNER_SIDE;
		if(verdict){continue;}else{rasterise = false;}

#else
            rasterise = false;
#endif
        }
    }
    return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
          	ivec2 pixelDifference = ivec2(abs(polygon.vertices[i].position.xy - point) * vec2(viewport));
          	int pointSize = viewport.x / 200;
            if( pixelDifference.x <= pointSize && pixelDifference.y <= pointSize) {
              return true;
            }
        }
    }
    return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {
    // https://en.wikipedia.org/wiki/Heron%27s_formula
    float ab = length(a - b);
    float bc = length(b - c);
    float ca = length(c - a);
    float s = (ab + bc + ca) / 2.0;
    return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
    float weightSum = 0.0;
    vec3 colorSum = vec3(0.0);
    vec3 positionSum = vec3(0.0);
    float depthSum = 0.0;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#if defined(INTERPOLATION) || defined(ZBUFFERING)
    // Put your code here
    // We need to get current and the other two vertices to form a triangular area:
    // The point together with another two vertices will form a sub-triangle whose area is the weight of A.
    Vertex A;
    A = getWrappedPolygonVertex(polygon,i);
    Vertex B,C;
    B = getWrappedPolygonVertex(polygon,i+1);
    C = getWrappedPolygonVertex(polygon,i+1+1);
    // In our case, the area of tri(ABC) is equal to the color weight of A.
    float A_weight= triangleArea(vec2(B.position[0],B.position[1]), vec2(C.position[0],C.position[1]),point);

#else
#endif
#ifdef ZBUFFERING
    // Put your code here
    // Both of pos and depth needs to divided by Z, since a projection transform.
    positionSum += A.position * A_weight / A.position[2];
    depthSum += (A_weight/triangleArea(vec2(B.position[0],B.position[1]),vec2(C.position[0],C.position[1]),vec2(A.position[0],A.position[1]))) /A.position[2];


#endif
#ifdef INTERPOLATION
    // Put your code here

    colorSum += (A_weight/triangleArea(vec2(B.position[0],B.position[1]),vec2(C.position[0],C.position[1]),vec2(A.position[0],A.position[1])))/A.position[2] * A.color;
    weightSum+= A_weight / A.position[2];

#endif
        }
    }

    Vertex result = polygon.vertices[0];

#ifdef INTERPOLATION
    // Put your code here
    result.color = colorSum / depthSum;

#endif
#ifdef ZBUFFERING
    // Put your code here
    result.position = positionSum / weightSum;

#endif
#if !defined(INTERPOLATION) && !defined(ZBUFFERING)
    // Put your code here
    return results;
#endif

  return result;
}

// Projection part

// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
    mat4 projectionMatrix = mat4(1);

  float aspect = float(viewport.x) / float(viewport.y);
  float imageDistance = 0.5;

#ifdef PROJECTION
    // Put your code here
    //The projection matrix takes 4 params, theta(fov),a(aspect),n(distance to near clipping plane) and f(distance to far clipping plane)
	// The value of theta_fov .65 is obtained by setting it to various possible value and pick one which gives best similar result to the one from the cw2 pdf
    float theta_fov = .65;
	// Set this to infinity atm.
    float f, d, n;
    f = 9999999.;
    d = 1.0 / tan(theta_fov/2.);
    n = imageDistance;
    projectionMatrix[0] = vec4(d/aspect, 0. ,0. ,0.);
    projectionMatrix[1] = vec4(0., d ,0. ,0.);
	// the value n and f by how they are defined are negative, so I put a abs() when assigning values to the thrid row of projection matrix.
    projectionMatrix[2] = vec4(0., 0. ,abs((n+f)/(n-f)) ,abs((2.*n*f)/(n-f)));
    projectionMatrix[3] = vec4(0., 0. ,-1. ,0.);

#endif

    return projectionMatrix;
}

// Used to generate a simple "look-at" camera.
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    mat4 viewMatrix = mat4(1);

#ifdef PROJECTION
    // Put your code here
    /*By follwing the lecture slides, to create the view matrix, we need to
        compute the followings : vec: u v n these vectors forms the Viewing
        co-coordinate, q and t for handling translation.
    */

    /*So far, we have VRP(where camera is located) and a vec TP(target point)
        By definition, we can get VPN(View plane normal) by taking the subtraction
        of VRP and TP
    */
    vec3 VPN, n, u, v, t;
    //View reference point q:
    //vec4 q = [0.,0.,0.,1.];
    VPN = TP - VRP;

    //By following the slides we can get n u v vectors:
    float n_norm = pow(VPN[0],2.) + pow(VPN[1],2.) + pow(VPN[2],2.);
	n_norm = sqrt(n_norm);
    n = VPN/n_norm;

    float u_norm = pow(cross(VUV,n)[0],2.) + pow(cross(VUV,n)[1],2.) + pow(cross(VUV,n)[2],2.);
  	u_norm = sqrt(u_norm);
    u = cross(VUV,n) / u_norm;
    v = cross(n,u);
	//dot(vrp,u) is equivalent to sum(q_i,u_i)
	  t = vec3(- dot(VRP, u), - dot(VRP, v), - dot(VRP, n));

    //We have computed everyting we need to finish the 4x4 view matirx:

	  viewMatrix[0] = vec4(u[0],v[0],n[0],0.);
    viewMatrix[1] = vec4(u[1],v[1],n[1],0.);
    viewMatrix[2] = vec4(u[2],v[2],n[2],0.);
    viewMatrix[3] = vec4(t,1.);

#endif
    return viewMatrix;
}



vec3 getCameraPosition() {
#ifdef ANIMATION
    // Put your code here
    return vec3(10.0 * sin(time), 10.0 * cos(time), 10.0 * tan(time));

#else
    return vec3(0, 0, 10);
#endif
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec3 projectVertexPosition(vec3 position) {

  // Set the parameters for the look-at camera.
    vec3 TP = vec3(0, 0, 0);
  	vec3 VRP = getCameraPosition();
    vec3 VUV = vec3(0, 1, 0);

    // Compute the view matrix.
    mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  // Compute the projection matrix.
    mat4 projectionMatrix = computeProjectionMatrix();

#ifdef PROJECTION
    // Put your code here
	// The transformation/matrix multiplication comes with orders, first transformation comes at the right most, which is next to the points' position vector.
	mat4 mixedTransformation = projectionMatrix * viewMatrix;
	// The given vec3 position needs an extra dimension to fit the transformation matirx
	vec4 pos_temp = vec4(position,1.);
	vec3 transformedPoint;
	// Applying rotation, translation, projection to the point P
	pos_temp = mixedTransformation * pos_temp;
  // the x y z should be written in the form (x/z,y/z,z/z)
	// Z : It scales back the point to the basic vectors' plane.
	float z = pos_temp[3];
	transformedPoint[0] = pos_temp[0]/z;transformedPoint[1] = pos_temp[1]/z;transformedPoint[2] = pos_temp[2]/z;
	return transformedPoint;
#else
    return position;
#endif
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
    copyPolygon(projectedPolygon, polygon);
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
        }
    }
}

// Draws a polygon by projecting, clipping, ratserizing and interpolating it
void drawPolygon(
  vec2 point,
  Polygon clipWindow,
  Polygon oldPolygon,
  inout vec3 color,
  inout float depth)
{
    Polygon projectedPolygon;
    projectPolygon(projectedPolygon, oldPolygon);

    Polygon clippedPolygon;
    sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

    if (isPointInPolygon(point, clippedPolygon)) {

        Vertex interpolatedVertex =
          interpolateVertex(point, projectedPolygon);
#if defined(ZBUFFERING)
    // Put your code here
    // find the shallowest vertex and assign the color/position.
    if (depth > interpolatedVertex.position[2]){
        color = interpolatedVertex.color;depth = interpolatedVertex.position.z;}

#else
      // Put your code to handle z buffering here
      color = interpolatedVertex.color;
      depth = interpolatedVertex.position.z;
#endif
   }

   if (isPointOnPolygonVertex(point, clippedPolygon)) {
        color = vec3(1);
   }
}

// Main function calls

void drawScene(vec2 pixelCoord, inout vec3 color) {
    color = vec3(0.3, 0.3, 0.3);

  	// Convert from GL pixel coordinates 0..N-1 to our screen coordinates -1..1
    vec2 point = 2.0 * pixelCoord / vec2(viewport) - vec2(1.0);

    Polygon clipWindow;
    clipWindow.vertices[0].position = vec3(-0.65,  0.95, 1.0);
    clipWindow.vertices[1].position = vec3( 0.65,  0.75, 1.0);
    clipWindow.vertices[2].position = vec3( 0.75, -0.65, 1.0);
    clipWindow.vertices[3].position = vec3(-0.75, -0.85, 1.0);
    clipWindow.vertexCount = 4;

  	// Draw the area outside the clip region to be dark
    color = isPointInPolygon(point, clipWindow) ? vec3(0.5) : color;

    const int triangleCount = 2;
    Polygon triangles[triangleCount];

    triangles[0].vertices[0].position = vec3(-2, -2, 0.0);
    triangles[0].vertices[1].position = vec3(4, 0, 3.0);
    triangles[0].vertices[2].position = vec3(-1, 2, 0.0);
    triangles[0].vertices[0].color = vec3(1.0, 0.5, 0.2);
    triangles[0].vertices[1].color = vec3(0.8, 0.8, 0.8);
    triangles[0].vertices[2].color = vec3(0.2, 0.5, 1.0);
    triangles[0].vertexCount = 3;

    triangles[1].vertices[0].position = vec3(3.0, 2.0, -2.0);
  	triangles[1].vertices[2].position = vec3(0.0, -2.0, 3.0);
    triangles[1].vertices[1].position = vec3(-1.0, 2.0, 4.0);
    triangles[1].vertices[1].color = vec3(0.2, 1.0, 0.1);
    triangles[1].vertices[2].color = vec3(1.0, 1.0, 1.0);
    triangles[1].vertices[0].color = vec3(0.1, 0.2, 1.0);
    triangles[1].vertexCount = 3;

    float depth = 10000.0;
    // Project and draw all the triangles
    for (int i = 0; i < triangleCount; i++) {
        drawPolygon(point, clipWindow, triangles[i], color, depth);
    }
}

void main() {
    drawScene(gl_FragCoord.xy, gl_FragColor.rgb);
    gl_FragColor.a = 1.0;
}
