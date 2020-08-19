#define SOLUTION_DE_CASTELJAU
#define SOLUTION_OUTLINE
#define SOLUTION_OUTLINE_G1
#define SOLUTION_SPOTS

uniform float element_idx;
uniform float left;

attribute float coord;
varying vec3 fragColor;
varying float fragCoord;
uniform float time;
vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
mat4 computeProjectionMatrix(float fov, float aspect, float zNear, float zFar) {
    float deltaZ = zFar - zNear;
    float cotangent = cos(fov * 0.5) / sin(fov * 0.5);

    mat4 projectionMatrix;
    projectionMatrix[0] = vec4(cotangent / aspect, 0.0, 0.0, 0.0);
    projectionMatrix[1] = vec4(0.0, cotangent, 0.0, 0.0);
    projectionMatrix[2] = vec4(0.0, 0.0, -(zFar + zNear) / deltaZ, -1.0);
    projectionMatrix[3] = vec4(0.0, 0.0, -2.0 * zNear * zFar / deltaZ, 0.0);

    return projectionMatrix;
}
// Used to generate a simple "look-at" camera.
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    // The VPN is pointing away from the TP. Can also be modeled the other way around.
    vec3 VPN = VRP - TP;
    // Generate the camera axes.
    vec3 n = normalize(VPN);
    vec3 u = normalize(cross(VUV, n));
    vec3 v = normalize(cross(n, u));

    mat4 modelViewMatrix;
	modelViewMatrix[0] = left*vec4(u[0], v[0], n[0], 0);
    modelViewMatrix[1] = vec4(u[1], v[1], n[1], 0);
    modelViewMatrix[2] = vec4(u[2], v[2], n[2], 0);
    modelViewMatrix[3] = vec4(- dot(VRP, u), - dot(VRP, v), - dot(VRP, n), 1);

    return modelViewMatrix;
}
vec3 linearBezier1D(vec3 y0, vec3 y1, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  // From slides p12 we can get the equation for linearBezier directly
	/*vec3 Q[2];
	Q[0] = y0;Q[1] = y1;
    for(int k = 0; k <=2; k++) {
      for(int i=2;i>=0;i--){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  vec3 p_t;
  vec3 p_1;
  vec3 p_2;
  p_t = y0*(1.-coord)+coord*y1;
  return p_t;
  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 quadraticBezier1D(vec3 y0, vec3 y1, vec3 y2, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  // After we have the function to compute segment, we can apply it to solve quadratic.
	/*vec3 Q[3];
	Q[0] = y0;Q[1] = y1;Q[2] = y2;
    for(int k = 0; k <=3; k++) {
      for(int i=3;i>=0;i--){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  vec3 p_t;
  vec3 p_1;
  vec3 p_2;
  // The idea is we have three points, two points form a line segment, we can first get two points from y0 to y2
  // Then using that two points to find the p_t.
  p_1 = linearBezier1D( y0,  y1,  coord);
  p_2 = linearBezier1D( y1,  y2,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;
  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 cubicBezier1D(vec3 y0, vec3 y1, vec3 y2, vec3 y3, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  // The commented code is my first attempt for this question, wonder whether this is recurrsion or not.
  // The commented code's equation comes from the following url:
  //http://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/de-casteljau.html
	/*vec3 Q[4];
	Q[0] = y0;Q[1] = y1;Q[2] = y2;Q[3] = y3;
    for(int k = 0; k <=4; k++) {
      for(int i=4;i>=0;i--){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  vec3 p_t;
  vec3 p_1;
  vec3 p_2;
  // The idea is the same for the rest part of this section.
  // Using the previous function do get two points, and using linearBezier to find the final point on the curve.
  p_1 = quadraticBezier1D( y0,  y1,  y2,  coord);
  p_2 = quadraticBezier1D( y1,  y2,  y3,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;

  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 quarticBezier1D(vec3 y0, vec3 y1, vec3 y2, vec3 y3, vec3 y4, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  /*vec3 Q[5];
  Q[0] = y0;Q[1] = y1;Q[2] = y2;Q[3] = y3;Q[4] = y4;
  for(int k = 0; k <=5; k++) {
    for(int i=5;i>=0;i++){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  /*vec3 p_t,p_1,p_2;
  p_1 = cubicBezier1D( y0,  y1,  y2,  y3,  coord);
  p_2 = cubicBezier1D( y1,  y2,  y3,  y4,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;*/
  vec3 p_t,p_1,p_2;
  p_1 = cubicBezier1D( y0,  y1,  y2,  y3,  coord);
  p_2 = cubicBezier1D( y1,  y2,  y3,  y4,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;



  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 quinticBezier1D(vec3 y0, vec3 y1, vec3 y2, vec3 y3, vec3 y4, vec3 y5, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
 /* vec3 Q[6];
  Q[0] = y0;Q[1] = y1;Q[2] = y2;Q[3] = y3;Q[4] = y4,Q[5]=y5;
  for(int k = 0; k <=6; k++) {
    for(int i=6;i>=0;i++){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/

  vec3 p_t,p_1,p_2;
  p_1 = quarticBezier1D( y0,  y1,  y2,  y3,  y4,  coord);
  p_2 = quarticBezier1D( y1,  y2,  y3,  y4,  y5,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;
  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 sexticBezier1D(vec3 y0, vec3 y1, vec3 y2, vec3 y3, vec3 y4, vec3 y5, vec3 y6, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  /*vec3 Q[7];
  Q[0] = y0;Q[1] = y1;Q[2] = y2;Q[3] = y3;Q[4] = y4,Q[5]=y5,Q[6] = y6;
  for(int k = 0; k <=7; k++) {
    for(int i=7;i>=0;i++){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  vec3 p_t,p_1,p_2;
  p_1 = quinticBezier1D( y0,  y1,  y2,  y3,  y4,  y5,  coord);
  p_2 = quinticBezier1D( y1,  y2,  y3,  y4,  y5,  y6,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;

  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}
vec3 septicBezier1D(vec3 y0, vec3 y1, vec3 y2, vec3 y3, vec3 y4, vec3 y5, vec3 y6, vec3 y7, float coord) {
  #ifdef SOLUTION_DE_CASTELJAU
  // put your code here
  /*vec3 Q[8];
  Q[0] = y0;Q[1] = y1;Q[2] = y2;Q[3] = y3;Q[4] = y4,Q[5]=y5,Q[6] = y6,Q[7]=y7;
  for(int k = 0; k <=8; k++) {
    for(int i=8;i>=0;i++){
      Q[i] = (1. - coord)*Q[i] + coord * Q[i+1];
    }
  }
  return Q[0];*/
  vec3 p_t,p_1,p_2;
  p_1 = sexticBezier1D( y0,  y1,  y2,  y3,  y4,  y5,  y6,  coord);
  p_2 = sexticBezier1D( y1,  y2,  y3,  y4,  y5,  y6,  y7,  coord);
  p_t = linearBezier1D( p_1,  p_2,  coord);
  return p_t;

  #else
  return vec3(0.0, 0.0, 0.0);
  #endif
}

vec3 createOutline(const float coord) {
  vec3 cp[25];
  cp[0] = vec3(0.0,0.2861,0.0);
  cp[1] = vec3(-0.024843,0.286072,0.0);
  cp[2] = vec3(-0.078544,0.097519,0.0);
  cp[3] = vec3(-0.104240,0.068146,0.0);
  cp[4] = vec3(-0.163855,0.000000,0.0);
  cp[5] = vec3(-0.237873,0.019290,0.0);
  cp[6] = vec3(-0.308851,0.070699,0.0);
  cp[7] = vec3(-0.373387,0.117443,0.0);
  cp[8] = vec3(-0.450749,0.220935,0.0);
  cp[9] = vec3(-0.407830,0.303048,0.0);
  cp[10] = vec3(-0.373392,0.368937,0.0);
  cp[11] = vec3(-0.266283,0.402792,0.0);
  cp[12] = vec3(-0.201575,0.419100,0.0);
  cp[13] = vec3(-0.327433,0.408588,0.0);
  cp[14] = vec3(-0.416736,0.403000,0.0);
  cp[15] = vec3(-0.447241,0.537507,0.0);
  cp[16] = vec3(-0.458375,0.586597,0.0);
  cp[17] = vec3(-0.464963,0.630227,0.0);
  cp[18] = vec3(-0.483701,0.677174,0.0);
  cp[19] = vec3(-0.495162,0.705888,0.0);
  cp[20] = vec3(-0.544062,0.777377,0.0);
  cp[21] = vec3(-0.527718,0.811812,0.0);
  cp[22] = vec3(-0.438397,1.000000,0.0);
  cp[23] = vec3(-0.060168,0.520334,0.0);
  cp[24] = vec3(-0.022714,0.452637,0.0);


  int seg_count = 8;
  int seg_ind = int(min(float(seg_count), floor(coord*float(seg_count))));
  float local_coord = coord * float(seg_count) - float(seg_ind);

  #ifdef SOLUTION_OUTLINE_G1
  // put your code here
  // What we need to do here is to put group3(seg_ind ==3) and group4 together(seg_ind ==4)
  // which means using sexticBezier1D to compute cp[9] to cp[15]
  if (seg_ind == 0)    return cubicBezier1D(cp[0], cp[1], cp[2], cp[3], local_coord);
  else if (seg_ind == 1)    return cubicBezier1D(cp[3], cp[4], cp[5], cp[6], local_coord);
  else if (seg_ind == 2)    return cubicBezier1D(cp[6], cp[7], cp[8], cp[9], local_coord);
  else if (seg_ind == 3)    return sexticBezier1D(cp[9], cp[10], cp[11], cp[12], cp[13], cp[14], cp[15], local_coord/2.);
  //we need to scale the local_coord first since its now for 2 groups' point.
  // For the group seg_ind == 4, we can get its local_coord by solving the followings:
  //(coord * seg_count_4 - seg_count_4)/2 + c = (coord * seg_count_3 - seg_count_3)/2
  // which gives 0.5 but can be written in a more general form: (float(seg_ind)-(float(seg_ind)-1.))/2.
  else if (seg_ind == 4)    return sexticBezier1D(cp[9], cp[10], cp[11], cp[12], cp[13], cp[14], cp[15], local_coord/2. + (float(seg_ind)-(float(seg_ind)-1.))/2.);
  else if (seg_ind == 5)    return cubicBezier1D(cp[15], cp[16], cp[17], cp[18], local_coord);
  else if (seg_ind == 6)    return cubicBezier1D(cp[18], cp[19], cp[20], cp[21], local_coord);
  else if (seg_ind == 7)    return cubicBezier1D(cp[21], cp[22], cp[23], cp[24], local_coord);
  #else
  #ifdef SOLUTION_OUTLINE
  // put your code here
  //50 lines below just gives a perfect example of how to group points together, so I just move then here and extends the points to 24
  // 24 control points, 4 points per group so put them into 6 + 1 group, we have extra 1 group because we need to make sure each curve is joint(head to tail)
  if (seg_ind == 0)    return cubicBezier1D(cp[0], cp[1], cp[2], cp[3], local_coord);
  else if (seg_ind == 1)    return cubicBezier1D(cp[3], cp[4], cp[5], cp[6], local_coord);
  else if (seg_ind == 2)    return cubicBezier1D(cp[6], cp[7], cp[8], cp[9], local_coord);
  else if (seg_ind == 3)    return cubicBezier1D(cp[9], cp[10], cp[11], cp[12], local_coord);
  else if (seg_ind == 4)    return cubicBezier1D(cp[12], cp[13], cp[14], cp[15], local_coord);
  else if (seg_ind == 5)    return cubicBezier1D(cp[15], cp[16], cp[17], cp[18], local_coord);
  else if (seg_ind == 6)    return cubicBezier1D(cp[18], cp[19], cp[20], cp[21], local_coord);

  #else
  return vec3(0.0, 0.0, 0.0);
  #endif // SOLUTION_OUTLINE
  #endif // SOLUTION_OUTLINE_G1
}

vec3 createSpot(const float coord) {
  vec3 cp[12];
  cp[0] = vec3(-0.246849,0.653297,0.0);
  cp[1] = vec3(-0.192073,0.601874,0.0);
  cp[2] = vec3(-0.166362,0.541509,0.0);
  cp[3] = vec3(-0.183130,0.524740,0.0);
  cp[9] = vec3(-0.394410,0.712545,0.0);

  cp[6] = vec3(-0.35, 0.55,0.0);

  cp[10] = vec3(-0.378760,0.727077,0.0);
  cp[11] = vec3(-0.301626,0.704720,0.0);

  #ifdef SOLUTION_SPOTS
  // put your code here
  // According to slides p18 we have the tangent of curve a end points(t=0 and t = 1):
  // For 3rd degree Bezier curve:
  // P'(t_0) = 3(p_1-p_0)
  // P'(t_1) = 3(p_3-p_2)
  // The coursework expect us to compute point 4 to point 8 while maintaining the tangent between point 6 and point 0
  // So we have the followings:
  // P'(p_3): 3(p_4-p_3) = - P'(p_9): - 3(p_10 - p_9) at t_0 for p_3 to p_6 and p_9 to p_0
  // p_4 = -p_10 + p_9 + p_3
  cp[4] = cp[3] + cp[9] - cp[10];

  // P'(p_6): 3(p_6-p_5) = - P'(p_0): - 3(p_0 - p_11) at t_1 for p_3 to p_6 and p_9 to p_0
  // p_5 = p_6 + p_0 - p_11
  cp[5] = cp[0] + cp[6] - cp[11];

  // P'(p_6): 3(p_7-p_6) = - P'(p_0): - 3(p_1 - p_0) at t_0 for p_6 to p_9 and p_0 to p_3
  // p_7 = -p_1 + p_0 + p_6
  cp[7] = cp[0] + cp[6] - cp[1];

  // P'(p_9): 3(p_9-p_8) = - P'(p_3): - 3(p_3 - p_2) at t_1 for p_6 to p_9 and p_0 to p_3
  // p_8 = -p_9 + p_3 - p_2
  cp[8] = cp[3] + cp[9] - cp[2];

  #else
  cp[4] = cp[3];
  cp[5] = cp[6];
  cp[7] = cp[6];
  cp[8] = cp[9];
  #endif


  int seg_count = 4;
  int seg_ind = int(min(float(seg_count), floor(coord*float(seg_count))));
  float local_coord = coord * float(seg_count) - float(seg_ind);

  if (seg_ind == 0)    return cubicBezier1D(cp[0], cp[1], cp[2], cp[3], local_coord);
  else if (seg_ind == 1)    return cubicBezier1D(cp[3], cp[4], cp[5], cp[6], local_coord);
  else if (seg_ind == 2)    return cubicBezier1D(cp[6], cp[7], cp[8], cp[9], local_coord);
  else if (seg_ind == 3)    return cubicBezier1D(cp[9], cp[10], cp[11], cp[0], local_coord);
}

void main(void) {
  vec3 TP = vec3(0, 0.5, 0);
  vec3 VRP = 1.8 * vec3(sin(time), 0.5, cos(time));
  vec3 VUV = vec3(0, 1, 0);
  mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  mat4 projectionMatrix = computeProjectionMatrix(0.6, 2.0, 0.5, 100.0);

  if (element_idx < 0.5)
	gl_Position = projectionMatrix * viewMatrix * vec4(createOutline(coord), 1.0);
  else
	gl_Position = projectionMatrix * viewMatrix * vec4(createSpot(coord), 1.0);

  fragColor = hsv2rgb(vec3(2.0 * coord, 1, 1));
  fragCoord = coord;
}
