// Blobs
// Eric Galin

const int Steps = 1000;
const float Epsilon = 0.05; // Marching epsilon
const float T=0.5;

const float rA=10.0; // Maximum ray marching or sphere tracing distance from origin
const float rB=40.0; // Minimum

// Transforms
vec3 rotateX(vec3 p, float a)
{
  float sa = sin(a);
  float ca = cos(a);
  return vec3(p.x, ca*p.y - sa*p.z, sa*p.y + ca*p.z);
}

vec3 rotateY(vec3 p, float a)
{
  float sa = sin(a);
  float ca = cos(a);
  return vec3(ca*p.x + sa*p.z, p.y, -sa*p.x + ca*p.z);
}

vec3 rotateZ(vec3 p, float a)
{
  float sa = sin(a);
  float ca = cos(a);
  return vec3(ca*p.x + sa*p.y, -sa*p.x + ca*p.y, p.z);
}


// Smooth falloff function
// r : small radius
// R : Large radius
float falloff( float r, float R )
{
  float x = clamp(r/R,0.0,1.0);
  float y = (1.0-x*x);
  return y*y*y;
}

// Primitive functions

// Point skeleton
// p : point
// c : center of skeleton
// e : energy associated to skeleton
// R : large radius
float point(vec3 p, vec3 c, float e, float R)
{
  return e*falloff(length(p-c),R);
}

// Bubble skeleton
// p : point
// c : center of skeleton
// e : energy associated to skeleton
// R : large radius
float bubble(vec3 p, vec3 c, float radius , float e, float R)
{
  float d = abs(length(p-c)-radius);
  return e*falloff(d, R);
}

// Blending
// a : field function of left sub-tree
// b : field function of right sub-tree
float Blend(float a,float b)
{
    return a+b;
}

// Union
// a : field function of left sub-tree
// b : field function of right sub-tree
float Union(float a,float b)
{
    return max(a,b);
}

// Intersection
// a : field function of left sub-tree
// b : field function of right sub-tree
float Intersection(float a,float b)
{
    return min(a,b);
}

// Difference
// a : field function of left sub-tree
// b : field function of right sub-tree
float Difference(float a,float b)
{
    return min(a, 2. * T - b);
}

// BlendN
// a : field function of left sub-tree
// b : field function of right sub-tree
float BlendN(float a,float b,float n)
{
    return pow((pow(a, n) + pow(b, n)), 1./n);
}

// Potential field of the object
// p : point
float object(vec3 p)
{
  p.z=-p.z;
  float v = Difference(bubble(p,vec3( 0.0, 1.0, 1.0),2.0 ,1.0,1.0),
                  point(p,vec3( 2.0, 0.0,-1.5),1.0,4.5));

  // v=Blend(v,point(p,vec3(-3.0, 2.0,-3.0),1.0,4.5));
  // v=Union(v,point(p,vec3(-1.0, -1.0, 0.0),1.0,4.5));
  return v-T;
}

// Calculate object normal
// p : point
vec3 ObjectNormal(in vec3 p )
{
  float eps = 0.0001;
  vec3 n;
  float v = object(p);
  n.x = object( vec3(p.x+eps, p.y, p.z) ) - v;
  n.y = object( vec3(p.x, p.y+eps, p.z) ) - v;
  n.z = object( vec3(p.x, p.y, p.z+eps) ) - v;
  return normalize(n);
}

// Trace ray using ray marching
// o : ray origin
// u : ray direction
// h : hit
// s : Number of steps
float Trace(vec3 o, vec3 u, out bool h,out int s)
{
  h = false;

    // Don't start at the origin, instead move a little bit forward
    float t=rA;

  for(int i=0; i<Steps; i++)
  {
    s=i;
    vec3 p = o+t*u;
    float v = object(p);
    // Hit object
      if (v > 0.0)
      {
          s=i;
          h = true;
          break;
      }
      // Move along ray
      t += Epsilon;
      // Escape marched far away
      if (t>rB)
      {
          break;
      }
  }
  return t;
}

// Trace ray using ray marching
// o : ray origin
// u : ray direction
// h : hit
// s : Number of steps
float SphereTrace(vec3 o, vec3 u, out bool h,out int s)
{
  h = false;

    // Don't start at the origin, instead move a little bit forward
    float t=rA;

  for(int i=0; i<Steps; i++)
  {
    s=i;
    vec3 p = o+t*u;
    float v = object(p);
    // Hit object
      if (v > 0.0)
      {
          s=i;
          h = true;
          break;
      }
      // Move along ray
      t += max(Epsilon,abs(v)/10.0);
      // Escape marched far away
      if (t>rB)
      {
          break;
      }
  }
  return t;
}


// Background color
vec3 background(vec3 rd)
{
  return mix(vec3(0.4, 0.3, 0.0), vec3(0.7, 0.8, 1.0), rd.y*0.5+0.5);
}

// Shading and lighting
// p : point,
// n : normal at point
vec3 Shade(vec3 p, vec3 n)
{
  // point light
  const vec3 lightPos = vec3(5.0, 5.0, 5.0);
  const vec3 lightColor = vec3(0.5, 0.5, 0.5);

  vec3 c = 0.25*background(n);
  vec3 l = normalize(lightPos - p);

  // Not even Phong shading, use weighted cosine instead for smooth transitions
  float diff = 0.5*(1.0+dot(n, l));

  c += diff*lightColor;

  return c;
}

// Shading with number of steps
vec3 ShadeSteps(int n)
{
   float t=float(n)/(float(Steps-1));
   return vec3(t,0.25+0.75*t,0.5-0.5*t);
}


void main()
{
  vec2 pixel = (gl_FragCoord.xy / iResolution.xy)*2.0-1.0;

  // compute ray origin and direction
  float asp = iResolution.x / iResolution.y;
  vec3 rd = normalize(vec3(asp*pixel.x, pixel.y, -4.0));
  vec3 ro = vec3(0.0, 0.0, 20.0);

  // vec2 mouse = iMouse.xy / iResolution.xy;
  float a=iGlobalTime*0.25;
  ro = rotateY(ro, a);
  rd = rotateY(rd, a);

  // Trace ray
  bool hit;

  // Number of steps
  int s;

  float t = SphereTrace(ro, rd, hit,s);
  vec3 pos=ro+t*rd;
  // Shade background
  vec3 rgb = background(rd);

  if (hit)
  {
    // Compute normal
    vec3 n = ObjectNormal(pos);

    // Shade object with light
    rgb = Shade(pos, n);
  }

  // Uncomment this line to shade image with false colors representing the number of steps
  //rgb = ShadeSteps(s);

  gl_FragColor=vec4(rgb, 1.0);
}

