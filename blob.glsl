// Blobs
// Eric Galin - Sphere tracing, lighting
// Gerard Geer - Text rendering
// Romain Maurizot & Thomas Margier

#define _f float
const lowp _f CH_A    = _f(0x69f99), CH_B    = _f(0x79797), CH_C    = _f(0xe111e),
       	  	  CH_D    = _f(0x79997), CH_E    = _f(0xf171f), CH_F    = _f(0xf1711),
		  	  CH_G    = _f(0xe1d96), CH_H    = _f(0x99f99), CH_I    = _f(0xf444f),
		  	  CH_J    = _f(0x88996), CH_K    = _f(0x95159), CH_L    = _f(0x1111f),
		  	  CH_M    = _f(0x9f999), CH_N    = _f(0x9bd99), CH_O    = _f(0x69996),
		  	  CH_P    = _f(0x79971), CH_Q    = _f(0x69b5a), CH_R    = _f(0x79759),
		  	  CH_S    = _f(0xe1687), CH_T    = _f(0xf4444), CH_U    = _f(0x99996),
		  	  CH_V    = _f(0x999a4), CH_W    = _f(0x999f9), CH_X    = _f(0x99699),
    	  	  CH_Y    = _f(0x99e8e), CH_Z    = _f(0xf843f), CH_0    = _f(0x6bd96),
		  	  CH_1    = _f(0x46444), CH_2    = _f(0x6942f), CH_3    = _f(0x69496),
		  	  CH_4    = _f(0x99f88), CH_5    = _f(0xf1687), CH_6    = _f(0x61796),
		  	  CH_7    = _f(0xf8421), CH_8    = _f(0x69696), CH_9    = _f(0x69e84),
		  	  CH_APST = _f(0x66400), CH_PI   = _f(0x0faa9), CH_UNDS = _f(0x0000f),
		  	  CH_HYPH = _f(0x00600), CH_TILD = _f(0x0a500), CH_PLUS = _f(0x02720),
		  	  CH_EQUL = _f(0x0f0f0), CH_SLSH = _f(0x08421), CH_EXCL = _f(0x33303),
		  	  CH_QUES = _f(0x69404), CH_COMM = _f(0x00032), CH_FSTP = _f(0x00002),
    	  	  CH_QUOT = _f(0x55000), CH_BLNK = _f(0x00000), CH_COLN = _f(0x00202),
			  CH_LPAR = _f(0x42224), CH_RPAR = _f(0x24442);
const lowp vec2 MAP_SIZE = vec2(4,5);
#undef flt

const int Steps = 1000;
const float Epsilon = 0.01; // Marching epsilon
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

// Segment skeleton
// p : point
// a : segment start
// b : segment end
// e : energy associated to skeleton
// R : large radius
float segment(vec3 p, vec3 a,vec3 b, float e, float R)
{
	vec3 u = (b - a) / length(b-a); // unit vector of the segment
    float l = dot((p - a), u);
    
    float d;
    if (l < 0.0)
    {
        d = length(p - a);
    }
    else if (l > length(b - a))
    {
        d = length(p - b);
    }
    else
    {
        d = sqrt(pow(length(p - a), 2.0) - pow(l, 2.0));
    }
    
    return e * falloff(d, R);
}

float circle(vec3 p, vec3 c, float r, vec3 n, float e, float R)
{
    n /= length(n); // enlever pour replier l'espace temps
    
    float h = dot((p - c), n); 
    float l = sqrt(pow(length(p - c), 2.0) - h*h);
    float m = l - r;
    float d = sqrt(h*h + m*m);
    
    return e * falloff(d, R);
}

float disk(vec3 p, vec3 c, float r, vec3 n, float e, float R)
{
    n /= length(n); // enlever pour replier l'espace temps
    float h = dot((p - c), n); 
    float l = sqrt(pow(length(p - c), 2.0) - h*h);
    float m = l - r;
    float d;
    
    if (m < 0.)
        d = abs(h);
    else
        d = sqrt(h*h + m*m);
    
    return e * falloff(d, R);
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
float Blend(float a, vec3 colorA, float b, vec3 colorB, out vec3 colorOut)
{
    colorOut = (a*colorA + b*colorB)/(a+b);
    return a+b;
}

// Union
// a : field function of left sub-tree
// b : field function of right sub-tree
float Union(float a, vec3 colorA, float b, vec3 colorB, out vec3 colorOut)
{
    if (a>b)
        colorOut = colorA;
    else
        colorOut = colorB;
    return max(a,b);
}

// Intersection
// a : field function of left sub-tree
// b : field function of right sub-tree
float Intersection(float a, vec3 colorA, float b, vec3 colorB, out vec3 colorOut)
{
    if (a>b)
        colorOut = colorA;
    else
        colorOut = colorB;
    return min(a,b);
}

// Difference
// a : field function of left sub-tree
// b : field function of right sub-tree
float Difference(float a, vec3 colorA, float b, vec3 colorB, out vec3 colorOut)
{
    if (a>b)
        colorOut = colorA;
    else
        colorOut = colorB;
    return min(a, 2. * T - b);
}

// BlendN
// a : field function of left sub-tree
// b : field function of right sub-tree
float BlendN(float a, vec3 colorA, float b, vec3 colorB, float n, out vec3 colorOut)
{
    colorOut = (a*colorA + b*colorB)/(a+b);
    return pow((pow(a, n) + pow(b, n)), 1./n);
}

// Potential field of the object
// p : point
float object(vec3 p, out vec3 colorOut)
{
    p.z=-p.z;

/* lavaLamp */

  vec3 colorA = vec3(1., 1., 0.);
  vec3 colorB = vec3(1., 0., 0.);
  vec3 coordA = vec3(0., 0.3, -1.2);
  vec3 coordB = vec3(0., -0.5, -1.5);

    float h1 = sin(iGlobalTime/2. + 10.);
    vec3 pos1 = vec3(0.,0.,h1);
    float h2 = sin(iGlobalTime/2. - 10.);
    vec3 pos2 = vec3(0.,0.,h2);

  float pointA = point(p, coordA*pos1, 2.0, 1.*abs(h1)+1.);
  float pointB = point(p, -coordB*pos2, 2.0, 1.*abs(h2)+1.);

  float v = BlendN(pointA, colorA, pointB, colorB, 2., colorOut);

  /*dog baloon*/

//   vec3 colorDog = vec3(1., 0., 0.);
//   vec3 seg4a = vec3(0., 0., -0.6);
//   vec3 seg4b = vec3(0., 0., 0.6);
  
//   vec3 seg3a = vec3(0., 0.3, -1.2);
//   vec3 seg3b = vec3(0., 0.75, -1.5);
  
//   vec3 seg2a = vec3(0., 1.2, -1.5);
//   vec3 seg2b = vec3(0., 1.2, -2.5);
  
//   vec3 seg5a = vec3(0., 0., 1.2);
//   vec3 seg5b = vec3(0., 0.27, 1.5);
  
//   vec3 seg6a = vec3(0., seg5b.y, 1.5);
//   vec3 seg6b = vec3(0., seg5b.y + 0.5, 2.);

//   vec3 seg1Aa = vec3(-0.2, -0.4, -0.7);
//   vec3 seg1Ab = vec3(-0.4, -0.9, -0.8);
//   vec3 seg1Ba = vec3(0.2, -0.4, -0.7);
//   vec3 seg1Bb = vec3(0.4, -0.9, -0.8);
 
//   vec3 seg7Aa = vec3(-0.2, -0.4, 0.7);
//   vec3 seg7Ab = vec3(-0.4, -0.9, 0.8);
//   vec3 seg7Ba = vec3(0.2, -0.4, 0.7);
//   vec3 seg7Bb = vec3(0.4, -0.9, 0.8);
 
//   vec3 seg8Aa = vec3(-0.2, 1.7, -1.7);
//   vec3 seg8Ab = vec3(-0.4, 2.1, -1.6);
//   vec3 seg8Ba = vec3(0.2, 1.7, -1.7);
//   vec3 seg8Bb = vec3(0.4, 2.1, -1.6);

//   float seg4 = segment(p, seg4a, seg4b, 1.0, 0.8);
//   float seg3 = segment(p, seg3a, seg3b, 1.0, 0.8);
//   float seg2 = segment(p, seg2a, seg2b, 1.0, 0.8);
//   float seg5 = segment(p, seg5a, seg5b, 1.0, 0.8);
//   float seg6 = segment(p, seg6a, seg6b, 1.0, 0.4);
//   float seg1A = segment(p, seg1Aa, seg1Ab, 1.0, 0.8);
//   float seg1B = segment(p, seg1Ba, seg1Bb, 1.0, 0.8);
//   float seg7A = segment(p, seg7Aa, seg7Ab, 1.0, 0.8);
//   float seg7B = segment(p, seg7Ba, seg7Bb, 1.0, 0.8);
//   float seg8A = segment(p, seg8Aa, seg8Ab, 1.0, 0.8);
//   float seg8B = segment(p, seg8Ba, seg8Bb, 1.0, 0.8);

//   float v = Union(seg3, colorDog, seg4, colorDog,colorOut);
//   v = Union(v, colorDog, seg2, colorDog,colorOut);
//   v = Union(v, colorDog, seg5, colorDog,colorOut);
//   v = Union(v, colorDog, seg6, colorDog,colorOut);
//   v = Union(v, colorDog, seg1A, colorDog,colorOut);
//   v = Union(v, colorDog, seg1B, colorDog,colorOut);
//   v = Union(v, colorDog, seg7A, colorDog,colorOut);
//   v = Union(v, colorDog, seg7B, colorDog,colorOut);
//   v = Union(v, colorDog, seg8A, colorDog,colorOut);
//   v = Union(v, colorDog, seg8B, colorDog,colorOut);

  return v-T;
}

// Calculate object normal
// p : point
vec3 ObjectNormal(in vec3 p)
{
  vec3 color;
  float eps = 0.0001;
  vec3 n;
  float v = object(p, color);
  n.x = object( vec3(p.x+eps, p.y, p.z), color ) - v;
  n.y = object( vec3(p.x, p.y+eps, p.z), color ) - v;
  n.z = object( vec3(p.x, p.y, p.z+eps), color ) - v;
  return normalize(n);
}

// Trace ray using ray marching
// o : ray origin
// u : ray direction
// h : hit
// s : Number of steps
float Trace(vec3 o, vec3 u, out bool h,out int s, out vec3 objColor)
{
  h = false;

    // Don't start at the origin, instead move a little bit forward
    float t=rA;

  for(int i=0; i<Steps; i++)
  {
    s=i;
    vec3 p = o+t*u;
    float v = object(p, objColor);
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
float SphereTrace(vec3 o, vec3 u, out bool h,out int s, out vec3 objColor)
{
  h = false;
    // Don't start at the origin, instead move a little bit forward
    float t=rA;

  for(int i=0; i<Steps; i++)
  {
    s=i;
    vec3 p = o+t*u;
    float v = object(p, objColor);
    // Hit object
      if (v > 0.0)
      {
          s=i;
          h = true;
          break;
      }
      // Move along ray
      t += max(Epsilon,abs(v)/4.0);
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
vec3 Shade(vec3 p, vec3 n, vec3 objColor)
{
  // point light
  const vec3 lightPos = vec3(5.0, 5.0, 5.0);
  const vec3 lightColor = vec3(0.5, 0.5, 0.5);

  vec3 c = 0.25*background(n);
  vec3 l = normalize(lightPos - p);

  // Not even Phong shading, use weighted cosine instead for smooth transitions
  float diff = 0.5*(1.0+dot(n, l));

  c += diff * lightColor * objColor;

  return c;
}

// Shading with number of steps
vec3 ShadeSteps(int n)
{
   float t=float(n)/(float(Steps-1));
   return vec3(t,0.25+0.75*t,0.5-0.5*t);
}


/*
	returns the status of a bit in a bitmap. This is done value-wise, so
	the exact representation of the float doesn't really matter.
*/
float getBit( in float map, in float index )
{
    // Ooh -index takes out that divide :)
    return mod( floor( map*exp2(-index) ), 2.0 );
}

/*
	Trades a float for a character bitmap. Here's to eliminating
	branches with step()!
*/
float floatToChar( in float x )
{
    float res = CH_BLNK;
    res += (step(-.5,x)-step(0.5,x))*CH_0;
    res += (step(0.5,x)-step(1.5,x))*CH_1;
    res += (step(1.5,x)-step(2.5,x))*CH_2;
    res += (step(2.5,x)-step(3.5,x))*CH_3;
    res += (step(3.5,x)-step(4.5,x))*CH_4;
    res += (step(4.5,x)-step(5.5,x))*CH_5;
    res += (step(5.5,x)-step(6.5,x))*CH_6;
    res += (step(6.5,x)-step(7.5,x))*CH_7;
    res += (step(7.5,x)-step(8.5,x))*CH_8;
    res += (step(8.5,x)-step(9.5,x))*CH_9;
    return res;
}

/*
	Draws a character, given its encoded value, a position, size and
	current [0..1] uv coordinate.
*/
float drawChar( in float char, in vec2 pos, in vec2 size, in vec2 uv )
{
    // Subtract our position from the current uv so that we can
    // know if we're inside the bounding box or not.
    uv-=pos;
    
    // Divide the screen space by the size, so our bounding box is 1x1.
    uv /= size;    
    
    // Create a place to store the result.
    float res;
    
    // Branchless bounding box check.
    res = step(0.0,min(uv.x,uv.y)) - step(1.0,max(uv.x,uv.y));
    
    // Go ahead and multiply the UV by the bitmap size so we can work in
    // bitmap space coordinates.
    uv *= MAP_SIZE;
    
    // Get the appropriate bit and return it.
    res*=getBit( char, 4.0*floor(uv.y) + floor(uv.x) );
    return clamp(res,0.0,1.0);
}

/*
	Prints a float as an int. Be very careful about overflow.
	This as a side effect will modify the character position,
	so that multiple calls to this can be made without worrying
	much about kerning.
*/
float drawIntCarriage( in int val, inout vec2 pos, in vec2 size, in vec2 uv, in int places )
{
    // Create a place to store the current values.
    float res = 0.0,digit = 0.0;
    // Surely it won't be more than 10 chars long, will it?
    // (MAX_INT is 10 characters)
    for( int i = 0; i < 10; ++i )
    {
        // If we've run out of film, cut!
        if(val == 0 && i >= places) break;
        // The current lsd is the difference between the current
        // value and the value rounded down one place.
        digit = float( val-(val/10)*10 );
        // Draw the character. Since there are no overlaps, we don't
        // need max().
        res += drawChar(floatToChar(digit),pos,size,uv);
        // Move the carriage.
        pos.x -= size.x*1.2;
        // Truncate away this most recent digit.
        val /= 10;
    }
    return res;
}

/*
	Draws an integer to the screen. No side-effects, but be ever vigilant
	so that your cup not overfloweth.
*/
float drawInt( in int val, in vec2 pos, in vec2 size, in vec2 uv )
{
    vec2 p = vec2(pos);
    float s = sign(float(val));
    val *= int(s);
    
    float c = drawIntCarriage(val,p,size,uv,1);
    return c + drawChar(CH_HYPH,p,size,uv)*max(0.0, -s);
}

/*
	Prints a fixed point fractional value. Be even more careful about overflowing.
*/
float drawFixed( in float val, in int places, in vec2 pos, in vec2 size, in vec2 uv )
{
    // modf() sure would be nice right about now.
    vec2 p = vec2(pos);
    float res = 0.0;
    
    // Draw the floating point part.
    res = drawIntCarriage( int( fract(val)*pow(10.0,float(places)) ), p, size, uv, places );
    // The decimal is tiny, so we back things up a bit before drawing it.
    p.x += size.x*.4;
    res = max(res, drawChar(CH_FSTP,p,size,uv)); p.x-=size.x*1.2;
    // And after as well.
    p.x += size.x *.1;
    // Draw the integer part.
    res = max(res, drawIntCarriage(int(floor(val)),p,size,uv,1));
	return res;
}

float text( in vec2 uv )
{
    // Set a general character size...
    vec2 charSize = vec2(.03, .0375);
    // and a starting position.
    vec2 charPos = vec2(-1, -1);
    // Draw some text!
    float chr = 0.0;
    // Bitmap text rendering!
    chr += drawChar( CH_G, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_UNDS, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_I, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_M, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_COLN, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += .04;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += .04;
    chr += drawFixed( iGlobalTime, 2, charPos, charSize, uv); charPos.x += .04;

    return chr;
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

  float txt = 0.;//text(pixel);
  vec3 rgb;
  vec3 objColor;
  if (txt == 0.0)
  {
    float t = SphereTrace(ro, rd, hit, s, objColor);
    vec3 pos=ro+t*rd;
    // Shade background
    rgb = background(rd);

    if (hit)
    {
      // Compute normal
      vec3 n = ObjectNormal(pos);

      // Shade object with light
      rgb = Shade(pos, n, objColor);
    }
  }
  else
  {
    rgb = vec3(txt, txt, txt);
  }


  // Uncomment this line to shade image with false colors representing the number of steps
  //rgb = ShadeSteps(s);
  gl_FragColor=vec4(rgb, 1.0);
}

