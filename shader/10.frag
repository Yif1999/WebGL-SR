precision highp float;
precision highp int;
#define HIGH_PRECISION
#define SHADER_NAME ShaderMaterial
#define USE_LIGHTMAP 1
#define USE_SHADOW_RAMP 1
#define USE_MATMAP 1
#define MAT_CHANNEL 0
#define USE_METALMAP 1
#define USE_FACEMAP 0
#define USE_UV2 1
#define USE_FADE 0
#define USE_DEPTH 0
#define USE_ALPHA 1
#define USE_TEXTURE 1
#define USE_MORPH 4
#define GAMMA_FACTOR 2
#define FLIP_SIDED
uniform mat4 viewMatrix;
uniform vec3 cameraPosition;
uniform bool isOrthographic;

vec4 LinearToLinear( in vec4 value ) {
	return value;
}
vec4 GammaToLinear( in vec4 value, in float gammaFactor ) {
	return vec4( pow( value.rgb, vec3( gammaFactor ) ), value.a );
}
vec4 LinearToGamma( in vec4 value, in float gammaFactor ) {
	return vec4( pow( value.rgb, vec3( 1.0 / gammaFactor ) ), value.a );
}
vec4 sRGBToLinear( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 LinearTosRGB( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}
vec4 RGBEToLinear( in vec4 value ) {
	return vec4( value.rgb * exp2( value.a * 255.0 - 128.0 ), 1.0 );
}
vec4 LinearToRGBE( in vec4 value ) {
	float maxComponent = max( max( value.r, value.g ), value.b );
	float fExp = clamp( ceil( log2( maxComponent ) ), -128.0, 127.0 );
	return vec4( value.rgb / exp2( fExp ), ( fExp + 128.0 ) / 255.0 );
}
vec4 RGBMToLinear( in vec4 value, in float maxRange ) {
	return vec4( value.rgb * value.a * maxRange, 1.0 );
}
vec4 LinearToRGBM( in vec4 value, in float maxRange ) {
	float maxRGB = max( value.r, max( value.g, value.b ) );
	float M = clamp( maxRGB / maxRange, 0.0, 1.0 );
	M = ceil( M * 255.0 ) / 255.0;
	return vec4( value.rgb / ( M * maxRange ), M );
}
vec4 RGBDToLinear( in vec4 value, in float maxRange ) {
	return vec4( value.rgb * ( ( maxRange / 255.0 ) / value.a ), 1.0 );
}
vec4 LinearToRGBD( in vec4 value, in float maxRange ) {
	float maxRGB = max( value.r, max( value.g, value.b ) );
	float D = max( maxRange / maxRGB, 1.0 );
	D = clamp( floor( D ) / 255.0, 0.0, 1.0 );
	return vec4( value.rgb * ( D * ( 255.0 / maxRange ) ), D );
}
const mat3 cLogLuvM = mat3( 0.2209, 0.3390, 0.4184, 0.1138, 0.6780, 0.7319, 0.0102, 0.1130, 0.2969 );
vec4 LinearToLogLuv( in vec4 value ) {
	vec3 Xp_Y_XYZp = cLogLuvM * value.rgb;
	Xp_Y_XYZp = max( Xp_Y_XYZp, vec3( 1e-6, 1e-6, 1e-6 ) );
	vec4 vResult;
	vResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;
	float Le = 2.0 * log2(Xp_Y_XYZp.y) + 127.0;
	vResult.w = fract( Le );
	vResult.z = ( Le - ( floor( vResult.w * 255.0 ) ) / 255.0 ) / 255.0;
	return vResult;
}
const mat3 cLogLuvInverseM = mat3( 6.0014, -2.7008, -1.7996, -1.3320, 3.1029, -5.7721, 0.3008, -1.0882, 5.6268 );
vec4 LogLuvToLinear( in vec4 value ) {
	float Le = value.z * 255.0 + value.w;
	vec3 Xp_Y_XYZp;
	Xp_Y_XYZp.y = exp2( ( Le - 127.0 ) / 2.0 );
	Xp_Y_XYZp.z = Xp_Y_XYZp.y / value.y;
	Xp_Y_XYZp.x = value.x * Xp_Y_XYZp.z;
	vec3 vRGB = cLogLuvInverseM * Xp_Y_XYZp.rgb;
	return vec4( max( vRGB, 0.0 ), 1.0 );
}
vec4 linearToOutputTexel( vec4 value ) { return LinearToLinear( value ); }

#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate(a) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement(a) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float max3( vec3 v ) { return max( max( v.x, v.y ), v.z ); }
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
struct GeometricContext {
	vec3 position;
	vec3 normal;
	vec3 viewDir;
#ifdef CLEARCOAT
	vec3 clearcoatNormal;
#endif
};
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {
	float distance = dot( planeNormal, point - pointOnPlane );
	return - distance * planeNormal + point;
}
float sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {
	return sign( dot( point - pointOnPlane, planeNormal ) );
}
vec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {
	return lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
float linearToRelativeLuminance( const in vec3 color ) {
	vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );
	return dot( weights, color.rgb );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
const float x_ = 1.0/255.0;
const float y_ = 1./65025.0;
const float z_ = 1./160581375.0;

float DecodeDepth( vec4 val ) {
	return mix(1000.,1000.*dot( val, vec4(1.0, x_, y_, z_ )),val.a);
}

vec4 EncodeDepth( float v ) {
		vec4 enc = vec4(1.0, 255.0, 65025.0, 160581375.0) * v /1000.;
		enc = fract(enc);
		enc -= enc.yzww * vec4(x_,x_,x_,0.0);
		return vec4(enc.xyz,1.);
}


uniform sampler2D diffuse;
uniform sampler2D outlineMap;
uniform vec2 resolution;

uniform float rimIntensity;

#if USE_LIGHTMAP > 0
uniform sampler2D lightMap;
#endif

#if USE_ALPHA
uniform sampler2D alphaMap;
#endif

#if USE_MATMAP == 1
uniform sampler2D matMap;
#endif

uniform float metallness1;
uniform float metallness2;
uniform float metallness3;
uniform float metallness4;
uniform float metallness5;
uniform float metallness6;
uniform float metallness7;
uniform float metallness8;

const float lightArea = 0.5;

#if USE_SHADOW_RAMP == 1
uniform sampler2D shadowRamp;

const float shadowRampWidth = .5;
const float shadowTransitionRange = 0.001;
#endif

#if USE_METALMAP == 1
uniform sampler2D metalMap;
uniform vec3 metalLightColor;
uniform vec3 metalDarkColor;
#endif

uniform vec3 lightDir;

uniform vec4 shadowColor1;
uniform vec4 shadowColor2;
uniform vec4 shadowColor3;
uniform vec4 shadowColor4;
uniform vec4 shadowColor5;
uniform vec4 shadowColor6;
uniform vec4 shadowColor7;
uniform vec4 shadowColor8;

uniform vec4 brightness;

uniform float exposure;
 

varying vec2 vUv;
varying vec4 vNormalAndDiff;
varying vec3 vViewDir;
varying vec4 vViewPosition;
varying vec3 vColor;
varying float vCameraAngle;

#if USE_UV2 == 1
  varying vec2 vUv2;
#endif

#if USE_DEPTH > 0
  uniform sampler2D faceDepthMap;
#endif

float specularFactor(vec3 N, vec3 H, float shininess){
	return pow(max(dot(N, H), 0.001), shininess);
}

vec4 color3 = vec4(1., 1., 1., 1.);

void fetchMaterialInfo(float a, out vec4 shadowMultiColor, out float metalness){
  if (a > 0.8){
    shadowMultiColor = shadowColor1;
    metalness = metallness1;
    color3 = vec4(1., 0., 0., 1.);
  }else if (a > .7){
    shadowMultiColor = shadowColor2;
    metalness = metallness2;
    color3 = vec4(0., 0., 1., 1.);
  }else if (a > .6){
    shadowMultiColor = shadowColor3;
    metalness = metallness3;
    color3 = vec4(1., 1., 0., 1.);
  }else if (a > .45){
    shadowMultiColor = shadowColor4;
    metalness = metallness4;
  }else if (a > .3){
    metalness = metallness5;
    shadowMultiColor = shadowColor5;
  }else if (a > .2){
    metalness = metallness6;
    shadowMultiColor = shadowColor6;
  }else if (a > .1){
    metalness = metallness7;
    shadowMultiColor = shadowColor7;
  }else{
    metalness = metallness8;
    shadowMultiColor = shadowColor8;
  }
}


#if USE_SHADOW_RAMP == 1
vec3 sampleShadowRamp(float id, float shadowRampUV){
  return texture2D(shadowRamp, vec2(shadowRampUV, id+.05)).rgb;  
}
#endif

void calcToonDiffuse(float factor, 
    float vertexAO, 
    float lightmapAO, 
		out float shadowStrength, 
    out float shadowRampUV){

    float D = lightmapAO*vertexAO;

    float threshold = factor;

    if (D < 0.05)
      threshold = 0.;
    else if (D > 0.95)
      threshold = 1.;
    else
      threshold = (factor + D) * 0.5;

    #if USE_SHADOW_RAMP == 1
        shadowStrength = (lightArea - threshold) / lightArea;
        shadowRampUV =  1. - min(shadowStrength/shadowRampWidth, 1.);          
        //shadowStrength = (lightArea - threshold) / shadowTransitionRange;
        shadowStrength = saturate(shadowStrength); // 阴影强度修正
    #endif
}


vec3 fun1(float param1, float param2, float param3){
    vec2 param14 = vec2(param1, param2);
    vec3 param18 = vec3(param14, param3);
    return param18;
}

vec3 fun3(vec3 color){
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(color.bg, K.wz), vec4(color.gb, K.xy), step(color.b, color.g));
    vec4 q = mix(vec4(p.xyw, color.r), vec4(color.r, p.yzx), step(p.x, color.r));	
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 fun4(vec3 hsv){
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

vec3 fun2(vec3 param4){
    vec3 param9 = fun3(param4);
    
    float param13 = pow(param9.g, .8);
    float param17 = pow(param9.b, 1.1);
    vec3 param19 = fun1(param9.r, param13, param17);
    vec3 param20 = fun4(param19);
    return param20;
}

void main(){
  #if USE_UV2 == 1
    vec2 uv = vNormalAndDiff.z > 0. ? vUv : vUv2;
  #else
    vec2 uv = vUv;
  #endif

  vec3 N = vNormalAndDiff.xyz;

  vec3 H = normalize(lightDir+vViewDir); 
  vec3 diffuseColor = texture2D(diffuse, uv).rgb;


  float dotNL = dot(vViewDir,N);
  
  #if USE_LIGHTMAP > 0
	  vec4 lightTexColor = texture2D(lightMap, uv);
  #else
    vec4 lightTexColor =vec4(0.);
  #endif

  // lightTexColor.r = 1. - lightTexColor.r;
  // lightTexColor.g = 0.;

  #if USE_MATMAP == 1
	  float matTexColor = texture2D(matMap, uv)[MAT_CHANNEL];
  #else
    float matTexColor = 0.;
  #endif

  float shadowStrength = 0.;
  float shadowRampUV = 1.; 
  vec4 shadowColor;
  float metalness;

  fetchMaterialInfo(matTexColor, shadowColor, metalness);

  if(vNormalAndDiff.z < 0.){
      N *= -1.;
  }
  #if USE_UV2 == 1
    calcToonDiffuse(abs(vNormalAndDiff.w), vColor.r, lightTexColor.g, shadowStrength, shadowRampUV);
  #else
    calcToonDiffuse(vNormalAndDiff.w, vColor.r, lightTexColor.g, shadowStrength, shadowRampUV);
  #endif
  #if USE_SHADOW_RAMP == 1
    shadowColor.rgb = sampleShadowRamp(matTexColor, shadowRampUV);
    // shadowColor.rgb = vec3(1., 0., 0.);
  #endif

  #if USE_METALMAP == 1
    vec3 viewNormal = mat3(viewMatrix)*N;
    vec2 metalMapUV = viewNormal.xy* 0.5 + 0.5;
    float metalMapValue = texture2D(metalMap, metalMapUV).r;
    vec3 inputMetalLightPartColor = mix(metalDarkColor, metalLightColor, metalMapValue);       
    //vec3 inputMetalLightPartColor = mix(vec3(1.,0.,0.), vec3(1.,1.,0.), metalMapValue);       
    float NoH = max(dot(N, H), .1); 
    float specular = pow(NoH, 20.);
    diffuseColor = mix(diffuseColor, (inputMetalLightPartColor), clamp(lightTexColor.r * lightTexColor.b * metalness * (1.+specular), 0., 1.));
    
  #else
    diffuseColor.rgb += lightTexColor.r * lightTexColor.b * dotNL * .15;
  #endif

 
  float rim = clamp(1.-dotNL,0.,1.);


  #if USE_FACEMAP == 0   
    #if USE_UV2 == 1
      diffuseColor *= shadowColor.rgb*(.8+.2*step(0.,vNormalAndDiff.z));
    #else
      diffuseColor *= shadowColor.rgb*(.6+.4*step(0.,vNormalAndDiff.z));
    #endif
  #else
    diffuseColor *= mix(shadowColor.rgb, vec3(1.), lightTexColor.g);
  #endif

  
  vec2 screenUV = gl_FragCoord.xy/resolution;
  float stroke = texture2D(outlineMap,screenUV).a;

  float nol = abs(vNormalAndDiff.w);
  float frontDiff = saturate(nol);
  float viewLerp = saturate((dot(lightDir.xz, vViewDir.xz)*0.5 + 0.5));
  stroke *= (frontDiff)*viewLerp*rimIntensity*.5;
  gl_FragColor = vec4(clamp(stroke+rim*shadowColor.a+diffuseColor,vec3(0.),vec3(1.)),1.);
  gl_FragColor.rgb = fun2(mix(gl_FragColor.rgb,brightness.rgb,smoothstep(0.,.4,brightness.a)*smoothstep(0.,1.-rim,brightness.a)));
  gl_FragColor.rgb *= (1.+exposure); 
  #if USE_FACEMAP == 1
    gl_FragColor.rgb = mix(gl_FragColor.rgb, shadowColor2.rgb, smoothstep(.5,1.,dot(vNormalAndDiff.xyz, vec3(0.,0.,1.)))*smoothstep(.9,1.,vCameraAngle) * lightTexColor.b); 
  #endif

  #if USE_DEPTH > 0
    vec4 faceDepthColor = texture2D(faceDepthMap, gl_FragCoord.xy/resolution);
    float faceWorldDepth = DecodeDepth(faceDepthColor);
    gl_FragColor.a *=  mix(1., clamp((faceWorldDepth - vViewPosition.z)/1200.,0.,1.), vCameraAngle*faceDepthColor.a);
  #endif
  
  #if USE_ALPHA == 1
    vec4 alphaColor = texture2D(alphaMap, uv);
    gl_FragColor.a = alphaColor.r;
    
  #endif
}
