precision highp float;
precision highp int;
#define HIGH_PRECISION
#define SHADER_NAME ShaderMaterial
#define MAT_CHANNEL 0
#define DISCARD 0
#define SIMPLE_MODE 0
#define USE_DEPTH 0
#define USE_TEXTURE 1
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


uniform vec3 strokeColor;
uniform vec4 brightness;
uniform float exposure;

uniform float opacity;

varying vec2 vUv;
varying vec3 vViewDir;
varying vec3 vNormal;


#if SIMPLE_MODE == 1
uniform sampler2D lightMap;
uniform vec3 strokeColor2;
uniform vec3 strokeColor3;
uniform vec3 strokeColor4;
uniform vec3 strokeColor5;
uniform vec3 strokeColor6;
uniform vec3 strokeColor7;

vec3 fetchStrokeColor(in float a){
  vec3 color;
  if (a > 0.8){
    color = strokeColor;
  }else if (a > .7){
    color = strokeColor2;
  }else if (a > .6){
    color = strokeColor3;
  }else if (a > .45){
    color = strokeColor4;
  }else if (a > .3){
    color = strokeColor5;
  }else if (a > .2){
    color = strokeColor6;
  }else if (a > .1){
    color = strokeColor7;
  }else{
    color = strokeColor;
  }
  return color;
}
#endif

#if USE_DEPTH > 0
  varying vec4 vViewPosition;
  varying float vCameraAngle;
  uniform sampler2D faceDepthMap;
  uniform vec2 resolution;
#endif

void main(){
  #if DISCARD == 1
    gl_FragColor = vec4(0.);    
    discard;
  #else
    #if SIMPLE_MODE == 1
      vec4 lightTexColor = texture2D(lightMap, vUv.xy); 
      vec3 color = fetchStrokeColor(lightTexColor[MAT_CHANNEL]);
    #else
      vec3 color = strokeColor;    
    #endif

    float rim = clamp(1.-dot(vViewDir, -vNormal),0.,1.);
    gl_FragColor = vec4(color,opacity);
    gl_FragColor.rgb = mix(gl_FragColor.rgb,brightness.rgb,smoothstep(0.,.4,brightness.a)*smoothstep(0.,1.-rim,brightness.a));//);
    gl_FragColor.rgb *= (1.+exposure);
  #endif

  #if USE_DEPTH > 0
    vec4 faceDepthColor = texture2D(faceDepthMap, gl_FragCoord.xy/resolution);
    float faceWorldDepth = DecodeDepth(faceDepthColor);
    gl_FragColor.a *= mix(1., clamp((faceWorldDepth - vViewPosition.z)/1200.,0.,1.),vCameraAngle*faceDepthColor.a);

    //gl_FragColor = faceDepthColor;
  #endif
}
