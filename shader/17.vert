precision highp float;
precision highp int;
#define HIGH_PRECISION
#define SHADER_NAME ShaderMaterial
#define MAT_CHANNEL 0
#define DISCARD 0
#define SIMPLE_MODE 2
#define USE_DEPTH 0
#define USE_TEXTURE 1
#define VERTEX_TEXTURES
#define GAMMA_FACTOR 2
#define MAX_BONES 1024
#define USE_SKINNING
#define BONE_TEXTURE
#define FLIP_SIDED
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform bool isOrthographic;
#ifdef USE_INSTANCING
	attribute mat4 instanceMatrix;
#endif
#ifdef USE_INSTANCING_COLOR
	attribute vec3 instanceColor;
#endif
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
#ifdef USE_TANGENT
	attribute vec4 tangent;
#endif
#if defined( USE_COLOR_ALPHA )
	attribute vec4 color;
#elif defined( USE_COLOR )
	attribute vec3 color;
#endif
#ifdef USE_MORPHTARGETS
	attribute vec3 morphTarget0;
	attribute vec3 morphTarget1;
	attribute vec3 morphTarget2;
	attribute vec3 morphTarget3;
	#ifdef USE_MORPHNORMALS
		attribute vec3 morphNormal0;
		attribute vec3 morphNormal1;
		attribute vec3 morphNormal2;
		attribute vec3 morphNormal3;
	#else
		attribute vec3 morphTarget4;
		attribute vec3 morphTarget5;
		attribute vec3 morphTarget6;
		attribute vec3 morphTarget7;
	#endif
#endif
#ifdef USE_SKINNING
	attribute vec4 skinIndex;
	attribute vec4 skinWeight;
#endif

#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	#ifdef BONE_TEXTURE
		uniform highp sampler2D boneTexture;
		uniform int boneTextureSize;
		mat4 getBoneMatrix( const in float i ) {
			float j = i * 4.0;
			float x = mod( j, float( boneTextureSize ) );
			float y = floor( j / float( boneTextureSize ) );
			float dx = 1.0 / float( boneTextureSize );
			float dy = 1.0 / float( boneTextureSize );
			y = dy * ( y + 0.5 );
			vec4 v1 = texture2D( boneTexture, vec2( dx * ( x + 0.5 ), y ) );
			vec4 v2 = texture2D( boneTexture, vec2( dx * ( x + 1.5 ), y ) );
			vec4 v3 = texture2D( boneTexture, vec2( dx * ( x + 2.5 ), y ) );
			vec4 v4 = texture2D( boneTexture, vec2( dx * ( x + 3.5 ), y ) );
			mat4 bone = mat4( v1, v2, v3, v4 );
			return bone;
		}
	#else
		uniform mat4 boneMatrices[ MAX_BONES ];
		mat4 getBoneMatrix( const in float i ) {
			mat4 bone = boneMatrices[ int(i) ];
			return bone;
		}
	#endif
#endif

#ifdef USE_MORPH
  attribute vec3 morphTarget0;
  attribute vec3 morphTarget1;
  attribute vec3 morphTarget2;
  attribute vec3 morphTarget3;
  #if USE_MORPH > 4
    attribute vec3 morphTarget4;
  #endif
  uniform float morphTargetInfluences[ USE_MORPH ];
#endif

const vec3 outlineWidthAdjustZs = vec3(0.01, 150., 200.);
const vec3 outlineWidthAdjustScales = vec3(0.105, 0.2, .3);

//const vec4 outlineWidthAdjustZs = vec4(0.01, 2., 6., 0.);
//const vec4 outlineWidthAdjustScales = vec4(0.105, 0.245, 0.6, 0.);

float lerpByZ(float startScale, float endScale, float startZ, float endZ, float z){
	float t = (z - startZ) / max(endZ - startZ, 0.001);
	t = clamp(t,0.,1.);
	return mix(startScale, endScale, t);
}

float outlineOffset(float z, float s){
	float fovScale = projectionMatrix[1][1];
	z *= 2.414 / fovScale; 
	vec2 zRange, scales;
	if (z < outlineWidthAdjustZs.y){
		zRange = outlineWidthAdjustZs.xy;
		scales = outlineWidthAdjustScales.xy;
	}else{
		zRange = outlineWidthAdjustZs.yz;
		scales = outlineWidthAdjustScales.yz;
	}
	float scale = lerpByZ(scales.x, scales.y, zRange.x, zRange.y, z);
  return scale*s*1.4;
}

attribute vec4 color;
attribute vec4 color2;

varying vec2 vUv;
varying vec3 vViewDir;
varying vec3 vNormal;

varying vec4 vViewPosition;
varying float vCameraAngle;

void main(){
    vUv = uv;
    vec3 objectNormal = color.xyz*2. - 1.;
    vec3 transformedNormal = normal;

    #ifdef USE_SKINNING
      mat4 boneMatX = getBoneMatrix( skinIndex.x );
      mat4 boneMatY = getBoneMatrix( skinIndex.y );
      mat4 boneMatZ = getBoneMatrix( skinIndex.z );
      mat4 boneMatW = getBoneMatrix( skinIndex.w );
      
      mat4 skinMatrix = mat4( 0.0 );
      skinMatrix += skinWeight.x * boneMatX;
      skinMatrix += skinWeight.y * boneMatY;
      skinMatrix += skinWeight.z * boneMatZ;
      skinMatrix += skinWeight.w * boneMatW;
      skinMatrix  = bindMatrixInverse * skinMatrix * bindMatrix;

      vec3 transformed = position;
        
      #ifdef USE_MORPH
        transformed += ( morphTarget0 ) * morphTargetInfluences[ 0 ];
        transformed += ( morphTarget1 ) * morphTargetInfluences[ 1 ];
        transformed += ( morphTarget2 ) * morphTargetInfluences[ 2 ];
        transformed += ( morphTarget3 ) * morphTargetInfluences[ 3 ];
        #if USE_MORPH > 4
          transformed += ( morphTarget4 ) * morphTargetInfluences[ 4 ];
        #endif
      #endif      
      
      vec3 dirNormal = normalize(normalMatrix * (skinMatrix * vec4(0.,0.,1.,0.)).xyz);
      objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;

      transformedNormal = vec4( skinMatrix * vec4( normal, 0.0 ) ).xyz;
     
      vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
      vec4 skinned = vec4( 0.0 );
      skinned += boneMatX * skinVertex * skinWeight.x;
      skinned += boneMatY * skinVertex * skinWeight.y;
      skinned += boneMatZ * skinVertex * skinWeight.z;
      skinned += boneMatW * skinVertex * skinWeight.w;
      transformed = ( bindMatrixInverse * skinned ).xyz; 


    #else
      vec3 transformed = vec3( position );
    #endif

    vViewPosition = modelViewMatrix * (vec4(transformed,1.));    
    vViewDir = normalize(-vViewPosition.xyz);
    
    vec3 N = normalize(normalMatrix * objectNormal);

    vNormal = normalize (normalMatrix * transformedNormal);

    N.z = 0.001;
		N = normalize(N);

    vec4 newViewPosition = vViewPosition;
   
    vCameraAngle = smoothstep(.5,1.,dot(vec3(0.,0.,1.), dirNormal));
    

    #if SIMPLE_MODE == 2
      float offset = outlineOffset(-newViewPosition.z , color2.a)* color2.r;
      newViewPosition.xyz += normalize(newViewPosition.xyz) * 2.* color2.r;    
    #else
      float offset = outlineOffset(-newViewPosition.z , color2.a);

      newViewPosition.xyz += normalize(newViewPosition.xyz)*.1;    
    #endif
   
   
    newViewPosition.xy += N.xy * offset;   

    vViewPosition = newViewPosition;

    gl_Position = projectionMatrix * newViewPosition;

    gl_Position.z = log2( max( 0.000001, gl_Position.w + 1.0 ) ) * 0.18237350163834035 - 1.0;
gl_Position.z *= gl_Position.w;

}
