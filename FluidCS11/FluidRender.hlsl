//--------------------------------------------------------------------------------------
// File: FluidRender.hlsl
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Particle Rendering
//--------------------------------------------------------------------------------------

struct Particle {
    float2 position;
    float2 velocity;
	float2 ttl;
};

struct ParticleDensity {
    float density;
};

StructuredBuffer<Particle> ParticlesRO : register( t0 );
StructuredBuffer<ParticleDensity> ParticleDensityRO : register( t1 );

cbuffer cbRenderConstants : register( b0 )
{
    matrix g_mViewProjection;
    float g_fParticleSize;
};

struct VSParticleOut
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct GSParticleOut
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 texcoord : TEXCOORD;
};


//--------------------------------------------------------------------------------------
// Visualization Helper
//--------------------------------------------------------------------------------------

static const float4 Rainbow[5] = {
    float4(0.2, 0.2, 0.2, 1), // red
    float4(1, 1, 0, 1), // orange
    float4(0, 1, 0, 1), // green
    float4(0, 1, 1, 1), // teal
    float4(0, 0, 1, 1), // blue
};

float4 VisualizeNumber(float n)
{
    return lerp( Rainbow[ floor(n * 4.0f) ], Rainbow[ ceil(n * 4.0f) ], frac(n * 4.0f) );
}

float4 VisualizeNumber(float n, float lower, float upper)
{
    return VisualizeNumber( saturate( (n - lower) / (upper - lower) ) );
}


//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------

VSParticleOut ParticleVS(uint ID : SV_VertexID)
{
    VSParticleOut Out = (VSParticleOut)0;
    Out.position = float3(ParticlesRO[ID].position, ParticlesRO[ID].ttl.y);
    Out.color = VisualizeNumber(ParticleDensityRO[ID].density, 1000.0f, 2000.0f);
    return Out;
}


//--------------------------------------------------------------------------------------
// Particle Geometry Shader
//--------------------------------------------------------------------------------------

static const float2 g_positions[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
static const float2 g_texcoords[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };

[maxvertexcount(4)]
void ParticleGS(point VSParticleOut In[1], inout TriangleStream<GSParticleOut> SpriteStream)
{
	if (In[0].position.z > 0)
	{
		[unroll]
		for (int i = 0; i < 4; i++)
		{
			GSParticleOut Out = (GSParticleOut)0;
			float4 position = float4(In[0].position.xy, 0, 1) + g_fParticleSize * float4(g_positions[i], 0, 0);
			Out.position = mul(position, g_mViewProjection);
			Out.color = In[0].color;
			Out.texcoord = g_texcoords[i];
			SpriteStream.Append(Out);
		}
		SpriteStream.RestartStrip();
	}
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------

float4 ParticlePS(GSParticleOut In) : SV_Target
{
    return In.color;
}
