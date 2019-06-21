//--------------------------------------------------------------------------------------
// File: DDSWithoutD3DX.hlsl
//
// The HLSL file for the DDSWithoutD3DX sample for the Direct3D 11 device
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

struct Particle
{
	float2 position;
	float2 velocity;
	float2 ttl;
};

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer cbPerObject : register( b0 )
{
    matrix  g_mWorldViewProjection  : packoffset( c0 );
	float4  g_Other					: packoffset( c4 );
}

RWStructuredBuffer<Particle> EmitterRW : register(u1);

//-----------------------------------------------------------------------------------------
// Textures and Samplers
//-----------------------------------------------------------------------------------------
Texture2D    g_txDiffuse : register( t0 );
SamplerState g_samLinear : register( s0 );

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float4 Position     : POSITION; // vertex position 
    float2 TextureUV    : TEXCOORD0;// vertex texture coords 
};

struct VS_OUTPUT
{
    float4 Position     : SV_POSITION; // vertex position 
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords
	float2 OriPosition	: TEXCOORD1;
};

//--------------------------------------------------------------------------------------
// This shader computes standard transform and lighting
//--------------------------------------------------------------------------------------
VS_OUTPUT RenderSceneVS( VS_INPUT input )
{
    VS_OUTPUT Output;
    //float4 Position = float4(0.6f + input.Position.xyz * 0.25f, input.Position.w);
	float4 Position = float4(0.7f + input.Position.xyz * 0.0025f, input.Position.w);

	Output.OriPosition = Position.xy;

    // Transform the position from object space to homogeneous projection space
    Output.Position = mul( Position, g_mWorldViewProjection );
    
    // Just copy the texture coordinate through
    Output.TextureUV = input.TextureUV; 
    
    return Output;    
}

//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by modulating the texture's
// color with diffuse material color
//--------------------------------------------------------------------------------------
float4 RenderScenePS( VS_OUTPUT In ) : SV_TARGET
{ 
	// Lookup mesh texture and modulate it with diffuse
	float2 AdjustUV = In.TextureUV; // + float2(g_Other.x, 0);
	float4 TexColor = g_txDiffuse.Sample(g_samLinear, AdjustUV);

	float delta = g_Other.x - TexColor.z;
	if (delta > -0.0001f && delta < 0.0001f && g_Other.z > 0)
	{
		unsigned int x = EmitterRW.IncrementCounter();
		EmitterRW[x].position = In.OriPosition;
		EmitterRW[x].velocity = float2(0, 0);
		EmitterRW[x].ttl = float2(0, g_Other.y);
	}

	TexColor = float4(1.0f, 0, 0, 1);
	if (delta > 0.1f)
	{
		TexColor.w = 1.0f - (delta - 0.1f) / 0.2f;
		if (delta > 0.3f)
			TexColor.w = 0;
	}
	
	return TexColor;
}
