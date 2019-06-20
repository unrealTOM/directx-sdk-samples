//--------------------------------------------------------------------------------------
// File: FluidCS11.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Smoothed Particle Hydrodynamics Algorithm Based Upon:
// Particle-Based Fluid Simulation for Interactive Applications
// Matthias Müller
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Optimized Grid Algorithm Based Upon:
// Broad-Phase Collision Detection with CUDA
// Scott Le Grand
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "resource.h"
#include "WaitDlg.h"
#include "DDSTextureLoader.h"
#include "SDKMesh.h"
#include "resource.h"

#include <algorithm>
#include <random>
#include <d3dx11effect.h>

#pragma warning( disable : 4100 )

using namespace DirectX;

struct Particle
{
    XMFLOAT2 vPosition;
    XMFLOAT2 vVelocity;
	XMFLOAT2 vTTL;
};

struct ParticleDensity
{
    FLOAT fDensity;
};

struct ParticleForces
{
    XMFLOAT2 vAcceleration;
};

struct UINT2
{
    UINT x;
    UINT y;
};

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

// Numthreads size for the simulation
const UINT SIMULATION_BLOCK_SIZE = 256;

// For this sample, only use power-of-2 numbers >= 8K and <= 64K
// The algorithm can be extended to support any number of particles
// But to keep the sample simple, we do not implement boundary conditions to handle it
const UINT NUM_PARTICLES_8K = 8 * 1024;
const UINT NUM_PARTICLES_16K = 16 * 1024;
const UINT NUM_PARTICLES_32K = 32 * 1024;
const UINT NUM_PARTICLES_64K = 64 * 1024;
UINT g_iNumParticles = NUM_PARTICLES_16K;
UINT g_iEmitterWidth = (UINT)sqrt((FLOAT)g_iNumParticles) / 2;

// Particle Properties
// These will control how the fluid behaves
FLOAT g_fInitialParticleSpacing = 0.0045f;
FLOAT g_fParticleGenerateInterval = 0.01f;
FLOAT g_fParticleMaxTTL = 5.0f;
FLOAT g_fSmoothlen = 0.012f;
FLOAT g_fPressureStiffness = 200.0f;
FLOAT g_fRestDensity = 1000.0f;
FLOAT g_fParticleMass = 0.0002f;
FLOAT g_fViscosity = 0.1f;
FLOAT g_fMaxAllowableTimeStep = 0.005f;
FLOAT g_fParticleRenderSize = 0.003f;

// Gravity Directions
const XMFLOAT2A GRAVITY_DOWN(0, -0.5f);
const XMFLOAT2A GRAVITY_UP(0, 0.5f);
const XMFLOAT2A GRAVITY_LEFT(-0.5f, 0);
const XMFLOAT2A GRAVITY_RIGHT(0.5f, 0);
XMFLOAT2A g_vGravity = GRAVITY_DOWN;

// Map Size
// These values should not be larger than 256 * fSmoothlen
// Since the map must be divided up into fSmoothlen sized grid cells
// And the grid cell is used as a 16-bit sort key, 8-bits for x and y
FLOAT g_fMapHeight = 1.2f;
FLOAT g_fMapWidth = (4.0f / 3.0f) * g_fMapHeight;

// Map Wall Collision Planes
FLOAT g_fWallStiffness = 3000.0f;
XMFLOAT3A g_vPlanes[4] = {
    XMFLOAT3A(1, 0, 0),
    XMFLOAT3A(0, 1, 0),
    XMFLOAT3A(-1, 0, g_fMapWidth),
    XMFLOAT3A(0, -1, g_fMapHeight)
};

//--------------------------------------------------------------------------------------
// Direct3D11 Global variables
//--------------------------------------------------------------------------------------
ID3D11ShaderResourceView* const     g_pNullSRV = nullptr;       // Helper to Clear SRVs
ID3D11UnorderedAccessView* const    g_pNullUAV = nullptr;       // Helper to Clear UAVs
ID3D11Buffer* const                 g_pNullBuffer = nullptr;    // Helper to Clear Buffers
UINT                                g_iNullUINT = 0;         // Helper to Clear Buffers

CDXUTDialogResourceManager          g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg                     g_D3DSettingsDlg;        // Device settings dialog
CDXUTDialog                         g_HUD;                   // manages the 3D   
CDXUTDialog                         g_SampleUI;              // dialog for sample specific controls

CDXUTSDKMesh                        g_BallMesh;				    // mesh
ID3D11VertexShader*                 g_pBallVertexShader11 = NULL;
ID3D11PixelShader*                  g_pBallPixelShader11 = NULL;
ID3D11InputLayout*                  g_pBallLayout11 = NULL;
ID3D11ShaderResourceView*           g_pBallSRV11 = NULL;
ID3D11SamplerState*                 g_pBallSamLinear = NULL;
ID3D11Buffer*                       g_pcbBallVSPerObject11 = NULL;

// Resources
CDXUTTextHelper*                    g_pTxtHelper = nullptr;

// Shaders
ID3D11VertexShader*                 g_pParticleVS = nullptr;
ID3D11GeometryShader*               g_pParticleGS = nullptr;
ID3D11PixelShader*                  g_pParticlePS = nullptr;

ID3D11ComputeShader*                g_pDensity_SimpleCS = nullptr;
ID3D11ComputeShader*                g_pForce_SimpleCS = nullptr;
ID3D11ComputeShader*                g_pIntegrateCS = nullptr;
ID3D11ComputeShader*                g_pEmitCS = nullptr;

// Structured Buffers
ID3D11Buffer*                       g_pParticles = nullptr;
ID3D11ShaderResourceView*           g_pParticlesSRV = nullptr;
ID3D11UnorderedAccessView*          g_pParticlesUAV = nullptr;

ID3D11Buffer*                       g_pEmitter[2] = { nullptr, nullptr };
ID3D11ShaderResourceView*           g_pEmitterSRV[2] = { nullptr, nullptr };
ID3D11UnorderedAccessView*          g_pEmitterUAV[2] = {nullptr, nullptr};

ID3D11Buffer*                       g_pSortedParticles = nullptr;
ID3D11ShaderResourceView*           g_pSortedParticlesSRV = nullptr;
ID3D11UnorderedAccessView*          g_pSortedParticlesUAV = nullptr;

ID3D11Buffer*                       g_pParticleDensity = nullptr;
ID3D11ShaderResourceView*           g_pParticleDensitySRV = nullptr;
ID3D11UnorderedAccessView*          g_pParticleDensityUAV = nullptr;

ID3D11Buffer*                       g_pParticleForces = nullptr;
ID3D11ShaderResourceView*           g_pParticleForcesSRV = nullptr;
ID3D11UnorderedAccessView*          g_pParticleForcesUAV = nullptr;

UINT g_EmitSlot = 0;
BOOL g_EmitWithCS = false;
XMMATRIX g_mViewProjection;

// Constant Buffer Layout
#pragma warning(push)
#pragma warning(disable:4324) // structure was padded due to __declspec(align())
__declspec(align(16)) struct CBSimulationConstants
{
    UINT iNumParticles;
	FLOAT fParticleMaxTTL;
    FLOAT fTimeStep;
    FLOAT fSmoothlen;
    FLOAT fPressureStiffness;
    FLOAT fRestDensity;
    FLOAT fDensityCoef;
    FLOAT fGradPressureCoef;
    FLOAT fLapViscosityCoef;
    FLOAT fWallStiffness;
    
	UINT iEmitterWidth;
	FLOAT fInitialParticleSpacing;

    XMFLOAT2A vGravity;
    XMFLOAT4A vGridDim;

    XMFLOAT3A vPlanes[4];
};

__declspec(align(16)) struct CBRenderConstants
{
    XMFLOAT4X4 mViewProjection;
    FLOAT fParticleSize;
};

__declspec(align(16)) struct CB_VS_PER_OBJECT
{
	XMFLOAT4X4 m_mWorldViewProjection;
	XMFLOAT4A m_Others; //x = m_fElapsedTime, y = m_fParticleMaxTTL
};
#pragma warning(pop)

// Constant Buffers
ID3D11Buffer*                       g_pcbSimulationConstants = nullptr;
ID3D11Buffer*                       g_pcbRenderConstants = nullptr;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4

#define IDC_RESETSIM              5
#define IDC_NUMPARTICLES          6
#define IDC_GRAVITY               7
#define IDC_SIMMODE               8

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );

bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext );

HRESULT CreateSimulationBuffers( ID3D11Device* pd3dDevice );
void InitApp();
void RenderText();

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device
    // that is available on the system depending on which D3D callbacks are set below

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    InitApp();
    DXUTInit( true, true ); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"FluidCS11" );
    DXUTCreateDevice( D3D_FEATURE_LEVEL_10_0, true, 1024, 768 );
    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    // Initialize dialogs
    g_D3DSettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );
    g_SampleUI.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); int iY = 20;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 0, iY, 170, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 0, iY += 26, 170, 22, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 0, iY += 26, 170, 22, VK_F2 );

    g_SampleUI.SetCallback( OnGUIEvent ); iY = 0;

    g_SampleUI.AddButton( IDC_RESETSIM, L"Reset Particles", 0, iY += 26, 170, 22 );
    
    g_SampleUI.AddComboBox( IDC_NUMPARTICLES, 0, iY += 26, 170, 22 );
    g_SampleUI.GetComboBox( IDC_NUMPARTICLES )->AddItem( L"8K Particles", UIntToPtr(NUM_PARTICLES_8K) );
    g_SampleUI.GetComboBox( IDC_NUMPARTICLES )->AddItem( L"16K Particles", UIntToPtr(NUM_PARTICLES_16K) );
    g_SampleUI.GetComboBox( IDC_NUMPARTICLES )->AddItem( L"32K Particles", UIntToPtr(NUM_PARTICLES_32K) );
    g_SampleUI.GetComboBox( IDC_NUMPARTICLES )->AddItem( L"64K Particles", UIntToPtr(NUM_PARTICLES_64K) );
    g_SampleUI.GetComboBox( IDC_NUMPARTICLES )->SetSelectedByData( UIntToPtr(g_iNumParticles) );

    g_SampleUI.AddComboBox( IDC_GRAVITY, 0, iY += 26, 170, 22 );
    g_SampleUI.GetComboBox( IDC_GRAVITY )->AddItem( L"Gravity Down", (void*)&GRAVITY_DOWN );
    g_SampleUI.GetComboBox( IDC_GRAVITY )->AddItem( L"Gravity Up", (void*)&GRAVITY_UP );
    g_SampleUI.GetComboBox( IDC_GRAVITY )->AddItem( L"Gravity Left", (void*)&GRAVITY_LEFT );
    g_SampleUI.GetComboBox( IDC_GRAVITY )->AddItem( L"Gravity Right", (void*)&GRAVITY_RIGHT );
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	pDeviceSettings->d3d11.CreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
    return true;
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 2, 0 );
    g_pTxtHelper->SetForegroundColor( Colors::Yellow );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
    g_pTxtHelper->DrawFormattedTextLine( L"%i Particles", g_iNumParticles );

    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    return 0;
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    switch( nControlID )
    {
            // Standard DXUT controls
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:
            DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:
            g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); break;
        case IDC_RESETSIM:
            CreateSimulationBuffers( DXUTGetD3D11Device() ); break;
        case IDC_NUMPARTICLES:
            g_iNumParticles = PtrToUint( ((CDXUTComboBox*)pControl)->GetSelectedData() );
            CreateSimulationBuffers( DXUTGetD3D11Device() );
            break;
        case IDC_GRAVITY:
            g_vGravity = *(const XMFLOAT2A*)((CDXUTComboBox*)pControl)->GetSelectedData(); break;
    }
}


//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    if ( DeviceInfo->ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x == FALSE )
        return false;

    return true;
}


//--------------------------------------------------------------------------------------
// Helper for creating constant buffers
//--------------------------------------------------------------------------------------
template <class T>
HRESULT CreateConstantBuffer(ID3D11Device* pd3dDevice, ID3D11Buffer** ppCB)
{
    HRESULT hr = S_OK;

    D3D11_BUFFER_DESC Desc;
    Desc.Usage = D3D11_USAGE_DEFAULT;
    Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    Desc.CPUAccessFlags = 0;
    Desc.MiscFlags = 0;
    Desc.ByteWidth = sizeof( T );
    V_RETURN( pd3dDevice->CreateBuffer( &Desc, nullptr, ppCB ) );

    return hr;
}


//--------------------------------------------------------------------------------------
// Helper for creating structured buffers with an SRV and UAV
//--------------------------------------------------------------------------------------
template <class T>
HRESULT CreateStructuredBuffer(
	ID3D11Device* pd3dDevice, UINT iNumElements, 
	ID3D11Buffer** ppBuffer, 
	ID3D11ShaderResourceView** ppSRV, 
	ID3D11UnorderedAccessView** ppUAV, 
	const T* pInitialData = nullptr,
	UINT flag = 0)
{
    HRESULT hr = S_OK;

    // Create SB
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.ByteWidth = iNumElements * sizeof(T);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufferDesc.StructureByteStride = sizeof(T);

    D3D11_SUBRESOURCE_DATA bufferInitData = {};
    bufferInitData.pSysMem = pInitialData;
    V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : nullptr, ppBuffer ) );

    // Create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.ElementWidth = iNumElements;
    V_RETURN( pd3dDevice->CreateShaderResourceView( *ppBuffer, &srvDesc, ppSRV ) );

    // Create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = iNumElements;
	uavDesc.Buffer.Flags = flag;
    V_RETURN( pd3dDevice->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV ) );

    return hr;
}

//--------------------------------------------------------------------------------------
// Create the ball rendering related resources
//--------------------------------------------------------------------------------------
HRESULT CreateBallResources(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	g_BallMesh.Destroy();

	SAFE_RELEASE(g_pBallSamLinear);
	SAFE_RELEASE(g_pcbBallVSPerObject11);
	SAFE_RELEASE(g_pBallSRV11);

	// Create state objects
	D3D11_SAMPLER_DESC samDesc;
	ZeroMemory(&samDesc, sizeof(samDesc));
	samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samDesc.MaxAnisotropy = 1;
	samDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	samDesc.MaxLOD = D3D11_FLOAT32_MAX;
	V_RETURN(pd3dDevice->CreateSamplerState(&samDesc, &g_pBallSamLinear));
	DXUT_SetDebugName(g_pBallSamLinear, "Linear");

	// Create constant buffers
	D3D11_BUFFER_DESC cbDesc;
	ZeroMemory(&cbDesc, sizeof(cbDesc));
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	cbDesc.ByteWidth = sizeof(CB_VS_PER_OBJECT);
	V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pcbBallVSPerObject11));
	DXUT_SetDebugName(g_pcbBallVSPerObject11, "CB_VS_PER_OBJECT");

	// load the mesh
	V_RETURN(g_BallMesh.Create(pd3dDevice, L"ball.sdkmesh"));

	// Load the texture
	V_RETURN(CreateDDSTextureFromFile(pd3dDevice, L"seafloor.dds", nullptr, &g_pBallSRV11));

	return hr;
}

//--------------------------------------------------------------------------------------
// Create the buffers used for the simulation data
//--------------------------------------------------------------------------------------
HRESULT CreateSimulationBuffers( ID3D11Device* pd3dDevice )
{
    HRESULT hr = S_OK;

    // Destroy the old buffers in case the number of particles has changed
    SAFE_RELEASE( g_pParticles );
    SAFE_RELEASE( g_pParticlesSRV );
    SAFE_RELEASE( g_pParticlesUAV );

	for (UINT i = 0; i < 2; ++i)
	{
		SAFE_RELEASE(g_pEmitter[i]);
		SAFE_RELEASE(g_pEmitterSRV[i]);
		SAFE_RELEASE(g_pEmitterUAV[i]);
	}

    SAFE_RELEASE( g_pSortedParticles );
    SAFE_RELEASE( g_pSortedParticlesSRV );
    SAFE_RELEASE( g_pSortedParticlesUAV );

    SAFE_RELEASE( g_pParticleForces );
    SAFE_RELEASE( g_pParticleForcesSRV );
    SAFE_RELEASE( g_pParticleForcesUAV );
    
    SAFE_RELEASE( g_pParticleDensity );
    SAFE_RELEASE( g_pParticleDensitySRV );
    SAFE_RELEASE( g_pParticleDensityUAV );

	std::ranlux24_base gen_;
	std::uniform_int_distribution<> random_dis_(0, g_iNumParticles);

    auto particles = std::make_unique<Particle[]>(g_iNumParticles);
    ZeroMemory( particles.get(), sizeof(Particle) * g_iNumParticles );

	if (g_EmitWithCS)
	{
		UINT Total = g_iEmitterWidth * g_iEmitterWidth / 8;
		for (UINT i = 0; i < g_iNumParticles; i++)
		{
			// Arrange the particles in a nice square
			UINT x = i % (g_iEmitterWidth * 2);
			UINT y = i / (g_iEmitterWidth * 2);

			UINT xx = x + (g_iEmitterWidth * 2);
			UINT yy = y + (g_iEmitterWidth * 2);

			particles[i].vPosition = XMFLOAT2(g_fInitialParticleSpacing * (FLOAT)xx, g_fInitialParticleSpacing * (FLOAT)yy);

			if (x < g_iEmitterWidth && y < g_iEmitterWidth)
			{
				//particles[i].vTTL.x = ((y * iStartingWidth / 2) + x) * g_fParticleGenerateInterval;
				particles[i].vTTL.x = (random_dis_(gen_) % Total + y * g_iEmitterWidth / 2) * g_fParticleGenerateInterval;
				particles[i].vTTL.y = g_fParticleMaxTTL;
			}
			else
			{
				particles[i].vTTL.x = 0;
				particles[i].vTTL.y = -1000;
			}
		}
	}
	else
	{
		for (UINT i = 0; i < g_iNumParticles; i++)
		{
			particles[i].vTTL.x = 0;
			particles[i].vTTL.y = -1;
		}
	}

    // Create Structured Buffers
    V_RETURN( CreateStructuredBuffer< Particle >( pd3dDevice, g_iNumParticles, &g_pParticles, &g_pParticlesSRV, &g_pParticlesUAV, particles.get() ) );
    DXUT_SetDebugName( g_pParticles, "Particles" );
    DXUT_SetDebugName( g_pParticlesSRV, "Particles SRV" );
    DXUT_SetDebugName( g_pParticlesUAV, "Particles UAV" );

	const CHAR* Emitter[2] = {"Emitter 1", "Emitter 2"};
	const CHAR* EmitterSRV[2] = {"Emitter SRV 1", "Emitter SRV 2"};
	const CHAR* EmitterUAV[2] = { "Emitter UAV 1", "Emitter UAV 2" };

	for (UINT i = 0; i < 2; ++i)
	{
		V_RETURN(CreateStructuredBuffer< Particle >(pd3dDevice, g_iNumParticles, &g_pEmitter[i], &g_pEmitterSRV[i], &g_pEmitterUAV[i], particles.get(), D3D11_BUFFER_UAV_FLAG_COUNTER));
		DXUT_SetDebugName(g_pEmitter[i], Emitter[i]);
		DXUT_SetDebugName(g_pEmitterSRV[i], EmitterSRV[i]);
		DXUT_SetDebugName(g_pEmitterUAV[i], EmitterUAV[i]);
	}

    V_RETURN( CreateStructuredBuffer< Particle >( pd3dDevice, g_iNumParticles, &g_pSortedParticles, &g_pSortedParticlesSRV, &g_pSortedParticlesUAV, particles.get()) );
    DXUT_SetDebugName( g_pSortedParticles, "Sorted" );
    DXUT_SetDebugName( g_pSortedParticlesSRV, "Sorted SRV" );
    DXUT_SetDebugName( g_pSortedParticlesUAV, "Sorted UAV" );

    V_RETURN( CreateStructuredBuffer< ParticleForces >( pd3dDevice, g_iNumParticles, &g_pParticleForces, &g_pParticleForcesSRV, &g_pParticleForcesUAV ) );
    DXUT_SetDebugName( g_pParticleForces, "Forces" );
    DXUT_SetDebugName( g_pParticleForcesSRV, "Forces SRV" );
    DXUT_SetDebugName( g_pParticleForcesUAV, "Forces UAV" );

    V_RETURN( CreateStructuredBuffer< ParticleDensity >( pd3dDevice, g_iNumParticles, &g_pParticleDensity, &g_pParticleDensitySRV, &g_pParticleDensityUAV ) );
    DXUT_SetDebugName( g_pParticleDensity, "Density" );
    DXUT_SetDebugName( g_pParticleDensitySRV, "Density SRV" );
    DXUT_SetDebugName( g_pParticleDensityUAV, "Density UAV" );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
    HRESULT hr;

    auto pd3dImmediateContext = DXUTGetD3D11DeviceContext();
    V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
    g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

    // Compile the Shaders
    ID3DBlob* pBlob = nullptr;

	// Create our vertex input layout
	const D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};

	// Shaders used for rendering the ball
	V_RETURN(DXUTCompileFromFile(L"DDSWithoutD3DX.hlsl", nullptr, "RenderSceneVS", "vs_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob));
	V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pBallVertexShader11));
	DXUT_SetDebugName(g_pBallVertexShader11, "RenderSceneVS");

	V_RETURN(pd3dDevice->CreateInputLayout(layout, ARRAYSIZE(layout), pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &g_pBallLayout11));
	DXUT_SetDebugName(g_pBallLayout11, "Primary");

	SAFE_RELEASE(pBlob);

	V_RETURN(DXUTCompileFromFile(L"DDSWithoutD3DX.hlsl", nullptr, "RenderScenePS", "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pBallPixelShader11));
	SAFE_RELEASE(pBlob);
	DXUT_SetDebugName(g_pBallPixelShader11, "RenderScenePS");

    // Rendering Shaders
    V_RETURN( DXUTCompileFromFile( L"FluidRender.hlsl", nullptr, "ParticleVS", "vs_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pParticleVS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticleVS, "ParticleVS" );

    V_RETURN( DXUTCompileFromFile( L"FluidRender.hlsl", nullptr, "ParticleGS", "gs_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreateGeometryShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pParticleGS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticleGS, "ParticleGS" );

    V_RETURN( DXUTCompileFromFile( L"FluidRender.hlsl", nullptr, "ParticlePS", "ps_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pParticlePS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticlePS, "ParticlePS" );

    // Compute Shaders
    const char* CSTarget = (pd3dDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0)? "cs_5_0" : "cs_4_0";
    
    CWaitDlg CompilingShadersDlg;
    CompilingShadersDlg.ShowDialog( L"Compiling Shaders..." );

	V_RETURN(DXUTCompileFromFile(L"FluidCS11.hlsl", nullptr, "EmitCS", CSTarget, D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob));
	V_RETURN(pd3dDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pEmitCS));
	SAFE_RELEASE(pBlob);
	DXUT_SetDebugName(g_pEmitCS, "EmitCS");

    V_RETURN( DXUTCompileFromFile( L"FluidCS11.hlsl", nullptr, "IntegrateCS", CSTarget, D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pIntegrateCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pIntegrateCS, "IntegrateCS" );

    V_RETURN( DXUTCompileFromFile( L"FluidCS11.hlsl", nullptr, "DensityCS_Simple", CSTarget, D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pDensity_SimpleCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pDensity_SimpleCS, "DensityCS_Simple" );

    V_RETURN( DXUTCompileFromFile( L"FluidCS11.hlsl", nullptr, "ForceCS_Simple", CSTarget, D3DCOMPILE_ENABLE_STRICTNESS, 0, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &g_pForce_SimpleCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pForce_SimpleCS, "ForceCS_Simple" );

    CompilingShadersDlg.DestroyDialog();

	// Create ball rendering related resources
	V_RETURN( CreateBallResources(pd3dDevice) );

    // Create the Simulation Buffers
    V_RETURN( CreateSimulationBuffers( pd3dDevice ) );

    // Create Constant Buffers
    V_RETURN( CreateConstantBuffer< CBSimulationConstants >( pd3dDevice, &g_pcbSimulationConstants ) );
    V_RETURN( CreateConstantBuffer< CBRenderConstants >( pd3dDevice, &g_pcbRenderConstants ) );

    DXUT_SetDebugName( g_pcbSimulationConstants, "Simluation" );
    DXUT_SetDebugName( g_pcbRenderConstants, "Render" );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
    g_HUD.SetSize( 170, 170 );
    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 170, pBackBufferSurfaceDesc->Height - 400 );
    g_SampleUI.SetSize( 170, 300 );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// GPU Fluid Simulation
//--------------------------------------------------------------------------------------
HRESULT RenderBall(ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime)
{
	HRESULT hr;

	UINT UAVInitialCounts = 0;
	static float s_TotalElapsedTime = 0;

	s_TotalElapsedTime += fElapsedTime;
	s_TotalElapsedTime -= (int)s_TotalElapsedTime;

	// Set the constant buffers
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	V_RETURN(pd3dImmediateContext->Map(g_pcbBallVSPerObject11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
	CB_VS_PER_OBJECT* pVSPerObject = (CB_VS_PER_OBJECT*)MappedResource.pData;
	XMStoreFloat4x4(&pVSPerObject->m_mWorldViewProjection, XMMatrixTranspose( g_mViewProjection ));
	pVSPerObject->m_Others.x = s_TotalElapsedTime;
	pVSPerObject->m_Others.y = g_fParticleMaxTTL;
	pVSPerObject->m_Others.z = g_fMapWidth;
	pVSPerObject->m_Others.w = g_fMapHeight;
	pd3dImmediateContext->Unmap(g_pcbBallVSPerObject11, 0);
	pd3dImmediateContext->VSSetConstantBuffers(0, 1, &g_pcbBallVSPerObject11);
	pd3dImmediateContext->PSSetConstantBuffers(0, 1, &g_pcbBallVSPerObject11);

	// Set render resources
	pd3dImmediateContext->IASetInputLayout(g_pBallLayout11);
	pd3dImmediateContext->VSSetShader(g_pBallVertexShader11, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_pBallPixelShader11, NULL, 0);
	pd3dImmediateContext->PSSetShaderResources(0, 1, &g_pBallSRV11);
	pd3dImmediateContext->PSSetSamplers(0, 1, &g_pBallSamLinear);
	pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(
		D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL, nullptr, nullptr, 1, 1, &g_pEmitterUAV[g_EmitSlot], &UAVInitialCounts);

	g_BallMesh.Render(pd3dImmediateContext);

	// Reset
	pd3dImmediateContext->PSSetShaderResources(0, 1, &g_pNullSRV);
	pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(
		D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL, nullptr, nullptr, 1, 1, &g_pNullUAV, &UAVInitialCounts);

	return hr;
}

//--------------------------------------------------------------------------------------
// GPU Fluid Simulation - Simple N^2 Algorithm
//--------------------------------------------------------------------------------------
void SimulateFluid_Simple( ID3D11DeviceContext* pd3dImmediateContext )
{
    UINT UAVInitialCounts = 0;

    // Setup
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbSimulationConstants );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pParticlesSRV );

	// Emit
	if (g_EmitWithCS)
	{
		pd3dImmediateContext->CSSetUnorderedAccessViews(1, 1, &g_pEmitterUAV[g_EmitSlot], &UAVInitialCounts);
		pd3dImmediateContext->CSSetShader(g_pEmitCS, nullptr, 0);
		pd3dImmediateContext->Dispatch(g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);
	}

    // Density
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pParticleDensityUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShader( g_pDensity_SimpleCS, nullptr, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

    // Force
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pParticleForcesUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 1, 1, &g_pParticleDensitySRV );
    pd3dImmediateContext->CSSetShader( g_pForce_SimpleCS, nullptr, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

    // Integrate
    pd3dImmediateContext->CopyResource( g_pSortedParticles, g_pParticles );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pParticlesUAV, &UAVInitialCounts );
	pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1, &g_pEmitterUAV[1 - g_EmitSlot], &UAVInitialCounts);
	pd3dImmediateContext->CSSetShaderResources( 2, 1, &g_pParticleForcesSRV );
    pd3dImmediateContext->CSSetShader( g_pIntegrateCS, nullptr, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );
}

//--------------------------------------------------------------------------------------
// GPU Fluid Simulation
//--------------------------------------------------------------------------------------
void SimulateFluid( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
    UINT UAVInitialCounts = 0;

    // Update per-frame variables
    CBSimulationConstants pData = {};

    // Simulation Constants
    pData.iNumParticles = g_iNumParticles;
	pData.fParticleMaxTTL = g_fParticleMaxTTL;
	// Clamp the time step when the simulation runs slowly to prevent numerical explosion
    pData.fTimeStep = std::min( g_fMaxAllowableTimeStep, fElapsedTime );
    pData.fSmoothlen = g_fSmoothlen;
    pData.fPressureStiffness = g_fPressureStiffness;
    pData.fRestDensity = g_fRestDensity;
    pData.fDensityCoef = g_fParticleMass * 315.0f / (64.0f * XM_PI * pow(g_fSmoothlen, 9));
    pData.fGradPressureCoef = g_fParticleMass * -45.0f / (XM_PI * pow(g_fSmoothlen, 6));
    pData.fLapViscosityCoef = g_fParticleMass * g_fViscosity * 45.0f / (XM_PI * pow(g_fSmoothlen, 6));

	pData.iEmitterWidth = g_iEmitterWidth;
	pData.fInitialParticleSpacing = g_fInitialParticleSpacing;

    pData.vGravity = g_vGravity;
    
    // Cells are spaced the size of the smoothing length search radius
    // That way we only need to search the 8 adjacent cells + current cell
    pData.vGridDim.x = 1.0f / g_fSmoothlen;
    pData.vGridDim.y = 1.0f / g_fSmoothlen;
    pData.vGridDim.z = 0;
    pData.vGridDim.w = 0;

    // Collision information for the map
    pData.fWallStiffness = g_fWallStiffness;
    pData.vPlanes[0] = g_vPlanes[0];
    pData.vPlanes[1] = g_vPlanes[1];
    pData.vPlanes[2] = g_vPlanes[2];
    pData.vPlanes[3] = g_vPlanes[3];

    pd3dImmediateContext->UpdateSubresource( g_pcbSimulationConstants, 0, nullptr, &pData, 0, 0 );

	SimulateFluid_Simple(pd3dImmediateContext);

    // Unset
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pNullUAV, &UAVInitialCounts );
	pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1, &g_pNullUAV, &UAVInitialCounts );
	pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 1, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 2, 1, &g_pNullSRV );
}


//--------------------------------------------------------------------------------------
// GPU Fluid Rendering
//--------------------------------------------------------------------------------------
void RenderFluid( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
    // Update Constants
    CBRenderConstants pData = {};

    XMStoreFloat4x4( &pData.mViewProjection, XMMatrixTranspose( g_mViewProjection ) );
    pData.fParticleSize = g_fParticleRenderSize;

    pd3dImmediateContext->UpdateSubresource( g_pcbRenderConstants, 0, nullptr, &pData, 0, 0 );

    // Set the shaders
    pd3dImmediateContext->VSSetShader( g_pParticleVS, nullptr, 0 );
    pd3dImmediateContext->GSSetShader( g_pParticleGS, nullptr, 0 );
    pd3dImmediateContext->PSSetShader( g_pParticlePS, nullptr, 0 );

    // Set the constant buffers
    pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
    pd3dImmediateContext->GSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
    pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );

    // Setup the particles buffer and IA
    pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pParticlesSRV );
    pd3dImmediateContext->VSSetShaderResources( 1, 1, &g_pParticleDensitySRV );
    pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pNullBuffer, &g_iNullUINT, &g_iNullUINT );
    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );

    // Draw the mesh
    pd3dImmediateContext->Draw( g_iNumParticles, 0 );

    // Unset the particles buffer
    pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pNullSRV );
    pd3dImmediateContext->VSSetShaderResources( 1, 1, &g_pNullSRV );
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext )
{
    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.OnRender( fElapsedTime );
        return;
    }

    // Clear the render target and depth stencil
    auto pRTV = DXUTGetD3D11RenderTargetView();
    pd3dImmediateContext->ClearRenderTargetView( pRTV, Colors::Black );

    auto pDSV = DXUTGetD3D11DepthStencilView();
    pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0, 0 );

	// Simple orthographic projection to display the entire map
	XMMATRIX mView = XMMatrixTranslation(-g_fMapWidth / 2.0f, -g_fMapHeight / 2.0f, 0);
	XMMATRIX mProjection = XMMatrixOrthographicLH(g_fMapWidth, g_fMapHeight, 0, 1);

	g_mViewProjection = mView * mProjection;

	if (!g_EmitWithCS)
		RenderBall( pd3dImmediateContext, fElapsedTime );

    SimulateFluid( pd3dImmediateContext, fElapsedTime );

    RenderFluid( pd3dImmediateContext, fElapsedTime );

	g_EmitSlot = 1 - g_EmitSlot;

    // Render the HUD
    DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
    g_HUD.OnRender( fElapsedTime );
    g_SampleUI.OnRender( fElapsedTime );
    RenderText();
    DXUT_EndPerfEvent();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_D3DSettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE( g_pTxtHelper );

	g_BallMesh.Destroy();

	SAFE_RELEASE(g_pBallVertexShader11);
	SAFE_RELEASE(g_pBallPixelShader11);
	SAFE_RELEASE(g_pBallLayout11);
	SAFE_RELEASE(g_pBallSRV11);
	SAFE_RELEASE(g_pBallSamLinear);

	SAFE_RELEASE(g_pcbBallVSPerObject11);

    SAFE_RELEASE( g_pcbSimulationConstants );
    SAFE_RELEASE( g_pcbRenderConstants );

    SAFE_RELEASE( g_pParticleVS );
    SAFE_RELEASE( g_pParticleGS );
    SAFE_RELEASE( g_pParticlePS );

	SAFE_RELEASE( g_pEmitCS );
    SAFE_RELEASE( g_pIntegrateCS );
    SAFE_RELEASE( g_pDensity_SimpleCS );
    SAFE_RELEASE( g_pForce_SimpleCS );

    SAFE_RELEASE( g_pParticles );
    SAFE_RELEASE( g_pParticlesSRV );
    SAFE_RELEASE( g_pParticlesUAV );

	for (UINT i = 0; i < 2; ++i)
	{
		SAFE_RELEASE(g_pEmitter[i]);
		SAFE_RELEASE(g_pEmitterSRV[i]);
		SAFE_RELEASE(g_pEmitterUAV[i]);
	}

    SAFE_RELEASE( g_pSortedParticles );
    SAFE_RELEASE( g_pSortedParticlesSRV );
    SAFE_RELEASE( g_pSortedParticlesUAV );

    SAFE_RELEASE( g_pParticleForces );
    SAFE_RELEASE( g_pParticleForcesSRV );
    SAFE_RELEASE( g_pParticleForcesUAV );
    
    SAFE_RELEASE( g_pParticleDensity );
    SAFE_RELEASE( g_pParticleDensitySRV );
    SAFE_RELEASE( g_pParticleDensityUAV );
}
