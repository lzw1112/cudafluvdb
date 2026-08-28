
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <helper_math.h>
#include "CudaTextureAccessor.h"
#include "CudaSurfaceAccessor.h"
__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufLoc, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
		float4 vel = tex.sample(loc.x, loc.y, loc.z);
		return make_float3(vel.x, vel.y, vel.z);
		};

	float3 loc = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);

	float3 vel1 = sample(texVel, loc);
	float3 vel2 = sample(texVel, loc - 0.5f * vel1);
	float3 vel3 = sample(texVel, loc - 0.75f * vel2);
	loc -= (2.f / 9.f) * vel1 + (1.f / 3.f) * vel2 + (4.f / 9.f) * vel3;

	sufLoc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
}
extern "C" void advect_kernel_extern(CudaTextureAccessor<float4> &texVel, CudaSurfaceAccessor<float4>& sufLoc, CudaSurfaceAccessor<char>& sufBound, unsigned int& n,int *a)
{
	//std::cout << texVel << std::endl;
	advect_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (texVel, sufLoc, sufBound, n);
	
}



template <class T>
__global__ void resample_kernel(CudaSurfaceAccessor<float4> sufLoc, CudaTextureAccessor<T> texClr, CudaSurfaceAccessor<T> sufClrNext, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float4 loc = sufLoc.read(x, y, z);
	T clr = texClr.sample(loc.x, loc.y, loc.z);
	sufClrNext.write(clr, x, y, z);
}
extern "C"  void resample_kernel_extern(CudaSurfaceAccessor<float4> &sufLoc, CudaTextureAccessor<float4> &texClr, CudaSurfaceAccessor<float4> &sufClrNext, unsigned int& n)
{
	resample_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufLoc, texClr, sufClrNext, n);
}


__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float vxp = sufVel.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
	float vyp = sufVel.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
	float vzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
	float vxn = sufVel.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
	float vyn = sufVel.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
	float vzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
	float div = (vxp - vxn + vyp - vyn + vzp - vzn) * 0.5f;
	sufDiv.write(div, x, y, z);
}
extern "C" void divergence_kernel_extern(CudaSurfaceAccessor<float4>& sufVel, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n)
{
	divergence_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufVel, sufDiv, n);
}




__global__ void jacobi_kernel(CudaSurfaceAccessor<float> sufDiv, CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufPreNext, unsigned int n) {
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;


	float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
	float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
	float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
	float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
	float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
	float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
	float div = sufDiv.read(x, y, z);
	float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
	sufPreNext.write(preNext, x, y, z);
}
extern "C" void jacobi_kernel_extern(CudaSurfaceAccessor<float>& sufDiv, CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufPreNext, unsigned int& n)
{
	jacobi_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufDiv, sufPre, sufPreNext, n);
}


__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;
	if (sufBound.read(x, y, z) < 0) return;

	float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
	float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
	float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
	float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
	float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
	float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
	float4 vel = sufVel.read(x, y, z);
	vel.x -= (pxp - pxn) * 0.5f;
	vel.y -= (pyp - pyn) * 0.5f;
	vel.z -= (pzp - pzn) * 0.5f;
	sufVel.write(vel, x, y, z);
}
extern "C" void subgradient_kernel_extern(CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float4>& sufVel, CudaSurfaceAccessor<char>& sufBound, unsigned int& n)
{
	subgradient_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufPre, sufVel, sufBound, n);
}


__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float* sum, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float div = sufDiv.read(x, y, z);
	atomicAdd(sum, div * div);
}
extern "C" void sumloss_kernel_extern(CudaSurfaceAccessor<float> &sufDiv, float *sum, unsigned int &n)
{
	sumloss_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufDiv, sum, n);
}



__global__ void rbgs_kernel_0(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
	int phase = 0;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;
	if ((x + y + z) % 2 != phase) return;

	float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
	float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
	float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
	float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
	float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
	float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
	float div = sufDiv.read(x, y, z);
	float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
	sufPre.write(preNext, x, y, z);
}
extern "C" void rbgs_kernel_0_extern(CudaSurfaceAccessor<float> &sufPre, CudaSurfaceAccessor<float> &sufDiv, unsigned int& tn)
{
	rbgs_kernel_0 << <dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8) >> > (sufPre, sufDiv, tn);
}

__global__ void rbgs_kernel_1(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
	int phase = 1;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;
	if ((x + y + z) % 2 != phase) return;

	float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
	float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
	float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
	float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
	float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
	float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
	float div = sufDiv.read(x, y, z);
	float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
	sufPre.write(preNext, x, y, z);
}
extern "C" void rbgs_kernel_1_extern(CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n)
{
	rbgs_kernel_1 << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufPre, sufDiv, n);
}



__global__ void restrict_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float ooo = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2);
	float ioo = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2);
	float oio = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2);
	float iio = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2);
	float ooi = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2 + 1);
	float ioi = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2 + 1);
	float oii = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2 + 1);
	float iii = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2 + 1);
	float preNext = (ooo + ioo + oio + iio + ooi + ioi + oii + iii);
	sufPreNext.write(preNext, x, y, z);
}
extern "C" void restrict_kernel_extern(CudaSurfaceAccessor<float>& sufPreNext, CudaSurfaceAccessor<float>& sufPre, unsigned int &n)
{
	restrict_kernel << <dim3((n / 2 + 7) / 8, (n*2 / 2 + 7) / 8, (n / 2 + 7) / 8), dim3(8, 8, 8) >> > (sufPreNext, sufPre, n / 2);
}

__global__ void residual_kernel(CudaSurfaceAccessor<float> sufRes, CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
	float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
	float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
	float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
	float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
	float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
	float pre = sufPre.read(x, y, z);
	float div = sufDiv.read(x, y, z);
	float res = pxp + pxn + pyp + pyn + pzp + pzn - 6.f * pre - div;
	sufRes.write(res, x, y, z);
}
extern "C" void residual_kernel_extern(CudaSurfaceAccessor<float>& sufRes, CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n)
{
	residual_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufRes, sufPre, sufDiv, n);
}


__global__ void fillzero_kernel(CudaSurfaceAccessor<float> sufPre, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	sufPre.write(0.f, x, y, z);
}
extern "C" void fillzero_kernel_extern(CudaSurfaceAccessor<float> &sufPre, unsigned int &n)
{
	fillzero_kernel << <dim3((n /2 + 7) / 8, (n/2  + 7) / 8, (n/2  + 7) / 8), dim3(8, 8, 8) >> > (sufPre, n / 2);
}
extern "C" void fillzero_kernel_extern_n(CudaSurfaceAccessor<float> &sufPre, unsigned int &n)
{
	fillzero_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (sufPre, n);
}

__global__ void prolongate_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= n || y >= n || z >= n) return;

	float preDelta = sufPre.read(x, y, z) * (0.5f / 8.f);
#pragma unroll
	for (int dz = 0; dz < 2; dz++) {
#pragma unroll
		for (int dy = 0; dy < 2; dy++) {
#pragma unroll
			for (int dx = 0; dx < 2; dx++) {
				float preNext = sufPreNext.read<cudaBoundaryModeZero>(x * 2 + dx, y * 2 + dy, z * 2 + dz);
				preNext += preDelta;
				sufPreNext.write<cudaBoundaryModeZero>(preNext, x * 2 + dx, y * 2 + dy, z * 2 + dz);
			}
		}
	}
}
extern "C" void prolongate_kernel_extern(CudaSurfaceAccessor<float>& sufPreNext, CudaSurfaceAccessor<float>& sufPre, unsigned int& n)
{
	prolongate_kernel << <dim3((n / 2 + 7) / 8, (n / 2 + 7) / 8, (n / 2 + 7) / 8), dim3(8, 8, 8) >> > (sufPreNext, sufPre, n / 2);
}