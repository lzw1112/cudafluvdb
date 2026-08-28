#pragma once

#include "CudaTextureAccessor.h"
#include <memory>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "sm_20_atomic_functions.h"
#include "CudaSurfaceAccessor.h"
#include "texture_indirect_functions.h"
#include "driver_types.h"
extern "C" void advect_kernel_extern(CudaTextureAccessor<float4> &texVel, CudaSurfaceAccessor<float4>& sufLoc, CudaSurfaceAccessor<char>& sufBound, unsigned int& n,int *a);

extern "C" void resample_kernel_extern(CudaSurfaceAccessor<float4>& sufLoc, CudaTextureAccessor<float4>& texClr, CudaSurfaceAccessor<float4>& sufClrNext, unsigned int& n);

extern "C" void divergence_kernel_extern(CudaSurfaceAccessor<float4>& sufVel, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n);

extern "C" void jacobi_kernel_extern(CudaSurfaceAccessor<float>& sufDiv, CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufPreNext, unsigned int& n);

extern "C" void subgradient_kernel_extern(CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float4>& sufVel, CudaSurfaceAccessor<char>& sufBound, unsigned int& n);

extern "C" void sumloss_kernel_extern(CudaSurfaceAccessor<float>& sufDiv, float* sum, unsigned int &n);

extern "C" void rbgs_kernel_0_extern(CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufDiv, unsigned int& tn);

extern "C" void rbgs_kernel_1_extern(CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n);

extern "C" void restrict_kernel_extern(CudaSurfaceAccessor<float>& sufPreNext, CudaSurfaceAccessor<float>& sufPre, unsigned int& n);

extern "C" void residual_kernel_extern(CudaSurfaceAccessor<float>& sufRes, CudaSurfaceAccessor<float>& sufPre, CudaSurfaceAccessor<float>& sufDiv, unsigned int& n);

extern "C" void fillzero_kernel_extern(CudaSurfaceAccessor<float> &sufPre, unsigned int &n);

extern "C" void fillzero_kernel_extern_n(CudaSurfaceAccessor<float>& sufPre, unsigned int& n);

extern "C" void prolongate_kernel_extern(CudaSurfaceAccessor<float>& sufPreNext, CudaSurfaceAccessor<float>& sufPre, unsigned int& n);
struct SmokeSim: DisableCopy
{

	unsigned int n;
	std::unique_ptr<CudaSurface<float4>> loc;//
	std::unique_ptr<CudaTexture<float4>> vel;//
	std::unique_ptr<CudaTexture<float4>> velNext;//
	std::unique_ptr<CudaTexture<float4>> clr;//
	std::unique_ptr<CudaTexture<float4>> clrNext;//

	//std::unique_ptr<CudaTexture<float>> tmp;//
	//std::unique_ptr<CudaTexture<float>> tmpNext;//
	std::unique_ptr<CudaSurface<float>> pre;//
	std::unique_ptr<CudaSurface<float>> div;//

	std::vector<std::unique_ptr<CudaSurface<float>>>res;//
	std::vector<std::unique_ptr<CudaSurface<float>>>res2;//
	std::vector<std::unique_ptr<CudaSurface<float>>>err2;//
	std::vector<unsigned int>sizes;//
	std::unique_ptr<CudaSurface<char>> bound;

	explicit SmokeSim(unsigned int _n, unsigned int _n0 = 16) :n(_n)
		, loc(std::make_unique<CudaSurface<float4>>(uint3{ n, n, n }))
		, vel(std::make_unique<CudaTexture<float4>>(uint3{ n, n, n }))
		, velNext(std::make_unique<CudaTexture<float4>>(uint3{ n, n, n }))
		, clr(std::make_unique<CudaTexture<float4>>(uint3{ n, n, n }))
		, clrNext(std::make_unique<CudaTexture<float4>>(uint3{ n,n,n }))
		, pre(std::make_unique<CudaSurface<float>>(uint3{ n,n,n }))
		, bound(std::make_unique<CudaSurface<char>>(uint3{ n, n, n }))
		//, tmp(std::make_unique<CudaTexture<float>>(uint3{ n, n, n }))
		//, tmpNext(std::make_unique<CudaTexture<float>>(uint3{ n, n, n }))
		, div(std::make_unique<CudaSurface<float>>(uint3{ n, n, n }))
	{
		//fillzero_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (pre->accessSurface(), n);
		fillzero_kernel_extern_n(pre->accessSurface(), n);
		unsigned int tn;
		for (tn = n; tn >= _n0; tn /= 2)
		{
			res.push_back(std::make_unique<CudaSurface<float>>(uint3{ tn,tn,tn }));
			res2.push_back(std::make_unique<CudaSurface<float>>(uint3{ tn / 2,tn / 2,tn / 2 }));
			err2.push_back(std::make_unique<CudaSurface<float>>(uint3{ tn / 2,tn / 2,tn / 2 }));
			sizes.push_back(tn);
		}
	}
	void smooth(CudaSurface<float>* v, CudaSurface<float>* f, unsigned int lev, int times = 4)
	{
		//std::cout << lev << std::endl;
		//for (unsigned int a : sizes)
		//{
		//	std::cout << a<<std::endl;
		//}
		//std::cout << sizes.size()<<std::endl;
		unsigned int tn = sizes[lev];
		//std::cout << "ok" << std::endl;
		for (int step = 0; step < times; step++) {
			/*rbgs_kernel<0> << <dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8) >> > (v->accessSurface(), f->accessSurface(), tn);

			rbgs_kernel<1> << <dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8) >> > (v->accessSurface(), f->accessSurface(), tn);*/
			rbgs_kernel_0_extern(v->accessSurface(), f->accessSurface(), tn);
			rbgs_kernel_1_extern(v->accessSurface(), f->accessSurface(), tn);
		}
	}

	void vcycle(unsigned int lev, CudaSurface<float>* v, CudaSurface<float>* f)
	{
		if (lev >= sizes.size()) {
			unsigned int tn = sizes.back() / 2;
			smooth(v, f, 3);
			return;
		}
		auto* r = res[lev].get();
		auto* r2 = res2[lev].get();
		auto* e2 = err2[lev].get();
		unsigned int tn = sizes[lev];
		smooth(v, f, lev);
		//residual_kernel << <dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8) >> > (r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
		 residual_kernel_extern(r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
		//restrict_kernel << <dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8) >> > (r2->accessSurface(), r->accessSurface(), tn / 2);
		 restrict_kernel_extern(r2->accessSurface(), r->accessSurface(), tn);
		//fillzero_kernel << <dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8) >> > (e2->accessSurface(), tn / 2);
		 fillzero_kernel_extern(e2->accessSurface(), tn );

		vcycle(lev + 1, e2, r2);
		//prolongate_kernel << <dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8) >> > (v->accessSurface(), e2->accessSurface(), tn / 2);
		prolongate_kernel_extern(v->accessSurface(), e2->accessSurface(), tn );
		smooth(v, f, lev);
		
	}

	void projection()
	{
		//heatup_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (vel->accessSurface(), tmp->accessSurface(), clr->accessSurface(), bound->accessSurface(), 0.05f, 0.018f, 0.004f, n);

		//divergence_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (vel->accessSurface(), div->accessSurface(), n);
		divergence_kernel_extern(vel->accessSurface(), div->accessSurface(), n);
		for (int i = 0; i < 400; i++)
		{
			//jacobi_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (div->accessSurface(), pre->accessSurface(), pre->accessSurface(), n);
			jacobi_kernel_extern(div->accessSurface(), pre->accessSurface(), pre->accessSurface(), n);
		}
		//subgradient_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (pre->accessSurface(), vel->accessSurface(), bound->accessSurface(), n);
		subgradient_kernel_extern(pre->accessSurface(), vel->accessSurface(), bound->accessSurface(), n);
	}

	void advection()
	{
		//advect_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (vel->accessTexture(), loc->accessSurface(), bound->accessSurface(), n);
		int a = 1;
		
		advect_kernel_extern(vel->accessTexture(), loc->accessSurface(), bound->accessSurface(), n,&a);
		
		//resample_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (loc->accessSurface(), vel->accessTexture(), velNext->accessSurface(), n);
		//resample_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (loc->accessSurface(), clr->accessTexture(), clrNext->accessSurface(), n);

		resample_kernel_extern(loc->accessSurface(), vel->accessTexture(), velNext->accessSurface(), n);
		resample_kernel_extern(loc->accessSurface(), clr->accessTexture(), clrNext->accessSurface(), n);


		std::swap(vel, velNext);
		std::swap(clr, clrNext);
	}

	void step(int times = 16)
	{
		for (int step = 0; step < times; step++)
		{
			//old_projection();
			projection();
			advection();
		}
	}
	float calc_loss()
	{
		//divergence_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (vel->accessSurface(), div->accessSurface(), n);
		divergence_kernel_extern(vel->accessSurface(), div->accessSurface(), n);
		float* sum;
		checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
		//sumloss_kernel << <dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8) >> > (div->accessSurface(), sum, n);
		sumloss_kernel_extern(div->accessSurface(), sum, n);
		auto r = cudaStreamSynchronize(0);
		float cpu;
		checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(sum));
		return cpu;
	}
};
//结构体结束


