#include "CudaSurfaceAccessor.h"
#include "CudaTextureAccessor.h"
#include <device_launch_parameters.h>
#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <helper_math.h>
#include "VDB_Write.h"
#include "SomkeSim.h"

#include <fstream>
#include <filesystem>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/openvdb.h>


int main()
{
	unsigned int n = 128;
	SmokeSim sim(n);

	{
		std::vector<float4> cpu(n * n * n);
		for (int z = 0; z < n; z++) {
			for (int y = 0; y < n; y++) {
				for (int x = 0; x < n; x++) {
					//float den = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
					float den = std::hypot(std::hypot(x - (int)n / 2, y - (int)n / 2), z - (int)n / 4) < n / 12 ? 1.f : 0.f;
					cpu[x + n * (y + n * z)] = make_float4(den, 0.f, 0.f, 0.f);
				}
			}
		}
		sim.clr->copyIn(cpu.data());


	}

	{
		std::vector<float4> cpu(n * n * n);
		for (int z = 0; z < n; z++) {
			for (int y = 0; y < n; y++) {
				for (int x = 0; x < n; x++) {
					//float tmp = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
					float tmp = std::hypot(std::hypot(x - (int)n / 2, y - (int)n / 2), z - (int)n / 4) < n / 12 ? 1.f : 0.f;
					cpu[x + n * (y + n * z)] = make_float4(tmp, 0.f, 0.f, 0.f);
				}
			}
		}
		sim.vel->copyIn(cpu.data());
		//SmokeSimvel_Extern(sim,cpu);
	}


	std::vector<std::shared_ptr<std::thread>> tpool;
	for (int frame = 1; frame <= 200; frame++)
	{
		std::vector<float4> cpuClr(n * n * n);
		sim.clr->copyOut(cpuClr.data());
		tpool.emplace_back(std::make_shared<std::thread>([cpuClr = std::move(cpuClr), frame, n]
			{
				writevdb<float, 1>("C:\\vs_project\\CudaRuntime2\\data\\" + std::to_string(1000 + frame).substr(1) + ".vdb", cpuClr.data(), n, n, n, sizeof(float4));
			}));

			printf("frame=%d, loss=%f\n", frame, sim.calc_loss());
			sim.step();
			//SmokeSim_step_Extern(sim);
	}

	for (auto& t : tpool)
		t->join();

}