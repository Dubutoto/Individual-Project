/*
 
  Copyright 2016 Tom Deakin, University of Bristol
 
  This file is part of mega-stream.
 
  mega-stream is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
 
  mega-stream is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License
  along with mega-stream.  If not, see <http://www.gnu.org/licenses/>.
 
 
  This aims to investigate the limiting factor for a simple kernel, in particular
  where bandwidth limits not to be reached, and latency becomes a dominating factor.
 
*/
 
#define VERSION "2.0"
 
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include <sycl/sycl.hpp>
 
// Index functions for SYCL
#define IDX2(i,j,ni) ((i)+(ni)*(j))
#define IDX3(i,j,k,ni,nj) ((i)+(ni)*IDX2((j),(k),(nj)))
#define IDX4(i,j,k,l,ni,nj,nk) ((i)+(ni)*IDX3((j),(k),(l),(nj),(nk)))
#define IDX5(i,j,k,l,m,ni,nj,nk,nl) ((i)+(ni)*IDX4((j),(k),(l),(m),(nj),(nk),(nl)))
#define IDX6(i,j,k,l,m,n,ni,nj,nk,nl,nm) ((i)+(ni)*IDX5((j),(k),(l),(m),(n),(nj),(nk),(nl),(nm)))
/*
  Arrays are defined in terms of 3 sizes: inner, middle and outer.
  The large arrays are of size inner*middle*middle*middle*outer.
  The medium arrays are of size inner*middle*middle*outer.
  The small arrays are of size inner and are indexed with 1 index.
 
*/
 
const int OUTER = 64; // 2^6
const int MIDDLE = 16; // 2^4
const int INNER = 128; // 2^7
const int VLEN = 8;
 
static_assert((VLEN > 0) && ((VLEN & (VLEN-1)) == 0), "VLEN must be a power of 2.");
 
/* Default alignment of 2 MB page boundaries */
#define ALIGNMENT 2*1024*1024
 
/* Tollerance with which to check final array values */
#define TOLR 1.0E-15
 
/* Starting values */
#define R_START 0.0
#define Q_START 0.01
#define X_START 0.02
#define Y_START 0.03
#define Z_START 0.04
#define A_START 0.06
#define B_START 0.07
#define C_START 0.08
 
/*
#ifdef __APPLE__
void* aligned_alloc(size_t alignment, size_t size) {
    void* mem = nullptr;
    posix_memalign(&mem, alignment, size);
    return mem;
}
#endif
*/
 
 
 
void kernel(
    sycl::queue* queue,
    const int Ng,
    const int Ni,
    const int Nj,
    const int Nk,
    const int Nl,
    const int Nm,
    double*  __restrict r,
    const double*  __restrict q,
    double*  __restrict x,
    double*  __restrict y,
    double*  __restrict z,
    const double* __restrict a,
    const double* __restrict b,
    const double* __restrict c,
    double*  __restrict sum
);
 
void initializeQandR(
    sycl::queue* queue,
    double* __restrict q,
    double* __restrict r,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nk,
    const int Nj,
    const int VLEN
);
 
void initializeX(
    sycl::queue* queue,
    double* __restrict x,
    const int Nm,
    const int Ng,
    const int Nk,
    const int Nj,
    const int VLEN
);
 
void initializeY(
    sycl::queue* queue,
    double* __restrict y,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nj,
    const int VLEN
);
 
void initializeZ(
    sycl::queue* queue,
    double* __restrict z,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nk,
    const int VLEN
);
 
void initializeABC(
    sycl::queue* queue,
    double* __restrict a,
    double* __restrict b,
    double* __restrict c,
    const int Ng,
    const int VLEN
);
 
void initializeSum(
    sycl::queue* queue,
    double* __restrict sum,
    const int Nm,
    const int Nl,
    const int Nk,
    const int Nj
);
 
void parse_args(int argc, char *argv[]);
 
// Default strides
int Ni = INNER;
int Nj = MIDDLE;
int Nk = MIDDLE;
int Nl = MIDDLE;
int Nm = OUTER;
int Ng;
 
/* Number of iterations to run benchmark */
int ntimes = 1000;
 
 
 
int main(int argc, char *argv[])
{
    std::cout << "MEGA-STREAM! - v" << VERSION << "\n\n";
 
    //device selector
     std::cout << "큐선언 시작 " "\n";
    sycl::queue queue(sycl::cpu_selector_v);
      std::cout << "큐선언 완료" "\n";
 //(sycl::gpu_selector_v)
    //device detecting code
    /*
    auto platforms = sycl::platform::get_platforms();
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
        auto devices = platform.get_devices();
        for (const auto& device : devices) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        }
    }
    */
   
    parse_args(argc, argv);
 
    std::cout << "Small arrays:  " << Ni << " elements\t\t"
              << (Ni * sizeof(double) * 1.0E-3) << " KB\n";
    std::cout << "Medium arrays: " << Ni << " x " << Nj << " x " << Nj << " x " << Nm
              << " elements\t" << (Ni * Nj * Nj * Nm * sizeof(double) * 1.0E-6) << " MB\n";
    std::cout << "Large arrays:  " << Ni << " x " << Nj << " x " << Nj << " x " << Nj << " x " << Nm
              << " elements\t" << (Ni * Nj * Nj * Nj * Nm * sizeof(double) * 1.0E-6) << " MB\n";
 
    const double footprint = (double)sizeof(double) * 1.0E-6 * (
            2.0*Ni*Nj*Nk*Nl*Nm +  // r, q
            Ni*Nj*Nk*Nm +         // x
            Ni*Nj*Nl*Nm +         // y
            Ni*Nk*Nl*Nm +         // z
            3.0*Ni +              // a, b, c
            Nj*Nk*Nl*Nm           // sum
    );
   
    std::cout << "Memory footprint: " << footprint << " MB\n";
 
    /* Total memory moved (in the best case) - the arrays plus an extra sum as update is += */
    const double moved = (double)sizeof(double) * 1.0E-6 * (
        Ni*Nj*Nk*Nl*Nm  + // read q
        Ni*Nj*Nk*Nl*Nm  + // write r
        Ni+Ni+Ni        + // read a, b and c
        2.0*Ni*Nj*Nk*Nm + // read and write x
        2.0*Ni*Nj*Nl*Nm + // read and write y
        2.0*Ni*Nk*Nl*Nm + // read and write z
        2.0*Nj*Nk*Nl*Nm   // read and write sum
    );
 
    /* Split inner-most dimension into VLEN-sized chunks */
    Ng = ((Ni + (VLEN-1)) & ~(VLEN-1)) / VLEN;
    std::cout << "Inner dimension split into " << Ng << " chunks of size " << VLEN << "\n";
    std::cout << "Running " << ntimes << " times\n\n";
 
    std::vector<double> timings(ntimes);
  // std::cout << "메모리할당 시작 " "\n";
 
    double* q = sycl::malloc_shared<double>(VLEN * Nj * Nk * Nl * Nm * Ng, queue);
    double* r = sycl::malloc_shared<double>(VLEN * Nj * Nk * Nl * Nm * Ng, queue);
 
    double* x = sycl::malloc_shared<double>(VLEN * Nj * Nk * Nm * Ng, queue);
    double* y = sycl::malloc_shared<double>(VLEN * Nj * Nk * Nm * Ng, queue);
    double* z = sycl::malloc_shared<double>(VLEN * Nj * Nk * Nm * Ng, queue);
 
    double* a = sycl::malloc_shared<double>(VLEN * Ng, queue);
    double* b = sycl::malloc_shared<double>(VLEN * Ng, queue);
    double* c = sycl::malloc_shared<double>(VLEN * Ng, queue);
 
    double* sum = sycl::malloc_shared<double>(Nj * Nk * Nl * Nm, queue);
     // std::cout << "메모리할당 완료 " "\n";
 
   
    initializeQandR(&queue, q, r, Nm, Ng, Nl, Nk, Nj, VLEN);
    initializeX(&queue, x, Nm, Ng, Nk, Nj, VLEN);
    initializeY(&queue, y, Nm, Ng, Nl, Nj, VLEN);
    initializeZ(&queue, z, Nm, Ng, Nl, Nk, VLEN);
    initializeABC(&queue, a, b, c, Ng, VLEN);
    initializeSum(&queue, sum, Nm, Nl, Nk, Nj);
    queue.wait();
 
  std::cout << "Initializaition complete " "\n";
    auto begin = std::chrono::high_resolution_clock::now();
    /* Run the kernel multiple times */
    for (int t = 0; t < ntimes; t++) {
        auto tick = std::chrono::high_resolution_clock::now();
 
        kernel(&queue, Ng, Ni, Nj, Nk, Nl, Nm, r, q, x, y, z, a, b, c, sum);
       
       /* Swap the pointers */
        double* tmp = q;  q = r; r = tmp;      
       
        auto tock = std::chrono::high_resolution_clock::now();
        timings[t] = std::chrono::duration<double>(tock - tick).count();
 
    }
     
    auto end = std::chrono::high_resolution_clock::now();
 
    /* Check the results - total of the sum array */
    double total = 0.0;
    for (int i = 0; i < Nj * Nk * Nl * Nm; i++) {
        total += sum[i];
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sum total: " << total << "\n";
 
    /* Print timings */
    double min = DBL_MAX;
    double max = 0.0;
    double avg = 0.0;
    for (int t = 1; t < ntimes; t++) {
        min = std::min(min, timings[t]);
        max = std::max(max, timings[t]);
        avg += timings[t];
    }
    avg /= (double)(ntimes - 1);
 
    std::cout << "\n";
    std::cout << "Bandwidth MB/s  Min time    Max time    Avg time\n";
    std::cout << std::fixed;
    std::cout << std::setprecision(1) << moved/min << "        "
              << std::setprecision(6) << min << "     "
              << max << "     " << avg << "\n";
    std::cout << "Total time: " << std::setprecision(6) << (std::chrono::duration<double>(end - begin).count()) << "\n";
// 왜 있는지 확인
    sycl::free(q, queue);
    sycl::free(r, queue);
    sycl::free(x, queue);
    sycl::free(y, queue);
    sycl::free(z, queue);
    sycl::free(a, queue);
    sycl::free(b, queue);
    sycl::free(c, queue);
    sycl::free(sum, queue);
    return EXIT_SUCCESS;
}
/**************************************************************************
* Initialization Fucntions
*************************************************************************/
 
void initializeQandR(
    sycl::queue* queue,
    double* __restrict q,
    double* __restrict r,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nk,
    const int Nj,
    const int VLEN
) {
    queue->parallel_for(
        sycl::range<3>{
            static_cast<size_t>(VLEN * Nj * Nk),
            static_cast<size_t>(Nl),
            static_cast<size_t>(Ng * Nm)
        },
        [=](sycl::id<3> idx) {
            int flat = idx.get(0);
            int l = idx.get(1);
            int combined = idx.get(2);
            int v = flat % VLEN;
            int j = (flat / VLEN) % Nj;
            int k = (flat / (VLEN * Nj)) % Nk;
            int g = combined % Ng;
            int m = combined / Ng;
            q[IDX6(v, j, k, l, g, m, VLEN, Nj, Nk, Nl, Ng)] = Q_START;
            r[IDX6(v, j, k, l, g, m, VLEN, Nj, Nk, Nl, Ng)] = R_START;
        }
    );
}
 
void initializeX(
    sycl::queue* queue,
    double* __restrict x,
    const int Nm,
    const int Ng,
    const int Nk,
    const int Nj,
    const int VLEN
) {
    queue->parallel_for(
        sycl::range<3>(VLEN * Nj * Nk, Ng, Nm),
        [=](sycl::id<3> idx) {
            int flat = idx.get(0);
            int g = idx.get(1);
            int m = idx.get(2);
            int v = flat % VLEN;
            int j = (flat / VLEN) % Nj;
            int k = (flat / (VLEN * Nj)) % Nk;
            x[IDX5(v, j, k, g, m, VLEN, Nj, Nk, Ng)] = X_START;
        }
    );
}
 
void initializeY(
    sycl::queue* queue,
    double* __restrict y,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nj,
    const int VLEN
) {
    queue->parallel_for(
        sycl::range<3>(VLEN * Nj * Nl, Ng, Nm),
        [=](sycl::id<3> idx) {
            int flat = idx.get(0);
            int g = idx.get(1);
            int m = idx.get(2);
            int v = flat % VLEN;
            int j = (flat / VLEN) % Nj;
            int l = (flat / (VLEN * Nj)) % Nl;
            y[IDX5(v, j, l, g, m, VLEN, Nj, Nl, Ng)] = Y_START;
        }
    );
}
 
void initializeZ(
    sycl::queue* queue,
    double* __restrict z,
    const int Nm,
    const int Ng,
    const int Nl,
    const int Nk,
    const int VLEN
) {
    queue->parallel_for(
        sycl::range<3>(VLEN * Nk * Nl, Ng, Nm),
        [=](sycl::id<3> idx) {
            int flat = idx.get(0);
            int g = idx.get(1);
            int m = idx.get(2);
            int v = flat % VLEN;
            int k = (flat / VLEN) % Nk;
            int l = (flat / (VLEN * Nk)) % Nl;
            z[IDX5(v, k, l, g, m, VLEN, Nk, Nl, Ng)] = Z_START;
        }
    );
}
 
void initializeABC(
    sycl::queue* queue,
    double* __restrict a,
    double* __restrict b,
    double* __restrict c,
    const int Ng,
    const int VLEN
) {
    queue->parallel_for(
        sycl::range<2>(VLEN, Ng),
        [=](sycl::id<2> idx) {
            int v = idx.get(0);
            int g = idx.get(1);
            a[IDX2(v, g, VLEN)] = A_START;
            b[IDX2(v, g, VLEN)] = B_START;
            c[IDX2(v, g, VLEN)] = C_START;
        }
    );
}
 
void initializeSum(
    sycl::queue* queue,
    double* __restrict sum,
    const int Nm,
    const int Nl,
    const int Nk,
    const int Nj
) {
    queue->parallel_for(
        sycl::range<1>(Nj * Nk * Nl * Nm), [=](sycl::id<1> idx) {
            int index = idx[0];
            int j = index % Nj;
            index /= Nj;
            int k = index % Nk;
            index /= Nk;
            int l = index % Nl;
            int m = index / Nl;
            sum[IDX4(j, k, l, m, Nj, Nk, Nl)] = 0.0;
        }
    );
}
 
/**************************************************************************
* Kernel
*************************************************************************/
 
void kernel(
    sycl::queue* queue,
    const int Ng,
    const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
    double* __restrict r,
    const double*  __restrict q,
    double*  __restrict x,
    double*  __restrict y,
    double*  __restrict z,
    const double*  __restrict a,
    const double*  __restrict b,
    const double*  __restrict c,
    double* __restrict sum
) {
    queue->parallel_for(sycl::range<1>(Nm * Ng * Nl * Nk * Nj), [=](sycl::id<1> id) {
            int idx = id[0];
            int m = idx / (Ng * Nl * Nk * Nj);
            int rem = idx % (Ng * Nl * Nk * Nj);
            int g = rem / (Nl * Nk * Nj);
            rem = rem % (Nl * Nk * Nj);
            int l = rem / (Nk * Nj);
            rem = rem % (Nk * Nj);
            int k = rem / Nj;
            int j = rem % Nj;
 
            double total = 0.0;
            for (int v = 0; v < VLEN; v++) {
                int global_idx = ((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN + v;
                int x_idx = (((m * Ng + g) * Nk + k) * Nj + j) * VLEN + v;
                int y_idx = (((m * Ng + g) * Nl + l) * Nj + j) * VLEN + v;
                int z_idx = (((m * Ng + g) * Nl + l) * Nk + k) * VLEN + v;
                int a_b_c_idx = g * VLEN + v;
 
                r[global_idx] =
                    q[global_idx] +
                    a[a_b_c_idx] * x[x_idx] +
                    b[a_b_c_idx] * y[y_idx] +
                    c[a_b_c_idx] * z[z_idx];
 
                x[x_idx] = 0.2 * r[global_idx] - x[x_idx];
                y[y_idx] = 0.2 * r[global_idx] - y[y_idx];
                z[z_idx] = 0.2 * r[global_idx] - z[z_idx];
 
                total += r[global_idx];
            }
            sum[((m * Nl + l) * Nk + k) * Nj + j] += total;
        });
       
         queue->wait();
}
 
void parse_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--outer") == 0)
        {
            Nm = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--inner") == 0)
        {
            Ni = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--middle") == 0)
        {
            int num = std::atoi(argv[++i]);
            Nj = num;
            Nk = num;
            Nl = num;
        }
        else if (strcmp(argv[i], "--Nj") == 0)
        {
            Nj = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--ntimes") == 0)
        {
            ntimes = std::atoi(argv[++i]);
            if (ntimes < 2)
            {
                std::cerr << "ntimes must be 2 or greater\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [OPTION]\n";
            std::cout << "\t --outer  n \tSet size of outer dimension\n";
            std::cout << "\t --inner  n \tSet size of middle dimensions\n";
            std::cout << "\t --middle n \tSet size of inner dimension\n";
            std::cout << "\t --Nj     n \tSet size of the j dimension\n";
            std::cout << "\t --ntimes n\tRun the benchmark n times\n";
            std::cout << "\n";
            std::cout << "\t Outer   is " << OUTER << " elements\n";
            std::cout << "\t Middle are " << MIDDLE << " elements\n";
            std::cout << "\t Inner   is " << INNER << " elements\n";
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unrecognised argument \"" << argv[i] << "\"\n";
            std::exit(EXIT_FAILURE);
        }
    }
}