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
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>


//Q
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

#ifndef VLEN
#define VLEN 8
#endif
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

#ifdef __APPLE__
void* aligned_alloc(size_t alignment, size_t size) {
    void* mem = nullptr; // C++에서 널 포인터로 nullptr 사용
    posix_memalign(&mem, alignment, size); // posix_memalign 함수는 그대로 사용
    return mem;
}
#endif // 미리 처리 지시문 종료

// Need to change the '__restrict' depends on compiler

void kernel(
    const int Ng, const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
    double* __restrict r,
    const double* __restrict q,
    double* __restrict x,
    double* __restrict y,
    double* __restrict z,
    const double* __restrict a,
    const double* __restrict b,
    const double* __restrict c,
    double* __restrict sum
);

/*
void kernel(
        const int Ng,
        const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
        double (* __restrict r)[Ng][Nl][Nk][Nj][VLEN],
        const double (* __restrict q)[Ng][Nl][Nk][Nj][VLEN],
        double (* __restrict x)[Ng][Nk][Nj][VLEN],
        double (* __restrict y)[Ng][Nl][Nj][VLEN],
        double (* __restrict z)[Ng][Nl][Nk][VLEN],
        const double (* __restrict a)[VLEN],
        const double (* __restrict b)[VLEN],
        const double (* __restrict c)[VLEN],
        double (* __restrict sum)[Nl][Nk][Nj]
); 
*/





void parse_args(int argc, char *argv[]);

// Default strides
int Ni = INNER;
int Nj = MIDDLE;
int Nk = MIDDLE;
int Nl = MIDDLE;
int Nm = OUTER;
int Ng;

/* Number of iterations to run benchmark */
int ntimes = 100;


int main(int argc, char *argv[])
{
    std::cout << "MEGA-STREAM! - v" << VERSION << "\n\n";

    parse_args(argc, argv);

    std::cout << "Small arrays:  " << Ni << " elements\t\t"
              << (Ni * sizeof(double) * 1.0E-3) << " KB\n";
    std::cout << "Medium arrays: " << Ni << " x " << Nj << " x " << Nj << " x " << Nm
              << " elements\t" << (Ni * Nj * Nj * Nm * sizeof(double) * 1.0E-6) << " MB\n";
    std::cout << "Large arrays:  " << Ni << " x " << Nj << " x " << Nj << " x " << Nj << " x " << Nm
              << " elements\t" << (Ni * Nj * Nj * Nj * Nm * sizeof(double) * 1.0E-6) << " MB\n";

    const double footprint = (double)sizeof(double) * 1.0E-6 * (
            2.0 * Ni * Nj * Nk * Nl * Nm +  // r, q
            Ni * Nj * Nk * Nm +             // x
            Ni * Nj * Nl * Nm +             // y
            Ni * Nk * Nl * Nm +             // z
            3.0 * Ni +                      // a, b, c
            Nj * Nk * Nl * Nm               // sum
    );
    std::cout << "Memory footprint: " << footprint << " MB\n";

    /* Total memory moved (in the best case) - the arrays plus an extra sum as update is += */
    const double moved = (double)sizeof(double) * 1.0E-6 * (
            Ni * Nj * Nk * Nl * Nm  + // read q
            Ni * Nj * Nk * Nl * Nm  + // write r
            Ni + Ni + Ni            + // read a, b and c
            2.0 * Ni * Nj * Nk * Nm + // read and write x
            2.0 * Ni * Nj * Nl * Nm + // read and write y
            2.0 * Ni * Nk * Nl * Nm + // read and write z
            2.0 * Nj * Nk * Nl * Nm   // read and write sum
    );
    /* Split inner-most dimension into VLEN-sized chunks */
    Ng = ((Ni + (VLEN-1)) & ~(VLEN-1)) / VLEN;
    std::cout << "Inner dimension split into " << Ng << " chunks of size " << VLEN << "\n";
    std::cout << "Running " << ntimes << " times\n\n";

    double timings[ntimes];

    double *q = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double) * VLEN * Nj * Nk * Nl * Nm * Ng));
    double *r = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double) * VLEN * Nj * Nk * Nl * Nm * Ng));

    double *x = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Nj*Nk*Nm*Ng));
    double *y = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Nj*Nl*Nm*Ng));
    double *z = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Nk*Nl*Nm*Ng));

    double *a = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Ng));
    double *b = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Ng));
    double *c = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*VLEN*Ng));

    double *sum = static_cast<double *>(aligned_alloc(ALIGNMENT, sizeof(double)*Nj*Nk*Nl*Nm));

#include <CL/sycl.hpp>
    using namespace cl::sycl;

// SYCL 큐 생성
    queue q;

// SYCL 버퍼 생성
    buffer<float, 1> q_buf(VLEN * Nj * Nk * Nl * Ng * Nm);
    buffer<float, 1> r_buf(VLEN * Nj * Nk * Nl * Ng * Nm);
    buffer<float, 1> x_buf(VLEN * Nj * Nk * Ng * Nm);
    buffer<float, 1> y_buf(VLEN * Nj * Nl * Ng * Nm);
    buffer<float, 1> z_buf(VLEN * Nk * Nl * Ng * Nm);
    buffer<float, 1> a_buf(VLEN * Ng);
    buffer<float, 1> b_buf(VLEN * Ng);
    buffer<float, 1> c_buf(VLEN * Ng);
    buffer<float, 1> sum_buf(Nj * Nk * Nl * Nm);

// 버퍼를 사용하여 데이터를 초기화하는 커널 제출
    q.submit([&](handler& h) {
        // 액세서(accessor) 생성
        auto q_acc = q_buf.get_access<access::mode::write>(h);
        auto r_acc = r_buf.get_access<access::mode::write>(h);
        auto x_acc = x_buf.get_access<access::mode::write>(h);
        auto y_acc = y_buf.get_access<access::mode::write>(h);
        auto z_acc = z_buf.get_access<access::mode::write>(h);
        auto a_acc = a_buf.get_access<access::mode::write>(h);
        auto b_acc = b_buf.get_access<access::mode::write>(h);
        auto c_acc = c_buf.get_access<access::mode::write>(h);
        auto sum_acc = sum_buf.get_access<access::mode::write>(h);

        // 병렬 실행 구조 정의
        h.parallel_for<class init_kernel>(range<1>(VLEN * Nj * Nk * Nl * Ng * Nm), [=](id<1> idx) {
            // 여기서 idx는 전체 1차원 인덱스, 다차원 배열 인덱스로 변환 필요
            int m = idx % Nm;
            int g = (idx / Nm) % Ng;
            int l = (idx / (Nm * Ng)) % Nl;
            int k = (idx / (Nm * Ng * Nl)) % Nk;
            int j = (idx / (Nm * Ng * Nl * Nk)) % Nj;
            int v = idx / (Nm * Ng * Nl * Nk * Nj);

            // 데이터 초기화
            q_acc[idx] = Q_START;
            r_acc[idx] = R_START;

            if (v < VLEN) { // VLEN에 맞추어 조건을 설정
                if (k < Nk && j < Nj && g < Ng && m < Nm) x_acc[idx] = X_START;
                if (l < Nl && j < Nj && g < Ng && m < Nm) y_acc[idx] = Y_START;
                if (l < Nl && k < Nk && g < Ng && m < Nm) z_acc[idx] = Z_START;
                if (g < Ng) {
                    a_acc[g * VLEN + v] = A_START;
                    b_acc[g * VLEN + v] = B_START;
                    c_acc[g * VLEN + v] = C_START;
                }
                if (j < Nj && k < Nk && l < Nl && m < Nm) sum_acc[(m * Nl + l) * Nk * Nj + k * Nj + j] = 0.0;
            }
        });
    }).wait(); // 모든 작업이 완료될 때까지 기다림


    double begin = omp_get_wtime();
    /* Run the kernel multiple times */
    for (int t = 0; t < ntimes; t++) {
        double tick = omp_get_wtime();


        kernel(Ng, Ni, Nj, Nk, Nl, Nm, r, q, x, y, z, a, b, c, sum);

        /* Swap the pointers */
        double *tmp = q; q = r; r = tmp;

        double tock = omp_get_wtime();
        timings[t] = tock-tick;

    }

    double end = omp_get_wtime();

    /* Check the results - total of the sum array */
    double total = 0.0;
    for (int i = 0; i < Nj * Nk * Nl * Nm; i++) 
        total += sum[i];
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
    std::cout << "Total time: " << std::setprecision(6) << (end - begin) << "\n";

    /* Free memory */
    free(q);
    free(r);
    free(x);
    free(y);
    free(z);
    free(a);
    free(b);
    free(c);
    free(sum);

    return EXIT_SUCCESS;
}

/**************************************************************************
 * Kernel
 *************************************************************************/
#include <immintrin.h>
void kernel(
    const int Ng, 
    const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
    double* __restrict r,
    const double* __restrict q,
    double* __restrict x,
    double* __restrict y,
    double* __restrict z,
    const double* __restrict a,
    const double* __restrict b,
    const double* __restrict c,
    double* __restrict sum
) {
    #pragma omp parallel for
    for (int m = 0; m < Nm; m++) {
        for (int g = 0; g < Ng; g++) {
            for (int l = 0; l < Nl; l++) {
                for (int k = 0; k < Nk; k++) {
                    for (int j = 0; j < Nj; j++) {
                        double total = 0.0;
                        _mm_prefetch((const char *)&q[((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN] + 32, _MM_HINT_T1);
                        #pragma vector nontemporal(r)
                        #pragma omp simd reduction(+:total) aligned(a, b, c, x, y, z, r, q:64)
                        for (int v = 0; v < VLEN; v++) {
                            // Set r
                            r[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v] =
                                q[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v] +
                                a[g * VLEN + v] * x[(((m * Ng + g) * Nk + k) * Nj + j) * VLEN + v] +
                                b[g * VLEN + v] * y[(((m * Ng + g) * Nl + l) * Nj + j) * VLEN + v] +
                                c[g * VLEN + v] * z[(((m * Ng + g) * Nl + l) * Nk + k) * VLEN + v];

                            // Update x, y and z
                            x[(((m * Ng + g) * Nk + k) * Nj + j) * VLEN + v] = 0.2 * r[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v] - x[(((m * Ng + g) * Nk + k) * Nj + j) * VLEN + v];
                            y[(((m * Ng + g) * Nl + l) * Nj + j) * VLEN + v] = 0.2 * r[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v] - y[(((m * Ng + g) * Nl + l) * Nj + j) * VLEN + v];
                            z[(((m * Ng + g) * Nl + l) * Nk + k) * VLEN + v] = 0.2 * r[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v] - z[(((m * Ng + g) * Nl + l) * Nk + k) * VLEN + v];

                            // Reduce over Ni
                            total += r[(((((m * Ng + g) * Nl + l) * Nk + k) * Nj + j) * VLEN) + v];
                        }

                        // Update sum
                        sum[((m * Nl + l) * Nk + k) * Nj + j] += total;
                    }
                } // Nk 
            } // Nl 
        } // Ng 
    } // Nm 
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

