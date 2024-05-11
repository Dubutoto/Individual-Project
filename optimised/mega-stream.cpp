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
inline size_t IDX2(size_t i, size_t j, size_t ni) {
    return i + (ni) * (j);
}

inline size_t IDX3(size_t i, size_t j, size_t k, size_t ni, size_t nj) {
    return i + (ni)*IDX2(j, k, nj);
}

inline size_t IDX4(size_t i, size_t j, size_t k, size_t l, size_t ni, size_t nj, size_t nk) {
    return i + (ni)*IDX3(j, k, l, nj, nk);
}

inline size_t IDX5(size_t i, size_t j, size_t k, size_t l, size_t m, size_t ni, size_t nj, size_t nk, size_t nl) {
    return i + (ni)*IDX4(j, k, l, m, nj, nk, nl);
}

inline size_t IDX6(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t ni, size_t nj, size_t nk, size_t nl, size_t nm) {
    return i + (ni)*IDX5(j, k, l, m, n, nj, nk, nl, nm);
}
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

/*
#ifdef __APPLE__
void* aligned_alloc(size_t alignment, size_t size) {
    void* mem = nullptr; 
    posix_memalign(&mem, alignment, size); 
    return mem;
}
#endif 
*/

 //device selector
   sycl::queue queue(sycl::cpu_selector_v);

void kernel_sycl(
    sycl::queue* queue,
    const int Ng, const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
    sycl::buffer<double>* buf_r,
    sycl::buffer<double>* buf_q,
    sycl::buffer<double>* buf_x,
    sycl::buffer<double>* buf_y,
    sycl::buffer<double>* buf_z,
    sycl::buffer<double>* buf_a,
    sycl::buffer<double>* buf_b,
    sycl::buffer<double>* buf_c,
    sycl::buffer<double>* buf_sum
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

    // q, r, x, y, z, a, b, c, sum에 대한 SYCL 버퍼 생성
    sycl::buffer<double> buf_q(VLEN * Nj * Nk * Nl * Nm * Ng);
    sycl::buffer<double> buf_r(VLEN * Nj * Nk * Nl * Nm * Ng);

    sycl::buffer<double> buf_x(VLEN * Nj * Nk * Nm * Ng);
    sycl::buffer<double> buf_y(VLEN * Nj * Nl * Nm * Ng);
    sycl::buffer<double> buf_z(VLEN * Nk * Nl * Nm * Ng);

    sycl::buffer<double> buf_a(VLEN * Ng);
    sycl::buffer<double> buf_b(VLEN * Ng);
    sycl::buffer<double> buf_c(VLEN * Ng);

    sycl::buffer<double> buf_sum(Nj * Nk * Nl * Nm);

   

    queue.submit([&](sycl::handler& handler) {
    auto acc_q = buf_q.get_access<sycl::access::mode::write>(handler);
    auto acc_r = buf_r.get_access<sycl::access::mode::write>(handler);
    
    // 글로벌 변수를 상수로 캡처
    const int local_VLEN = VLEN;
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const int local_Nl = Nl;
    const int local_Ng = Ng;
    const int local_Nm = Nm;
    
    handler.parallel_for<class init_qr>(
        sycl::range<3>{static_cast<size_t>(local_VLEN * local_Nj * local_Nk), static_cast<size_t>(local_Nl), static_cast<size_t>(local_Ng * local_Nm)},
        [=](sycl::id<3> idx) {
            // 이전과 동일한 계산, 하지만 로컬 상수 사용
            int flat = idx.get(0);
            int l = idx.get(1);
            int combined = idx.get(2);
            int v = flat % local_VLEN;
            int j = (flat / local_VLEN) % local_Nj;
            int k = (flat / (local_VLEN * local_Nj)) % local_Nk;
            int g = combined % local_Ng;
            int m = combined / local_Ng;
            acc_q[IDX6(v,j,k,l,g,m,local_VLEN,local_Nj,local_Nk,local_Nl,local_Ng)] = Q_START;
            acc_r[IDX6(v,j,k,l,g,m,local_VLEN,local_Nj,local_Nk,local_Nl,local_Ng)] = R_START;
            }
        );
    });

    // acc_x 초기화 수정
    queue.submit([&](sycl::handler& handler) {
    auto acc_x = buf_x.get_access<sycl::access::mode::write>(handler);
    const int local_VLEN = VLEN;
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const int local_Ng = Ng;
    const int local_Nm = Nm;

    handler.parallel_for<class init_x>(sycl::range<3>(local_VLEN * local_Nj * local_Nk, local_Ng, local_Nm), [=](sycl::id<3> idx) {
        int flat = idx.get(0);
        int g = idx.get(1);
        int m = idx.get(2);
        int v = flat % local_VLEN;
        int j = (flat / local_VLEN) % local_Nj;
        int k = (flat / (local_VLEN * local_Nj)) % local_Nk;
        acc_x[IDX5(v,j,k,g,m,local_VLEN,local_Nj,local_Nk,local_Ng)] = X_START;
    });
});

// acc_y 초기화 수정
queue.submit([&](sycl::handler& handler) {
    auto acc_y = buf_y.get_access<sycl::access::mode::write>(handler);
    const int local_VLEN = VLEN;
    const int local_Nj = Nj;
    const int local_Nl = Nl;
    const int local_Ng = Ng;
    const int local_Nm = Nm;

    handler.parallel_for<class init_y>(sycl::range<3>(local_VLEN * local_Nj * local_Nl, local_Ng, local_Nm), [=](sycl::id<3> idx) {
        int flat = idx.get(0);
        int g = idx.get(1);
        int m = idx.get(2);
        int v = flat % local_VLEN;
        int j = (flat / local_VLEN) % local_Nj;
        int l = (flat / (local_VLEN * local_Nj)) % local_Nl;
        acc_y[IDX5(v,j,l,g,m,local_VLEN,local_Nj,local_Nl,local_Ng)] = Y_START;
    });
});

// acc_z 초기화 수정
queue.submit([&](sycl::handler& handler) {
    auto acc_z = buf_z.get_access<sycl::access::mode::write>(handler);
    const int local_VLEN = VLEN;
    const int local_Nk = Nk;
    const int local_Nl = Nl;
    const int local_Ng = Ng;
    const int local_Nm = Nm;

    handler.parallel_for<class init_z>(sycl::range<3>(local_VLEN * local_Nk * local_Nl, local_Ng, local_Nm), [=](sycl::id<3> idx) {
        int flat = idx.get(0);
        int g = idx.get(1);
        int m = idx.get(2);
        int v = flat % local_VLEN;
        int k = (flat / local_VLEN) % local_Nk;
        int l = (flat / (local_VLEN * local_Nk)) % local_Nl;
        acc_z[IDX5(v,k,l,g,m,local_VLEN,local_Nk,local_Nl,local_Ng)] = Z_START;
    });
});

// acc_a, acc_b, acc_c 초기화 수정
queue.submit([&](sycl::handler& handler) {
    auto acc_a = buf_a.get_access<sycl::access::mode::write>(handler);
    auto acc_b = buf_b.get_access<sycl::access::mode::write>(handler);
    auto acc_c = buf_c.get_access<sycl::access::mode::write>(handler);
    const int local_VLEN = VLEN;
    const int local_Ng = Ng;

    handler.parallel_for<class init_abc>(sycl::range<2>(local_VLEN, local_Ng), [=](sycl::id<2> idx) {
        int v = idx.get(0);
        int g = idx.get(1);
        acc_a[IDX2(v,g,local_VLEN)] = A_START;
        acc_b[IDX2(v,g,local_VLEN)] = B_START;
        acc_c[IDX2(v,g,local_VLEN)] = C_START;
    });
});

// acc_sum 초기화 수정
queue.submit([&](sycl::handler& handler) {
    auto acc_sum = buf_sum.get_access<sycl::access::mode::write>(handler);
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const int local_Nl = Nl;
    const int local_Nm = Nm;

    handler.parallel_for<class init_sum>(sycl::range<3>(local_Nj * local_Nk, local_Nl, local_Nm), [=](sycl::id<3> idx) {
        int flat = idx.get(0);
        int l = idx.get(1);
        int m = idx.get(2);
        int j = flat % local_Nj;
        int k = flat / local_Nj;
        acc_sum[IDX4(j,k,l,m,local_Nj,local_Nk,local_Nl)] = 0.0;
    });
}); /* End of parallel region */
    
    auto begin = std::chrono::high_resolution_clock::now();
    /* Run the kernel multiple times */
    for (int t = 0; t < ntimes; t++) {
        auto tick = std::chrono::high_resolution_clock::now();

       // kernel(Ng, Ni, Nj, Nk, Nl, Nm, r, q, x, y, z, a, b, c, sum);
        kernel_sycl(&queue, Ng, Ni, Nj, Nk, Nl, Nm, &buf_r, &buf_q, &buf_x, &buf_y, &buf_z, &buf_a, &buf_b, &buf_c, &buf_sum);
        /* Swap the pointers */
        //double *tmp = buf_q; buf_q = buf_r; buf_r = tmp;
        auto tmp = std::move(buf_r);
        buf_r = std::move(buf_q);
        buf_q = std::move(tmp);

        auto tock = std::chrono::high_resolution_clock::now();
        timings[t] = std::chrono::duration<double>(tock - tick).count();

    }

    auto end = std::chrono::high_resolution_clock::now();

    /* Check the results - total of the sum array */
    double total = 0.0;

    {
    // 결과를 저장할 버퍼 생성
    sycl::buffer<double, 1> buf_total(&total, sycl::range<1>(1));
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const int local_Nl = Nl;
    const int local_Nm = Nm;
        // 큐에 커맨드 그룹 제출
        queue.submit([&](sycl::handler& cgh) {
            // 읽기 접근자
            auto acc_sum = buf_sum.get_access<sycl::access::mode::read>(cgh);
            // 쓰기 접근자
            auto acc_total = buf_total.get_access<sycl::access::mode::write>(cgh);

            // 커널 실행
            cgh.single_task([=]() {
                double local_total = 0.0;
                for (int i = 0; i < local_Nj*local_Nk*local_Nl*local_Nm; i++) {
                    local_total += acc_sum[i];
                }
                acc_total[0] = local_total;
            });
        });
    } // 버퍼가 범위를 벗어나면 데이터는 자동으로 복사됩니다.

    // 커맨드 큐의 모든 작업이 완료되기를 기다림
    queue.wait();

    // 결과 출력
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

    queue.wait();

    return EXIT_SUCCESS;
}

/**************************************************************************
 * Kernel
 *************************************************************************/

void kernel_sycl(
    sycl::queue* queue,
    const int Ng, const int Ni, const int Nj, const int Nk, const int Nl, const int Nm,
    sycl::buffer<double>* buf_r,
    sycl::buffer<double>* buf_q,
    sycl::buffer<double>* buf_x,
    sycl::buffer<double>* buf_y,
    sycl::buffer<double>* buf_z,
    sycl::buffer<double>* buf_a,
    sycl::buffer<double>* buf_b,
    sycl::buffer<double>* buf_c,
    sycl::buffer<double>* buf_sum
) {
    // 커널 실행
    queue->submit([&](sycl::handler& h) {
        auto r = buf_r->get_access<sycl::access::mode::read_write>(h);
        auto q = buf_q->get_access<sycl::access::mode::read>(h);
        auto x = buf_x->get_access<sycl::access::mode::read_write>(h);
        auto y = buf_y->get_access<sycl::access::mode::read_write>(h);
        auto z = buf_z->get_access<sycl::access::mode::read_write>(h);
        auto a = buf_a->get_access<sycl::access::mode::read>(h);
        auto b = buf_b->get_access<sycl::access::mode::read>(h);
        auto c = buf_c->get_access<sycl::access::mode::read>(h);
        auto sum = buf_sum->get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(Nm * Ng * Nl * Nk * Nj), [=](sycl::id<1> id) {
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

