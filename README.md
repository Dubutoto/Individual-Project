# Exploring HPC algorithms in SYCL
 This project would explore how we could make interesting and challenging parallel algorithms, such as transport sweeps, to the latest heterogeneous parallel programming APIs in SYCL. The project would explore how to map the parallelism the mega-stream/mega-sweep benchmark codes to SYCL, and analyse the performance of the code across different HPC-optimised CPU and GPU architectures.
 ![poster](https://github.com/user-attachments/assets/8542b3ae-a9e6-40ba-97c9-c4d79f30b0a1)


# mega-stream

The **mega-stream** mini-app initially aimed to test the theory that streaming many arrays (with different sizes) causes memory bandwidth limits not to be reached, resulting in latency becoming a dominant factor. We ran a kernel with a similar form to STREAM Triad, but with more than 3 input arrays. Additionally, we run a small reduction, requiring results of the Triad-style computation.

This was then extended to also include a finite difference style update on the medium-sized arrays.

## The Kernel

The main kernel consists of 8 multi-dimensional arrays with the following properties:

- **r** and **q** are large
- **x**, **y**, and **z** are medium
- **a**, **b**, and **c** are small

The computational kernel is found inside a triple-nested loop and can be expressed as:

```plaintext
r(i,j,k,l,m) = q(i,j,k,l,m) + a(i)*x(i,j,k,m) + b(i)*y(i,j,l,m) + c(i)*z(i,k,l,m)
x(i,j,k,m) = 0.2*r(i,j,k,l,m) - x(i,j,k,m)
y(i,j,l,m) = 0.2*r(i,j,k,l,m) - y(i,j,l,m)
z(i,k,l,m) = 0.2*r(i,j,k,l,m) - z(i,k,l,m)
sum(j,k,l,m) = SUM(r(:,j,k,l,m))

Building
The benchmark should build with make, and by default uses the Intel compiler. This can be changed by specifying CC, for example make CC=cc. Additional options can be passed to the Makefile as make OPTIONS=.

Notes
The Fortran version does not have any command line argument checking. A baseline and an optimised version are kept in this repository.
