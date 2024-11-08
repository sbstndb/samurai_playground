// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include "CLI/CLI.hpp"

#include "samurai/mr/adapt.hpp"
#include "samurai/mr/mesh.hpp"
#include "samurai/schemes/fv.hpp"
#include "samurai/samurai.hpp"


#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"



#include <chrono>

#include <filesystem>
namespace fs = std::filesystem;


[[gnu::noinline]] void compute_raw(double* d_y, double a, double *d_x, double *d_b, std::size_t size ){
    for (int i = 0 ; i < size ; i++){
            d_y[i] = a * d_x[i] + d_b[i] ;

    }
}




int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);
    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Add@sbstndbs -------------------------" << std::endl;
    //--------------------//
    // Program parameters //
    //--------------------//
    double left_box  = -1;
    double right_box = 1;

    //min_level == max_level in this example
//    std::size_t level = 10;
    std::size_t level = 11 ; 

    double a = 2.0 ; 

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, level, level};

    auto size = mesh.nb_cells();
    std::cout << "Size : " << size << std::endl ; 


    // allocate raw pointers
    double* d_x = (double*)malloc(size * sizeof(double)) ; 
    double* d_b = (double*)malloc(size * sizeof(double)) ;
    double* d_y = (double*)malloc(size * sizeof(double)) ;

    // init raw pointer
    for (int i = 0 ; i < size ; i++){
	d_x[i] = 1.0 ; 
	d_b[i] = 1.0 ; 
	d_y[i] = 1.0 ; 
    }	

    // compute raw pointer
    auto start_raw = std::chrono::high_resolution_clock::now() ;
    compute_raw(d_y, a, d_x, d_b, size);
    auto end_raw = std::chrono::high_resolution_clock::now() ;
    auto duration_raw = end_raw - start_raw;

    

    // verif + avoid automatic code deletion
    std::cout << " Result of the last cell : " << std::endl ;
    std::cout << " -- Raw pointer : "   << d_y[size-1]          << std::endl ;

    free(d_x) ; 
    free(d_b) ; 
    free(d_y) ; 

    std::cout << " Time for Raw p.  : " << duration_raw.count() 	<< std::endl ;
    
    samurai::finalize();
    return 0;
}
