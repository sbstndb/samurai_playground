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

[[gnu::noinline]] void compute_stdvector(std::vector<double>& y_vector, 
			double a, 
			std::vector<double>& x_vector, 
			std::vector<double>& b_vector, 
			std::size_t size){
    for (int i = 0 ; i < size ; i++){
            y_vector[i] = a * x_vector[i] + b_vector[i] ;
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


    // allocate c++ vectors
    std::vector<double> x_vector(size, 1.0); 
    std::vector<double> b_vector(size, 1.0) ;
    std::vector<double> y_vector(size, 1.0) ;    

    // compute c++ vector
    auto start_vector = std::chrono::high_resolution_clock::now() ;
    compute_stdvector(y_vector, a, x_vector, b_vector, size);
    auto end_vector = std::chrono::high_resolution_clock::now() ;
    auto duration_vector = end_vector - start_vector;

    // verif + avoid automatic code deletion
    std::cout << " Result of the last cell : " << std::endl ;
    std::cout << " -- C++ vector : "    << y_vector[size-1]     << std::endl ;

    std::cout << " Time for Vector  : " << duration_vector.count() 	<< std::endl ;
    
    samurai::finalize();
    return 0;
}
