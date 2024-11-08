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

template <typename T, typename T2>
[[gnu::noinline]] void compute_xtensor(T& y_tensor,
			double a, 
			T2& x_tensor, 
			T2& b_tensor,
			std::size_t size
		){
    y_tensor = xt::eval(a * x_tensor + b_tensor) ;
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


    // allocate xtensor vectors
    auto x_tensor = xt::ones<double>({size}) ; 
    auto b_tensor = xt::ones<double>({size}) ;
    xt::xarray<double> y_tensor = xt::ones<double>({size}) ;

    
    // compute xtensor
    auto start_xtensor = std::chrono::high_resolution_clock::now() ;
    compute_xtensor(y_tensor, a, x_tensor, b_tensor, size);
    auto end_xtensor = std::chrono::high_resolution_clock::now() ;
    auto duration_xtensor = end_xtensor - start_xtensor;

    // verif + avoid automatic code deletion
    std::cout << " Result of the last cell : " << std::endl ;
    std::cout << " -- Xtensor : "       << y_tensor[size-1]     << std::endl ;

    std::cout << " Time for Xtensor : " << duration_xtensor.count() 	<< std::endl ;
	
    samurai::finalize();
    return 0;
}
