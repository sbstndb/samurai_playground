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
    std::size_t level = 10;

    double a = 2.0 ; 

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, level, level};

    auto size = mesh.nb_cells();
    std::cout << "Size : " << size << std::endl ; 

    // allocate samurai fields
    auto x = samurai::make_field<double, 1>("x", mesh);
    auto b = samurai::make_field<double, 1>("b", mesh);
    auto y = samurai::make_field<double, 1>("y", mesh);

    // allocate xtensor vectors
    auto x_tensor = xt::ones<double>({size}) ; 
    auto b_tensor = xt::ones<double>({size}) ;
    xt::xarray<double> y_tensor = xt::ones<double>({size}) ;

    // allocate c++ vectors
    std::vector<double> x_vector(size, 1.0); 
    std::vector<double> b_vector(size, 1.0) ;
    std::vector<double> y_vector(size, 1.0) ;    

    // init samurai fields
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               x[cell] = 1.0;
			       b[cell] = 1.0;
			       y[cell] = 1.0;
                           });

    // compute samurai 
    auto start_samurai = std::chrono::high_resolution_clock::now() ; 
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               y[cell] = a * x[cell] + b[cell] ; 
                           });
    auto end_samurai = std::chrono::high_resolution_clock::now() ;
    auto duration_samurai = end_samurai - start_samurai;
    
    // compute xtensor
    auto start_xtensor = std::chrono::high_resolution_clock::now() ;
    y_tensor = xt::eval(a * x_tensor + b_tensor) ; 
    auto end_xtensor = std::chrono::high_resolution_clock::now() ;
    auto duration_xtensor = end_xtensor - start_xtensor;

    // compute c++ vector
    auto start_vector = std::chrono::high_resolution_clock::now() ;
    for (int i = 0 ; i < size ; i++){
	    y_vector[i] = a * x_vector[i] + b_vector[i] ;
    }
    auto end_vector = std::chrono::high_resolution_clock::now() ;
    auto duration_vector = end_vector - start_vector;


    std::cout << " Time for Samurai : " << duration_samurai.count() << std::endl ; 
    std::cout << " Time for Xtensor : " << duration_xtensor.count() << std::endl ;
    std::cout << " Time for Vector  : " << duration_vector.count() << std::endl ;
	
    samurai::finalize();
    return 0;
}
