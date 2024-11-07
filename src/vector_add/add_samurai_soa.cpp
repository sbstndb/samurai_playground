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

template <typename T, typename M>
[[gnu::noinline]] void compute_samurai(
		T& y, 
		double a, 
		T& x, 
		T& b,
		M& mesh,
		std::size_t size
		){
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               y[cell] = a * x[cell] + b[cell] ;
                           });
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

    // allocate samurai fields
    auto x = samurai::make_field<double, 1, true>("x", mesh);
    auto b = samurai::make_field<double, 1, true>("b", mesh);
    auto y = samurai::make_field<double, 1, true>("y", mesh);

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
    compute_samurai(y, a, x, b,  mesh, size);
    auto end_samurai = std::chrono::high_resolution_clock::now() ;
    auto duration_samurai = end_samurai - start_samurai;
    

    // verif + avoid automatic code deletion
    std::cout << " Result of the last cell : " << std::endl ;
    std::cout << " -- Samurai SOA: "       << std::endl ;

    std::cout << " Time for Samurai SOA: " << duration_samurai.count() 	<< std::endl ; 
    
    samurai::finalize();
    return 0;
}
