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

template <typename T>
[[gnu::noinline]] void compute_xtensor(T& y_tensor,
			double a, 
			T& x_tensor, 
			T& b_tensor,
			std::size_t size
		){
	y_tensor = xt::eval(a * x_tensor + b_tensor) ;
}

[[gnu::noinline]] void compute_stdvector(std::vector<double>& y_vector, 
			double a, 
			std::vector<double>& x_vector, 
			std::vector<double>& b_vector, 
			std::size_t size){
    for (int i = 0 ; i < size ; i++){
            y_vector[i] = a * x_vector[i] + b_vector[i] ;
    }
}

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

    // allocate samurai fields
    auto x = samurai::make_field<double, 1>("x", mesh);
    auto b = samurai::make_field<double, 1>("b", mesh);
    auto y = samurai::make_field<double, 1>("y", mesh);

    // allocate xtensor vectors
//    auto x_tensor = xt::ones<double>({size}) ; 
//    auto b_tensor = xt::ones<double>({size}) ;
//  !!!!!! i found that use auto is dangerous and can leads to unvectorized code
    xt::xarray<double> x_tensor, b_tensor, y_tensor ; 
    x_tensor = xt::ones<double>({size}) ; 
    b_tensor = xt::ones<double>({size}) ;
    y_tensor = xt::ones<double>({size}) ;

    // allocate c++ vectors
    std::vector<double> x_vector(size, 1.0); 
    std::vector<double> b_vector(size, 1.0) ;
    std::vector<double> y_vector(size, 1.0) ;    

    // allocate raw pointers
    double* d_x = (double*)malloc(size * sizeof(double)) ; 
    double* d_b = (double*)malloc(size * sizeof(double)) ;
    double* d_y = (double*)malloc(size * sizeof(double)) ;
    
    
    // init samurai fields
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               x[cell] = 1.0;
			       b[cell] = 1.0;
			       y[cell] = 1.0;
                           });

    // init raw pointer
    for (int i = 0 ; i < size ; i++){
	d_x[i] = 1.0 ; 
	d_b[i] = 1.0 ; 
	d_y[i] = 1.0 ; 
    }	


    // compute samurai 
    auto start_samurai = std::chrono::high_resolution_clock::now() ; 
    compute_samurai(y, a, x, b,  mesh, size);
    auto end_samurai = std::chrono::high_resolution_clock::now() ;
    auto duration_samurai = end_samurai - start_samurai;
    
    // compute xtensor
    auto start_xtensor = std::chrono::high_resolution_clock::now() ;
    compute_xtensor(y_tensor, a, x_tensor, b_tensor, size);
    auto end_xtensor = std::chrono::high_resolution_clock::now() ;
    auto duration_xtensor = end_xtensor - start_xtensor;

    // compute c++ vector
    auto start_vector = std::chrono::high_resolution_clock::now() ;
    compute_stdvector(y_vector, a, x_vector, b_vector, size);
    auto end_vector = std::chrono::high_resolution_clock::now() ;
    auto duration_vector = end_vector - start_vector;


    // compute raw pointer
    auto start_raw = std::chrono::high_resolution_clock::now() ;
    compute_raw(d_y, a, d_x, d_b, size);
    auto end_raw = std::chrono::high_resolution_clock::now() ;
    auto duration_raw = end_raw - start_raw;

    

    // verif + avoid automatic code deletion
    std::cout << " Result of the last cell : " << std::endl ;
    std::cout << " -- Samurai : "       << std::endl ;
    std::cout << " -- Xtensor : "       << y_tensor[size-1]     << std::endl ;
    std::cout << " -- C++ vector : "    << y_vector[size-1]     << std::endl ;
    std::cout << " -- Raw pointer : "   << d_y[size-1]          << std::endl ;

    free(d_x) ; 
    free(d_b) ; 
    free(d_y) ; 

    std::cout << " Time for Samurai : " << duration_samurai.count() 	<< std::endl ; 
    std::cout << " Time for Xtensor : " << duration_xtensor.count() 	<< std::endl ;
    std::cout << " Time for Vector  : " << duration_vector.count() 	<< std::endl ;
    std::cout << " Time for Raw p.  : " << duration_raw.count() 	<< std::endl ;
    
	
    samurai::finalize();
    return 0;
}
