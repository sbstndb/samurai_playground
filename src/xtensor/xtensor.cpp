#include <iostream>
#include <vector>


#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xadapt.hpp>


void first_example(){
        xt::xarray<double> arr1 {
                {1.0, 2.0, 3.0},
                {1.1, 2.2, 3.3},
                {1.1, 1.1, 1.1}
        };
        xt::xarray<double> arr2 {1.1,2.2,3.3} ;
        xt::xarray<double> res = xt::view(arr1, 1) + arr2 ;
        std::cout << " Res : " << res << std::endl ;
}

void reshape(){
	xt::xarray<int> arr1 {
		{1,2,3,4,5,6,7,8,9}
	};
	arr1.reshape({3,3}) ; 
	std::cout << "Array : " << arr1 << std::endl ;
}

void index_access(){
        xt::xarray<double> arr1 {
                {1.0, 2.0, 3.0},
                {1.1, 2.2, 3.3},
                {1.1, 1.1, 1.1}
        };
	std::cout << " Index (0, 0) : " << arr1(0,0) << std::endl ; 
}

void broadcasting(){
	xt::xarray<double> arr1 {
		1.0, 2.0, 3.0, 4.0
	};
	xt::xarray<double> arr2 {
		1.0,2.0,3.0,4.0
	};
	arr2.reshape({4,1});
	xt::xarray<double> res = xt::pow(arr1, arr2) ; 
	std::cout << " Resultat Broadcasting : " << res << std::endl ; 
}

void expression(){
	xt::xarray<double> a = {1,2,3} ; 
	xt::xarray<double> b = {4,5,6};
	xt::xarray<double>&& res = xt::eval(a + b) ;
	std::cout << "Resultat expression : " << res << std::endl ; 	
}

void memory_layout(){
	// bug : doc do not compile :/
//	std::vector<size_t> shape = { 3, 2, 4 };
//	std::vector<size_t> strides = { 8, 4, 1 };
//	xt::xarray<double, xt::layout_type::dynamic> a(shape, strides);
}

void run_compile_time(){
	std::array<size_t, 3> shape = {3,2,2} ;
//	xt::xarray<double,3> xarray(shape) ; 
	xt::xtensor<double,3> xtensor(shape) ; 
	xt::xtensor_fixed<double, xt::xshape<3,2,2>> xtensor_fixed() ; 
	std::cout << "Xtensor : " << xtensor << std::endl ; 
	std::cout << "Xtensor fixed : "<< xtensor_fixed <<std::endl ; // issue here, why ? 
}

void adapting_std(){
	std::vector<double> std = {1, 2, 3, 4, 5, 6} ; 
	std::vector<std::size_t> shape = {2,3};
	auto adapted = xt::adapt(std, shape) ; 
	std::cout << "Adapted tensor from vector : " << adapted << std::endl ; 
}

void adapting_c(){
	int size = 10 ; 
	double * data = (double*)malloc(size * sizeof(double)) ;
	for (int i = 0 ; i < size ; i++){
		data[i] = (double)i; 
	}
	std::vector<std::size_t> shape = {size} ; 
	auto a = xt::adapt(data, size, xt::no_ownership(), shape);
	std::cout << "Adapted tensor from C : " << a << std::endl ; 
	// what to do here ?? before the malloc
	free(data) ; 
}



int main(){

	first_example() ; 
	reshape() ; 
	index_access() ; 
	broadcasting() ; 
	expression() ; 
	run_compile_time();
	adapting_std() ; 
	adapting_c() ;
}
