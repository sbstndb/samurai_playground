
add_executable(vector_add add.cpp)
add_executable(vector_add_samurai add_samurai.cpp)
add_executable(vector_add_xtensor add_xtensor.cpp)
add_executable(vector_add_vector add_vector.cpp)
add_executable(vector_add_raw add_raw.cpp)
add_executable(vector_add_samurai_soa add_samurai_soa.cpp)


target_link_libraries(vector_add PRIVATE samurai::samurai samurai::libdeps xtensor)

target_link_libraries(vector_add_samurai PRIVATE samurai::samurai samurai::libdeps xtensor)
target_link_libraries(vector_add_xtensor PRIVATE samurai::samurai samurai::libdeps xtensor)
target_link_libraries(vector_add_vector PRIVATE samurai::samurai samurai::libdeps xtensor)
target_link_libraries(vector_add_raw PRIVATE samurai::samurai samurai::libdeps xtensor)
target_link_libraries(vector_add_samurai_soa PRIVATE samurai::samurai samurai::libdeps xtensor)


set(samurai_INCLUDE_DIRS "/home/sbstndbs/spack/opt/spack/linux-fedora40-skylake/gcc-14.2.1/samurai-master-oqjpnkbwdt7luewuiaqdt67drnavt5wi/include")
target_include_directories(vector_add PRIVATE ${samurai_INCLUDE_DIRS})
target_include_directories(vector_add_samurai PRIVATE ${samurai_INCLUDE_DIRS})
target_include_directories(vector_add_xtensor PRIVATE ${samurai_INCLUDE_DIRS})
target_include_directories(vector_add_vector PRIVATE ${samurai_INCLUDE_DIRS})
target_include_directories(vector_add_raw PRIVATE ${samurai_INCLUDE_DIRS})
target_include_directories(vector_add_samurai_soa PRIVATE ${samurai_INCLUDE_DIRS})


message("Samurai include directories: ${samurai_INCLUDE_DIRS}")
