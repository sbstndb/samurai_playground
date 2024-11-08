
add_executable(xt xtensor.cpp)


target_link_libraries(xt PRIVATE samurai::samurai samurai::libdeps xtensor)



set(samurai_INCLUDE_DIRS "/home/sbstndbs/spack/opt/spack/linux-fedora40-skylake/gcc-14.2.1/samurai-master-oqjpnkbwdt7luewuiaqdt67drnavt5wi/include")
target_include_directories(xt PRIVATE ${samurai_INCLUDE_DIRS})


message("Samurai include directories: ${samurai_INCLUDE_DIRS}")
