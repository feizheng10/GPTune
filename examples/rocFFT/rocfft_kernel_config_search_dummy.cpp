//
// dummy code to mock rocfft_config_search
//
//   -- build: g++ rocfft_kernel_config_search_dummy.cpp -o rocfft_config_search
//
//
//
//

#include <iostream>
#include <random>

int main() {
  // Create a random device to seed the random number generator
  std::random_device rd;

  // Use Mersenne Twister 19937 generator
  std::mt19937 gen(rd());

  // Define a distribution to produce numbers in the range [0, 1)
  std::normal_distribution<> dis(0.0, 1.0);

  // Generate a random number
  double random_number = dis(gen);

  // Print the random number
  std::cout << "kernel, " << random_number << std::endl;

  return 0;
}
