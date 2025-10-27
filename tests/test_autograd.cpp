#include <iostream>
#include <iomanip>
#include "engine.hpp"
#include "nn.hpp"

using namespace micrograd;

int main() {
    std::cout << "Testing autograd engine..." << std::endl;

    // Test 1: Simple operations
    std::cout << "\n=== Test 1: Basic operations ===" << std::endl;
    Value a(Eigen::MatrixXd::Constant(1, 1, 2.0));
    Value b(Eigen::MatrixXd::Constant(1, 1, 3.0));
    Value c = a * b + a.pow(2);
    c.backward();

    std::cout << "a = 2.0, b = 3.0" << std::endl;
    std::cout << "c = a * b + a^2 = " << c.data(0, 0) << std::endl;
    std::cout << "dc/da = " << a.grad(0, 0) << " (expected: 7.0)" << std::endl;
    std::cout << "dc/db = " << b.grad(0, 0) << " (expected: 2.0)" << std::endl;

    // Test 2: Matrix operations
    std::cout << "\n=== Test 2: Matrix multiplication ===" << std::endl;
    Eigen::MatrixXd x_data(2, 3);
    x_data << 1, 2, 3,
              4, 5, 6;
    Eigen::MatrixXd w_data(3, 2);
    w_data << 1, 2,
              3, 4,
              5, 6;

    Value x(x_data);
    Value w(w_data);
    Value y = x.matmul(w);

    std::cout << "X shape: " << x.rows() << "x" << x.cols() << std::endl;
    std::cout << "W shape: " << w.rows() << "x" << w.cols() << std::endl;
    std::cout << "Y shape: " << y.rows() << "x" << y.cols() << std::endl;
    std::cout << "Y =\n" << y.data << std::endl;

    // Test 3: ReLU
    std::cout << "\n=== Test 3: ReLU activation ===" << std::endl;
    Eigen::MatrixXd relu_data(1, 4);
    relu_data << -2, -1, 1, 2;
    Value relu_input(relu_data);
    Value relu_output = relu_input.relu();

    std::cout << "Input: " << relu_input.data << std::endl;
    std::cout << "ReLU output: " << relu_output.data << std::endl;

    // Test 4: Small neural network
    std::cout << "\n=== Test 4: Small MLP forward pass ===" << std::endl;
    MLP model(4, {8, 2});

    Eigen::MatrixXd input(3, 4);  // batch of 3 samples
    input << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0;

    Value input_val(input);
    Value output = model.forward(input_val);

    std::cout << "Model: 4 -> 8 -> 2" << std::endl;
    std::cout << "Input shape: " << input_val.rows() << "x" << input_val.cols() << std::endl;
    std::cout << "Output shape: " << output.rows() << "x" << output.cols() << std::endl;
    std::cout << "Output:\n" << output.data << std::endl;

    // Test 5: Backward pass
    std::cout << "\n=== Test 5: Backward pass ===" << std::endl;
    output.backward();
    std::cout << "Gradient at input:\n" << input_val.grad << std::endl;

    std::cout << "\n✅ All tests completed successfully!" << std::endl;

    return 0;
}
