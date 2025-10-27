#include <iostream>
#include <iomanip>
#include "engine.hpp"
#include "nn.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

using namespace micrograd;

int main() {
    std::cout << "Testing training loop with gradients..." << std::endl;

    // Create a small MLP
    MLP model(4, {8, 3});
    CrossEntropyLoss criterion;
    SGD optimizer(model.parameters(), 0.1);

    // Create dummy data (3 samples, 4 features, 3 classes)
    Eigen::MatrixXd X(3, 4);
    X << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0;

    Eigen::VectorXi y(3);
    y << 0, 1, 2;

    std::cout << "\nInitial model parameters:" << std::endl;
    auto params = model.parameters();
    std::cout << "Number of parameter matrices: " << params.size() << std::endl;
    std::cout << "First weight matrix (first 2x2 block):\n" << params[0]->data.block(0, 0, 2, 2) << std::endl;

    // Training for a few iterations
    std::cout << "\nTraining for 5 iterations..." << std::endl;
    for (int iter = 0; iter < 5; ++iter) {
        optimizer.zero_grad();

        Value inputs(X);
        Value logits = model.forward(inputs);
        Value loss = criterion.forward(logits, y);

        loss.backward();

        // Check if gradients are computed BEFORE step
        std::cout << "Iter " << iter + 1 << " - Loss: " << std::fixed << std::setprecision(4) << loss.data(0, 0);
        std::cout << ", Grad norm (first param): " << params[0]->grad.norm() << std::endl;

        optimizer.step();
    }

    std::cout << "\nAfter training:" << std::endl;
    std::cout << "First weight matrix (first 2x2 block):\n" << params[0]->data.block(0, 0, 2, 2) << std::endl;

    // Verify parameters changed
    std::cout << "\n✅ If parameters changed and loss decreased, training works correctly!" << std::endl;

    return 0;
}
