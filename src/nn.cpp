#include "nn.hpp"
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>

namespace micrograd {

void Module::zero_grad() {
    for (auto* p : parameters()) {
        p->zero_grad();
    }
}

void Module::save_weights(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << " for writing" << std::endl;
        return;
    }

    const auto params = parameters();
    const int num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(int));

    for (auto* p : params) {
        int rows = p->data.rows();
        int cols = p->data.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(p->data.data()), rows * cols * sizeof(double));
    }

    file.close();
    std::cout << "Model saved to " << path << std::endl;
}

void Module::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << " for reading" << std::endl;
        return;
    }

    int num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(int));

    auto params = parameters();
    if (num_params != params.size()) {
        std::cerr << "Error: Architecture mismatch. Model has " << params.size()
                  << " parameters, but file has " << num_params << std::endl;
        return;
    }

    for (auto* p : params) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));

        if (rows != p->data.rows() || cols != p->data.cols()) {
            std::cerr << "Error: Shape mismatch for parameter" << std::endl;
            return;
        }

        file.read(reinterpret_cast<char*>(p->data.data()), rows * cols * sizeof(double));
    }

    file.close();
    std::cout << "Model loaded from " << path << std::endl;
}

Layer::Layer(const int nin, const int nout, const bool nonlin)
    : nonlin(nonlin) {
    // He initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double stddev = std::sqrt(2.0 / nin);
    std::normal_distribution<> d(0.0, stddev);

    Eigen::MatrixXd w_data = Eigen::MatrixXd::Zero(nin, nout);
    for (int i = 0; i < nin; ++i) {
        for (int j = 0; j < nout; ++j) {
            w_data(i, j) = d(gen);
        }
    }

    w = std::make_shared<Value>(w_data);
    w->set_self(w);

    b = std::make_shared<Value>(Eigen::MatrixXd::Zero(1, nout));
    b->set_self(b);
}

Value Layer::forward(const Value& x) const {
    Value z = x.matmul(*w) + *b;
    return nonlin ? z.relu() : z;
}

std::vector<Value*> Layer::parameters() {
    return {w.get(), b.get()};
}

MLP::MLP(int nin, const std::vector<int>& nouts) {
    std::vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    for (size_t i = 0; i < nouts.size(); ++i) {
        bool is_last = (i == nouts.size() - 1);
        layers.emplace_back(sz[i], sz[i + 1], !is_last);
    }
}

Value MLP::forward(const Value& x) const {
    Value out = x;
    for (auto& layer : layers) {
        out = layer.forward(out);
    }
    return out;
}

std::vector<Value*> MLP::parameters() {
    std::vector<Value*> params;
    for (auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

} // namespace micrograd
