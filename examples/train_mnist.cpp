#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include "engine.hpp"
#include "nn.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "mnist_loader.hpp"

using namespace micrograd;

const double LEARNING_RATE = 0.01;
const double MOMENTUM = 0.9;
const int EPOCHS = 20;
const int BATCH_SIZE = 128;
const std::string DATASET_ROOT = "/home/minh/datasets/MNIST/";
const std::string WEIGHTS_PATH = "../mnist_mlp.bin";

int main() {
    std::cout << "Loading MNIST dataset..." << std::endl;

    MNISTLoader train_loader, val_loader;

    if (!train_loader.load(DATASET_ROOT + "train-images-idx3-ubyte",
                           DATASET_ROOT + "train-labels-idx1-ubyte")) {
        std::cerr << "Failed to load training data" << std::endl;
        return 1;
    }

    if (!val_loader.load(DATASET_ROOT + "t10k-images-idx3-ubyte",
                         DATASET_ROOT + "t10k-labels-idx1-ubyte")) {
        std::cerr << "Failed to load validation data" << std::endl;
        return 1;
    }

    std::cout << "MNIST dataset loaded" << std::endl;

    // Create model: 784 -> 32 -> 16 -> 10
    MLP model(784, {32, 16, 10});
    CrossEntropyLoss criterion;
    NesterovSGD optimizer(model.parameters(), LEARNING_RATE, MOMENTUM);

    std::cout << "Model created with " << model.parameters().size() << " parameter matrices" << std::endl;

    std::vector<double> train_acc_log, val_acc_log, train_loss_log;
    double best_val_acc = 0.0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double train_acc = 0.0;
        double train_loss = 0.0;
        int num_train_batches = train_loader.get_num_batches(BATCH_SIZE);

        // Training
        for (int batch_idx = 0; batch_idx < num_train_batches; ++batch_idx) {
            Eigen::MatrixXd batch_images;
            Eigen::VectorXi batch_labels;
            train_loader.get_batch(batch_idx, BATCH_SIZE, batch_images, batch_labels);

            Value inputs(batch_images);
            Value logits = model.forward(inputs);
            Value loss = criterion.forward(logits, batch_labels);

            train_loss += loss.data(0, 0);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Compute accuracy
            Eigen::MatrixXd probs = softmax(logits);
            int correct = 0;
            for (int i = 0; i < probs.rows(); ++i) {
                int pred_label;
                probs.row(i).maxCoeff(&pred_label);
                if (pred_label == batch_labels(i)) {
                    correct++;
                }
            }
            train_acc += static_cast<double>(correct) / probs.rows();

            if ((batch_idx + 1) % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << " [" << batch_idx + 1 << "/" << num_train_batches << "]"
                          << " Loss: " << std::fixed << std::setprecision(4) << loss.data(0, 0) << std::endl;
            }
        }

        train_acc /= num_train_batches;
        train_loss /= num_train_batches;
        train_acc_log.push_back(train_acc);
        train_loss_log.push_back(train_loss);

        std::cout << "Epoch " << epoch + 1 << " - Train Loss: " << std::fixed << std::setprecision(4)
                  << train_loss << ", Train Acc: " << std::setprecision(2)
                  << train_acc * 100 << "%" << std::endl;

        // Validation
        double val_acc = 0.0;
        int num_val_batches = val_loader.get_num_batches(BATCH_SIZE);

        for (int batch_idx = 0; batch_idx < num_val_batches; ++batch_idx) {
            Eigen::MatrixXd batch_images;
            Eigen::VectorXi batch_labels;
            val_loader.get_batch(batch_idx, BATCH_SIZE, batch_images, batch_labels);

            Value inputs(batch_images);
            Value logits = model.forward(inputs);

            Eigen::MatrixXd probs = softmax(logits);
            int correct = 0;
            for (int i = 0; i < probs.rows(); ++i) {
                int pred_label;
                probs.row(i).maxCoeff(&pred_label);
                if (pred_label == batch_labels(i)) {
                    correct++;
                }
            }
            val_acc += static_cast<double>(correct) / probs.rows();
        }

        val_acc /= num_val_batches;
        val_acc_log.push_back(val_acc);

        std::cout << "Epoch " << epoch + 1 << " - Val Acc: " << std::fixed << std::setprecision(2)
                  << val_acc * 100 << "%" << std::endl;

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            model.save_weights(WEIGHTS_PATH);
        }

        std::cout << std::endl;
    }

    std::cout << "Training complete! Best validation accuracy: "
              << std::fixed << std::setprecision(2) << best_val_acc * 100 << "%" << std::endl;

    return 0;
}
