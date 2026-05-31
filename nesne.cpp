#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <ctime>

// EXCEPTÝON SINIFLAR

class DimensionMismatchException : public std::runtime_error {
public:
    explicit DimensionMismatchException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class FileException : public std::runtime_error {
public:
    explicit FileException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// MATRÝX SINIFI

class Matrix {
private:
    int rows, cols;
    double** data;

    void allocate() {
        data = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            data[i] = new double[cols];
            for (int j = 0; j < cols; ++j)
                data[i][j] = 0.0;
        }
    }

    void clear() {
        if (data != NULL) {
            for (int i = 0; i < rows; ++i) {
                if (data[i] != NULL) {
                    delete[] data[i];
                }
            }
            delete[] data;
        }
        data = NULL;
        rows = cols = 0;
    }

public:
    Matrix(int r = 0, int c = 0) : rows(r), cols(c), data(NULL) {
        if (rows > 0 && cols > 0) allocate();
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(NULL) {
        if (rows > 0 && cols > 0) {
            allocate();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    data[i][j] = other.data[i][j];
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            clear();
            rows = other.rows;
            cols = other.cols;
            if (rows > 0 && cols > 0) {
                allocate();
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        data[i][j] = other.data[i][j];
            }
        }
        return *this;
    }

    ~Matrix() { clear(); }

    int getRows() const { return rows; }
    int getCols() const { return cols; }

    double& operator()(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols)
            throw std::out_of_range("Matrix indeks hatali");
        return data[r][c];
    }

    const double& operator()(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols)
            throw std::out_of_range("Matrix indeks hatali");
        return data[r][c];
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw DimensionMismatchException("Matris toplama boyut hatasi");

        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw DimensionMismatchException("Matris carpim boyut hatasi");

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                for (int k = 0; k < cols; ++k)
                    result.data[i][j] += data[i][k] * other.data[k][j];

        return result;
    }

    void saveToFile(const std::string& filename) const {
        std::ofstream file(filename.c_str());
        if (!file.is_open())
            throw FileException("Dosya yazma icin acilamadi");

        file << rows << " " << cols << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                file << data[i][j] << " ";
            file << "\n";
        }
        file.close();
    }

    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename.c_str());
        if (!file.is_open())
            throw FileException("Dosya okuma icin acilamadi");

        int r, c;
        file >> r >> c;

        clear();
        rows = r;
        cols = c;
        if (rows > 0 && cols > 0) {
            allocate();

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    file >> data[i][j];
        }
        file.close();
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::fixed << std::setprecision(4) << data[i][j] << " ";
            std::cout << "\n";
        }
    }
};

// AKTIVASYON FONKSIYONU ARAYUZU (IActivation)

class IActivation {
public:
    virtual ~IActivation() {}
    virtual Matrix activate(const Matrix& input) = 0;
    virtual std::string getName() const = 0;
};

// Sigmoid aktivasyonu: f(x) = 1 / (1 + e^(-x))
class Sigmoid : public IActivation {
public:
    Matrix activate(const Matrix& input) {
        Matrix result(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); ++i) {
            for (int j = 0; j < input.getCols(); ++j) {
                double x = input(i, j);
                if (x > 0) {
                    double exp_neg_x = std::exp(-x);
                    result(i, j) = 1.0 / (1.0 + exp_neg_x);
                } else {
                    double exp_x = std::exp(x);
                    result(i, j) = exp_x / (1.0 + exp_x);
                }
            }
        }
        return result;
    }

    std::string getName() const { return "Sigmoid"; }
};

// ReLU aktivasyonu: f(x) = max(0, x)
class ReLU : public IActivation {
public:
    Matrix activate(const Matrix& input) {
        Matrix result(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); ++i) {
            for (int j = 0; j < input.getCols(); ++j) {
                double x = input(i, j);
                result(i, j) = (x > 0.0) ? x : 0.0;
            }
        }
        return result;
    }

    std::string getName() const { return "ReLU"; }
};

// Tanh aktivasyonu: f(x) = tanh(x)
class Tanh : public IActivation {
public:
    Matrix activate(const Matrix& input) {
        Matrix result(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); ++i) {
            for (int j = 0; j < input.getCols(); ++j) {
                result(i, j) = std::tanh(input(i, j));
            }
        }
        return result;
    }

    std::string getName() const { return "Tanh"; }
};

// TEMEL KATMAN SINIFI (BaseLayer)

class BaseLayer {
public:
    virtual ~BaseLayer() {}
    virtual Matrix forward(const Matrix& input) = 0;
    virtual std::string getType() const = 0;
};

// YOÐUN  KATMAN SINIFI

class DenseLayer : public BaseLayer {
private:
    Matrix weights;
    Matrix bias;
    IActivation* activation;
    int inputDim;
    int outputDim;

    void randomInit() {
        srand((unsigned)time(NULL));
        double limit = std::sqrt(6.0 / (inputDim + outputDim));

        for (int i = 0; i < weights.getRows(); ++i) {
            for (int j = 0; j < weights.getCols(); ++j) {
                double r = (double)rand() / RAND_MAX;
                weights(i, j) = 2.0 * limit * (r - 0.5);
            }
        }

        for (int i = 0; i < bias.getRows(); ++i) {
            bias(i, 0) = 0.0;
        }
    }

public:
    DenseLayer(int input_dim, int output_dim, IActivation* act)
        : weights(output_dim, input_dim),
          bias(output_dim, 1),
          activation(act),
          inputDim(input_dim),
          outputDim(output_dim) {

        if (activation == NULL) {
            throw std::invalid_argument("Aktivasyon NULL olamaz");
        }
        randomInit();
    }

    ~DenseLayer() {
        if (activation != NULL) {
            delete activation;
            activation = NULL;
        }
    }

    Matrix forward(const Matrix& input) {
        if (input.getRows() != inputDim) {
            throw DimensionMismatchException("DenseLayer girdi boyutu uyusmadi");
        }
        Matrix Z = weights * input;
        return activation->activate(Z + bias);
    }

    std::string getType() const { return "DenseLayer"; }

    void printInfo() const {
        std::cout << "DenseLayer: " << inputDim << " -> " << outputDim
                  << " (Aktivasyon: " << activation->getName() << ")\n";
    }

    void saveWeights(const std::string& filename) const {
        weights.saveToFile(filename);
    }

    void loadWeights(const std::string& filename) {
        weights.loadFromFile(filename);
    }
};

// SÝNÝR AÐI SINIFI (NeuralNetwork)

class NeuralNetwork {
private:
    std::vector<BaseLayer*> layers;

public:
    NeuralNetwork() {}

    ~NeuralNetwork() {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] != NULL) {
                delete layers[i];
                layers[i] = NULL;
            }
        }
        layers.clear();
    }

    NeuralNetwork(const NeuralNetwork&) { }
    NeuralNetwork& operator=(const NeuralNetwork&) { return *this; }

    void addLayer(BaseLayer* layer) {
        if (layer == NULL) {
            throw std::invalid_argument("Layer NULL olamaz");
        }
        layers.push_back(layer);
    }

    Matrix predict(const Matrix& input) {
        if (layers.empty()) {
            throw std::logic_error("Network bostur");
        }

        Matrix current = input;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] != NULL) {
                current = layers[i]->forward(current);
            }
        }
        return current;
    }

    void summary() const {
        std::cout << "\n====== Network Yapisi ======\n";
        std::cout << "Toplam Katman: " << layers.size() << "\n";
        std::cout << "----------------------------\n";

        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] != NULL) {
                std::cout << "Katman " << (i + 1) << ": "
                          << layers[i]->getType() << "\n";

                DenseLayer* dl = dynamic_cast<DenseLayer*>(layers[i]);
                if (dl != NULL) {
                    dl->printInfo();
                }
            }
        }
        std::cout << "============================\n\n";
    }

    void saveModel(const std::string& filename) const {
        std::ofstream file(filename.c_str());
        if (!file.is_open()) {
            throw FileException("Model dosyasi yazýlamadi");
        }

        file << layers.size() << "\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            DenseLayer* dl = dynamic_cast<DenseLayer*>(layers[i]);
            if (dl != NULL) {
                char buf[50];
                sprintf(buf, "weights_layer_%d.txt", (int)i);
                dl->saveWeights(buf);
                file << buf << "\n";
            }
        }
        file.close();
        std::cout << "Model kaydedildi: " << filename << "\n";
    }
};

// CSV OKUYUCU

class CSVReader {
private:
    std::string filename;

public:
    CSVReader(const std::string& file) : filename(file) {}

    std::vector<std::vector<double> > read() {
        std::ifstream file(filename.c_str());
        if (!file.is_open()) {
            throw FileException("CSV dosyasi acilamadi");
        }

        std::vector<std::vector<double> > data;
        std::string line;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                if (!cell.empty()) {
                    try {
                        double value = std::atof(cell.c_str());
                        row.push_back(value);
                    } catch (...) {
                        continue;
                    }
                }
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        file.close();
        return data;
    }

    void printData(const std::vector<std::vector<double> >& data) const {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                std::cout << std::fixed << std::setprecision(4) << data[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
};

// UNIT TEST FONKSIYONLARI

void testMatrixOperations() {
    std::cout << "\n--- Matrix Islemleri Testi ---\n";

    try {
        std::cout << "[TEST 1] Matrix olusturma... ";
        Matrix m(2, 2);
        m(0, 0) = 1.0; m(0, 1) = 2.0;
        m(1, 0) = 3.0; m(1, 1) = 4.0;
        std::cout << "PASS\n";

        std::cout << "[TEST 2] Matrix carpimi... ";
        Matrix result = m * m;
        if (result.getRows() == 2 && result.getCols() == 2) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }

        std::cout << "[TEST 3] Matrix toplama... ";
        Matrix result2 = m + m;
        if (result2.getRows() == 2 && result2.getCols() == 2) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }

        std::cout << "[TEST 4] Boyut hatasi kontrolu... ";
        bool caught = false;
        try {
            Matrix bad(3, 3);
            Matrix m2(2, 2);
            Matrix wrong = bad * m2;
        } catch (const DimensionMismatchException&) {
            caught = true;
        }
        std::cout << (caught ? "PASS\n" : "FAIL\n");
    }
    catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << "\n";
    }
}

void testActivationFunctions() {
    std::cout << "\n--- Aktivasyon Fonksiyonlari Testi ---\n";

    try {
        std::cout << "[TEST 1] Sigmoid(0) = 0.5... ";
        Sigmoid sig;
        Matrix in(1, 1);
        in(0, 0) = 0.0;
        Matrix out = sig.activate(in);
        if (std::abs(out(0, 0) - 0.5) < 1e-9) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }

        std::cout << "[TEST 2] ReLU(-5) = 0... ";
        ReLU relu;
        in(0, 0) = -5.0;
        out = relu.activate(in);
        if (out(0, 0) == 0.0) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }

        std::cout << "[TEST 3] ReLU(5) = 5... ";
        in(0, 0) = 5.0;
        out = relu.activate(in);
        if (out(0, 0) == 5.0) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }

        std::cout << "[TEST 4] Tanh aktivasyonu... ";
        Tanh tanh_fn;
        in(0, 0) = 0.0;
        out = tanh_fn.activate(in);
        if (out(0, 0) == 0.0) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << "\n";
    }
}

void testNeuralNetwork() {
    std::cout << "\n--- Neural Network Testi ---\n";

    try {
        std::cout << "[TEST] Network olusturma ve tahmin yapma...\n";

        NeuralNetwork nn;

        IActivation* relu = new ReLU();
        IActivation* sigmoid = new Sigmoid();

        nn.addLayer(new DenseLayer(3, 4, relu));
        nn.addLayer(new DenseLayer(4, 1, sigmoid));

        nn.summary();

        Matrix input(3, 1);
        input(0, 0) = 1.0;
        input(1, 0) = 0.5;
        input(2, 0) = -0.3;

        std::cout << "Girdi matrisi:\n";
        input.print();

        Matrix output = nn.predict(input);
        std::cout << "\nCikti matrisi:\n";
        output.print();

        std::cout << "PASS\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << "\n";
    }
}

// ANA PROGRAM

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "   OOP-Brain - Gelistirilmis Versiyon\n";
        std::cout << "========================================\n";

        testMatrixOperations();
        testActivationFunctions();
        testNeuralNetwork();

        std::cout << "\n--- Model Persistence Testi ---\n";
        Matrix m(2, 2);
        m(0, 0) = 1.5; m(0, 1) = 2.5;
        m(1, 0) = 3.5; m(1, 1) = 4.5;

        std::cout << "Kaydedilen matris:\n";
        m.print();

        m.saveToFile("model_weights.txt");
        std::cout << "\nMatris dosyaya kaydedildi.\n";

        Matrix loaded(2, 2);
        loaded.loadFromFile("model_weights.txt");

        std::cout << "\nYuklenen matris:\n";
        loaded.print();

        std::cout << "\n========================================\n";
        std::cout << "   Tum testler basarili tamamlandi!\n";
        std::cout << "========================================\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\nHata: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

