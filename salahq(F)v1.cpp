#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <algorithm>

using namespace std;

__host__ __device__ bool BinarySearch(const uint32_t *arr, size_t arrSize, uint32_t target) {
    int left = 0;
    int right = arrSize - 1;

    while (left <= right) {
        int mid = (right + left) / 2;

        if (arr[mid] == target) {return true;}
        else if (arr[mid] < target) {left = mid + 1;}
        else {right = mid - 1;}
    }

    return false;
}

__host__ bool BinarySearch(const vector<uint32_t> &arr, uint32_t target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left <= right) {
        int mid = (right + left) / 2;

        if (arr[mid] == target) {return true;}
        else if (arr[mid] < target) {left = mid + 1;}
        else {right = mid - 1;}
    }

    return false;
}

//Check if a bitset is a subset of another bitset.
__host__ __device__ bool IsSubset(const uint32_t &subset, const uint32_t &superset) {
    return (subset & superset) == subset;
}

//Check if a uint32_t* is a subset of another uint32_t*. This also works as a find() function.
__host__ __device__ bool IsSubset(const uint32_t *subset, const uint32_t *superset, size_t subsetSize, size_t supersetSize) {
    for (int isub = 0; isub < subsetSize; isub++) {
        bool found = BinarySearch(superset, supersetSize, subset[isub]);
        if (!found) {return false;}
    }

    return true;
}

//Check if a vector<uint32_t> is a subset of another vector<uint32_t>.
__host__ bool IsSubset(const vector<uint32_t> &subset, const vector<uint32_t> &superset) {
    for (int isub = 0; isub < subset.size(); isub++) {
        bool found = BinarySearch(superset, subset[isub]);
        if (!found) {return false;}
    }

    return true;
}

//Generate powerset as array of bitmasks on host.
__host__ vector<uint32_t> PowerSetHost(size_t N) {
    size_t powersetSize = 1 << N;
    vector<uint32_t> result(powersetSize);

    for (int i = 0; i < powersetSize; i++) {result[i] = i;}


    return result;
}

//Generate powerset as array of bitmasks on device.
__device__ uint32_t *PowerSetDevice(size_t N) {
    size_t powersetSize = 1 << N;
    uint32_t *result = (uint32_t *)malloc(powersetSize * sizeof(uint32_t));

    for (int i = 0; i < powersetSize; i++) {result[i] = i;}


    return result;
}

//Compute lowerset as array of bitmasks on host.
__host__ vector<uint32_t> LowerSetHost(size_t N, const vector<uint32_t> &generators) {
    vector<uint32_t> result;
    vector<uint32_t> poset = PowerSetHost(N);

    if (generators.empty()) {
        return poset;
    }

    for (const uint32_t &gen: generators) {
        for (const uint32_t &subset: poset) {

            if (IsSubset(subset, gen)) {
                if (!BinarySearch(result, subset)) {result.push_back(subset);}
            }

        }
    }


    return result;
}

//Generate the ith subset of a powerset, and return its size.
__device__ tuple<uint32_t*, size_t> IthPowerDevice(const uint32_t *set, size_t setSize, int idx) {
    int size = __popc(idx);
    if (size == 0) {
        return make_tuple(nullptr, 0);
    }
    uint32_t *result = (uint32_t *)malloc(size * sizeof(uint32_t));

    size_t count = 0;
    for (int i = 0; i < setSize; i++) {
        if (idx & (1 << i)) {
            result[count] = set[i];
            count++;
        }
    }


    return make_tuple(result, count);
}

//Compute upperset as array of bitmasks on the device.
__device__ tuple<uint32_t*, size_t> UpperSetDevice(size_t N, const uint32_t *generators, size_t generatorsSize) {
    size_t MAX_SIZE = (1 << N);
    uint32_t *result = (uint32_t *)malloc(MAX_SIZE * sizeof(uint32_t));
    uint32_t *poset = PowerSetDevice(N);

    if (generatorsSize == 0) {
        return make_tuple(poset, MAX_SIZE);
    }

    size_t count = 0;
    for (int igen = 0; igen < generatorsSize; igen++) {
        uint32_t gen = generators[igen];

        for (int ipos = 0; ipos < MAX_SIZE; ipos++) {
            uint32_t subset = poset[ipos];

            if (IsSubset(gen, subset)) {
                bool IsIn = BinarySearch(result, count, subset);
                if (!IsIn) {
                    result[count] = subset;
                    count++;
                }
            }
        }
    }

    uint32_t *RESULT = (uint32_t *)malloc(count * sizeof(uint32_t));
    memcpy(RESULT, result, count * sizeof(uint32_t));


    return make_tuple(RESULT, count);
}

//Lambda function for a given cover, on the device.
__device__ auto CoverFuncDevice(const uint32_t *cover, size_t coverSize) {
    return [cover, coverSize] __device__ (double p) -> double {
        double result = 0;
        for (int i = 0; i < coverSize; i++) {
            uint32_t c = cover[i];
            int size = __popc(c);
            result += pow(p, size);
        }

        return result - 0.5;
    };
}

//Bisect method on device.
template <typename Func>
__device__ double bisect(Func f, double a, double b, double tol=1e-3) {
    if (f(a) * f(b) >= 0) {
        return NAN;
    }

    double c = a;
    while ((b - a) >= tol) {
        c = (a + b) / 2;
        if (f(c) == 0.0) {break;}
        else if (f(c) * f(a) < 0) {b = c;}
        else {a = c;}
    }


    return c;
}

__global__ void Test(size_t N, const uint32_t *generators, size_t generatorsSize, uint32_t *LSet, size_t LSetSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pow(2, LSetSize)) {
        uint32_t *cover = nullptr;
        size_t coverSize = 0;

        tuple<uint32_t*, size_t> X = IthPowerDevice(LSet, LSetSize, idx);
        uint32_t *potentialCover = get<0>(X);
        size_t potentialCoverSize = get<1>(X);


        if (potentialCover != nullptr) {
            tuple<uint32_t*, size_t> Y = UpperSetDevice(N, potentialCover, potentialCoverSize);
            uint32_t *USet = get<0>(Y);
            size_t USetSize = get<1>(Y);

            //if issubset(generators, USet) {use potential cover otherwise nullptr}
            if (IsSubset(generators, USet, generatorsSize, USetSize)) {
                cover = potentialCover;
                coverSize = potentialCoverSize;
            }
        }

        auto f = CoverFuncDevice(cover, coverSize);
        bool EmptyIn = BinarySearch(cover, coverSize, 0);
        double best_p = EmptyIn? 0: bisect(f, 0.0, 1.0);

        if (best_p > 0.6) {printf("Best p: %f\n", best_p);}
    }
}

int main() {
    system("rm main32");

    size_t N = 6; //{a, b, c, d, e, f}

    vector<uint32_t> generators = {{0b111100}};
    sort(generators.begin(), generators.end());
    vector<uint32_t> lowerset = LowerSetHost(N, generators);

    uint32_t *DGen;
    uint32_t *DLSet;

    cudaMalloc(&DGen, generators.size() * sizeof(uint32_t));
    cudaMalloc(&DLSet, lowerset.size() * sizeof(uint32_t));

    cudaMemcpy(DGen, generators.data(), generators.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(DLSet, lowerset.data(), lowerset.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    //idx must range of 2 ^ lowerset size
    int THREAD_RANGE = pow(2, lowerset.size());
    int threadsPerBlock = 256;
    int numBlocks = (THREAD_RANGE + threadsPerBlock - 1) / threadsPerBlock;

    auto start = chrono::high_resolution_clock::now();
    Test<<<numBlocks, threadsPerBlock>>>(N, DGen, generators.size(), DLSet, lowerset.size());
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

  cudaFree(DGen);
  cudaFree(DLSet);

    chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << "s" << endl;

    return 0;

}