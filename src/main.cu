#include "StatsCollector.hpp"
#include <argparse/argparse.hpp>
#include <cinttypes>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "CellularAutomata/CADataDomainComparator.cuh"
#include "CellularAutomata/CASolverFactory.cuh"
#include "GPUBenchmark.cuh"

const char *logFileName = "log.txt";

struct MainArgs
{
    int n;
    int mode;
    int steps;
    int deviceId;
    float density;
    int seed;
    int doVerify;
};

void defineArguments(argparse::ArgumentParser &program);
MainArgs parseArguments(argparse::ArgumentParser &program);

int main(int argc, char **argv)
{
    MainArgs args;
    argparse::ArgumentParser program("CAT: Celular Automata on Tensor Cores", "1.0.0", argparse::default_arguments::all,
                                     true);
    defineArguments(program);
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(1);
    }
    args = parseArguments(program);
    debugInit(1, logFileName);

    CASolver *solver = CASolverFactory::createSolver(args.mode, args.deviceId, args.n, RADIUS);

    if (solver == nullptr)
    {
        printf("main(): solver is NULL\n");
        exit(1);
    }
    GPUBenchmark *benchmark = new GPUBenchmark(solver, args.n, args.steps, args.seed, args.density);

    benchmark->run();

    fDebug(1, benchmark->getStats()->printStats());
    benchmark->getStats()->printShortStats();

    if (args.doVerify)
    {
        printf("\n[VERIFY] verifying...\n\n");
        CASolver *referenceSolver = CASolverFactory::createSolver(0, 0, args.n, RADIUS);
        if (referenceSolver == nullptr)
        {
            printf("main(): solver is NULL\n");
            exit(1);
        }
        GPUBenchmark *referenceBenchmark =
            new GPUBenchmark(referenceSolver, args.n, args.steps, args.seed, args.density);
        lDebug(1, "***** Verifyng *****");
        referenceBenchmark->run();

        lDebug(1, "Cheking results...");
        CADataDomainComparator *comparator = new CADataDomainComparator(solver, referenceSolver);

        if (!comparator->compareCurrentStates())
        {
            printf("\n[VERIFY] verification FAILED!.\n\n");
            exit(1);
        }
        else
        {
            printf("\n[VERIFY] verification successful.\n\n");
        }
    }
}

void defineArguments(argparse::ArgumentParser &program)
{
    MainArgs args;
    program.add_argument("n").help("Size of the data domain").action([](const std::string &value) {
        return std::stoi(value);
    });
    program.add_argument("solver")
        .help("Solver to use:\n\t0 - BASE: Global Memory\n\t1 - SHARED: Shared /memory\n\t2 - COARSE: Thread Coarsening "
              "\n\t3 - CAT: Fast Tensor Core\n\t4 - MCELL: Multiple cells per thread\n\t5 - PACK: uint64 "
              "Packet Coding")
        .action([](const std::string &value) { return std::stoi(value); });

    program.add_argument("steps").help("Number of steps of the CA simulation").action([](const std::string &value) {
        return std::stoi(value);
    });

    program.add_argument("-g", "--deviceId")
        .help("Device ID")
        .action([](const std::string &value) { return std::stoi(value); })
        .default_value(0);

    program.add_argument("-d", "--density").help("Density of the data domain").action([](const std::string &value) {
        return std::stof(value);
    });

    program.add_argument("--seed")
        .help("Seed for the random number generator")
        .action([](const std::string &value) { return std::stoi(value); })
        .default_value(0);

    program.add_argument("--doVerify")
        .help("Verify the results? WARNING: memory requirements double")
        .default_value(false)
        .implicit_value(true);
}

MainArgs parseArguments(argparse::ArgumentParser &program)
{
    MainArgs args;
    args.n = program.get<int>("n");
    args.mode = program.get<int>("solver");
    args.steps = program.get<int>("steps");
    args.deviceId = program.get<int>("-g");
    args.density = program.get<float>("-d");
    args.seed = program.get<int>("--seed");
    args.doVerify = program.get<bool>("--doVerify");
    return args;
}