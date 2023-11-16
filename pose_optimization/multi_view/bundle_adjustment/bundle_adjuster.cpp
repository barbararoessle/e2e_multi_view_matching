#include <iostream>

#include <boost/filesystem.hpp>

#include <ba_problem.h>

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    if (argc != 2)
    {
        std::cerr << "Usage: bundle_adjuster <path to read and write>\n";
        return 1;
    }

    const boost::filesystem::path path(argv[1]);
    BaProblem ba_problem((path / "ba_in.csv").string());

    Solve(ba_problem);

    ba_problem.WriteResult((path / "ba_out.csv").string());
    return 0;
}
