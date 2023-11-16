#include <iostream>

#include <boost/filesystem.hpp>

#include <ba_init.h>

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    if (argc != 2)
    {
        std::cerr << "Usage: ba_initializer <path to read and write>\n";
        return 1;
    }

    const boost::filesystem::path path(argv[1]);
    BaInit ba_init((path / "ba_init_in.csv").string());

    ba_init.Run();

    ba_init.WriteResult((path / "ba_init_out.csv").string());
    return 0;
}
