#include <iostream>
#include <CL/cl.hpp>
#include <kernelStrings.hpp>

using namespace std;

class clBase
{
    vector<cl::Platform> platforms;             /** List of available platforms */
    vector<cl::Context> contexts;               /** A context for every platform */
    vector<cl::CommandQueue> queues;            /** A command-queue for every device */

};