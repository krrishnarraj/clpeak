#include <clpeak.h>

#define MSTRINGIFY(A) #A

static const char* stringifiedKernels = 
#include "bandwidth_kernels.cl"
#include "compute_sp_kernels.cl"
#include "compute_dp_kernels.cl"
;

int clPeak::parseArgs(int argc, char **argv)
{
    return 0;
}


int clPeak::runAll()
{
    try
    {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for(int p=0; p < (int)platforms.size(); p++)
        {
            cout << NEWLINE "Platform: " << platforms[p].getInfo<CL_PLATFORM_NAME>() << endl;
            
            cl_context_properties cps[3] = { 
                    CL_CONTEXT_PLATFORM, 
                    (cl_context_properties)(platforms[p])(), 
                    0 
                };
            cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
            vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
            
            cl::Program::Sources source(1, make_pair(stringifiedKernels, (strlen(stringifiedKernels)+1)));
            cl::Program prog = cl::Program(ctx, source);
            
            try {
                prog.build(devices);
            }
            catch (cl::Error error){
                cerr << TAB "Build Log: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;
                throw error;
            }
            
            for(int d=0; d < (int)devices.size(); d++)
            {
                cout << TAB "Device: " << devices[d].getInfo<CL_DEVICE_NAME>() << endl;
                cout << TAB TAB "Driver version: " << devices[d].getInfo<CL_DRIVER_VERSION>() << endl;
                
                device_info_t devInfo = getDeviceInfo(devices[d]);
                cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d]);
                
                runBandwidthTest(queue, prog, devInfo);
                runComputeSP(queue, prog, devInfo);
                runComputeDP(queue, prog, devInfo);
                
                cout << NEWLINE;
            }
        }
    }
    catch(cl::Error error)
    {
        cerr << error.what() << "( " << error.err() << " )" << endl;
        return -1;
    }
    
    return 0;
}

