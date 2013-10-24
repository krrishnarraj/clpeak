MSTRINGIFY(

__kernel void bandwidth(__global float *arr, uint N)
{
    float x = arr[get_global_id(0)];
    arr[get_global_id(0)] = x * x;
}



)

