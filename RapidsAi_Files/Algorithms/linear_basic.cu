#include<bits/stdc++.h> // header file for all c++ libraries
using namespace std;   // stdout library for printing values 
#include <iostream> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <list>
#include <thrust/copy.h>

__device__ volatile thrust::device_vector<float>derror;          // array to store all error values
thrust::host_vector<float>herror;          // array to store all error values


__device__ float err;
__device__ float b0 = 0;                   //initializing b0
__device__ float b1 = 0;                   //initializing b1
__device__ float alpha = 0.01;             //intializing error rate

__global__ void train(float *x, float *y){
    thrust::device_vector<float>dlist;
    /*Training Phase*/
    for (int i = 0; i < 20; i ++) {   // since there are 5 values and we want 4 epochs so run for loop for 20 times
        int idx = i % 5;              //for accessing index after every epoch
        float p = b0 + b1 * x[idx];  //calculating prediction
        err = p - y[idx];              // calculating error
        b0 = b0 - alpha * err;         // updating b0
        b1 = b1 - alpha * err * x[idx];// updating b1
        printf("B0=%f  B1=%f  error=%f \n", b0,b1,err);
        //cout<<"B0="<<b0<<" "<<"B1="<<b1<<" "<<"error="<<err<<endl;// printing values after every updation
        derror.push_back(err);
    }
    //thrust::copy(dlist.begin(),dlist.end(), derror.begin());
}

bool custom_sort(float a, float b) /* this custom sort function is defined to 
                                     sort on basis of min absolute value or error*/
{
    float  a1=abs(a-0);
    float  b1=abs(b-0);
    return a1<b1;
}




int main()
{
        
    /*Intialization Phase*/
    size_t bytes = 5*sizeof(float);

    float *x = (float*)malloc(bytes);
    float *y = (float*)malloc(bytes);

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    for (int i = 0; i<5; i++) {
        x[i] = i+1;
        y[i] = i+1;
        
    }

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
    /*Initializing Done*/

    train<<<1, 1>>>(x, y);
    std::sort(herror.begin(),herror.end(),custom_sort);//sorting based on error values
    thrust::copy(derror.begin(), derror.end(), herror.begin());
    cout<<"Final Values are: "<<"B0="<<b0<<" "<<"B1="<<b1<<" "<<"error="<<herror[0]<<endl;



    cudaDeviceSynchronize();
    /*Testing Phase*/
    cout<<"Enter a test x value";
    float test;
    cin>>test;
    float pred=b0+b1*test;
    cout<<endl;
    cout<<"The value predicted by the model= "<<pred;


}