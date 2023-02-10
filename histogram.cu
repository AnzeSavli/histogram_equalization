#include <stdlib.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

__global__ void histo_kernel(unsigned char *imageIn, long long img_size, unsigned long long *histo){

    __shared__ unsigned long long temp_histo[GRAYLEVELS];
    temp_histo[threadIdx.x] = 0;

    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x;
    while (i < img_size) {
        atomicAdd(&temp_histo[imageIn[i]], 1);
        i += step;
    }

    __syncthreads();

    atomicAdd(&histo[threadIdx.x], temp_histo[threadIdx.x]);
}


__global__ void cdf_kernel(unsigned long long *cdf_out, unsigned long long *histo_in){

    __shared__ unsigned long long temp[GRAYLEVELS]; 

    int thread_id = threadIdx.x;
    int offset = 1; 

    temp[2 * thread_id] = histo_in[2 * thread_id];
    temp[2 * thread_id + 1] = histo_in[ 2 * thread_id + 1]; 
 	
    for (int d = GRAYLEVELS>>1; d > 0; d >>= 1){ 
        __syncthreads();
        if (thread_id < d){ 
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;  
            temp[bi] += temp[ai];
        } 
        offset *= 2;
    } 

    if (thread_id == 0){
        temp[GRAYLEVELS - 1] = 0;
    }
 	
    for (int d = 1; d < GRAYLEVELS; d *= 2){      
        offset >>= 1;      
        __syncthreads();      
        if (thread_id < d){ 
            int ai = offset * (2 * thread_id + 1) - 1;     
            int bi = offset* ( 2 * thread_id + 2) - 1; 
            unsigned long long t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }
    __syncthreads(); 

    cdf_out[2 * thread_id] = temp[2 * thread_id]; 
    cdf_out[2 * thread_id + 1] = temp[2 * thread_id + 1]; 
}

__global__ void findmin_kernel(unsigned long long* cdf, unsigned long long *min_cdf){

    __shared__ unsigned long long temp[GRAYLEVELS];

    int thread_id = threadIdx.x;
    temp[thread_id] = cdf[thread_id];

    __syncthreads();

    for (int d = 1; d < GRAYLEVELS; d *= 2) {
        if (thread_id % (2*d) == 0) {
            if (temp[thread_id] != 0 && temp[thread_id + d] != 0) temp[thread_id] = min(temp[thread_id], temp[thread_id + d]);
            else temp[thread_id] = max(temp[thread_id], temp[thread_id + d]);
        }
        __syncthreads();
    }

    if(thread_id == 0){
        *min_cdf = temp[thread_id];
    }

}

__device__ unsigned char scale(unsigned long long cdf, unsigned long cdfmin, unsigned long imageSize){

    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS - 1));

    return (int)scale;
}

__global__ void transform_kernel(unsigned char *imageOut, unsigned char *imageIn, int width, int height, long long img_size, unsigned long long *cdf, unsigned long long *min_cdf) {

    int xi = threadIdx.x + blockIdx.x * blockDim.x;

    if (xi < img_size){
        imageOut[xi] = scale(cdf[imageIn[xi]], *min_cdf, img_size);
    }
}


int main(void) {
    
    float timefor10 = 0;

    for(int i=0; i<10; i++){
        
    int width, height, cpp;

    unsigned char *imageIn = stbi_load("window-neq.jpg", &width, &height, &cpp, DESIRED_NCHANNELS);

    if(imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);

    long long IMG_SIZE = width * height;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    

    unsigned char *dev_buffer;
    unsigned long long *dev_histo;
    unsigned long long histo[GRAYLEVELS];

    cudaMalloc((void**)&dev_buffer, IMG_SIZE);
    cudaMemcpy(dev_buffer, imageIn, IMG_SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, GRAYLEVELS * sizeof(long long));
    cudaMemset(dev_histo, 0, GRAYLEVELS * sizeof(long long));

    cudaEventRecord(start, 0);

    histo_kernel<<<IMG_SIZE/256, 256>>>(dev_buffer, IMG_SIZE, dev_histo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTimeHisto;

    cudaEventElapsedTime(&elapsedTimeHisto, start, stop);
    printf("Time to generate histogram (CUDA): %3.1f ms\n", elapsedTimeHisto);
    timefor10 += elapsedTimeHisto;

    cudaMemcpy(histo, dev_histo, GRAYLEVELS * sizeof(long long),cudaMemcpyDeviceToHost);
    

    cudaFree(dev_buffer);
    cudaFree(dev_histo);


    unsigned long long *dev_cdf;
    unsigned long long *dev_cdf_histo;
    unsigned long long cdf_cuda[GRAYLEVELS];

    cudaMalloc((void**)&dev_cdf_histo, GRAYLEVELS * sizeof(long long));
    cudaMemcpy(dev_cdf_histo, histo, GRAYLEVELS * sizeof(long long), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_cdf, GRAYLEVELS * sizeof(long long));
    cudaMemset(dev_cdf, 0, GRAYLEVELS * sizeof(long long));

    cudaEventRecord(start, 0);

    cdf_kernel<<<blocks * 2, 128>>>(dev_cdf, dev_cdf_histo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTimeCDF;

    cudaEventElapsedTime(&elapsedTimeCDF, start, stop);
    printf("Time to generate CDF (CUDA): %3.1f ms\n", elapsedTimeCDF);    
    timefor10 += elapsedTimeCDF;

    cudaMemcpy(cdf_cuda, dev_cdf, GRAYLEVELS * sizeof(long long),cudaMemcpyDeviceToHost);


    cudaFree(dev_cdf);
    cudaFree(dev_cdf_histo);    


    unsigned char *dev_img_in;
    unsigned char *dev_img_out;
    unsigned long long *dev_cdf_cuda;
    unsigned long long *dev_min_cdf;

    unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));

    cudaMalloc((void**)&dev_min_cdf, sizeof(long long));
    cudaMemcpy(dev_min_cdf, 0, sizeof(long long), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&dev_img_in, IMG_SIZE * sizeof(char));
    cudaMemcpy(dev_img_in, imageIn, IMG_SIZE * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_cdf_cuda, GRAYLEVELS * sizeof(long long));
    cudaMemcpy(dev_cdf_cuda, cdf_cuda, GRAYLEVELS * sizeof(long long), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_img_out, IMG_SIZE * sizeof(char));

    cudaEventRecord(start, 0);

    findmin_kernel<<<blocks * 2, 256>>>(dev_cdf_cuda, dev_min_cdf);
    transform_kernel<<<int(IMG_SIZE/256)+1, 256>>>(dev_img_out, dev_img_in, width, height, IMG_SIZE, dev_cdf_cuda, dev_min_cdf);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTimeTransform;

    cudaEventElapsedTime(&elapsedTimeTransform, start, stop);
    printf("Time to transform (CUDA): %3.1f ms\n", elapsedTimeTransform);
    timefor10 += elapsedTimeTransform;

    cudaMemcpy(imageOut, dev_img_out, IMG_SIZE * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(dev_img_in);
    cudaFree(dev_img_out);
    cudaFree(dev_cdf_cuda);

    stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
    stbi_write_jpg("out.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(imageIn);
    free(imageOut);

    }

    printf("Povprečen čas izvajanja (10krat) z uporabo CUDE: %5.5f ms\n", timefor10/10.0);

	return 0;
}



