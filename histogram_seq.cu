#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

unsigned long findMin(unsigned long* cdf){
    
    unsigned long min = 0;
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}

unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1));
    
    return (int)scale;
}

void CalculateHistogram(unsigned char* image, int width, int heigth, unsigned long* histogram){
    
    //Clear histogram:
    for (int i=0; i<GRAYLEVELS; i++) {
        histogram[i] = 0;
    }
    
    //Calculate histogram
    for (int i=0; i<heigth; i++) {
        for (int j=0; j<width; j++) {
            histogram[image[i*width + j]]++;
        }
    }
}

void CalculateCDF(unsigned long* histogram, unsigned long* cdf){
    
    // clear cdf:
    for (int i=0; i<GRAYLEVELS; i++) {
        cdf[i] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int i=1; i<GRAYLEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
}


void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int heigth, unsigned long* cdf){
     
    unsigned long imageSize = width * heigth;
    
    unsigned long cdfmin = findMin(cdf);
    
    //Equalize
    for (int i=0; i<heigth; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
    }
}




int main(void)
{
    double timefor10 = 0;

    for(int i=0; i<10;i++){
        // Read image from file
        int width, height, cpp;
        // read only DESIRED_NCHANNELS channels from the input image:
        unsigned char *imageIn = stbi_load("window-neq.jpg", &width, &height, &cpp, DESIRED_NCHANNELS);
        if(imageIn == NULL) {
            printf("Error in loading the image\n");
            return 1;
        }
        printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);
        

        //Allocate memory for raw output image data, histogram, and CDF 
        unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));
        unsigned long *histogram= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
        unsigned long *CDF= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));


        time_t start = clock();
        // Histogram equalization steps: 
        // 1. Create the histogram for the input grayscale image.
        CalculateHistogram(imageIn, width, height, histogram);
        // 2. Calculate the cumulative distribution histogram.
        CalculateCDF(histogram, CDF);
        // 3. Calculate the new gray-level values through the general histogram equalization formula 
        //    and assign new pixel values
        Equalize(imageIn, imageOut, width, height, CDF);

        timefor10 += (double)((clock() - start) * 1000 / CLOCKS_PER_SEC);

        // write output image:
        stbi_write_png("out.png", width, height, DESIRED_NCHANNELS, imageOut, width * DESIRED_NCHANNELS);
        stbi_write_jpg("out.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

        //Free memory
        free(imageIn);
        free(imageOut);
        free(histogram);
        free(CDF);
    }

    printf("Time spent: %5.5f ms \n", (double)(timefor10 / 10.0));

	return 0;
}



