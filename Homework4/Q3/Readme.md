# Image Processing With PyCuda

This code is an example of how to use GPU to convert color images to grayscale.



- For each image, the function

  ```
  enhanceImage
  ```

  does three steps enhancement:

  - calculate the histogram of 256 color used in the image
  - accumulate the histogram calculated in step one
  - assign each pixel of the image according to the inversed histogram

- In step one and three, we could parallelize the program in GPU. Step two is a simple 256 iteration loop which should be pretty fast on CPU and there is data dependency, therefore we’d better do it in CPU.

- To do the acceleration in GPU, my implementation uses one thread per pixel of the image. Each thread reads the pixel out and adds its value to the appropriated position in the histogram.

  - Threads in the same block can share the local memory which is faster than the global memory, so in each block the threads are using the shared memory to calculate the histogram.
  - After the local shared histogram is calculated out, the first 256 threads add the value of the local histogram to the global one.

- The data movement between CPU and GPU via Cuda APIs:

  - `cudaMelloc` can allocate the space in GPU’s display memory
  - `cudaMellocpy` can copy the data from the main memory to the display memory and vice versa.
  - `cudaFree` can release the memory space used in GPU

- Optimization:

  - Use shared memory as much as possible. Because the shared memory in a block is 100x faster than the global memory.
  - In my implementation, `atomicAdd` is used, because the histogram might be accessed by more than one thread. However, this could be a performance bottleneck of the overall GPU program. So I tried to use a large global memory to avoid the conflicts, but the memory was becoming the new performance bottleneck. To keep the program neat, I still decided to use `atomicAdd`.



### Reference:

http://bfeng.github.io/blog/2014/07/09/image-processing-with-pycuda/