module = SourceModule("""
    __global__ void histograms(float * hist_rgb, unsigned char * pix, int total)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if(idx >= total) return;

      __shared__ float local_hist[256];

      if(threadIdx.x < 256)
        local_hist[threadIdx.x] = 0;

      __syncthreads();

      int pixel = pix[idx];
      atomicAdd((float *)&local_hist[pixel], 256.0/total);

      if(threadIdx.x < 256)
        atomicAdd((float *)&hist_rgb[threadIdx.x], local_hist[threadIdx.x]);
    }
    __global__ void enhance(unsigned char * pix, float * hist_rgb, int total)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if(idx >= total) return;
      int pixel = pix[idx];
      pix[idx] = (int)(hist_rgb[pixel]);
    }
    """)

histograms     = module.get_function("histograms")
enhance    = module.get_function("enhance")


THREADS_PER_BLOCK = 512

def gpu_enhanceImage(pic):
    # convert to array
    pix = numpy.array(pic)
    width, height = pic.size
    total = width*height

    # ------------------------------------------------
    # accelerate the following part as much as possible using GPU
    #
    # compute histograms for RGB components separately
    # the value range for each pixel is [0,255]
    hist_rgb = [0]*256

    hist_rgb_buffer = numpy.float32(hist_rgb)
    pix = pix.astype(numpy.uint8)

    hist_rgb_gpu = cuda.mem_alloc(hist_rgb_buffer.nbytes)
    pix_gpu = cuda.mem_alloc(pix.nbytes)

    cuda.memcpy_htod(hist_rgb_gpu, hist_rgb_buffer)
    cuda.memcpy_htod(pix_gpu, pix)

    histograms(hist_rgb_gpu, pix_gpu,
         numpy.int32(total),
         block=(THREADS_PER_BLOCK, 1, 1),
         grid=((total + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,1))

    cuda.memcpy_dtoh(hist_rgb_buffer, hist_rgb_gpu)

    hist_rgb = hist_rgb_buffer

    # compute the accumulative histograms
    for intensity in range(1,256):
      temp = hist_rgb[intensity] + hist_rgb[intensity-1]
      # take care of rounding error
      temp = min(temp, 256 - 1)
      hist_rgb[intensity] = temp

    # enhance the picture according to the inversed histgram
    cuda.memcpy_htod(hist_rgb_gpu, hist_rgb)

    enhance(pix_gpu, hist_rgb_gpu,
      numpy.int32(total),
      block=(THREADS_PER_BLOCK, 1, 1),
      grid=((total + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, 1))

    cuda.memcpy_dtoh(pix, pix_gpu)
    # -----------------------------------------------

    # save the picture
    pic = Image.fromarray(pix)
    return pic