import time
import numpy as np 
import pycuda.driver as cuda 
import pycuda.autoinit 
from pycuda.compiler import SourceModule 

BLOCK_SIZE = 256 

n = 1600
ni = np.int32(n) 

# matrix A 
a = np.random.randn(n, n)*10
a = a.astype(np.float32) 

# matrix B 
b = np.random.randn(n, n)*10
b = b.astype(np.float32) 

# matrix B 
c = np.empty([n, n]) 
c = c.astype(np.float32) 

# allocate memory on device 
a_gpu = cuda.mem_alloc(a.nbytes) 
b_gpu = cuda.mem_alloc(b.nbytes) 
c_gpu = cuda.mem_alloc(c.nbytes) 

# copy matrix to memory 
cuda.memcpy_htod(a_gpu, a) 
cuda.memcpy_htod(b_gpu, b) 

# compile kernel 
mod = SourceModule("""
__global__ void matmul(int row_num_b, float *a, float *b, float *c)
{
  int n = row_num_b * row_num_b;
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n)
  {
    int quot=i/row_num_b;
    int rem=i%row_num_b;
    double temp=0;
    for(int j=0;j<row_num_b;j++)
    {
      temp+=a[row_num_b*quot + j] * b[row_num_b*j + rem];
    }
    c[quot*row_num_b + rem]=(float)temp;
  }
  
  
}""") 

# get function 
matmul = mod.get_function("matmul"); 


# set grid size 
grid = int(ni * ni / BLOCK_SIZE)
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,1,1), grid=(grid,1)); 
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,1,1), grid=(grid,1)); 
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,1,1), grid=(grid,1)); 

# call gpu function 
start = time.time() 
for i in range(3):
    matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,1,1), grid=(grid,1)); 

# copy back the result 
cuda.memcpy_dtoh(c, c_gpu) 
end = time.time() 
print ("Time: %.5f s"%(end-start))
















