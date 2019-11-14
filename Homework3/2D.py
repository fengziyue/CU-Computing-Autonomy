import time
import numpy as np 
import pycuda.driver as cuda 
import pycuda.autoinit 
from pycuda.compiler import SourceModule 

BLOCK_SIZE = 32 

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
__global__ void matmul2(int n, float *a, float *b, float *c)
{
  int row_num_a,col_num_a,row_num_b,col_num_b,row_num_c,col_num_c;
  row_num_a=col_num_a=row_num_b=col_num_b=row_num_c=col_num_c = n;
  __shared__ float A[32][32];//2D array for storing shared matrix values of A & B
  __shared__ float B[32][32];//Process subMatrix in block

  
  int Col=blockIdx.x*blockDim.x+threadIdx.x;//Col and Row Ids of threads
  int Row=blockIdx.y*blockDim.y+threadIdx.y;
  double temp = 0;
  for (int i = 0; i < (col_num_a-1)/blockDim.x+1; ++i) 
  {
     if (Row < row_num_a && i*blockDim.x+threadIdx.x < col_num_a)
        A[threadIdx.y][threadIdx.x] = a[Row*col_num_a + i*blockDim.x+threadIdx.x];//Memory Fetch from a
     else
        A[threadIdx.y][threadIdx.x] = 0;//In case the block dim is not a multiple of matrix

     if (Col < col_num_b && i*blockDim.x+threadIdx.y < row_num_b)
        B[threadIdx.y][threadIdx.x] = b[(i*blockDim.x+threadIdx.y)*col_num_b+Col];//Memory Fetch from b
     else
        B[threadIdx.y][threadIdx.x] = 0;

     __syncthreads();//Wait for all matrix loads to shared memory - then proceed with for loop
      if (Row < row_num_c && Col < col_num_c)
         
      for (int j = 0; j < blockDim.x; ++j)//Matrix multiplication
              temp += A[threadIdx.y][j] * B[j][threadIdx.x];
     __syncthreads();
  }
    if(Row<row_num_c && Col<col_num_c)//If the matrix is needed, then do this
       c[Row*col_num_c+Col] = (float)temp;//Save to c
  
  
}""") 

# get function 
matmul = mod.get_function("matmul2"); 


# set grid size 
grid = int(ni / BLOCK_SIZE)

# call gpu function 
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(grid,grid)); 
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(grid,grid)); 
matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(grid,grid)); 

start = time.time() 
for i in range(3):
    matmul(ni, a_gpu, b_gpu, c_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(grid,grid)); 


# copy back the result 
cuda.memcpy_dtoh(c, c_gpu) 
end = time.time() 
print ("Time: %.5f s"%(end-start))


















