#include <stdio.h> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <mpi.h> 
#include <string.h> 
// Define output file name 
#define OUTPUT_FILE "stencil.pgm" 
#define NDIMS 1  /* setting the number of dimensions in the grid with a macro */ 
#define MASTER 0 
 
#include "helper_functions.h" 
 
int main(int argc, char* argv[]) 
{ 
  // Check usage 
  if (argc != 4) { 
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]); 
    exit(EXIT_FAILURE); 
  } 
 
  int nx = atoi(argv[1]); /* problem dimensions */ 
  int ny = atoi(argv[2]); 
  /* Add padding to the image to avoid out of range indexing at borders */ 
  int rows = ny+2; 
  int cols = nx+2; 
  int niters = atoi(argv[3]); 
  int periods[NDIMS]; 
  int dims[NDIMS]; 
  int reorder = 1; 
  int rank; 
  int size; 
  MPI_Comm CART_COMM_WORLD; 
  // Initalise periods and dims 
  for(int ii=0; ii<NDIMS; ii++){ 
    periods[ii] = 0; 
    dims[ii]    = 0; 
  } 
 
 
  // Allocate image 
  float *image        = malloc(sizeof(float) * rows * cols); /* rows & cols have padding */ 
  float *tmp_image    = malloc(sizeof(float) * rows * cols); /* rows & cols have padding */ 
  float *result_image = malloc(sizeof(float) * rows * cols); /* rows & cols have padding */ 
 
  // Initialise image 
  init_images(nx, ny, cols, rows, image, tmp_image); 
  zero_image(cols,rows,result_image); 
  output_image("init.pgm",nx,ny,cols,rows,image); 
 
  MPI_Init(&argc, &argv); 
 
  MPI_Comm_size( MPI_COMM_WORLD, &size ); 
 
  MPI_Dims_create(size, NDIMS, dims); 
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &CART_COMM_WORLD); 
 
  MPI_Comm_rank( CART_COMM_WORLD, &rank ); 
 
  // Initialise problem dimensions 
  int local_rows = ny+2; /* rows already includes padding */ 
  int local_ny   = ny; 
  int local_nx   = calc_local_cols(nx, size, rank); 
  int local_cols = local_nx+2; /* with halo padding */ 
 
 
  // Initialise local_image 
  float *local_image     = malloc(sizeof(float)*local_rows*local_cols); 
  float *local_tmp_image = malloc(sizeof(float)*local_rows*local_cols); 
 
  memset(local_image,    0,sizeof(float)*local_rows*local_cols); 
  memset(local_tmp_image,0,sizeof(float)*local_rows*local_cols); 
 
  int *displs = malloc(sizeof(int)*size); 
  int *counts = malloc(sizeof(int)*size); 
  int halo_counts[2] = {local_rows, local_rows};
  int halo_send_displs[2] = {local_rows,local_rows*(local_cols-2)};
  int halo_recv_displs[2] = {0,local_rows*(local_cols-1)};
 
  calc_scatter_params(rows, nx, size, displs, counts); 
 
 
  MPI_Scatterv( 
              image+rows,counts,displs, 
              MPI_FLOAT, 
              local_image+local_rows, 
              counts[rank], 
              MPI_FLOAT, 
              MASTER,CART_COMM_WORLD); 
 

  // Call the stencil kernel 
  MPI_Barrier(CART_COMM_WORLD);
  double tic = wtime(); 
  for (int t = 0; t < niters; ++t) { 
 
    MPI_Neighbor_alltoallv(local_image,halo_counts, halo_send_displs, MPI_FLOAT,
                          local_image,halo_counts,halo_recv_displs,MPI_FLOAT,CART_COMM_WORLD);
    stencil(local_nx,local_ny,local_cols,local_rows,local_image,local_tmp_image);
    MPI_Neighbor_alltoallv(local_tmp_image,halo_counts,halo_send_displs,MPI_FLOAT,
                          local_tmp_image,halo_counts,halo_recv_displs,MPI_FLOAT,CART_COMM_WORLD);
    stencil(local_nx,local_ny,local_cols,local_rows,local_tmp_image,local_image);
  }
  double toc = wtime(); 
 
 
  MPI_Barrier(CART_COMM_WORLD); 
  MPI_Gatherv(local_image+local_rows,local_nx*local_rows,MPI_FLOAT,result_image+rows,counts,displs,MPI_FLOAT,MASTER,CART_COMM_WORLD); 

  // Output 
  if (rank == MASTER){ 
    printf("\n------------------------------------\n"); 
    printf(" runtime: %lf s\n", toc - tic); 
    printf("------------------------------------\n"); 
    output_image("stencil.pgm",nx,ny,cols,rows,result_image); 
    // output_image("stencil.pgm",nx,ny,cols,rows,tmp_image); 
  } 
  MPI_Finalize(); 
  // free(image); 
  // free(tmp_image); 
  // free(local_image); 
  // free(result_image); 
  // free(counts); 
  // free(displs); 
  // free(sendbuf); 
  // free(recvbuf); 
  exit(EXIT_SUCCESS); 
} 
