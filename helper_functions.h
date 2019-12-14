int calc_local_cols(const int cols, const int size, const int rank){
  int local_cols = cols/size;
  int remainder  = cols%size;
  if (remainder != 0 && rank < remainder) local_cols++;
  return local_cols;
}

void calc_scatter_params(const int rows,const int cols,const int size, int *displs, int *sendcounts){

  int rank;
  int rank_cols;
  for(i=0; rank < size; rank++){
    rank_cols = calc_local_cols(cols,size,rank);
    // add this to displs[] array
    sendcounts[rank] = rank_cols * rows;
  }

  for(rank=0; rank < size; rank++){
    if (rank == 0) displs[rank] = 0;
    else {
      displs[rank] = displs[rank-1] + sendcounts[rank-1];
    }
  }
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* restrict tmp_image)
{
  //omp_nodes_cores = 56 ???? environment variable in terminal
  for (int i = 1; i < nx + 1; ++i) {
    #pragma vector aligned
    #pragma omp parallel for
     for (int j = 1; j < ny + 1; ++j){
      tmp_image[j + i * height] =  image[j + i * height] * 0.6f
      + (image[j - 1 + i * height]
      + image[j + 1 + i * height]
      + image[j + (i - 1) * height]
      + image[j + (i + 1) * height]) * 0.1f;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* restrict image, float* restrict tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0f;
      tmp_image[j + i * height] = 0.0f;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0f;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* restrict image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0f * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
