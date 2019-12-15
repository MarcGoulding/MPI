int calc_local_cols(const int cols, const int size, const int rank){
  int local_cols = cols/size;
  int remainder  = cols%size;
  if (remainder != 0 && rank < remainder) local_cols++;
  return local_cols;
}

void calc_scatter_params(const int rows,const int cols,const int size, int *displs, int *sendcounts){

  int rank, rank_cols;
  for(rank=0; rank < size; rank++){
    rank_cols = calc_local_cols(cols,size,rank);
    sendcounts[rank] = rank_cols * rows;
    if (rank == 0) displs[rank] = 0;
    else {
      displs[rank] = displs[rank-1] + sendcounts[rank-1];
    }
  }
}

void stencil(const int nx, const int ny, const int width, const int height, float* image, float* tmp_image)
{
  // #pragma omp parallel for collapse(2)
  for (int i = 1; i < nx + 1; ++i) {
    #pragma vector aligned
    for (int j = 1; j < ny + 1; ++j){
      tmp_image[j + i * height] =  image[j + i * height] * 0.6f
      + (image[j - 1 + i * height]
      +  image[j + 1 + i * height]
      +  image[j + (i - 1) * height]
      +  image[j + (i + 1) * height]) * 0.1f;
    }
  }
}
// ORIGINAL:
// void stencil(const int nx, const int ny, const int width, const int height,
//              float* image, float* tmp_image)
// {
//   for (int j = 1; j < ny + 1; ++j) {
//     for (int i = 1; i < nx + 1; ++i) {
//       tmp_image[j + i * height] =  image[j     + i       * height] * 3.0 / 5.0;
//       tmp_image[j + i * height] += image[j     + (i - 1) * height] * 0.5 / 5.0;
//       tmp_image[j + i * height] += image[j     + (i + 1) * height] * 0.5 / 5.0;
//       tmp_image[j + i * height] += image[j - 1 + i       * height] * 0.5 / 5.0;
//       tmp_image[j + i * height] += image[j + 1 + i       * height] * 0.5 / 5.0;
//     }
//   }
// }


void zero_image(const int width, const int height, float* image){
  for (int j=0; j<width; j++){
    for (int i=0; i<height;i++){
      image[i+j*height] = 0.0f;
    }
  }
}
// Create the input image
void init_images(const int nx, const int ny, const int width, const int height, float* restrict image, float* restrict tmp_image)
{

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
void output_image(const char* file_name, const int nx, const int ny, const int width, const int height, float* restrict image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", file_name);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;

  // Calculate padding
  int startx,starty,endx,endy;
  if (width != nx){
    startx=1;
    endx=nx+1;
  }else{
    startx=0;
    endx=nx;
  }
  if (height != ny){
    starty=1;
    endy=ny+1;
  }else{
    starty=0;
    endy=ny;
  }

  for (int j = starty; j < endy; ++j) {
    for (int i = startx; i < endx; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = starty; j < endy; ++j) {
    for (int i = startx; i < endx; ++i) {
      fputc((char)(255.0f * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// ORIGINAL
void output_imageORIGINAL(const char* file_name, const int nx, const int ny,
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

