#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define MAX(A, B) ( (A)>(B) ? (A) : (B) )

#define idx1(i1, i0, Nd0) \
            ((i1) * (Nd0) + (i0))

#define idx2(i2, i1, i0, Nd1, Nd0) \
            ( (i2) * ( (Nd1) * (Nd0) ) + (i1) * (Nd0) + (i0) )

#define idx3( i3, i2, i1, i0, Nd2, Nd1, Nd0 ) \
           ( (i3) * ( (Nd2) * (Nd1) * (Nd0) ) + (i2) * ( (Nd1) * (Nd0) ) + (i1) * (Nd0) + (i0) )

#define idx4( i4, i3, i2, i1, i0, Nd3, Nd2, Nd1, Nd0)  \
           ( (i4) * ( (Nd3) * (Nd2) * (Nd1) * (Nd0) ) + (i3) * ( (Nd2) * (Nd1) * (Nd0) ) + (i2) * ((Nd1) * (Nd0)) + (i1) * (Nd0) + (i0) )

#define WX 2
#define WY 2

#define CONV_Nif1 1
#define CONV_Nix1 32
#define CONV_Niy1 32

#define CONV_Nkf1 1
#define CONV_Nkx1 5
#define CONV_Nky1 5

#define CONV_Nof1 6
#define CONV_Nox1 28
#define CONV_Noy1 28

#define CONV_Nif2 6
#define CONV_Nix2 14
#define CONV_Niy2 14

#define CONV_Nkf2 6
#define CONV_Nkx2 5
#define CONV_Nky2 5

#define CONV_Nof2 16
#define CONV_Nox2 10
#define CONV_Noy2 10

#define POOL_Nif3 16
#define POOL_Nix3 5
#define POOL_Niy3 5


#define FC_Nif1 400
#define FC_Nof1 120

#define FC_Nif2 120
#define FC_Nof2 84

#define FC_Nif3 84
#define FC_Nof3 10





#define STRIDE 1

double *loadData(char *path, int Nd3, int Nd2, int Nd1, int Nd0);
double *conv(double *ifm, double *w, double *bias, int Nof, int Nox, int Noy, int Nix, int Niy, int Nkf, int Nkx, int Nky, int s);
double *maxPool(double *input, int Nof, int Nox, int Noy, int Nix, int Niy, int wx, int wy);
double *fullyConnect(double *ifm, double *w, double *bias, int Nof, int Nif);
int Relu(double *input, int Nd2, int Nd1, int Nd0);
int show(double *arr, char *name, int dx, int dy);

int main()
{
    double *ifm1, *ifm2, *ifm3, *ifm4, *ifm5, *ifm6;
    double *ofm1, *ofm2;
    double *convker1, *convker2, *convB1, *convB2;
    double *fcw1, *fcw2, *fcw3, *fcb1, *fcb2, *fcb3;

    char path_ifm[]        = "./data/conv_input1.txt";
    char path_ker1[]       = "./data/conv_weight1.txt";
    char path_ker2[]       = "./data/conv_weight2.txt";
    char path_conv_bias1[] = "./data/conv_bias1.txt";
    char path_conv_bias2[] = "./data/conv_bias2.txt";
    char path_fc_weight1[] = "./data/fc_weight1.txt";
    char path_fc_weight2[] = "./data/fc_weight2.txt";
    char path_fc_weight3[] = "./data/fc_weight3.txt";
    char path_fc_bias1[]   = "./data/fc_bias1.txt";
    char path_fc_bias2[]   = "./data/fc_bias2.txt";
    char path_fc_bias3[]   = "./data/fc_bias3.txt";

    ifm1     = loadData(path_ifm,          1, CONV_Nif1, CONV_Nix1, CONV_Niy1);
    convker1 = loadData(path_ker1, CONV_Nof1, CONV_Nif1, CONV_Nkx1, CONV_Nky1);
    convker2 = loadData(path_ker2, CONV_Nof2, CONV_Nkf2, CONV_Nkx2, CONV_Nky2);
    convB1   = loadData(path_conv_bias1, CONV_Nof1, 1, 1, 1);
    convB2   = loadData(path_conv_bias2, CONV_Nof2, 1, 1, 1);

    fcw1    = loadData(path_fc_weight1, FC_Nof1, FC_Nif1, 1, 1);
    fcw2    = loadData(path_fc_weight2, FC_Nof2, FC_Nif2, 1, 1);
    fcw3    = loadData(path_fc_weight3, FC_Nof3, FC_Nif3, 1, 1);
    fcb1    = loadData(path_fc_bias1, FC_Nof1, 1, 1, 1);
    fcb2    = loadData(path_fc_bias2, FC_Nof2, 1, 1, 1);
    fcb3    = loadData(path_fc_bias3, FC_Nof3, 1, 1, 1);


    ofm1 = conv(ifm1, convker1, convB1, CONV_Nof1, CONV_Nox1, CONV_Noy1, CONV_Nix1, CONV_Niy1, CONV_Nkf1, CONV_Nkx1, CONV_Nky1, STRIDE);
    Relu(ofm1, CONV_Nof1, CONV_Nox1, CONV_Noy1);
    ifm2 = maxPool(ofm1, CONV_Nif2, CONV_Nix2, CONV_Niy2, CONV_Nox1, CONV_Noy1, WX, WY);
    
    ofm2 = conv(ifm2, convker2, convB2, CONV_Nof2, CONV_Nox2, CONV_Noy2, CONV_Nix2, CONV_Niy2, CONV_Nkf2, CONV_Nkx2, CONV_Nky2, STRIDE);
    Relu(ofm2, CONV_Nof2, CONV_Nox2, CONV_Noy2);
    ifm3 = maxPool(ofm2, POOL_Nif3, POOL_Nix3, POOL_Niy3, CONV_Nox2, CONV_Noy2, WX, WY);

    //show(ifm1, "ifm1", 32, 32);
    //show(ofm1, "ofm1 0", 28, 28);
    //show(ofm1+(28*28), "ofm1 0", 28, 28);
    //show(ofm1+2*(28*28), "ofm1 2", 28, 28);
    //show(ofm1+3*(28*28), "ofm1 3", 28, 28);
    //show(ofm1+4*(28*28), "ofm1 4", 28, 28);
    //show(ofm1+5*(28*28), "ofm1 5", 28, 28);
    
    ifm4 = fullyConnect(ifm3, fcw1, fcb1, FC_Nof1, FC_Nif1);
    Relu(ifm4, 1, 1, FC_Nif1);
    
    ifm5 = fullyConnect(ifm4, fcw2, fcb2, FC_Nof2, FC_Nif2);
    Relu(ifm5, 1, 1,FC_Nif2);

    ifm6 = fullyConnect(ifm5, fcw3, fcb3, FC_Nof3, FC_Nif3);
    
    show(ifm6, "result", 10, 1);
    
    return 0;
}


int Relu(double* input, int Nd2, int Nd1, int Nd0)
{
    int idx2, idx1, idx0;
    int i;
    for (idx2 = 0; idx2 < Nd2; idx2++)
    {
        for (idx1 = 0; idx1 < Nd1; idx1++)
        {
            for (idx0 = 0; idx0 < Nd0; idx0++)
            {
                i = idx2( idx2, idx1, idx0, Nd1, Nd0);
                if (input[i] <= 0.0)
                {
                    input[i] = 0.0;
                }
            }
        }
    }
    return 0;
}

double* maxPool(double *input, int Nof, int Nox, int Noy, int Nix, int Niy, int wx, int wy)
{
    double t = 0.0;
    int of, ox, oy, ix, iy;
    Nix = (Nix / wx);
    Niy = (Niy / wy);
    double* output = (double *) malloc(Nof * Nox * Noy * sizeof(double));
    printf("Nof=%d Nox=%d Noy=%d = %d\n", Nof, Nox, Noy, Nof * Nox * Noy);
    printf("Nof=%d Nix=%d Niy=%d = %d\n", Nof, Nix, Niy, Nof * Nix * Niy);
    for (of = 0; of < Nof; of++)
    {
        for (ox = 0; ox < Nox; ox++)
        {
            for (oy = 0; oy < Noy; oy++)
            {
                t = 0.0;
                for (ix = 0; ix < wx; ix++)
                {
                    for (iy = 0; iy < wy; iy++)
                    {
                        t = MAX(t, input[idx4(of, ox, ix, oy, iy, Nix, wx, Niy, wy)]);
                    }
                }
                output[ idx2(of, ox, oy, Nox, Noy) ] = t;
            }
        }
    }
    return output;
}

double* conv(double* ifm, double *w, double *bias, int Nof, int Nox, int Noy, int Nix, int Niy, int Nkf, int Nkx, int Nky, int s)
{
    int of, ox, oy, oz, kf, kx, ky;
    double *ofm = (double *)malloc(Nof * Nox * Noy * sizeof(double));
    printf("Nof:%d Nox:%d Noy:%d = %d\n", Nof, Nox, Noy, Nof * Nox * Noy);
    printf("Nkf:%d Nkx:%d Nky:%d = %d\n", Nkf, Nkx, Nky, Nkf * Nkx * Nky);
    printf("Nix:%d Niy:%d \n", Nix, Niy);
    memset((void *)ofm, 0, Nof * Nox * Noy * sizeof(double));
    printf("memset ok\n");
    for (of = 0; of < Nof; of++) {
        for (ox = 0; ox < Nox; ox++) {
            for (oy = 0; oy < Noy; oy++) {

                for (kf = 0; kf < Nkf; kf++) {
                    for (kx = 0; kx < Nkx; kx++) {
                        for (ky = 0; ky < Nky; ky++) {
                            ofm[idx2(of, ox, oy, Nox, Noy)] += \
                            ifm[idx2(kf, s * ox + kx, s * oy + ky, Nix, Niy)] * w[idx3(of, kf, kx, ky, Nkf, Nkx, Nky)];
                        }
                    }
                }
                ofm[idx2(of, ox, oy, Nox, Noy)] += bias[of];
                //printf("of:%d ox:%d oy:%d bias:%.1lf = %.4lf\n", of, ox, oy, bias[of], ofm[idx2(of, ox, oy, Nox, Noy)]);
            }
        }
    }
    return ofm;
}

/*
int loadIFM(double *OFM, int Nif,int Niy, int Nix) {

}
*/

double* fullyConnect(double *ifm, double *w, double *bias, int Nof, int Nif) {
    int ix, ox;
    double *ofm = (double *)malloc(Nof * sizeof(double));
    memset((void*)ofm, 0, Nof * sizeof(double));
    for (ox = 0; ox < Nof; ox++)
    {
        for (ix = 0; ix < Nif; ix++) {
            ofm[ox] += ifm[ix] * w[idx1(ox, ix, Nif)];
        }
        ofm[ox] += bias[ox];
    }
    return ofm;
}

double* loadData(char *path, int Nd3, int Nd2, int Nd1, int Nd0) {
    FILE *fp = NULL;
    int kf, kx, ky;
    double *t = (double *)malloc( Nd3 * Nd2 * Nd1 * Nd0 * sizeof(double));
    double d;

    ssize_t nread;
    size_t len;
    int i = 0, j = 0;
    char c;
    char line[128];
    printf("open %s\n", path);

    if ((fp = fopen(path, "r")) == NULL)
    {
        printf("open %s file error!!\n", path);
        return NULL;
    }
    printf("open file...\n");

    
    while ((nread = fscanf(fp, "%c", &c)) != -1) {
        double d = 0.0;
        if (c == '\n')
        {
            d = strtod( line, line+i);
            t[j] = d;
            //printf("%lf\n", t[j]);
            i = 0;
            j++;
        }
        else
        {
            line[i] = c;
            i++;
        }
    }

    fclose(fp);
    printf("%s load %d data success !\n", path, j);
    return t;
}



int show(double *arr, char* name, int dx, int dy) {
    int i, j;
    printf("\n%s====================\n", name);
    for (j = 0; j < dy; j++)
    {
        for (i=0; i<dx; i++) 
        {
            if(*(arr + (j * dx) + i) < 0.0) {
                printf("%2.1f ", *(arr + (j * dx) + i));
            } else {
                printf(" %2.1f ", *(arr + (j * dx) + i));
            }
        }
        printf("\n");
    }
    return 0;
}