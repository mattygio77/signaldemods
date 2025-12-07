
#include <iostream>
#include <vector>
#include <complex>
#include <fstream>

#include <sndfile.h>
#include <fftw3.h>

using std::cout;
using std::endl;
using std::vector;
using std::complex;

int main(int argc, char** argv){

    const char* filename;
    if (argc < 2 ){
       filename = "../BPSK-Decode/BPSK_IQ_Fs48KHz.wav";
    } else {
       filename = argv[1];
    }

    cout << filename << endl;
    SF_INFO info{};

    SNDFILE* file = sf_open(filename, SFM_READ, &info);

    if (!file) {
        std::cerr << "Failed to open: " << sf_strerror(nullptr) << "\n";
        return 1;
    }

    cout << "Channels:       " << info.channels << "\n";
    cout << "Sample Rate:    " << info.samplerate << "\n";
    cout << "Frames:         " << info.frames << "\n";
    cout << "Format:         " << info.format << "\n";

    int format = info.format & SF_FORMAT_SUBMASK;

    switch (format) {
        case SF_FORMAT_PCM_U8:
            std::cout << "Unsigned 8-bit PCM\n";
            break;
        case SF_FORMAT_PCM_16:
            std::cout << "Signed 16-bit PCM\n";
            break;
        case SF_FORMAT_PCM_24:
            std::cout << "Signed 24-bit PCM\n";
            break;
        case SF_FORMAT_PCM_32:
            std::cout << "Signed 32-bit PCM\n";
            break;
        case SF_FORMAT_FLOAT:
            std::cout << "32-bit float PCM\n";
            break;
    }

    vector<complex<int16_t>> samples(info.frames * info.channels / 2);
    sf_count_t readCount = sf_read_short(file, reinterpret_cast<short*>(samples.data()), samples.size()*2);
    sf_close(file);

    std::cout << "Read " << readCount << " samples\n";

    std::cout << "\nFirst 5 stereo frames (L, R):\n";
    for (int i = 0; i < 10 && i < readCount; i ++) {
        int16_t left  = samples[i].real();
        int16_t right = samples[i].imag();
        std::cout << left << ", " << right << "\n";
    }

    int N = info.frames;
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for (int n = 0; n < N; n++) {
        in[n][0] = samples[n].real();   // real part
        in[n][1] = samples[n].imag();   // imag part
    }

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    std::ofstream specFile("spectrum.csv");

    vector<double> mag_shifted(info.frames);
    // cout << "FFT Results: " << endl;
    for(int idx = 0; idx < N; idx++){
        double re = out[idx][0];
        double im = out[idx][1];
        mag_shifted[(idx + N/2)%N] = std::sqrt(re*re + im*im);
        // if(idx < 10){
        //     cout << "Idx: " << idx << ", " << "Shifted Index: " << (idx + N/2)%N << endl;
        // }
        // specFile << mag_shifted[(idx +N/2)%N] << "\n";
    }

    for (int idx = 0; idx < mag_shifted.size(); idx++){
        specFile << mag_shifted[idx] << "\n";
    }

    specFile.close();

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

}