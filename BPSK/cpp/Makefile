all:
	g++ bpsk_demod.cc -I/opt/homebrew/Cellar/libsndfile/1.2.2_1/include \
					  -I/opt/homebrew/Cellar/fftw/3.3.10_2/include \
					  -L/opt/homebrew/Cellar/fftw/3.3.10_2/lib -lfftw3 \
					  -L/opt/homebrew/Cellar/libsndfile/1.2.2_1/lib -lsndfile \
					  -o bpsk_demod.o
	./bpsk_demod.o
