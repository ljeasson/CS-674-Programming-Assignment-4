lib:
	gcc -Wall -shared -o fft.so -fPIC fft.c
	gcc -Wall -shared -o boxmuller.so -fPIC boxmuller.c
