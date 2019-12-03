lib:
	gcc -Wall -shared -o fft.so -fPIC fft.c
	gcc -Wall -shared -o boxmuller.so -fPIC -lm boxmuller.c
	gcc -Wall -shared -o spatialfilter.so -fPIC spatialfilter.c
