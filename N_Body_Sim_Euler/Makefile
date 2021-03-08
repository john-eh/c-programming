CFLAGS=-Wall
INCLUDES=-I/opt/X11/include
LDFLAGS=-L/opt/X11/lib -lX11 -lm -O3

N_Body_Problem_Euler: N_Body_Problem_Euler.o graphics.o
	gcc -o N_Body_Problem_Euler N_Body_Problem_Euler.o graphics.o $(LDFLAGS) 

N_Body_Problem_Euler.o: N_Body_Problem_Euler.c graphics.h
	gcc $(CFLAGS) $(INCLUDES) -c N_Body_Problem_Euler.c

graphics.o: graphics.c graphics.h
	gcc  -c graphics.c

clean:
	rm -f ./N_Body_Problem_Euler *.o
