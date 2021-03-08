CFLAGS=-Wall
INCLUDES=-I/opt/X11/include
LDFLAGS=-L/opt/X11/lib -lX11 -lm -O3 -lpthread -Wall

Barnes_Hut_Sim: Barnes_Hut_Sim.o graphics.o
	gcc -o Barnes_Hut_Sim Barnes_Hut_Sim.o graphics.o $(LDFLAGS) 

Barnes_Hut_Sim.o: Barnes_Hut_Sim.c graphics.h
	gcc $(CFLAGS) $(INCLUDES) -c Barnes_Hut_Sim.c

graphics.o: graphics.c graphics.h
	gcc  -c graphics.c

clean:
	rm -f ./Barnes_Hut_Sim *.o result.gal
