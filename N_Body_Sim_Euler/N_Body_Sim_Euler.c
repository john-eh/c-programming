
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "graphics.h"

typedef struct particle {
	double *xpos;
	double *ypos;
	double *mass;
	double *xvel;
	double *yvel;
	double *brightness;
} particle;

double gravity;
double epsilon = 1E-3;

const float circleRadius=0.0025,circleColor=0;
const int windowWidth=800;

//double compute_energy(){}

int set_particles(particle  particle_list, int noParticles, const char * title){

	FILE * fp;
	fp = fopen(title, "r+");
  	if(!fp) {
    	printf("Error reading from file: failed to open input file '%s'.\n", title);
    	return -1;
  	}

  	double read_array[6];
  	for (int i = 0; i < noParticles; ++i){

  		fread(read_array, 6*sizeof(double), 1, fp);

  		 particle_list.xpos[i] = read_array[0];
  		 particle_list.ypos[i] = read_array[1];
  		 particle_list.mass[i] = read_array[2];
  		 particle_list.xvel[i] = read_array[3];
  		 particle_list.yvel[i] = read_array[4];
  		 particle_list.brightness[i] = read_array[5];
    }

  	fclose(fp);

  	return 0;
}

static void compute_forces(double x_acceleration[], double *y_acceleration,  particle particle_list, int noOfParticles){

	for(int ii = 0; ii < noOfParticles; ++ii){

	double x_acc = 0;
	double y_acc = 0;
	for(int jj = 0; (jj < noOfParticles) ; ++jj){
		if(jj == ii){
			continue;
			}
			double xdist, ydist, rdist, rdist3 = 0;		
			xdist = particle_list.xpos[ii] - particle_list.xpos[jj];
			ydist = particle_list.ypos[ii] - particle_list.ypos[jj];
			rdist = sqrt(xdist*xdist + ydist*ydist);
			//	rdist3 = pow(rdist + epsilon, -3);
			rdist3 = 1.0/((rdist + epsilon)*(rdist + epsilon)*(rdist + epsilon));

			x_acc += particle_list.mass[jj]*xdist*rdist3;
			y_acc += particle_list.mass[jj]*ydist*rdist3;

		}
	x_acceleration[ii] = -gravity*x_acc;
	y_acceleration[ii] = -gravity*y_acc;
	}
}

void euler_step(particle  particle_list, double stepSize, int noOfParticles, int nSteps, double x_acceleration[], double *y_acceleration){

	compute_forces(x_acceleration, y_acceleration, particle_list, noOfParticles);

	double new_xvel, new_yvel, new_xpos, new_ypos;
 	for (int i = 0; i < noOfParticles; ++i){

		new_xvel = particle_list.xvel[i] + stepSize*x_acceleration[i];
 		new_yvel = particle_list.yvel[i] + stepSize*y_acceleration[i];

 		new_xpos = particle_list.xpos[i] + stepSize*new_xvel;
		new_ypos = particle_list.ypos[i] + stepSize*new_yvel;

		particle_list.xvel[i] = new_xvel;
		particle_list.yvel[i] = new_yvel;
 		particle_list.xpos[i] = new_xpos;
		particle_list.ypos[i] = new_ypos;
	}

}

void save_to_file(particle particle_list, int noOfParticles){

	FILE * fp;
   	fp = fopen("result.txt", "w+");

	for (int i = 0; i < noOfParticles; ++i){
	  	// fwrite(&particle_list.xpos[i], sizeof(double), 1, fp);
	  	// fwrite(&particle_list.ypos[i], sizeof(double), 1, fp);
	  	// fwrite(&particle_list.mass[i], sizeof(double), 1, fp);
	  	// fwrite(&particle_list.xvel[i], sizeof(double), 1, fp);
	  	// fwrite(&particle_list.yvel[i], sizeof(double), 1, fp);
	  	// fwrite(&particle_list.brightness[i], sizeof(double), 1, fp);

	  	fprintf(fp, "%f %f %f %f \n", particle_list.xpos[i], particle_list.ypos[i], particle_list.xvel[i], particle_list.yvel[i]);

	}
   	fclose(fp);
}

double compute_energy( particle particle_list, int noOfParticles){

    double hamiltonian = 0;

	for(int ii = 0; ii < noOfParticles; ++ii){

		double vec_norm = particle_list.xvel[ii]*particle_list.xvel[ii] + particle_list.yvel[ii]*particle_list.yvel[ii];
		double kin_energy = vec_norm/(2.0*particle_list.mass[ii]);
		double pot_energy = 0;

		for(int jj = 0; (jj < noOfParticles) ; ++jj){
			if(jj == ii){
				continue;
			}
				double xdist, ydist, rdist = 0;		
				xdist = particle_list.xpos[ii] - particle_list.xpos[jj];
				ydist = particle_list.ypos[ii] - particle_list.ypos[jj];
				rdist = sqrt(xdist*xdist + ydist*ydist);
			
				pot_energy += particle_list.mass[ii]/rdist;
			
		}

		pot_energy = -gravity*particle_list.mass[ii]*pot_energy;
		
		hamiltonian += kin_energy + pot_energy;
	}
 hamiltonian = hamiltonian/noOfParticles;
 return hamiltonian;
}



int main(int argc, char  *argv[]) {

	//clock_t tic = clock();
	if( argc != 6){
		printf("input arguments should be 5 (int noOfParticles char fileName int nSteps double deltaK int graphics)\n");
		return -1;
	}

	int noOfParticles = atoi(argv[1]);
	const char * fileName = argv[2];
	int nSteps = atoi(argv[3]);
	double delta_k = atof(argv[4]);
	int graphics = atoi(argv[5]);
	gravity = 100.0/noOfParticles;
	float L=1, W=1;

  	particle particle_list;
  	particle_list.xpos = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_list.ypos = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_list.mass = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_list.xvel = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_list.yvel = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_list.brightness = (double *)malloc( sizeof(double)*noOfParticles ); 
  	//double energy[nSteps];

  	if(  set_particles(particle_list, noOfParticles, fileName) != 0){
  		printf("Error: Couldn't read input file\n");
		return -1;
	}


	double *x_acceleration, *y_acceleration;
	x_acceleration = (double *)malloc( sizeof(double)*noOfParticles ); 
	y_acceleration = (double *)malloc( sizeof(double)*noOfParticles ); 


	if (graphics == 1){
  		InitializeGraphics(argv[0],windowWidth,windowWidth);
  		SetCAxes(0,1);

		int j = 0;
		 while(!CheckForQuit() && j < 200) {
			ClearScreen();
			for (int i = 0; i < noOfParticles;++i){
				DrawCircle(particle_list.xpos[i],particle_list.ypos[i], L, W, circleRadius, circleColor);
			//DrawCircle(particle_list.xpos[i],particle_list.ypos[i], L, W, particle_list.mass[i], particle_list.brightness[i]);

			}
			Refresh();
			usleep(15000);
			euler_step(particle_list, delta_k, noOfParticles,  nSteps, x_acceleration,y_acceleration);
			j++;
		}

	} else{

	  	for (int i = 0; i < nSteps; ++i){
	  		euler_step(particle_list, delta_k, noOfParticles,  nSteps, x_acceleration,y_acceleration);
  		}
	}

  	save_to_file(particle_list, noOfParticles);

 	free( particle_list.xpos );
  	free( particle_list.ypos );
  	free( particle_list.mass );
  	free( particle_list.xvel );
  	free( particle_list.yvel );
  	free( particle_list.brightness );

  	free( x_acceleration);
  	free( y_acceleration);
  	if (graphics == 1)
  	{
 		FlushDisplay();
  		CloseDisplay();
  	}
  	 //  clock_t toc = clock();
   	// printf("Elapsed time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
 	return 0;
}


