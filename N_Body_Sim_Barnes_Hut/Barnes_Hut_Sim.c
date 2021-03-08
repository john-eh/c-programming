#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h> 
#include <stdio.h>
#include <sys/sysinfo.h>
#include "graphics.h"

typedef struct thread_data {
	struct quad_node * head	;
	double * force_x;
	double * force_y;	
	double * target_x;
	double * target_y;
	int length;
	int index;
} thread_data;

typedef struct force2D {
	double x_force;
	double y_force;
} force2D; 

typedef struct mass2D {
	double center_of_mass_x;
	double center_of_mass_y;
	double mass;
} mass2D; 

// Structure for nodes in quad-tree 
typedef struct quad_node {
	int particle_no, index;
	double center_of_mass_x, center_of_mass_y, mass;
	double x_center, y_center, height;
	double particle_x, particle_y, particle_mass;

	struct quad_node * nw;
	struct quad_node * ne;
	struct quad_node * se;
	struct quad_node * sw;
} quad_node;

/* Structure of arrays for list of particles */
typedef struct particle_array {
	double *xpos;
	double *ypos;
	double *mass; 
	double *xvel;
	double *yvel;
	double *brightness;
} particle_array;

double gravity;
double epsilon = 1E-3;
double theta_max;
//int graphics;

const float circleRadius=0.0025,circleColor=0;
const int windowWidth=800;

int read_from_file(particle_array particle_array, int noParticles, const char * title){
	
	FILE * fp;
	fp = fopen(title, "r+");
  	if(!fp) {
    	printf("Error reading from file: failed to open input file '%s'.\n", title);
    	return -1;
  	}

    double read_array[6];
  	for (int i = 0; i < noParticles; ++i){

  		fread(read_array, 6*sizeof(double), 1, fp);

  		 particle_array.xpos[i] = read_array[0];
  		 particle_array.ypos[i] = read_array[1];
  		 particle_array.mass[i] = read_array[2];
  		 particle_array.xvel[i] = read_array[3];
  		 particle_array.yvel[i] = read_array[4];
  		 particle_array.brightness[i] = read_array[5];
    }
    fclose(fp);

  	return 0;
}

void save_to_file( particle_array particle_array, int noOfParticles){
	FILE * fp;
	fp = fopen("result.gal", "w+");

	for (int i = 0; i < noOfParticles; ++i){

	  	fwrite(&particle_array.xpos[i], sizeof(double), 1, fp);
	  	fwrite(&particle_array.ypos[i], sizeof(double), 1, fp);
	  	fwrite(&particle_array.mass[i], sizeof(double), 1, fp);
	  	fwrite(&particle_array.xvel[i], sizeof(double), 1, fp);
	  	fwrite(&particle_array.yvel[i], sizeof(double), 1, fp);
	  	fwrite(&particle_array.brightness[i], sizeof(double), 1, fp);

	}

	free(particle_array.xpos);
	free(particle_array.ypos);
	free(particle_array.mass);
	free(particle_array.xvel);
	free(particle_array.yvel);
	free(particle_array.brightness);

  	fclose(fp);
}

int find_quad(quad_node * current, double xpos, double ypos ){
	/* Function to find which quadrant particle with position (xpos,ypos) is in
	** Returns 0 for north-east, 1 for north-west, 2 for south-east, 3 for south-west
	** If particle is not in current -1 is returned */

	int res = 0;  

	if( xpos < current->x_center - current->height || xpos > current->x_center + current->height 
	 || ypos < current->y_center - current->height || ypos > current->y_center + current->height ){
		printf("ERROR : quad is [%0.3f,%0.3f]x[%0.3f,%0.3f] and pos is (%0.3f,%0.3f)  \n", current->x_center - current->height ,
	   current->x_center + current->height, current->y_center - current->height, current->y_center+current->height, xpos, ypos);
		return -1;
	}
	
	if (xpos < current->x_center){
		res += 1;
  } if (ypos < current->y_center){
    	res += 2;  
  }	

  return res;
}

 void print_tree(quad_node ** head){
 /* Prints current tree starting with head, 
 **	mostly for debugging */

	if ( *head == NULL){ return; }

	quad_node * current = *head;
	 
	print_tree(&current->nw);
	print_tree(&current->ne);
	print_tree(&current->se);
	print_tree(&current->sw);

  	printf("x: [%0.5f-%0.5f] y: [%0.5f-%0.5f] no: %d  Mass: %f   center: [%f,%f] ", current->x_center - current->height, 
  				current->x_center+ current->height, current->y_center - current->height, current->x_center + current->height, 
  								current->particle_no, current->mass, current->center_of_mass_x, current->center_of_mass_y);
  	
  	if (current->particle_no == 1){
  	printf(" pos: (%0.3f,%0.3f) ",current->particle_x, current->particle_y );
  	}
  	printf("\n");
	
}

void  _delete_tree(quad_node * current){

	quad_node * test = current;

	if ( current == NULL){
		return;
	}

	_delete_tree(test->nw);
	
	_delete_tree(test->ne);
	 
	_delete_tree(test->se);
	
	_delete_tree(test->sw);

	free(current);
	
}

void delete_tree(quad_node ** current){
	/* Uses function _delete_tree and sets pointer to NULL */

	_delete_tree(*current);
	*current = NULL;
}

quad_node * new_node(double x_center, double y_center, double height, double particle_x, 
double particle_y, double particle_mass, int index ){
	/* Creates new quad-node with variables for position and size of box and particle values */

	quad_node * node;
	node =  (quad_node*)malloc( sizeof(quad_node));
	node->particle_no = 1;

	node->index = index;
	node->height = height;
	node->x_center = x_center;
	node->y_center = y_center;

	node->particle_x = particle_x;
	node->particle_y = particle_y;
	node->particle_mass =particle_mass;

	node->nw = NULL; node->ne = NULL; node->sw = NULL; node->se = NULL;
	return &(*node);
}


quad_node * insert(quad_node * head, double x_center, double y_center, double height, 
double new_particle_x, double new_particle_y, double new_particle_mass, int new_index	){
	/*	Inserts new particle into quad-tree */

	/* If current node is NULL, a new node is created */
 	if (head == NULL){

 		return new_node(x_center, y_center, height, new_particle_x, new_particle_y, new_particle_mass , new_index);
 	} 

 	quad_node * current = head;

 	/* If node currently has 1 particle, both old and new particles are passed down to further nodes */
 	 if( ( head )->particle_no == 1 ){

 	 	int new_particle_pos = find_quad(current,new_particle_x, new_particle_y );
 	 	int old_particle_pos = find_quad(current, current->particle_x, current->particle_y );

  	 	if (new_particle_pos == 0){

 	 		current->ne = insert( current->ne, 	x_center + 0.5*height, 	y_center + 0.5*height, 	0.5 * height, 
 	 											new_particle_x, new_particle_y, new_particle_mass, new_index);
 	 	} else if (new_particle_pos == 1){

 	  		current->nw = insert( current->nw, 	x_center - 0.5*height, 	y_center + 0.5*height, 	0.5 * height, 
 	  											new_particle_x, new_particle_y, new_particle_mass, new_index);
 	  	} else if (new_particle_pos == 2){

 	  		current->se = insert( current->se, 	x_center + 0.5*height, 	y_center - 0.5* height, 	0.5 * height, 
 	  											new_particle_x, new_particle_y, new_particle_mass, new_index);
		}  else if (new_particle_pos == 3){

 	  		current->sw = insert( current->sw, 	x_center - 0.5*height, 	y_center - 0.5*height, 	0.5 * height, 
 	  											new_particle_x, new_particle_y, new_particle_mass, new_index);
		  }

		if (old_particle_pos == 0){
 			current->ne = insert(current->ne, 	x_center + 0.5*height, 	y_center + 0.5*height, 	0.5 * height, 
 											current->particle_x, current->particle_y, current->particle_mass, current->index);
 		} else if (old_particle_pos == 1){
 			current->nw = insert(current->nw,  	x_center - 0.5*height, 	y_center + 0.5*height, 	0.5 * height, 
 											current->particle_x, current->particle_y, current->particle_mass, current->index);
 		} else if (old_particle_pos == 2){
 			current->se = insert(current->se,  	x_center + 0.5*height, 	y_center - 0.5*height, 	0.5 * height, 
 											current->particle_x, current->particle_y, current->particle_mass, current->index);
		}  else if (old_particle_pos == 3){
 			current->sw = insert(current->sw, 	x_center - 0.5*height, 	y_center - 0.5*height, 	0.5 * height, 
 											current->particle_x, current->particle_y, current->particle_mass, current->index);
		}
		(current)->particle_no +=1;
 		return current;
 	} 

 	/* If node includes multiple particles, current particle is passed down to next node in line */
 	else {

 		int new_particle_pos = find_quad(current, new_particle_x, new_particle_y );

 		if (new_particle_pos == 0){
 			current->ne = insert(current->ne, x_center + 0.5*height, y_center + 0.5*height, 0.5 * height, 
 														new_particle_x, new_particle_y, new_particle_mass, new_index);
 		} else if (new_particle_pos == 1){
 			current->nw = insert(current->nw, x_center - 0.5*height, y_center + 0.5*height, 0.5 * height, 
 														new_particle_x, new_particle_y, new_particle_mass, new_index);
 		} else if (new_particle_pos == 2){
 			current->se = insert(current->se, x_center + 0.5*height, y_center - 0.5*height, 0.5 * height, 
 														new_particle_x, new_particle_y, new_particle_mass, new_index);
		}  else if (new_particle_pos == 3){
 			current->sw = insert(current->sw, x_center - 0.5*height, y_center - 0.5*height, 0.5 * height, 
 														new_particle_x, new_particle_y, new_particle_mass, new_index);
		}
		current->particle_no += 1;

 	}

 	return current;
 }


 mass2D _calc_mass(quad_node * current){
 	/* Calculates mass of node and returns it to subsequent node */

 	mass2D mass2D_array;

 	/* If node has a particle the mass is the same as particle */
	if(current->particle_no == 1){
		current->center_of_mass_x = current->particle_x;
		current->center_of_mass_y = current->particle_y;			
		current->mass = current->particle_mass;

		mass2D_array.center_of_mass_x = current->center_of_mass_x ;
		mass2D_array.center_of_mass_y = current->center_of_mass_y;
		mass2D_array.mass = current->mass;

	/* Otherwise mass is sum of all child-nodes masses, which are computed the same way */
	} else {
		mass2D nw_array, ne_array, se_array, sw_array;

		nw_array.center_of_mass_x = nw_array.center_of_mass_y = nw_array.mass = 0;
		ne_array.center_of_mass_x = ne_array.center_of_mass_y = ne_array.mass = 0;
		sw_array.center_of_mass_x = sw_array.center_of_mass_y = sw_array.mass = 0;
		se_array.center_of_mass_x = se_array.center_of_mass_y = se_array.mass = 0;

		if (current->nw != NULL){
			nw_array = _calc_mass(current->nw); 
	  } if (current->ne != NULL){
			ne_array = _calc_mass(current->ne); 
	  }	if (current->sw != NULL){
		    sw_array = _calc_mass(current->sw); 
	  }	if (current->se != NULL){
			se_array = _calc_mass(current->se); 
		}	

		current->mass = nw_array.mass +  ne_array.mass +  se_array.mass +  sw_array.mass;

		current->center_of_mass_x = nw_array.mass*nw_array.center_of_mass_x + 
									ne_array.mass*ne_array.center_of_mass_x + 
									se_array.mass*se_array.center_of_mass_x + 
									sw_array.mass*sw_array.center_of_mass_x;

		current->center_of_mass_y = nw_array.mass*nw_array.center_of_mass_y + 
									ne_array.mass*ne_array.center_of_mass_y + 
									se_array.mass*se_array.center_of_mass_y + 
									sw_array.mass*sw_array.center_of_mass_y;


		current->center_of_mass_x = current->center_of_mass_x / current->mass;
		current->center_of_mass_y = current->center_of_mass_y / current->mass;
		
		mass2D_array.center_of_mass_x = current->center_of_mass_x; 
		mass2D_array.center_of_mass_y = current->center_of_mass_y;
		mass2D_array.mass = current->mass; 
	}
	return mass2D_array;
}


void calc_mass( quad_node * current ){
	/* Calculates mass for head node, as it does not return any value */

	if(current->particle_no == 1){
		current->center_of_mass_x = current->particle_x;
		current->center_of_mass_y = current->particle_y;			
		current->mass = current->particle_mass;
	} 
	else {

		mass2D nw_array, ne_array, se_array, sw_array;

		nw_array.center_of_mass_x = nw_array.center_of_mass_y = nw_array.mass = 0;
		ne_array.center_of_mass_x = ne_array.center_of_mass_y = ne_array.mass = 0;
		sw_array.center_of_mass_x = sw_array.center_of_mass_y = sw_array.mass = 0;
		se_array.center_of_mass_x = se_array.center_of_mass_y = se_array.mass = 0;

		if (current->nw != NULL){
			nw_array =  _calc_mass(current->nw); 
	  } if (current->ne != NULL){
			ne_array = _calc_mass(current->ne); 
	  } if (current->sw != NULL){
		    sw_array = _calc_mass(current->sw); 
	  } if (current->se != NULL){
			se_array = _calc_mass(current->se); 
	  }	

	  	/* Mass and mass-centre are calculated based on sub-nodes values */
		current->mass = nw_array.mass + ne_array.mass + se_array.mass + sw_array.mass;

		current->center_of_mass_x = nw_array.mass*nw_array.center_of_mass_x + 
									ne_array.mass*ne_array.center_of_mass_x + 
									se_array.mass*se_array.center_of_mass_x + 
									sw_array.mass*sw_array.center_of_mass_x;

		current->center_of_mass_y = nw_array.mass*nw_array.center_of_mass_y + 
									ne_array.mass*ne_array.center_of_mass_y + 
									se_array.mass*se_array.center_of_mass_y + 
									sw_array.mass*sw_array.center_of_mass_y;

		current->center_of_mass_x = current->center_of_mass_x / current->mass;
		current->center_of_mass_y = current->center_of_mass_y / current->mass;		
	}
}

force2D calc_forces (quad_node * current , double target_x, double target_y, int target_index){
	/* Calculates forces on particle with coords target_x and target_y from current node */
	
 	force2D force2d_array;
 	double rdist, rdist3, d;

 	/* If only one particle in node, forces are directly calculated */
 	if(current->particle_no == 1){

 		if( current->index != target_index ){
 			
 			double xdist = target_x- current->particle_x;
 			double ydist = target_y - current->particle_y;
		
			rdist = sqrt(xdist*xdist + ydist*ydist);
			rdist3 = 1.0/((rdist + epsilon)*(rdist + epsilon)*(rdist + epsilon));

 			force2d_array.x_force = current->particle_mass* xdist * rdist3;
 			force2d_array.y_force = current->particle_mass* ydist * rdist3;	
 			return force2d_array; 	
 		
 		} else {
			force2d_array.x_force = 0; force2d_array.y_force = 0; 	
			return force2d_array;
 		}

 	/* If multiple particles, check theta if node can be computed as one particle */
 	} else {
 		double xdist = target_x- current->center_of_mass_x;
 		double ydist = target_y - current->center_of_mass_y;
 		rdist = sqrt(xdist*xdist + ydist*ydist);
 		d = 2*current->height;

 		if ( d < rdist*theta_max ){	       		
			rdist3 = 1.0/((rdist + epsilon)*(rdist + epsilon)*(rdist + epsilon));
       		force2d_array.x_force = current->mass*xdist*rdist3;
 			force2d_array.y_force = current->mass*ydist*rdist3;	
 			return force2d_array;
 		} 
 		/* If not, forces are sum of child-nodes forces */
 		else{
			
			force2D nw_array, ne_array, se_array, sw_array;

			nw_array.x_force = 0; nw_array.y_force = 0; 
			ne_array.x_force = 0; ne_array.y_force = 0; 
		    se_array.x_force = 0; se_array.y_force = 0; 
			sw_array.x_force = 0; sw_array.y_force = 0; 
		
			if (current->nw != NULL ){
				nw_array = calc_forces(current->nw, target_x, target_y, target_index); 
		  } if (current->ne != NULL ){
				ne_array = calc_forces(current->ne, target_x, target_y, target_index); 
		  } if (current->se != NULL   ){
				se_array = calc_forces(current->se, target_x, target_y, target_index); 
		  } if (current->sw != NULL  ){
				sw_array = calc_forces(current->sw, target_x, target_y, target_index); 
		  }		

			force2d_array.x_force = nw_array.x_force + ne_array.x_force + se_array.x_force + sw_array.x_force;
			force2d_array.y_force = nw_array.y_force + ne_array.y_force + se_array.y_force + sw_array.y_force;
 		}

 	}

 	return force2d_array;
 }

void  *calc_forces_thread(void *threadarg){
	/* pthread function, computes forces acting on all particles with coords in target_x, target_y 
	 * Values are written to array of force structs which are accessible in euler_step  */
 	thread_data *my_data=(thread_data *)threadarg;
 	quad_node * current = my_data->head;
 	double * target_x = my_data->target_x; 
 	double * target_y = my_data->target_y;
 	int length = my_data->length;

 	struct force2D* force2d_array2 = malloc(length * sizeof(struct force2D));

	for (int i = 0; i < length; ++i)
	 	{
	 		
	 		force2d_array2[i] = calc_forces( current, target_x[i], target_y[i] , -1);
	 		
	 		my_data->force_x[i] = force2d_array2[i].x_force; my_data->force_y[i] = force2d_array2[i].y_force; 
	 	} 	
	 free(force2d_array2);
	 pthread_exit(NULL);
 }

void euler_step( particle_array particle_array , double delta_k , int noOfParticles, int threadNo, int steps, int graphics){

	quad_node *head;
	thread_data thread_data_array[threadNo];
	pthread_t threads[threadNo];
	int thread_length = noOfParticles/threadNo;
	float L=1, W=1;

	// initialize and allocate memory for arrays that will be needed
	double * xpos_array[threadNo];
	double * ypos_array[threadNo];
	double * force_array[2];
	force_array[0] = malloc( noOfParticles * sizeof(double));
	force_array[1] = malloc( noOfParticles * sizeof(double));
	for (int i = 0; i < threadNo; ++i){
		xpos_array[i] = malloc( thread_length * sizeof(double));
		ypos_array[i] = malloc( thread_length * sizeof(double));
		thread_data_array[i].force_x = malloc( thread_length * sizeof(double));
		thread_data_array[i].force_y = malloc( thread_length * sizeof(double));	
	}



	for (int i = 0; i < steps; ++i){

	 	// 1. build quadtree
	 	head = new_node( 0.5, 0.5, 0.5, particle_array.xpos[0], particle_array.ypos[0], particle_array.mass[0], 0);

	 	for (int i = 1; i < noOfParticles; ++i){
		 	insert( head, 0.5, 0.5, 0.5, particle_array.xpos[i] ,particle_array.ypos[i], particle_array.mass[i], i);
		}

	 	// 2. calculate mass and mass centers
		calc_mass( head );

		// 3. calculate forces
		for (int j = 0; j < threadNo; ++j){
				
			// store x and y positions in struct that can be passed to pthread function
			memcpy(xpos_array[j], particle_array.xpos + j*thread_length, sizeof(double)*thread_length );
			thread_data_array[j].target_x = xpos_array[j];
			memcpy(ypos_array[j], particle_array.ypos + j*thread_length, sizeof(double)*thread_length );
			thread_data_array[j].target_y = ypos_array[j];

			thread_data_array[j].length = thread_length;
			thread_data_array[j].index = j; 
			thread_data_array[j].head = head;

			//Each thread calculates forces for allocated particles
			pthread_create(&threads[j], NULL, calc_forces_thread, (void *)&thread_data_array[ j ]);
		
		} 

		for (int i = 0; i < threadNo; ++i){

			pthread_join(threads[i], NULL );
			
			for (int j = 0; j < thread_length; ++j){			
				force_array[0][thread_length*i + j] = thread_data_array[i].force_x[j];
				force_array[1][thread_length*i + j]=  thread_data_array[i].force_y[j];
			}
		}

		 // 4. take euler step
		  for (int i = 0; i < noOfParticles; ++i){

		  	particle_array.xvel[i] -= (delta_k *gravity * force_array[0][i]) ;
		  	particle_array.yvel[i] -= (delta_k *gravity * force_array[1][i] ) ;

		  	particle_array.xpos[i] +=  delta_k * particle_array.xvel[i];
		  	particle_array.ypos[i] +=  delta_k * particle_array.yvel[i];

		  }

	 	// 5. delete quadtree
		delete_tree(&head);

		if (graphics == 1){
			ClearScreen();
			for (int i = 0; i < noOfParticles;++i){
				DrawCircle(particle_array.xpos[i],particle_array.ypos[i], L, W, circleRadius, circleColor);
			}
			Refresh();
			usleep(15000);
		}


	}

	// Frees all after steps are taken
	free(force_array[0]);
	free(force_array[1]);

	for (int i = 0; i < threadNo; ++i){
		free( xpos_array[i] );
		free( ypos_array[i] );
		free(thread_data_array[i].force_x);
		free(thread_data_array[i].force_y);	
	}
 }

int main(int argc, char const *argv[]){

	if( argc != 7 && argc != 8  ){
		printf("Error, wrong input \n");
		printf("input arguments should be  6 or 7 (int noOfParticles, char fileName,");
		printf("int nSteps, double deltaK, double delta, int graphics, ( and optional int threads) ) \n");
		return -1;
	}

	int noOfParticles = atoi(argv[1]);
	const char * fileName = argv[2];
	int nSteps = atoi(argv[3]);
	double delta_k = atof(argv[4]);
	double theta = atof(argv[5]);
	int graphics = atoi(argv[6]);
	int threadNo;
	if (argc == 8){
		threadNo = atoi(argv[7]);
    } else{ 
   		double a = noOfParticles; double b = get_nprocs();
  		if ( ceil(a/b) == floor(a/b)	) {
  	  		threadNo =  get_nprocs();
  	  	} else {
			threadNo =  1;
		}
	}

	gravity = 100.0/noOfParticles;
	theta_max = theta;

	if (graphics == 1){
  		InitializeGraphics((char *)argv[0],windowWidth,windowWidth);
  		SetCAxes(0,1);	
  	}


  	particle_array particle_array;
  	particle_array.xpos = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_array.ypos = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_array.mass = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_array.xvel = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_array.yvel = (double *)malloc( sizeof(double)*noOfParticles ); 
  	particle_array.brightness = (double *)malloc( sizeof(double)*noOfParticles ); 
  	

	if (read_from_file(particle_array, noOfParticles, fileName) == -1){
		printf("problem reading %s\n", fileName );
		return -1;
	}

	euler_step( particle_array, delta_k , noOfParticles, threadNo , nSteps, graphics);
	
	save_to_file(particle_array, noOfParticles);

	if (graphics == 1){
 		FlushDisplay();
  		CloseDisplay();
  	}

	pthread_exit(NULL);

	return 0;
}