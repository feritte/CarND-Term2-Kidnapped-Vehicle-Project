/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
	
	const double init_weight = 1.0;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i<num_particles; i++){
		Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
		particles.push_back(p);
		weights.push_back(init_weight);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (unsigned i = 0; i < num_particles; i++) {

    	// calculate new state
    	if (fabs(yaw_rate) < 0.00001) {
      		particles[i].x += velocity * delta_t * cos(particles[i].theta);
      		particles[i].y += velocity * delta_t * sin(particles[i].theta);
    	}
    	else {
      		particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      		particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      		particles[i].theta += yaw_rate * delta_t;
    	}

    	// add noise
    	particles[i].x += dist_x(gen);
    	particles[i].y += dist_y(gen);
    	particles[i].theta += dist_theta(gen);
  	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs obs = observations[i];
		double min_dist = numeric_limits<double>::max();
		int map_id = -1;
		for (unsigned int j = 0; j < predicted.size(); j++) {
			// grab current prediction
			LandmarkObs pred = predicted[j];
			
			// get distance between current/predicted landmarks
			double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

			// find the predicted landmark nearest the current observed landmark
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = pred.id;
			}
		}


		observations[i].id = map_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (unsigned i=0; i<num_particles; i++){

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// 1. transform observations to map coordinates
		vector<LandmarkObs> observations_map;

		for (unsigned j=0; j<observations.size(); j++){
			int id_obs = observations[j].id;
			double x_obs = observations[j].x;
			double y_obs = observations[j].y;

			double x_map = p_x + x_obs*cos(p_theta) - y_obs*sin(p_theta);
			double y_map = p_y + x_obs*sin(p_theta) + y_obs*cos(p_theta);

			LandmarkObs o_new = {id_obs, x_map, y_map};
			observations_map.push_back(o_new);
		}
		// 2. find landmarks within the particle's range
		vector<LandmarkObs> landmarks_in_range;;
		for (unsigned j=0; j<map_landmarks.landmark_list.size(); j++){
			int lm_id = map_landmarks.landmark_list[j].id_i;
			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;

			if (dist(p_x, p_y, lm_x, lm_y) < sensor_range){
				landmarks_in_range.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}

		// 3. find which landmark is likely being observed : Data Association based on Nearest Neighbor 
		dataAssociation(landmarks_in_range, observations_map);

		// 4. Calculating the Particle's Final Weight based on the difference between particle observation and actual observation
		particles[i].weight = 1.0;

		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double na = 2.0 * std_x * std_x;
		double nb = 2.0 * std_y * std_y;
		double gauss_norm = 2.0 * M_PI * std_x * std_y;

		for (unsigned j=0; j<observations_map.size(); j++){
			int id_obs = observations_map[j].id;
			double x_obs = observations_map[j].x;
			double y_obs = observations_map[j].y;

			double pr_x, pr_y;
			for (unsigned int k = 0; k < landmarks_in_range.size(); k++) {
        		if (landmarks_in_range[k].id == id_obs) {
          			pr_x = landmarks_in_range[k].x;
          			pr_y = landmarks_in_range[k].y;
          			break;
        		}
      		}
      		double obs_w = 1/gauss_norm * exp( - (pow(pr_x-x_obs,2)/na + (pow(pr_y-y_obs,2)/nb)) );

      		// product of this obersvation weight with total observations weight
      		particles[i].weight *= obs_w;
		}

		weights[i] = particles[i].weight;
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;

  	// get all of the current weights
  	vector<double> weights;
  	for (int i = 0; i < num_particles; i++) {
    	weights.push_back(particles[i].weight);
  	}
	// discrete_distribution
  	discrete_distribution<int> index(weights.begin(), weights.end());
  	for (unsigned j=0; j<num_particles;j++){
  		const int i = index(gen);
  		new_particles.push_back(particles[i]);
  	}
	particles = new_particles;	
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
