/*
 * run_stopper.cpp
 *
 *  Created on: Oct 27, 2016
 *      Author: viki
 */

#include "Stopper.h"

int main(int argc, char **argv) {
    // Initiate new ROS node named "stopper"
    ros::init(argc, argv, "stopper");

    // Create new stopper object
    Stopper stopper;

    // Start the movement
    stopper.startMoving();

    return 0;
};



