//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Initial experiments implementing PFNN player 
// 
// Phase-Functioned Neural Networks for Character Control
//
// PFNN system designed by Daniel Holden
//
// http://theorangeduck.com/page/phase-functioned-neural-networks-character-control
//
// Implemented in HiFi by David Wooldridge, November 2017
//
//
//

print("PFNN: Starting up.");

// Urgh! I give up! Hard coding locallib functions for now
//Script.require("./libraries/pfnnApi.js");
//Script.include(Script.resolvePath('./libraries/Matrix3.js'));
//Script.include('./libraries/Matrix3.js');

var glm = Script.require('https://git.io/glm-js.min.js');
quat_exp = function(vectorThree) {
    var w = glm.length(vectorThree);
    var q = w < 0.01 ? 
        glm.quat(1, 0, 0, 0) : 
        glm.quat(
            Math.cos(w),
            vectorThree.x * (Math.sin(w) / w),
            vectorThree.y * (Math.sin(w) / w),
            vectorThree.z * (Math.sin(w) / w)
        );
    return q / Math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}


// precomputation techniques (aka mode) for PFNN
const MODE_CONSTANT = 0; // default
const MODE_LINEAR = 1;   // not yet supported (uses interpolation to trade off memory and processing)
const MODE_CUBIC = 2;    // not yet supported (uses interpolation to trade off memory and processing)

const PATH_TO_DATA = "./pfnn-data/";


PFNN = function(precomputation_technique) {

    var that = {};
    //var data_url = "http://davedub.co.uk/highfidelity/pfnn/";
    
    var mode = precomputation_technique;

    // array sizes
    const XDIM = 342; // need to check / adjust for HiFi avatar number of joints
    const YDIM = 311;
    const HDIM = 512;

    that.XDIM = 342; // need to check / adjust for HiFi avatar number of joints
    that.YDIM = 311;
    that.HDIM = 512;

    // TODO: Check if array initialisation is still necessary here

    // PFNN inputs and outputs
    var Xmean = new Array(XDIM);
    var Xstd = new Array(XDIM);
    var Ymean = new Array(YDIM);
    var Ystd = new Array(YDIM);

    // containers for weights (loaded as separate files)
    var W0; // array size dependent on precomputation technique, so not initialised yet
    var W1; // array size dependent on precomputation technique, so not initialised yet
    var W2; // array size dependent on precomputation technique, so not initialised yet

    // 
    var b0; // array size dependent on precomputation technique, so not initialised yet
    var b1; // array size dependent on precomputation technique, so not initialised yet
    var b2; // array size dependent on precomputation technique, so not initialised yet

    // PFNN input and output arrays
    var Xp = new Array(XDIM);
    var Yp = new Array(YDIM);

    // 
    var H0 = new Array(HDIM);
    var H1 = new Array(HDIM);

    // PFNN weights
    var W0p = new Array(HDIM); // these are multi-dimensional arrays
    var W1p = new Array(HDIM); // these are multi-dimensional arrays
    var W2p = new Array(YDIM); // these are multi-dimensional arrays

    // PFNN biases
    var b0p = new Array(HDIM);
    var b1p = new Array(HDIM);
    var b2p = new Array(YDIM);

    that.initialise = function() {

        // PFNN inputs and outputs
        Xmean = Script.require(Script.resolvePath(PATH_TO_DATA + "Xmean.json"));
        Xstd = Script.require(Script.resolvePath(PATH_TO_DATA + "Xstd.json"));
        Ymean = Script.require(Script.resolvePath(PATH_TO_DATA + "Ymean.json"));
        Ystd = Script.require(Script.resolvePath(PATH_TO_DATA + "Ystd.json"));

        // set initial state for PFNN output (Yp)
        for (i = 0 ; i < YDIM ; i++) {
            Yp[i] = Ymean[i];
        }

        switch (mode) {

            case MODE_CONSTANT:

                const ARRAY_SIZE = 50;

                W0 = new Array(ARRAY_SIZE); 
                W1 = new Array(ARRAY_SIZE); 
                W2 = new Array(ARRAY_SIZE);
                b0 = new Array(ARRAY_SIZE); 
                b1 = new Array(ARRAY_SIZE); 
                b2 = new Array(ARRAY_SIZE);

                print("PFNN: Loading data");
                //print("Loading data");
                for (var i = 0 ; i < ARRAY_SIZE ; i++) {   
                    var index;
                    if (i < 10) {
                        index = "00" + i;
                    } else {
                        index = "0" + i;
                    }   
                    //print("Set index to " + index);
                    W0[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "W0_" + index + ".json"));
                    W1[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "W0_" + index + ".json"));
                    W2[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "W0_" + index + ".json"));
                    b0[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "b0_" + index + ".json"));
                    b1[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "b1_" + index + ".json"));
                    b2[i] = Script.require(Script.resolvePath(PATH_TO_DATA + "b2_" + index + ".json"));
                    //print("Loaded " + (i + 1) + " PFNN weights");
                    print("PFNN: Loading: " + (100 * (i + 1) / ARRAY_SIZE) + "% done");           
                }
                //print("Data loaded");
                print("PFNN: Data loaded");
                break;
        }
    }

    that.updateXpInput = function(index, value) {
        Xp[index] = value;
    }

    //that.dumpXp = function() {
    //    print("Xp is now: " + JSON.stringify(Xp, null, " "));
    //}

    that.getYp = function() {
        return Yp;
    }

    //var debugbool = true;

    // Predict the next frame. P is the current phase
    that.predict = function(P) {

        var pamount;
        var pindex_0, pindex_1, pindex_2, pindex_3;

        //if (debugbool) {
        //    print("Xp is: " + JSON.stringify(Xp, null, " "));
        //}
        //Xp = (Xp - Xmean) / Xstd;
        Xp = arrayFunctions.arrayDivide((arrayFunctions.arraySubtract(Xp, Xmean)), Xstd);
        //Xp = (arrayFunctions.arraySubtract(Xp, Xmean));
       // Xp = arrayFunctions.arrayDivide(Xp, Xstd);

        //if (debugbool) {
        //    debugbool = false;
        //    print("Xp is now: " + JSON.stringify(Xp, null, " "));
        //}

        //print("Predict called with phase equal to " + P.toFixed(2));
/*
        switch (mode) {

            case MODE_CONSTANT:
                pindex_1 = (int)((P / (2*M_PI)) * 50);
                H0 = (W0[pindex_1].matrix() * Xp.matrix()).array() + b0[pindex_1]; ELU(H0);
                H1 = (W1[pindex_1].matrix() * H0.matrix()).array() + b1[pindex_1]; ELU(H1);
                Yp = (W2[pindex_1].matrix() * H1.matrix()).array() + b2[pindex_1];
                break;
    
            default:
                break;
        }

        Yp = (Yp * Ystd) + Ymean;
*/
    }    

    return that;    
}

Trajectory = function() {

    var that = {};

    var debugBool = true;
    
    const LENGTH = 120; // There are LENGTH trajectory points
    const MARKER_RATIO = 10; // 3D overlays are drawn every MARKER_RATIO sample points. MARKER_RATIO / LENGTH must be an integer
    const MOVE_THRESHOLD = 0.01; //0.075; // movement dead zone
    const MARKER_SIZE = 0.1;

    // following terrain involves LENGTH intersection calls - too many for HiFi JS?
    var terrain_following = false;

    // these are the parameters (Xp values) that are fed into the PFNN
    var positions = new Array(LENGTH);     // Vec3
    var directions = new Array(LENGTH);    // Vec3
    var rotations = new Array(LENGTH);     // Vec3
    var heights = new Array(LENGTH);       // float

    var gait_stand = new Array(LENGTH);    // float
    var gait_walk = new Array(LENGTH);     // float
    var gait_jog = new Array(LENGTH);      // float
    var gait_crouch = new Array(LENGTH);   // float
    var gait_jump = new Array(LENGTH);     // float
    var gait_bump = new Array(LENGTH);     // float

    var target_dir = { x: 0, y: 0, z: 1 }; // Vec3
    var target_vel = { x: 0, y: 0, z: 0 }; // Vec3
    // End of PFNN Xp parameters

    // trajectory overlays
    var location_overlays = new Array(LENGTH / MARKER_RATIO);
    var direction_overlays = new Array(LENGTH / MARKER_RATIO);   

    // getters
    that.getLength = function() {
        return LENGTH;
    }

    that.hips_to_ground = function(pick_ray_origin) {
        var pick_ray = { origin: pick_ray_origin, direction: { x:0, y:-1, z:0 } };
        return Entities.findRayIntersection(pick_ray, true).distance; 
    } 

    that.initialise = function() {
        
        // initialise trajectory     
        for (var i = 0 ; i < LENGTH ; i++) {
            positions[i] = Vec3.sum(MyAvatar.position, { x:0, y:-character.getAvatarHipsToFeet(), z:0 });
            rotations[i] = Quat.multiply(MyAvatar.orientation, Quat.getFront);  // Quat.safeEulerAngles(MyAvatar.orientation);
            //directions[i] = Quat.getFront(MyAvatar.headOrientation, Quat.getFront); // Quat.safeEulerAngles(MyAvatar.orientation);{ x: 0, y: 0, z: 1 };
            directions[i] = Quat.multiply(MyAvatar.orientation, Quat.getFront);
            heights[i] = that.hips_to_ground(MyAvatar.position);
            gait_stand[i] = 0.0;
            gait_walk[i] = 0.0;
            gait_jog[i] = 0.0;
            gait_crouch[i] = 0.0;
            gait_jump[i] = 0.0;
            gait_bump[i] = 0.0;
        }
        target_dir = { x: 0, y: 0, z: 1 }; 
        target_vel = { x: 0, y: 0, z: 0 };  
        
        // initialse overlays
        for (var i = 0 ; i < LENGTH / MARKER_RATIO ; i++) {
            var location_overlay = Overlays.addOverlay("sphere", {
                position: positions[i],
                rotation: rotations[i],
                color: {
                    //red: gait_jump[i],
                    //green: gait_bump[i],
                    //blue: gait_crouch[i]
                    red: 0,
                    green: 0,
                    blue: 0
                },
                alpha: 1.0,
                visible: true,
                isSolid: true,
                size: MARKER_SIZE,
                scale: MARKER_SIZE,
                //isFacingAvatar: true,
                drawInFront: false
            });
            var direction_overlay = Overlays.addOverlay("line3d", {
                    start: positions[i],
                    end: Vec3.sum(positions[i], Vec3.multiply(0.25, directions[i])),
                    color: { red: 0, green: 0, blue: 0},
                    alpha: 1,
                    lineWidth: 5,
                    drawInFront: false
            });
            // store overlays
            location_overlays[i] = location_overlay;  
            direction_overlays[i] = direction_overlay;
        }       
    }    

    that.update = function(delta_time) {

        //print("delta_time is " + delta_time)

        //print("Distance from surface is " + avatarHipsToFeet + "m.");
        var current_height = that.hips_to_ground(MyAvatar.position);
        var current_position = Vec3.sum(MyAvatar.position, { x:0, y:-current_height, z:0 }); 
        //print("Current trajectory marker position is { x:" + current_position.x.toFixed(5) + ", y:" + current_position.y.toFixed(5) + ", z:" + current_position.z.toFixed(5) + " }");
        var current_rotation = Quat.getFront(MyAvatar.orientation); // Quat.inverse(MyAvatar.orientation); // Vec3.multiplyQbyV(Quat.inverse(MyAvatar.orientation), MyAvatar.velocity); // Vec3.normalize({ x: MyAvatar.bodyYaw, y:0, z:0 }); //  MyAvatar.orientation; 
        var current_direction = Quat.getFront(MyAvatar.headOrientation); //Vec3.normalize({ x: MyAvatar.headYaw, y: 0, z: 0 });  

        //print("Current height: " + current_height);

        // update current trajectory
        positions[LENGTH / 2] = current_position;
        rotations[LENGTH / 2] = current_rotation;
        heights[LENGTH / 2] = current_height;
        directions[LENGTH / 2] = current_direction;

        var current_speed = Vec3.length(MyAvatar.velocity);
        gait_stand[LENGTH / 2] = current_speed > MOVE_THRESHOLD ? 1 : 0;
        gait_walk[LENGTH / 2] = current_speed < MOVE_THRESHOLD ? 1 : 0;
        gait_jog[LENGTH / 2] = gait_jog[i+1];
        gait_crouch[LENGTH / 2] = gait_crouch[i+1];
        gait_jump[LENGTH / 2] = gait_jump[i+1];
        gait_bump[LENGTH / 2] = gait_bump[i+1];

        target_dir = { x: 0, y: 0, z: 1 }; 
        target_vel = { x: 0, y: 0, z: 0 }; 

        // update future trajectory
        for (var i = LENGTH / 2 + 1 ; i < LENGTH ; i++) {
            // distance to next trajectory marker
            var step_forward = (i - (LENGTH / 2 + 1)) / (MARKER_RATIO * 4);
            var next_position = Vec3.sum(current_position, Vec3.multiply(step_forward, MyAvatar.velocity));
            // Get the terrain height for this position
            var forward_probe_position = { x: next_position.x, y: MyAvatar.position.y, z: next_position.z };
            var next_height = terrain_following ? 
                              that.hips_to_ground(forward_probe_position) :
                              current_height; //  debug only
            heights[i] = next_height;
            next_position.y = forward_probe_position.y - next_height;
            positions[i] = next_position;
        }

        // render trajectory
        for (var i = 0 ; i < LENGTH ; i+= 10) {
            Overlays.editOverlay(location_overlays[i / MARKER_RATIO], {
                position: positions[i],
                rotations: rotations[i],
            });
            Overlays.editOverlay(direction_overlays[i / MARKER_RATIO], {
                start: positions[i],
                end: Vec3.sum(positions[i], Vec3.multiply(0.25, rotations[i]))
            });
        }

        // update past trajectory
        for (var i = 0 ; i < LENGTH / 2 ; i++) {
            positions[i] = positions[ i + 1 ];
            rotations[i] = rotations[ i + 1 ];
            directions[i] = directions[i + 1];
            heights[i] = heights[ i + 1 ];
            gait_stand[i] = gait_stand[i+1];
            gait_walk[i] = gait_walk[i+1];
            gait_jog[i] = gait_jog[i+1];
            gait_crouch[i] = gait_crouch[i+1];
            gait_jump[i] = gait_jump[i+1];
            gait_bump[i] = gait_bump[i+1];
        }    

        // Now to load up our PFNN X (input) values...
        //pfnn.updateXpInput(index, value)

        // Load up the positions and gaits
        for ( i = 0 ; i < 132 ; i++ ) {
            if (i < 12) {
                pfnn.updateXpInput(i, positions[i].x);
            } else if (i < 24) {
                pfnn.updateXpInput(i, positions[i - 12].z);
            } else if (i < 36) {
                pfnn.updateXpInput(i, directions[i - 24].x);
            } else if (i < 48) {
                pfnn.updateXpInput(i, directions[i - 36].z);
            } else if (i < 60) {
                pfnn.updateXpInput(i, gait_stand[i - 48]);
            } else if (i < 72) {
                pfnn.updateXpInput(i, gait_walk[i - 60]);
            } else if (i < 84) {
                pfnn.updateXpInput(i, gait_jog[i - 72]);
            } else if (i < 96) {
                pfnn.updateXpInput(i, gait_crouch[i - 84]);
            } else if (i < 108) {
                pfnn.updateXpInput(i, gait_jump[i - 96]);
            } else if (i < 120) { // this section is unused (could be used for flying gait in the future)
                pfnn.updateXpInput(i, 0);
            }
        }
        // Load up the joint current and previous positions
        var i = 120;
        var jointPositions = character.getJointPositions();
        for (joint in PFNNArmature) {
        //for ( i = 120 ; i < pfnn.XDIM - 36 ; i+=6 ) {
            //var jointNumber = (i - 120);
            var characterJointPositions = character.getJointPositions();
            if (debugBool) { 
                //print("Putting values for " + PFNNArmature[joint] + " into Xp position " + i + ".");
            }
            
            pfnn.updateXpInput(i,   jointPositions[joint].x);
            pfnn.updateXpInput(i+1, jointPositions[joint].y);
            pfnn.updateXpInput(i+2, jointPositions[joint].z);
            
            pfnn.updateXpInput(i+3, 3.1415926536);
            pfnn.updateXpInput(i+4, 3.1415926536);
            pfnn.updateXpInput(i+5, 3.1415926536);
            /**/
            i+=6;
        }
        if (debugBool) {
            debugBool = false;
            //pfnn.dumpXp();
        }

        pfnn.predict(0); // character.phase);       
    }

    that.clean_up = function() {
        for (var i = 0 ; i < LENGTH ; i++) {
            Overlays.deleteOverlay(location_overlays[i]);
            Overlays.deleteOverlay(direction_overlays[i]);
        }
    }

    return that;
};

Character = function() {

    var that = {};

    const JOINT_NUM = 31; // including End Sites (for some reason) - not 25;

    // Initialise (by calling initialise) after trajectory and pfnn are instantiated
    var joint_positions = new Array(JOINT_NUM);
    var joint_velocities = new Array(JOINT_NUM);
    var joint_rotations = new Array(JOINT_NUM);

    // Extras added by Dave
    var avatarHipsToFeet = 1.0;

    that.getJointPositions = function() {
        return joint_positions;
    }

    that.getAvatarHipsToFeet = function() {
        return avatarHipsToFeet;
    }

    that.initialise = function() {
        print("Character initialise called.");
        MyAvatar.orientation = Quat.fromPitchYawRollDegrees(0, 0, 0);
        var pick_ray = {origin: MyAvatar.position, direction: { x:0, y:-1, z:0 }};
        avatarHipsToFeet = Entities.findRayIntersection(pick_ray, true).distance;
        print("Hips to feet comes in at " + avatarHipsToFeet.toFixed(3) + "m.");
        MyAvatar.position = { x: 0, y: avatarHipsToFeet, z: 0 };
        print("Set MyAvatar.position to { x:0, y:avatarHipsToFeet, z:0");
        print("Avatar root at { x:" + MyAvatar.position.x.toFixed(3) +
              ", y:" + (MyAvatar.position.y - avatarHipsToFeet).toFixed(3) + 
              ", z:" + MyAvatar.position.z.toFixed(3) + " }");
        print("Avatar hips at { x:" + MyAvatar.position.x.toFixed(3) +
              ", y:" + MyAvatar.position.y.toFixed(3) + 
              ", z:" + MyAvatar.position.z.toFixed(3) + " }");

        var trajectoryLength = trajectory.getLength();
        //var rootPosition = Vec3.sum(MyAvatar.position, { x:0, y:-character.getAvatarHipsToFeet(), z:0 });
        var rootPosition = glm.vec3(MyAvatar.position.x, MyAvatar.position.y, MyAvatar.position.z).add(glm.vec3(0, -character.getAvatarHipsToFeet(), 0))
        //print("Character.initialise: rootPosition is { x:" + rootPosition.x + ", y:" + rootPosition.y + ", z:" + rootPosition.z + " }");
        var rootRotation = glm.mat3();
        var Yp = pfnn.getYp();
        var debugBool = true;
        for (i = 0; i < trajectoryLength; i++) {

            var opos = 8 + (((trajectoryLength / 2) / 10) * 4) + (JOINT_NUM * 3 * 0);
            var ovel = 8 + (((trajectoryLength / 2) / 10) * 4) + (JOINT_NUM * 3 * 1);
            var orot = 8 + (((trajectoryLength / 2) / 10) * 4) + (JOINT_NUM * 3 * 2);

/*
glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));            

xPos is 0
yPos is 93.1293793
zPos is 0
xRot is 0.0562893562
yRot is -0.00699363137
zRot is -0.000217798908
xVel is -0.00416085264YpOpos
zVel is 1.23609018
*/

            //from: glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
            //from: HumbleTim pos = glm.vec3(glm.mat4(root_rotation).mul(glm.vec4(glm.vec3(),1))).add(root_position);
            var oposX = parseFloat(Yp[opos + i * 3 + 0]);
            var oposY = parseFloat(Yp[opos + i * 3 + 1]);
            var oposZ = parseFloat(Yp[opos + i * 3 + 2]);
            var oposVec3 = glm.vec3(oposX, oposY, oposZ);
            var oposVec4 = glm.vec4(oposVec3, 1);
            oposVec4 = glm.mat4(rootRotation).mul(oposVec4);
            oposVec3 = glm.vec3(oposVec4);
            var pos = oposVec3.add(rootPosition);

            // from: glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
            var ovelX = parseFloat(Yp[ovel + i * 3 + 0]);
            var ovelY = parseFloat(Yp[ovel + i * 3 + 1]);
            var ovelZ = parseFloat(Yp[ovel + i * 3 + 2]);
            var ovelVec3 = glm.vec3(ovelX, ovelY, ovelZ);
            var ovelVec4 = glm.vec4(ovelVec3, 1);
            ovelVec4 = glm.mat4(rootRotation).mul(ovelVec4);
            var vel = glm.vec3(ovelVec4);

            // from: glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));
            var orotX = parseFloat(Yp[orot + i * 3 + 0]);
            var orotY = parseFloat(Yp[orot + i * 3 + 1]);
            var orotZ = parseFloat(Yp[orot + i * 3 + 2]);
            var orotVec3 = glm.vec3(orotX, orotY, orotZ);
            //var orotVec4 = quat_exp(orotVec3);
            //var orotMat3 = glm.mat3.(orotVec4);
            //var rot = glm.mat3(orotMat3).mul(rootRotation);

            var rot = rootRotation;




            if(debugBool) {
                debugBool = false;
                print("\nJOINT_NUM is " + JOINT_NUM.toFixed(2) +
                      "\ntrajectoryLength is " + trajectoryLength.toFixed(2) + 
                      "\nopos is " + opos.toFixed(2) + 
                      "\novel is " + ovel.toFixed(2) + 
                      "\norot is " + orot.toFixed(2) +

                      "\nxPos is " + Yp[opos + i * 3 + 0] +
                      "\nyPos is " + Yp[opos + i * 3 + 1] +
                      "\nzPos is " + Yp[opos + i * 3 + 2] +

                      "\nxRot is " + Yp[orot + i * 3 + 0] +
                      "\nyRot is " + Yp[orot + i * 3 + 1] +
                      "\nzRot is " + Yp[orot + i * 3 + 2] +

                      "\nxVel is " + Yp[ovel + i * 3 + 0] +
                      "\nyVel is " + Yp[ovel + i * 3 + 1] +
                      "\nzVel is " + Yp[ovel + i * 3 + 2] +


                      "\npos is " + pos.json +
                      "\nvel is " + vel.json +
                      "\nrot is " + rot.json +


                      " ");
            } 
/*
            var pos = (root_rotation * glm::vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
            var vel = (root_rotation * glm::vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
            var rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));            
            
            var pos = Vec3.cross(rootRotation, Vec3.sum( { x:Yp[opos + i * 3 + 0], y: Yp[opos + i * 3 + 1], z: Yp[opos + i * 3 + 2] }, rootPosition));
            var vel = Vec3.cross(rootRotation, { x:Yp[ovel + i * 3 + 0], y:Yp[ovel + i * 3 + 1], z:Yp[ovel + i * 3 + 2] });
            var rot = Vec3.cross(rootRotation, { x:Yp[orot + i * 3 + 0], y:Yp[orot + i * 3 + 1], z:Yp[orot + i * 3 + 2] });

            joint_positions[i] = pos;
            joint_velocities[i] = vel;
            joint_rotations[i] = rot;
            */
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // There are a number of differences between the PFNN demo character and the HiFi avi character
    // Ignoring finger / thumb bones, the PFNN demo character has the following extra bones:
    //
    //    ROOT
    //    Neck1
    //    LHipJoint
    //    RHipJoint
    //    End Site (x5)
    // 
    // The joint data from these will be combined into parent / children joints. 
    //
    // Moreover, the skeleton structure differs quite significantly - the PFNN demo skeleton *looks*
    // like Carnegie Melon data, which has always been difficult to deal with...
    //
    //


    //that.phase = 0;

    var debugBool = true;

    // Here we update the HiFiArmature with current values
    that.update = function() {
        for (joint in HiFiArmature) {
            HiFiArmature[joint].prv = 0;
            HiFiArmature[joint].pos = MyAvatar.getJointPosition(joint);
        }
        var jointIndex = 0;
        for (joint in PFNNArmature) {

            joint_velocities[jointIndex] = { X:0, y:0, z:0 };

            // Translate between armatures
            switch (PFNNArmature[joint]) {

                case "ROOT" :
                    joint_positions[jointIndex] = Vec3.sum(HiFiArmature["Hips"].pos, avatarHipsToFeet);
                    break;

                case "Neck" :
                case "Neck1" :
                    joint_positions[jointIndex] = Vec3.multiply(HiFiArmature["Neck"].pos, 0.5);
                    break;

                case "LeftHipJoint" :
                case "LeftUpLeg" :
                    joint_positions[jointIndex] = Vec3.multiply(HiFiArmature["LeftUpLeg"].pos, 0.5);
                    break;

                case "RightHipJoint" :
                case "RightUpLeg" :
                    joint_positions[jointIndex] = Vec3.multiply(HiFiArmature["RightUpLeg"].pos, 0.5);
                    break;

                case "LowerBack" :
                    joint_positions[jointIndex] = HiFiArmature["Spine"].pos;
                    break;

                case "Spine" :
                    joint_positions[jointIndex] = HiFiArmature["Spine1"].pos;
                    break;

                case "Spine1" :
                    joint_positions[jointIndex] = HiFiArmature["Spine2"].pos;
                    break;

                case "End Site" :
                    joint_positions[jointIndex] = { x:0, y:0, z:0 };
                    break;

                default :
                    joint_positions[jointIndex] = HiFiArmature[PFNNArmature[joint]].pos;
                    break;
            }
            if (debugBool) {
                //print("Set PFNNArmature " + PFNNArmature[joint] + " to " + 
                //      joint_positions[jointIndex].x + ", " +
                //      joint_positions[jointIndex].y + ", " +
                //      joint_positions[jointIndex].z);
            }
            jointIndex++;
        }
        if (debugBool) {
            debugBool = false;
        }
    }
    return that;
};


arrayFunctions = (function() {

    return {

        // subtract arrayTwo from arrayOne and return the result
        arraySubtract: function(arrayOne, arrayTwo) {

            if (!(arrayOne.length == arrayTwo.length)) {
                return ["Error: The arrays must be the same size."]
            } else {
                var arrayOneMinusArrayTwo = new Array(arrayOne.length);
                for (i in arrayOne) {
                    arrayOneMinusArrayTwo[i] = arrayOne[i] - arrayTwo[i];
                }
                return arrayOneMinusArrayTwo;
            }
        },

        // divide arrayOne by arrayTwo and return the result
        arrayDivide: function(arrayOne, arrayTwo) {
            //print("arrayDivide: array one is " + arrayOne.length + " and array two is " + arrayTwo.length);
            if (arrayOne.length != arrayTwo.length) {
                return ["Error: The arrays must be the same size."]
            } else {
                var arrayOneDividedByArrayTwo = [];
                for (i in arrayOne) {
                    arrayOneDividedByArrayTwo[i] = arrayOne[i] / arrayTwo[i];
                }
                return arrayOneDividedByArrayTwo;
            }
        }
    }
})(); // end object literal

/*
//processKeyPressEvent = function(event) {
function processKeyPressEvent(event) {
    if (event.isAutoRepeat) {  // isAutoRepeat is true when held down (or when Windows feels like it)
        print("Event is autorepeat");
        return;
    }
    var pick_ray = {origin: MyAvatar.position, direction: { x:0, y:-1, z:0 }};
    var avatarHipsToFeet = Entities.findRayIntersection(pick_ray, true).distance;
    if (event.text == "P") {
        print("Avatar root at { x:" + MyAvatar.position.x.toFixed(3) +
              ", y:" + (MyAvatar.position.y - avatarHipsToFeet).toFixed(3) + 
              ", z:" + MyAvatar.position.z.toFixed(3) + " }");
        print("Avatar hips at { x:" + MyAvatar.position.x.toFixed(3) +
              ", y:" + MyAvatar.position.y.toFixed(3) + 
              ", z:" + MyAvatar.position.z.toFixed(3) + " }");
    } else if (event.text == "O") {
        MyAvatar.position = { x: 0, y: avatarHipsToFeet, z: 0 };
        print("Set MyAvatar.position to { x:0, y:avatarHipsToFeet, z:0");
    } else if (event.text == "I") {
        print("Hips to feet comes in at " + avatarHipsToFeet.toFixed(3) + "m.");
    }
}
*/

var HiFiArmature = {
    "Hips" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftUpLeg" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftLeg" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftFoot" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftToeBase" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightUpLeg" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightLeg" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightFoot" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightToebase" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "Spine" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "Spine1" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "Spine2" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "Neck" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "Head" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftShoulder" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftArm" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftForeArm" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "LeftHand" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightShoulder" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightArm" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightForeArm" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    },
    "RightHand" : {
        pos : {
            x : 0,
            y : 0,
            z : 0
        },
        prv : {
            x : 0,
            y : 0,
            z : 0
        }
    }
};


var PFNNArmature = [
    "ROOT",
    "Hips",
    "LeftHipJoint",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "End Site",
    "RightHipJoint",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToebase",
    "End Site",
    "LowerBack",
    "Spine",
    "Spine1",
    "Neck",
    "Neck1",
    "Head",
    "End Site",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "End Site",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "End Site"
]

var prerotations = {
    "name":"prerotations",
    "version":"1.0",
    "joints":{
        "Hips":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftUpLeg":{
            "x":0,
            "y":0,
            "z":-180,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightUpLeg":{
            "x":0,
            "y":0,
            "z":180,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftLeg":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightLeg":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftFoot":{
            "x":80,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightFoot":{
            "x":80,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftToeBase":{
            "x":28.5194,
            "y":-1.43586,
            "z":0.685672
        },
        "RightToeBase":{
            "x":28.0842,
            "y":1.45959,
            "z":-0.687243
        },
        "Spine":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "Spine1":{
            "x":0,
            "y":0,
            "z":0
        },
        "Spine2":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftShoulder":{
            "x":-75,
            "y":85,
            "z":180,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightShoulder":{
            "x":-75,
            "y":-85,
            "z":-180,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftArm":{
            "x":-17,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightArm":{
            "x":-17,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftForeArm":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightForeArm":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "LeftHand":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "RightHand":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "Neck":{
            "x":0,
            "y":0,
            "z":0,
            "pitch":0,
            "yaw":0,
            "roll":0,
            "pitchPhase":0,
            "yawPhase":0,
            "rollPhase":0
        },
        "Head":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftHandPinky1":{
            "x":-1.59,
            "y":0,
            "z":1.34
        },
        "LeftHandPinky2":{
            "x":3.84,
            "y":0,
            "z":0
        },
        "LeftHandPinky3":{
            "x":4.91,
            "y":0,
            "z":0
        },
        "LeftHandPinky4":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftHandRing1":{
            "x":-1.12,
            "y":0,
            "z":0.34
        },
        "LeftHandRing2":{
            "x":4.35,
            "y":0,
            "z":0
        },
        "LeftHandRing3":{
            "x":-2.89,
            "y":0,
            "z":0
        },
        "LeftHandRing4":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftHandMiddle1":{
            "x":1.12,
            "y":0,
            "z":3.67
        },
        "LeftHandMiddle2":{
            "x":5.14,
            "y":0,
            "z":0
        },
        "LeftHandMiddle3":{
            "x":1.71,
            "y":0,
            "z":0
        },
        "LeftHandMiddle4":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftHandIndex1":{
            "x":3.8,
            "y":0,
            "z":0.48
        },
        "LeftHandIndex2":{
            "x":3,
            "y":0,
            "z":0
        },
        "LeftHandIndex3":{
            "x":7.94,
            "y":0,
            "z":0
        },
        "LeftHandIndex4":{
            "x":0,
            "y":0,
            "z":0
        },
        "LeftHandThumb1":{
            "x":18.28,
            "y":0,
            "z":31.87
        },
        "LeftHandThumb2":{
            "x":5.96,
            "y":0,
            "z":0
        },
        "LeftHandThumb3":{
            "x":-7.57,
            "y":0,
            "z":0
        },
        "LeftHandThumb4":{
            "x":0,
            "y":0,
            "z":0
        },
        "RightHandPinky1":{
            "x":-3.8,
            "y":0,
            "z":-3.1
        },
        "RightHandPinky2":{
            "x":0.85,
            "y":0,
            "z":0
        },
        "RightHandPinky3":{
            "x":0.86,
            "y":0,
            "z":0
        },
        "RightHandPinky4":{
            "x":0,
            "y":0,
            "z":0
        },
        "RightHandRing1":{
            "x":-0.97,
            "y":0,
            "z":-1.13
        },
        "RightHandRing2":{
            "x":-1.24,
            "y":0,
            "z":0
        },
        "RightHandRing3":{
            "x":-4.53,
            "y":0,
            "z":0
        },
        "RightHandRing4":{
            "x":0,
            "y":0,
            "z":0
        },
        "RightHandMiddle1":{
            "x":-3.31,
            "y":0,
            "z":0
        },
        "RightHandMiddle2":{
            "x":3.83,
            "y":0,
            "z":0
        },
        "RightHandMiddle3":{
            "x":6.04,
            "y":0,
            "z":0
        },
        "RightHandMiddle4":{
            "x":0,
            "y":0,
            "z":0
        },
        "RightHandIndex1":{
            "x":0.77,
            "y":0,
            "z":0.09
        },
        "RightHandIndex2":{
            "x":4.03,
            "y":0,
            "z":0
        },
        "RightHandIndex3":{
            "x":1.18,
            "y":0,
            "z":0
        },
        "RightHandIndex4":{
            "x":0,
            "y":0,
            "z":0
        },
        "RightHandThumb1":{
            "x":19.72,
            "y":0,
            "z":-30.89
        },
        "RightHandThumb2":{
            "x":2.35,
            "y":0,
            "z":0
        },
        "RightHandThumb3":{
            "x":-2.13,
            "y":0,
            "z":0
        },
        "RightHandThumb4":{
            "x":0,
            "y":0,
            "z":0
        }
    }
}

// ECMAScript 6 specification ready string.contains() function
if (!('contains' in String.prototype)) {
    String.prototype.contains = function(str, startIndex) {
        return ''.indexOf.call(this, str, startIndex) !== -1;
    };
}

/**/
// Display joints info, set t-stance
var numJoints = 0;
for (joint in prerotations.joints) {
    //print("Joint: " + joint + " : " + prerotations.joints[joint]["x"] + ", " + prerotations.joints[joint].y + ", " + prerotations.joints[joint].z);
    MyAvatar.setJointRotation(joint, Quat.fromPitchYawRollDegrees(
        prerotations.joints[joint]["x"],
        prerotations.joints[joint]["y"],
        prerotations.joints[joint]["z"])
    );
    if ((!joint.contains("LeftHand")  || joint == "LeftHand" ) &&
        (!joint.contains("RightHand") || joint == "RightHand")) {
        print("Joint " + joint + " found.");
        numJoints++;        
    } 
}
print("Avatar has " + numJoints + " joints (plus end sites).");



// Load the phase functioned neural network
pfnn = PFNN(0);

// Load the character's movement trajectory
trajectory = Trajectory();

// Load the character
character = Character();

// Initialise everything
pfnn.initialise();          // character.initialise uses pfnn.getYp, so need to initialise first
trajectory.initialise();    // character.initialise uses trajectory.getLength, so need to initialise first
character.initialise();

// Input 
//Controller.keyPressEvent.connect(processKeyPressEvent);

// Main loop
Script.update.connect(function(delta_time) {
    character.update();
	trajectory.update(delta_time);
});

// Tidy up
Script.scriptEnding.connect(function () {
    print("PFNN: Closing down");
    MyAvatar.clearJointsData();
    trajectory.clean_up();
    //Controller.keyPressEvent.disconnect(processKeyPressEvent);
});
