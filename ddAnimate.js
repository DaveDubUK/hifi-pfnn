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

print("PFNN: Starting up. Yp experiments ready version");

// Urgh! I give up! Hard coding locallib functions for now
//Script.require("./libraries/pfnnApi.js");
//Script.include(Script.resolvePath('./libraries/Matrix3.js'));
//Script.include('./libraries/Matrix3.js');

var glm = Script.require('https://git.io/glm-js.min.js');

// Helper functions

MyAvatar.$getParentJointIndex = function (index) {
    if (!isFinite(index)) throw "TODO: jointName; please use jointIndex";
    var bone = MyAvatar.getSkeleton()[index];
    return bone ? bone.parentIndex : -1;
};

var quatExp = function (vectorThree) {
    var w = glm.length(vectorThree);
    var q = w < 0.01 ?
        glm.quat(1, 0, 0, 0) :
        glm.quat(
            Math.cos(w),
            vectorThree.x * (Math.sin(w) / w),
            vectorThree.y * (Math.sin(w) / w),
            vectorThree.z * (Math.sin(w) / w)
        );
    // print(JSON.stringify(q / Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z))+" fkdjsf");
    return glm.normalize(q);
};

var mixDirections = function (x, y, a) {
    var x_q = glm.angleAxis(Math.atan2(x.x, x.z), glm.vec3(0, 1, 0));
    var y_q = glm.angleAxis(Math.atan2(y.x, y.z), glm.vec3(0, 1, 0));
    var z_q = glm.slerp(x_q, y_q, a);
    return z_q * glm.quat.getFront;
    //return glm.Quat.multiply(z_q, Quat.getFront);
    //return z_q * glm.vec3(0, 0, 1);
}

var options = {
    extraVelocitySmooth: 0.9,
    extraDirectionSmooth: 0.9,
    extraVelocitySmooth: 0.9,
    extraStrafeSmooth: 0.9,
    extraCrouchedSmooth: 0.9,
    extraGaitSmooth: 0.1,
    extraJointSmooth: 0.5
}

// precomputation techniques (aka mode) for PFNN
const MODE_CONSTANT = 0; // default
const MODE_LINEAR = 1;   // not yet supported (uses interpolation to trade off memory and processing)
const MODE_CUBIC = 2;    // not yet supported (uses interpolation to trade off memory and processing)

const PATH_TO_DATA = "./pfnn-data/";


PFNN = function (precomputationTechnique) {

    var that = {};
    //var data_url = "http://davedub.co.uk/highfidelity/pfnn/";

    var mode = precomputationTechnique;

    // array sizes
    const XDIM = 342; // will need to adjust for HiFi avatar number of joints
    const YDIM = 311;
    const HDIM = 512;

    that.XDIM = 342; // will need to adjust for HiFi avatar number of joints
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

    that.initialise = function () {

        // PFNN inputs and outputs
        Xmean = Script.require(Script.resolvePath(PATH_TO_DATA + "Xmean.json"));
        Xstd = Script.require(Script.resolvePath(PATH_TO_DATA + "Xstd.json"));
        Ymean = Script.require(Script.resolvePath(PATH_TO_DATA + "Ymean.json?" + Math.random()));
        Ystd = Script.require(Script.resolvePath(PATH_TO_DATA + "Ystd.json"));

        // set initial state for PFNN output (Yp)
        for (i = 0; i < YDIM; i++) {
            Yp[i] = Ymean[i];
        }
        return;
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
                for (var i = 0; i < ARRAY_SIZE; i++) {
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

    //that.updateXpInput = function(index, value) {
    //    Xp[index] = value;
    //}

    //that.dumpXp = function() {
    //    print("Xp is now: " + JSON.stringify(Xp, null, " "));
    //}

    that.getYp = function () {
        return Ymean;
    }

    //var debugbool = true;

    // Predict the next frame. P is the current phase
    that.predict = function (P) {

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

Trajectory = function () {

    var that = {};

    var debugBool = true;

    that.LENGTH = 120; // There are LENGTH trajectory points
    that.MARKER_RATIO = 10; // 3D overlays are drawn every MARKER_RATIO sample points. MARKER_RATIO / LENGTH must be an integer
    that.MOVE_THRESHOLD = 0.01; //0.075; // movement dead zone
    that.MARKER_SIZE = 0.1;

    // following terrain involves LENGTH intersection calls - too many for HiFi JS?
    that.terrainFollowing = false;

    // these are the parameters (Xp values) that are fed into the PFNN
    that.positions = new Array(that.LENGTH);     // Vec3
    that.directions = new Array(that.LENGTH);    // Vec3
    that.rotations = new Array(that.LENGTH);     // Vec3
    that.heights = new Array(that.LENGTH);       // float

    that.gaitStand = new Array(that.LENGTH);    // float
    that.gaitWalk = new Array(that.LENGTH);     // float
    that.gaitJog = new Array(that.LENGTH);      // float
    that.gaitCrouch = new Array(that.LENGTH);   // float
    that.gaitJump = new Array(that.LENGTH);     // float
    that.gaitBump = new Array(that.LENGTH);     // float

    that.targetDir = {x: 0, y: 0, z: 1}; // Vec3
    that.targetVel = {x: 0, y: 0, z: 0}; // Vec3
    // End of PFNN Xp parameters

    // trajectory overlays
    that.locationOverlays = new Array(that.LENGTH / that.MARKER_RATIO);
    that.directionOverlays = new Array(that.LENGTH / that.MARKER_RATIO);

    // getters
    that.getLength = function () {
        return that.LENGTH;
    }

    that.initialise = function () {

        // initialise trajectory    

        var pickRay = {origin: MyAvatar.position, direction: {x: 0, y: -1, z: 0}};
        var hipsToFeet = Entities.findRayIntersection(pickRay, true).distance;

        for (var i = 0; i < that.LENGTH; i++) {
            that.positions[i] = Vec3.sum(MyAvatar.position, {x: 0.0, y: -hipsToFeet, z: 0.0});
            that.rotations[i] = Quat.multiply(MyAvatar.orientation, Quat.getFront);  // Quat.safeEulerAngles(MyAvatar.orientation);
            //directions[i] = Quat.getFront(MyAvatar.headOrientation, Quat.getFront); // Quat.safeEulerAngles(MyAvatar.orientation);{ x: 0, y: 0, z: 1 };
            that.directions[i] = Quat.multiply(MyAvatar.orientation, Quat.getFront);
            that.heights[i] = character.avatarHipsToFeet(MyAvatar.position);
            that.gaitStand[i] = 1.0;
            that.gaitWalk[i] = 0.0;
            that.gaitJog[i] = 0.0;
            that.gaitCrouch[i] = 0.0;
            that.gaitJump[i] = 0.0;
            that.gaitBump[i] = 0.0;
        }
        that.targetDir = {x: 0, y: 0, z: 1};
        that.targetVel = {x: 0, y: 0, z: 0};

        // initialse overlays
        for (var i = 0; i < that.LENGTH / that.MARKER_RATIO; i++) {
            var locationOverlay = Overlays.addOverlay("sphere", {
                position: that.positions[i],
                rotation: that.rotations[i],
                color: {
                    //red: gaitJump[i],
                    //green: gaitBump[i],
                    //blue: gaitCrouch[i]
                    red: 0,
                    green: 0,
                    blue: 0
                },
                alpha: 1.0,
                visible: true,
                isSolid: true,
                size: that.MARKER_SIZE,
                scale: that.MARKER_SIZE,
                //isFacingAvatar: true,
                drawInFront: false
            });
            var directionOverlay = Overlays.addOverlay("line3d", {
                start: that.positions[i],
                end: Vec3.sum(that.positions[i], Vec3.multiply(0.25, that.directions[i])),
                color: {red: 0, green: 0, blue: 0},
                alpha: 1,
                lineWidth: 5,
                drawInFront: false
            });
            // store overlays
            that.locationOverlays[i] = locationOverlay;
            that.directionOverlays[i] = directionOverlay;
        }
    }

    that.clean_up = function () {
        for (var i = 0; i < that.LENGTH; i++) {
            Overlays.deleteOverlay(that.locationOverlays[i]);
            Overlays.deleteOverlay(that.directionOverlays[i]);
        }
    }

    return that;
};

Character = function () {

    var that = {};

    that.JOINT_NUM = 31; // including End Sites (for some reason) - not 25;

    var phase = 0;

    // Initialise (by calling initialise) after trajectory and pfnn are instantiated
    that.jointPositions = new Array(that.JOINT_NUM);
    that.jointVelocities = new Array(that.JOINT_NUM);
    that.jointRotations = new Array(that.JOINT_NUM);
    that.jointRotationsGlobal = new Array(that.JOINT_NUM);
    that.jointRestGlobal = new Array(that.JOINT_NUM);

    that.strafeTarget = 0;
    that.strafeAmount = 0;

    // Extras added by Dave
    that.avatarHipsToFeet = function (pickRayOrigin) {
        var pickRay = {origin: pickRayOrigin, direction: {x: 0, y: -1, z: 0}};
        return Entities.findRayIntersection(pickRay, true).distance;
    }

    that.getJointPositions = function () {
        return jointPositions;
    }

    that.initialise = function () {
        MyAvatar.position = glm.vec3(0, 0, 0);
        var hipsToFeet = this.avatarHipsToFeet(MyAvatar.position);
        print("Character initialise called.");
        MyAvatar.orientation = Quat.fromPitchYawRollDegrees(0, 0, 0);
        print("Hips to feet comes in at " + hipsToFeet.toFixed(3) + "m.");
        MyAvatar.position = {x: 0, y: hipsToFeet, z: 0};
        print("Set MyAvatar.position to { x:0, y:hipsToFeet, z:0");
        print("Avatar root at { x:" + MyAvatar.position.x.toFixed(3) +
            ", y:" + (MyAvatar.position.y - hipsToFeet).toFixed(3) +
            ", z:" + MyAvatar.position.z.toFixed(3) + " }");
        print("Avatar hips at { x:" + MyAvatar.position.x.toFixed(3) +
            ", y:" + MyAvatar.position.y.toFixed(3) +
            ", z:" + MyAvatar.position.z.toFixed(3) + " }");

        var trajectoryLength = trajectory.getLength();
        //var rootPosition = Vec3.sum(MyAvatar.position, { x:0, y:-character.getAvatarHipsToFeet(), z:0 });
        var rootPosition = glm.vec3(MyAvatar.position.x, MyAvatar.position.y, MyAvatar.position.z).add(glm.vec3(0, -hipsToFeet, 0))
        //print("Character.initialise: rootPosition is { x:" + rootPosition.x + ", y:" + rootPosition.y + ", z:" + rootPosition.z + " }");
        var rootRotation = glm.mat3();
        var Yp = pfnn.getYp();
        var debugBool = true;
        for (i = 0; i < that.JOINT_NUM; i++) {

            var opos = 8 + (((trajectoryLength / 2) / 10) * 4) + (that.JOINT_NUM * 3 * 0);
            var ovel = 8 + (((trajectoryLength / 2) / 10) * 4) + (that.JOINT_NUM * 3 * 1);
            var orot = 8 + (((trajectoryLength / 2) / 10) * 4) + (that.JOINT_NUM * 3 * 2);

            //from: glm.vec3 pos = (root_rotation * glm.vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
            var oposX = parseFloat(Yp[opos + i * 3 + 0]);
            var oposY = parseFloat(Yp[opos + i * 3 + 1]);
            var oposZ = parseFloat(Yp[opos + i * 3 + 2]);
            var oposVec3 = glm.vec3(oposX, oposY, oposZ);
            var oposVec4 = glm.vec4(oposVec3, 1);
            oposVec4 = glm.mat4(rootRotation).mul(oposVec4);
            oposVec3 = glm.vec3(oposVec4);
            var pos = oposVec3.add(rootPosition);

            // from: glm.vec3 vel = (root_rotation * glm.vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
            var ovelX = parseFloat(Yp[ovel + i * 3 + 0]);
            var ovelY = parseFloat(Yp[ovel + i * 3 + 1]);
            var ovelZ = parseFloat(Yp[ovel + i * 3 + 2]);
            var ovelVec3 = glm.vec3(ovelX, ovelY, ovelZ);
            var ovelVec4 = glm.vec4(ovelVec3, 1);
            ovelVec4 = glm.mat4(rootRotation).mul(ovelVec4);
            var vel = glm.vec3(ovelVec4);

            // from: glm.mat3 rot = (root_rotation * glm.toMat3(quatExp(glm.vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));
            var orotX = parseFloat(Yp[orot + i * 3 + 0]);
            var orotY = parseFloat(Yp[orot + i * 3 + 1]);
            var orotZ = parseFloat(Yp[orot + i * 3 + 2]);
            var orotVec3 = glm.vec3(orotX, orotY, orotZ);
            var orotVec4 = quatExp(orotVec3);
            var orotMat3 = glm.mat3(glm.toMat4(orotVec4));
            var rot = glm.mat3(orotMat3).mul(rootRotation);

            phase = 0.0;

            // the 'should be' values copied over from C++ code values
            if (debugBool) {
                debugBool = false;
                print("\nJOINT_NUM is " + that.JOINT_NUM.toFixed(2) + " should be 31" +
                    "\ntrajectoryLength is " + trajectoryLength.toFixed(2) + " should be 120" +
                    "\nopos is " + opos.toFixed(2) + " should be 32" +
                    "\novel is " + ovel.toFixed(2) + " should be 125" +
                    "\norot is " + orot.toFixed(2) + " should be 218" +

                    "\npos.x is " + Yp[opos + i * 3 + 0] + " should be 0" +
                    "\npos.y is " + Yp[opos + i * 3 + 1] + " should be 93.1293793" +
                    "\npos.z is " + Yp[opos + i * 3 + 2] + " should be 0" +

                    //"\nrot.x is " + Yp[orot + i * 3 + 0] + " should be 31" +
                    //"\nrot.y is " + Yp[orot + i * 3 + 1] + " should be 31" +
                    //"\nrot.z is " + Yp[orot + i * 3 + 2] + " should be 31" +

                    //"\nxVel is " + Yp[ovel + i * 3 + 0] + " should be 31" +
                    //"\nyVel is " + Yp[ovel + i * 3 + 1] + " should be 31" +
                    //"\nzVel is " + Yp[ovel + i * 3 + 2] + " should be 31" +


                    "\npos is " + pos.json + " should be {x=0.000000000 y=93.1293793 z=0.000000000 ...}" +
                    "\nvel is " + vel.json + " should be {x=-0.00416085264 y=-0.00193083170 z=1.23609018 ...}" +
                    "\nrot is " + rot.json + " should be + {x=0.999902189 y=-0.00122115412 z=0.0139327878...}, {x=-0.000351825875 y=0.993669689 z=0.112340435...},{x=-0.0139817735 y=-0.112334356 z=0.993572116...}" +


                    " ");
            }
        }
    }

    that.forwardKinematics = function () {
        var test = {};
        for (var i = 0; i <= that.JOINT_NUM; i++) {
            var jointIndex = Object.keys(HiFiArmature).indexOf(PFNNArmature[i]);
            that.jointRotationsGlobal[i] = that.jointRotations[i];
            that.jointRestGlobal[i] = MyAvatar.getDefaultJointRotation(jointIndex);
            var j = MyAvatar.$getParentJointIndex(jointIndex);
            while (j != -1) {
                that.jointRotationsGlobal[i] = Quat.multiply(that.jointRotationsGlobal[i], that.jointRotations[PFNNArmature[MyAvatar.getJointNames()[j]]]);
                that.jointRestGlobal[i] = Quat.multiply(that.jointRestGlobal[i], MyAvatar.getDefaultJointRotation(j));
                j = MyAvatar.$getParentJointIndex(j);
            }
            test[MyAvatar.getJointNames()[jointIndex]] = that.jointRotationsGlobal[i];
            MyAvatar.setJointRotation(jointIndex, that.jointRotationsGlobal[i]);//, that.jointRestGlobal[i]));
        }
        /*
        Script.require('https://htio.keybase.pub/hljs.js')({
            title: 'Dump',
            style: 'font-size: 8px;',
            code: JSON.stringify(test,0,2),
        });
        Script.stop();*/
    }

    return that;
};


arrayFunctions = (function () {

    return {

        // subtract arrayTwo from arrayOne and return the result
        arraySubtract: function (arrayOne, arrayTwo) {

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
        arrayDivide: function (arrayOne, arrayTwo) {
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


/*var HiFiArmature = {
    "Hips": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftUpLeg": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftLeg": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftFoot": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftToeBase": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightUpLeg": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightLeg": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightFoot": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightToebase": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "Spine": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "Spine1": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "Spine2": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "Neck": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "Head": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftShoulder": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftArm": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftForeArm": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "LeftHand": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightShoulder": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightArm": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightForeArm": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    },
    "RightHand": {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    }
};*/

var HiFiArmature = {};
var tempJoints = MyAvatar.getJointNames();
tempJoints.forEach(function (joint) {
    HiFiArmature[joint] = {
        pos: {
            x: 0,
            y: 0,
            z: 0
        },
        prv: {
            x: 0,
            y: 0,
            z: 0
        }
    };
});

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
    "name": "prerotations",
    "version": "1.0",
    "joints": {
        "Hips": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftUpLeg": {
            "x": 0,
            "y": 0,
            "z": -180,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightUpLeg": {
            "x": 0,
            "y": 0,
            "z": 180,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftLeg": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightLeg": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftFoot": {
            "x": 80,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightFoot": {
            "x": 80,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftToeBase": {
            "x": 28.5194,
            "y": -1.43586,
            "z": 0.685672
        },
        "RightToeBase": {
            "x": 28.0842,
            "y": 1.45959,
            "z": -0.687243
        },
        "Spine": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "Spine1": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "Spine2": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftShoulder": {
            "x": -75,
            "y": 85,
            "z": 180,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightShoulder": {
            "x": -75,
            "y": -85,
            "z": -180,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftArm": {
            "x": -17,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightArm": {
            "x": -17,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftForeArm": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightForeArm": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "LeftHand": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "RightHand": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "Neck": {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "pitchPhase": 0,
            "yawPhase": 0,
            "rollPhase": 0
        },
        "Head": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftHandPinky1": {
            "x": -1.59,
            "y": 0,
            "z": 1.34
        },
        "LeftHandPinky2": {
            "x": 3.84,
            "y": 0,
            "z": 0
        },
        "LeftHandPinky3": {
            "x": 4.91,
            "y": 0,
            "z": 0
        },
        "LeftHandPinky4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftHandRing1": {
            "x": -1.12,
            "y": 0,
            "z": 0.34
        },
        "LeftHandRing2": {
            "x": 4.35,
            "y": 0,
            "z": 0
        },
        "LeftHandRing3": {
            "x": -2.89,
            "y": 0,
            "z": 0
        },
        "LeftHandRing4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftHandMiddle1": {
            "x": 1.12,
            "y": 0,
            "z": 3.67
        },
        "LeftHandMiddle2": {
            "x": 5.14,
            "y": 0,
            "z": 0
        },
        "LeftHandMiddle3": {
            "x": 1.71,
            "y": 0,
            "z": 0
        },
        "LeftHandMiddle4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftHandIndex1": {
            "x": 3.8,
            "y": 0,
            "z": 0.48
        },
        "LeftHandIndex2": {
            "x": 3,
            "y": 0,
            "z": 0
        },
        "LeftHandIndex3": {
            "x": 7.94,
            "y": 0,
            "z": 0
        },
        "LeftHandIndex4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "LeftHandThumb1": {
            "x": 18.28,
            "y": 0,
            "z": 31.87
        },
        "LeftHandThumb2": {
            "x": 5.96,
            "y": 0,
            "z": 0
        },
        "LeftHandThumb3": {
            "x": -7.57,
            "y": 0,
            "z": 0
        },
        "LeftHandThumb4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "RightHandPinky1": {
            "x": -3.8,
            "y": 0,
            "z": -3.1
        },
        "RightHandPinky2": {
            "x": 0.85,
            "y": 0,
            "z": 0
        },
        "RightHandPinky3": {
            "x": 0.86,
            "y": 0,
            "z": 0
        },
        "RightHandPinky4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "RightHandRing1": {
            "x": -0.97,
            "y": 0,
            "z": -1.13
        },
        "RightHandRing2": {
            "x": -1.24,
            "y": 0,
            "z": 0
        },
        "RightHandRing3": {
            "x": -4.53,
            "y": 0,
            "z": 0
        },
        "RightHandRing4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "RightHandMiddle1": {
            "x": -3.31,
            "y": 0,
            "z": 0
        },
        "RightHandMiddle2": {
            "x": 3.83,
            "y": 0,
            "z": 0
        },
        "RightHandMiddle3": {
            "x": 6.04,
            "y": 0,
            "z": 0
        },
        "RightHandMiddle4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "RightHandIndex1": {
            "x": 0.77,
            "y": 0,
            "z": 0.09
        },
        "RightHandIndex2": {
            "x": 4.03,
            "y": 0,
            "z": 0
        },
        "RightHandIndex3": {
            "x": 1.18,
            "y": 0,
            "z": 0
        },
        "RightHandIndex4": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "RightHandThumb1": {
            "x": 19.72,
            "y": 0,
            "z": -30.89
        },
        "RightHandThumb2": {
            "x": 2.35,
            "y": 0,
            "z": 0
        },
        "RightHandThumb3": {
            "x": -2.13,
            "y": 0,
            "z": 0
        },
        "RightHandThumb4": {
            "x": 0,
            "y": 0,
            "z": 0
        }
    }
}

// ECMAScript 6 specification ready string.contains() function
if (!('contains' in String.prototype)) {
    String.prototype.contains = function (str, startIndex) {
        return ''.indexOf.call(this, str, startIndex) !== -1;
    };
}

/**/
// Display joints info, set t-stance
var numJoints = 0;
/*for (joint in prerotations.joints) {
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
print("Avatar has " + numJoints + " joints (plus end sites).");*/


var update = function (deltaTime) {

    // this is effectively used to comment out code that needs work
    // whilst ensuring that it at least makes sense to the JS 'compiler'
    var codeHadNoIssues = false;

    //////////////////////////////////////////////////////////////////
    // Update Target Direction / Velocity: C++ pre_render line 1458 //
    //////////////////////////////////////////////////////////////////

    var trajectoryTargetDirectionNew = Quat.getFront(MyAvatar.headOrientation);
    /* Using the head orientation makes a lot fo sense to me, but we will probablly 
    // need to port the C++ version to match variables:
    // glm::vec3 trajectory_target_direction_new = glm::normalize(glm::vec3(camera->direction().x, 0.0, camera->direction().z));
    var upVector = glm.vec3(0, 1, 0);
    var horizontalAngle = Math.atan2(trajectoryTargetDirectionNew.x,
                                     trajectoryTargetDirectionNew.z);
    var trajectoryTargetRotationAngle = glm.rotate(horizontalAngle, upVector);
    var trajectoryTargetRotation = glm.mat3(trajectoryTargetRotationAngle);
    print("trajectoryTargetRotationAngle is " + trajectoryTargetRotationAngle); */
    var trajectoryTargetRotation = Quat.getFront(MyAvatar.orientation);
    // targetVelSpeed not needed, as is only used to calculate velocity, hopefully we can use MyAvatar.velocity for now
    //var targetVelSpeed = MyAvatar.walkSpeed;
    //trajectory.targetVel = glm.mix(trajectory.targetVel, trajectoryTargetVelocityNew, options.extraVelocitySmooth);    
    var trajectoryTargetVelocityNew = MyAvatar.velocity;
    trajectory.targetVel = trajectoryTargetVelocityNew;

    // dummy values
    character.strafeTarget = 0;
    character.strafeAmount = 0;

    var trajectoryTargetVelocityDir = Vec3.length(trajectory.targetVel) < 1e-05 ? trajectory.targetDir : Vec3.normalize(trajectory.targetVel);
    trajectoryTargetDirectionNew = mixDirections(trajectoryTargetVelocityDir, trajectoryTargetDirectionNew, character.strafeAmount);

    // Current issue: mixDirections is returning null...
    //print("trajectoryTargetDirectionNew: " + trajectoryTargetDirectionNew);

    if (codeHadNoIssues) {

        trajectory.targetDir = mixDirections(trajectory.targetDir, trajectoryTargetDirectionNew, options.extraDirectionSmooth);

        ////////////////////////////////////////////////////
        // Update Gait: C++ pre_render starting line 1483 //
        ////////////////////////////////////////////////////
        if (glm.length(trajectory.targetVel) < 0.1) {
            // Standing
            var standAmount = 1.0 - glm.clamp(glm.length(trajectory.targetVel) / 0.1, 0.0, 1.0);
            trajectory.gaitStand[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitStand[trajectory.LENGTH / 2], standAmount, options.extraGaitSmooth);
            trajectory.gaitWalk[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitWalk[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJog[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJog[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitCrouch[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitCrouch[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitBump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitBump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
        } else if (character.crouchedAmount > 0.1) {
            // Crouching
            trajectory.gaitStand[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitStand[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitWalk[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitWalk[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJog[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJog[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitCrouch[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitCrouch[trajectory.LENGTH / 2], character.crouchedAmount, options.extraGaitSmooth);
            trajectory.gaitJump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitBump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitBump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
        } else if (false) {
            // Jogging - set false for now, will need to experiment with threshold
            trajectory.gaitStand[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitStand[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitWalk[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitWalk[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJog[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJog[trajectory.LENGTH / 2], 1.0, options.extraGaitSmooth);
            trajectory.gaitCrouch[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitCrouch[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitBump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitBump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
        } else {
            // Walking
            trajectory.gaitStand[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitStand[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitWalk[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitWalk[trajectory.LENGTH / 2], 1.0, options.extraGaitSmooth);
            trajectory.gaitJog[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJog[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitCrouch[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitCrouch[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitJump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitJump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
            trajectory.gaitBump[trajectory.LENGTH / 2] = glm.mix(trajectory.gaitBump[trajectory.LENGTH / 2], 0.0, options.extraGaitSmooth);
        }

        ////////////////////////////////////////////////////////
        // Predict Future Trajectory C++ pre_render line 1519 //
        ////////////////////////////////////////////////////////
        var trajectoryPositionsBlend = new Array(trajectory.LENGTH);
        trajectoryPositionsBlend[trajectory.LENGTH / 2] = trajectory.positions[trajectory.LENGTH / 2];

        for (i = trajectory.LENGTH / 2 + 1; i < trajectory.LENGTH; i++) {

            var biasPos = character.responsive ? glm.mix(2.0, 2.0, character.strafeAmount) : glm.mix(0.5, 1.0, character.strafeAmount);
            var biasDir = character.responsive ? glm.mix(5.0, 3.0, character.strafeAmount) : glm.mix(2.0, 0.5, character.strafeAmount);

            var scalePos = (1.0 - Math.pow(1.0 - ((i - trajectory.LENGTH / 2) / (trajectory.LENGTH / 2)), biasPos));
            var scaleDir = (1.0 - Math.pow(1.0 - ((i - trajectory.LENGTH / 2) / (trajectory.LENGTH / 2)), biasDir));

            trajectoryPositionsBlend[i] = trajectoryPositionsBlend[i - 1] + glm.mix(
                trajectory.positions[i] - trajectory.positions[i - 1],
                trajectory.target_vel,
                scalePos);

            /* Collide with walls */
            for (j = 0; j < areas.numWalls(); j++) {
                var trjpoint = glm.vec2(trajectoryPositionsBlend[i].x, trajectoryPositionsBlend[i].z);
                if (glm.length(trjpoint - ((areas.wallStart[j] + areas.wallStop[j]) / 2.0)) >
                    glm.length(areas.wallStart[j] - areas.wallStop[j])) {
                    continue;
                }
                var segpoint = segment_nearest(areas.wallStart[j], areas.wallStop[j], trjpoint);
                var segdist = glm.length(segpoint - trjpoint);
                if (segdist < areas.wallWidth[j] + 100.0) {
                    var prjpoint0 = (areas.wallWidth[j] + 0.0) * glm.normalize(trjpoint - segpoint) + segpoint;
                    var prjpoint1 = (areas.wallWidth[j] + 100.0) * glm.normalize(trjpoint - segpoint) + segpoint;
                    var prjpoint = glm.mix(prjpoint0, prjpoint1, glm.clamp((segdist - areas.wallWidth[j]) / 100.0, 0.0, 1.0));
                    trajectoryPositionsBlend[i].x = prjpoint.x;
                    trajectoryPositionsBlend[i].z = prjpoint.y;
                }
            }

            trajectory.directions[i] = mixDirections(trajectory.directions[i], trajectory.targetDir, scaleDir);
            trajectory.heights[i] = trajectory.heights[trajectory.LENGTH / 2];
            trajectory.gaitStand[i] = trajectory.gaitStand[trajectory.LENGTH / 2];
            trajectory.gaitWalk[i] = trajectory.gaitWalk[trajectory.LENGTH / 2];
            trajectory.gaitJog[i] = trajectory.gaitJog[trajectory.LENGTH / 2];
            trajectory.gaitCrouch[i] = trajectory.gaitCrouch[trajectory.LENGTH / 2];
            trajectory.gaitJump[i] = trajectory.gaitJump[trajectory.LENGTH / 2];
            trajectory.gaitBump[i] = trajectory.gaitBump[trajectory.LENGTH / 2];
        }

        for (i = trajectory.LENGTH / 2 + 1; i < trajectory.LENGTH; i++) {
            trajectory.positions[i] = trajectoryPositionsBlend[i];
        }

        // Work on these aspects isn't essential for getting a standing / walking demo together
        // but will need to be implemented at some point
        // (flying will be needed too)
        /* Jumps 
        for (i = trajectory.LENGTH / 2; i < trajectory.LENGTH; i++) {
            trajectory.gaitJump[i] = 0.0;
            for (j = 0; j < areas.numJumps(); j++) {
                var dist = glm.length(trajectory.positions[i] - areas.jump_pos[j]);
                trajectory.gaitJump[i] = Math.max(trajectory.gaitJump[i],
                    1.0 - glm.clamp((dist - areas.jumpSize[j]) / areas.jump_falloff[j], 0.0, 1.0));
            }
        }*/

        /* Crouch Area 
        for (i = trajectory.LENGTH / 2; i < trajectory.LENGTH; i++) {
            for (j = 0; j < areas.numCrouches(); j++) {
                var dist_x = abs(trajectory.positions[i].x - areas.crouch_pos[j].x);
                var dist_z = abs(trajectory.positions[i].z - areas.crouch_pos[j].z);
                var height = (Math.sin(trajectory.positions[i].x / Areas::CROUCHWAVE) + 1.0) / 2.0;
                trajectory.gaitCrouch[i] = glm.mix(1.0 - height, trajectory.gaitCrouch[i],
                    glm.clamp(
                    ((dist_x - (areas.crouchSize[j].x / 2)) +
                        (dist_z - (areas.crouchSize[j].y / 2))) / 100.0, 0.0, 1.0));
            }
        }*/

        /* Walls 
        for (i = 0; i < trajectory.LENGTH; i++) {
            trajectory.gait_bump[i] = 0.0;
            for (int j = 0; j < areas.numWalls(); j++) {
                var trjpoint = glm.vec2(trajectory.positions[i].x, trajectory.positions[i].z);
                var segpoint = segment_nearest(areas.wallStart[j], areas.wallStop[j], trjpoint);
                var segdist = glm.length(segpoint - trjpoint);
                trajectory.gait_bump[i] = glm.max(trajectory.gait_bump[i], 1.0 - glm.clamp((segdist - areas.wallWidth[j]) / 10.0, 0.0, 1.0));
            }
        }*/

        /* Trajectory Rotation */
        for (i = 0; i < trajectory.LENGTH; i++) {
            trajectory.rotations[i] = glm.mat3(glm.rotate(atan2f(
                trajectory.directions[i].x,
                trajectory.directions[i].z), glm.vec3(0, 1, 0)));
        }

        /* Trajectory Heights */
        for (i = trajectory.LENGTH / 2; i < trajectory.LENGTH; i++) {
            trajectory.positions[i].y = heightmap.sample(glm.vec2(trajectory.positions[i].x, trajectory.positions[i].z));
        }

        trajectory.heights[trajectory.LENGTH / 2] = 0.0;
        for (i = 0; i < trajectory.LENGTH; i += 10) {
            trajectory.heights[trajectory.LENGTH / 2] += (trajectory.positions[i].y / ((trajectory.LENGTH) / 10));
        }

        var root_position = glm.vec3(
            trajectory.positions[trajectory.LENGTH / 2].x,
            trajectory.heights[trajectory.LENGTH / 2],
            trajectory.positions[trajectory.LENGTH / 2].z);

        var root_rotation = trajectory.rotations[trajectory.LENGTH / 2];

        /* Input Trajectory Positions / Directions */
        for (i = 0; i < trajectory.LENGTH; i += 10) {
            var w = (trajectory.LENGTH) / 10;
            var pos = glm.inverse(root_rotation) * (trajectory.positions[i] - root_position);
            var dir = glm.inverse(root_rotation) * trajectory.directions[i];
            pfnn.Xp((w * 0) + i / 10) = pos.x;
            pfnn.Xp((w * 1) + i / 10) = pos.z;
            pfnn.Xp((w * 2) + i / 10) = dir.x;
            pfnn.Xp((w * 3) + i / 10) = dir.z;
        }

        /* Input Trajectory Gaits */
        for (i = 0; i < trajectory.LENGTH; i += 10) {
            w = (trajectory.LENGTH) / 10;
            pfnn.Xp((w * 4) + i / 10) = trajectory.gaitStand[i];
            pfnn.Xp((w * 5) + i / 10) = trajectory.gaitWalk[i];
            pfnn.Xp((w * 6) + i / 10) = trajectory.gaitJog[i];
            pfnn.Xp((w * 7) + i / 10) = trajectory.gaitCrouch[i];
            pfnn.Xp((w * 8) + i / 10) = trajectory.gaitJump[i];
            pfnn.Xp((w * 9) + i / 10) = 0.0; // Unused.
        }

        /* Input Joint Previous Positions / Velocities / Rotations */
        var prev_root_position = glm.vec3(
            trajectory.positions[trajectory.LENGTH / 2 - 1].x,
            trajectory.heights[trajectory.LENGTH / 2 - 1],
            trajectory.positions[trajectory.LENGTH / 2 - 1].z);

        var prevRootRotation = trajectory.rotations[trajectory.LENGTH / 2 - 1];

        for (i = 0; i < character.JOINT_NUM; i++) {
            var o = (((trajectory.LENGTH) / 10) * 10);
            var pos = glm.inverse(prevRootRotation) * (character.jointPositions[i] - prev_root_position);
            var prv = glm.inverse(prevRootRotation) * character.jointVelocities[i];
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 0) + i * 3 + 0) = pos.x;
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 0) + i * 3 + 1) = pos.y;
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 0) + i * 3 + 2) = pos.z;
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 1) + i * 3 + 0) = prv.x;
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 1) + i * 3 + 1) = prv.y;
            pfnn.Xp(o + (character.JOINT_NUM * 3 * 1) + i * 3 + 2) = prv.z;
        }

        /* Input Trajectory Heights */
        for (i = 0; i < trajectory.LENGTH; i += 10) {
            var o = (((trajectory.LENGTH) / 10) * 10) + character.JOINT_NUM * 3 * 2;
            var w = (trajectory.LENGTH) / 10;
            var position_r = trajectory.positions[i] + (trajectory.rotations[i] * glm.vec3(trajectory.width, 0, 0));
            var position_l = trajectory.positions[i] + (trajectory.rotations[i] * glm.vec3(-trajectory.width, 0, 0));
            pfnn.Xp(o + (w * 0) + (i / 10)) = heightmap.sample(glm.vec2(position_r.x, position_r.z)) - root_position.y;
            pfnn.Xp(o + (w * 1) + (i / 10)) = trajectory.positions[i].y - root_position.y;
            pfnn.Xp(o + (w * 2) + (i / 10)) = heightmap.sample(glm.vec2(position_l.x, position_l.z)) - root_position.y;
        }

        //////////////////////////////////////////////////
        // Perform Regression C++ _pre_render line 1680 //
        //////////////////////////////////////////////////

        // At this point we should have a fully populated set of input values for the PFNN (pfnn.Xp)
        // and be ready to compare values with the C++ code. This will probably need to entail some sort
        // of csv console output that can be pasted into a spreadsheet, as it's gonna be huge!

        pfnn.predict(character.phase);

        //////////////////////////////////////////////////////////////////////////////////////
        // Build Local Transforms - apply pfnn.Yp to our character C++ pre_render line 1705 //
        //////////////////////////////////////////////////////////////////////////////////////

        /*
        ** Blending Between the predicted positions and
        ** the previous positions plus the velocities
        ** smooths out the motion a bit in the case
        ** where the two disagree with each other.
        */

        character.forward_kinematics();


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
        // The joint data from these needs to be combined into parent / children joints. 
        //
        // Moreover, the skeleton structure differs quite significantly - the PFNN demo skeleton *looks*
        // like Carnegie Melon data, which has always been difficult to deal with...
        //
        //

        // Here we will update the HiFiArmature with values from Yp
        // This is a very rough, UNTESTED attempt at retargetting
    } // end of codeHadNoIssues


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // At this point, Yp contains the default pose, so we can apply it to the HiFi armature and *should* see the default (Ymean) pose //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    for (joint in HiFiArmature) {
        HiFiArmature[joint].prv = 0;
        HiFiArmature[joint].pos = MyAvatar.getJointPosition(joint);
    }
    var jointIndex = 0;

    var trajectoryLength = trajectory.getLength();
    //var rootPosition = Vec3.sum(MyAvatar.position, { x:0, y:-character.getAvatarHipsToFeet(), z:0 });
    var rootPosition = glm.vec3(MyAvatar.position.x, MyAvatar.position.y, MyAvatar.position.z).add(glm.vec3(0, -character.avatarHipsToFeet(), 0))
    //print("Character.initialise: rootPosition is { x:" + rootPosition.x + ", y:" + rootPosition.y + ", z:" + rootPosition.z + " }");
    var rootRotation = glm.mat3();
    var Yp = pfnn.getYp();
    var debugBool = true;

    for (var jointIndex = 0; jointIndex <= character.JOINT_NUM; jointIndex++) {

        var opos = 8 + (((trajectoryLength / 2) / 10) * 4) + (character.JOINT_NUM * 3 * 0);
        var ovel = 8 + (((trajectoryLength / 2) / 10) * 4) + (character.JOINT_NUM * 3 * 1);
        var orot = 8 + (((trajectoryLength / 2) / 10) * 4) + (character.JOINT_NUM * 3 * 2);

        /*
        glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos + i * 3 + 0), Yp(opos + i * 3 + 1), Yp(opos + i * 3 + 2))) + root_position;
        glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel + i * 3 + 0), Yp(ovel + i * 3 + 1), Yp(ovel + i * 3 + 2)));
        glm::mat3 rot = (root_rotation * glm::toMat3(quatExp(glm::vec3(Yp(orot + i * 3 + 0), Yp(orot + i * 3 + 1), Yp(orot + i * 3 + 2)))));

        xPos is 0
        yPos is 93.1293793
        zPos is 0
        xRot is 0.0562893562
        yRot is -0.00699363137
        zRot is -0.000217798908
        xVel is -0.00416085264YpOpos
        zVel is 1.23609018
        */

        // from: glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos + jointIndex * 3 + 0), Yp(opos + jointIndex * 3 + 1), Yp(opos + jointIndex * 3 + 2))) + root_position;
        // from: HumbleTim pos = glm.vec3(glm.mat4(root_rotation).mul(glm.vec4(glm.vec3(),1))).add(root_position);
        var oposX = parseFloat(Yp[opos + jointIndex * 3 + 0]);
        var oposY = parseFloat(Yp[opos + jointIndex * 3 + 1]);
        var oposZ = parseFloat(Yp[opos + jointIndex * 3 + 2]);
        var oposVec3 = glm.vec3(oposX, oposY, oposZ);
        var oposVec4 = glm.vec4(oposVec3, 1);
        oposVec4 = glm.mat4(rootRotation).mul(oposVec4);
        oposVec3 = glm.vec3(oposVec4);
        var pos = oposVec3.add(rootPosition);

        // from: glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel + jointIndex * 3 + 0), Yp(ovel + jointIndex * 3 + 1), Yp(ovel + jointIndex * 3 + 2)));
        var ovelX = parseFloat(Yp[ovel + jointIndex * 3 + 0]);
        var ovelY = parseFloat(Yp[ovel + jointIndex * 3 + 1]);
        var ovelZ = parseFloat(Yp[ovel + jointIndex * 3 + 2]);
        var ovelVec3 = glm.vec3(ovelX, ovelY, ovelZ);
        var ovelVec4 = glm.vec4(ovelVec3, 1);
        ovelVec4 = glm.mat4(rootRotation).mul(ovelVec4);
        var vel = glm.vec3(ovelVec4);

        // from: glm::mat3 rot = (root_rotation * glm::toMat3(quatExp(glm::vec3(Yp(orot + jointIndex * 3 + 0), Yp(orot + jointIndex * 3 + 1), Yp(orot + jointIndex * 3 + 2)))));
        // print(orot + jointIndex * 3);
        var orotX = parseFloat(Yp[orot + jointIndex * 3 + 0]);
        var orotY = parseFloat(Yp[orot + jointIndex * 3 + 1]);
        var orotZ = parseFloat(Yp[orot + jointIndex * 3 + 2]);
        var orotVec3 = glm.vec3(orotX, orotY, orotZ);

        // print("joint "+joint);
        // print("pfnnjoint "+PFNNArmature[joint]);
        var jointName;

        character.jointVelocities[jointIndex] = {x: 0, y: 0, z: 0};

        // Translate between armatures
        switch (PFNNArmature[jointIndex]) {

            case "ROOT" :
                jointName = "Hips";
                character.jointPositions[jointIndex] = Vec3.sum(HiFiArmature["Hips"].pos, character.avatarHipsToFeet(MyAvatar.position));
                break;

            case "Neck" :
            case "Neck1" :
                jointName = "Neck";
                character.jointPositions[jointIndex] = Vec3.multiply(HiFiArmature["Neck"].pos, 0.5);
                break;

            //case "LeftHipJoint" :
            case "LeftUpLeg" :
                jointName = "LeftUpLeg";
                character.jointPositions[jointIndex] = Vec3.multiply(HiFiArmature["LeftUpLeg"].pos, 0.5);
                break;

            //case "RightHipJoint" :
            case "RightUpLeg" :
                jointName = "RightUpLeg";
                character.jointPositions[jointIndex] = Vec3.multiply(HiFiArmature["RightUpLeg"].pos, 0.5);
                break;

            case "LowerBack" :
                jointName = "Spine";
                character.jointPositions[jointIndex] = HiFiArmature["Spine"].pos;
                break;

            case "Spine" :
                jointName = "Spine1";
                character.jointPositions[jointIndex] = HiFiArmature["Spine1"].pos;
                break;

            case "Spine1" :
                jointName = "Spine2";
                character.jointPositions[jointIndex] = HiFiArmature["Spine2"].pos;
                break;

            case "End Site" :
                jointName = "End Site";
                character.jointPositions[jointIndex] = {x: 0, y: 0, z: 0};
                break;

            default :
                if (!!HiFiArmature[PFNNArmature[jointIndex]]) {
                    character.jointPositions[jointIndex] = HiFiArmature[PFNNArmature[jointIndex]].pos;
                    jointName = PFNNArmature[jointIndex];
                } else {
                    jointName = "End Site";
                    character.jointPositions[jointIndex] = {x: 0, y: 0, z: 0};
                }
                break;
        }


        if (!!HiFiArmature[jointName]) {
            var temp = prerotations.joints[jointName];
            orotVec3 = glm.add(orotVec3,glm.radians(glm.vec3(temp.x,temp.y,temp.z)));
        } else {
        }

        // print(glm.to_string(orotVec3));
        var orotVec4 = quatExp(orotVec3);
        var orotMat3 = glm.mat3(glm.toMat4(orotVec4));
        var rot = glm.mul(orotMat3, rootRotation);

        print(JSON.stringify(rot));

        // var rot = rootRotation;



        var mat4 = glm.mat4(
            rot[0][0], rot[1][0], rot[2][0], pos[0],
            rot[0][1], rot[1][1], rot[2][1], pos[1],
            rot[0][2], rot[1][2], rot[2][2], pos[2],
            0, 0, 0, 1
        )

        var transpose = glm.transpose(mat4);

        var jointAbsolutePose = JSON.parse(glm.quat(transpose).json);


        if (jointName !== "End Site") {

            var jointAbsolutePose2 = glm.quat(glm.mat4(rot)).json;

            if (jointName === "Hips") {
                MyAvatar.setJointRotation(jointIndex, jointAbsolutePose2);
            } else {
                character.jointRotationsGlobal[jointIndex] = jointAbsolutePose;
            }
        }
    }

    for (var i = 0; i <= character.JOINT_NUM; i++) {
        if (i == 0) {
            character.jointRotations[i] = character.jointRotationsGlobal[i];
        } else {
            var jointIndex = Object.keys(HiFiArmature).indexOf(PFNNArmature[i]);
            character.jointRotations[i] = Quat.multiply(character.jointRotationsGlobal[i], Quat.inverse(character.jointRotationsGlobal[PFNNArmature[MyAvatar.getJointNames()[MyAvatar.$getParentJointIndex(jointIndex)]]]));
        }
    }

    character.forwardKinematics();

    /* IK? (probably not, at least to start with */


    /////////////////////////////////////////////
    // Render Trajectory: C++ render line 2323 //
    /////////////////////////////////////////////
    var currentDirection = trajectoryTargetDirectionNew;
    var currentHeight = character.avatarHipsToFeet(MyAvatar.position);
    var currentPosition = Vec3.sum(MyAvatar.position, {x: 0, y: -currentHeight, z: 0});
    var currentRotation = trajectoryTargetRotation;// Quat.inverse(MyAvatar.orientation); // Vec3.multiplyQbyV(Quat.inverse(MyAvatar.orientation), MyAvatar.velocity); // Vec3.normalize({ x: MyAvatar.bodyYaw, y:0, z:0 }); //  MyAvatar.orientation;

    // update current trajectory
    trajectory.positions[trajectory.LENGTH / 2] = currentPosition;
    trajectory.rotations[trajectory.LENGTH / 2] = currentRotation;
    trajectory.heights[trajectory.LENGTH / 2] = currentHeight;
    trajectory.directions[trajectory.LENGTH / 2] = currentDirection;

    targetDir = {x: 0, y: 0, z: 1};
    targetVel = {x: 0, y: 0, z: 0};

    // update future trajectory
    for (var i = trajectory.LENGTH / 2 + 1; i < trajectory.LENGTH; i++) {
        // distance to next trajectory marker
        var stepForward = (i - (trajectory.LENGTH / 2 + 1)) / (trajectory.MARKER_RATIO * 4);
        var nextPosition = Vec3.sum(currentPosition, Vec3.multiply(stepForward, MyAvatar.velocity));
        // Get the terrain height for this position
        var forwardProbePosition = {x: nextPosition.x, y: MyAvatar.position.y, z: nextPosition.z};
        var nextHeight = trajectory.terrainFollowing ?
            character.avatarHipsToFeet(forwardProbePosition) :
            currentHeight; //  debug only
        trajectory.heights[i] = nextHeight;
        nextPosition.y = forwardProbePosition.y - nextHeight;
        trajectory.positions[i] = nextPosition;
    }

    // update overlays
    for (var i = 0; i < trajectory.LENGTH; i += 10) {
        Overlays.editOverlay(trajectory.locationOverlays[i / trajectory.MARKER_RATIO], {
            position: trajectory.positions[i],
            rotations: trajectory.rotations[i],
        });
        Overlays.editOverlay(trajectory.directionOverlays[i / trajectory.MARKER_RATIO], {
            start: trajectory.positions[i],
            end: Vec3.sum(trajectory.positions[i], Vec3.multiply(0.25, trajectory.rotations[i]))
        });
    }

    // Update Past Trajectory: C++ post_render line 1954
    for (var i = 0; i < trajectory.LENGTH / 2; i++) {
        trajectory.positions[i] = trajectory.positions[i + 1];
        trajectory.rotations[i] = trajectory.rotations[i + 1];
        trajectory.directions[i] = trajectory.directions[i + 1];
        trajectory.heights[i] = trajectory.heights[i + 1];
        trajectory.gaitStand[i] = trajectory.gaitStand[i + 1];
        trajectory.gaitWalk[i] = trajectory.gaitWalk[i + 1];
        trajectory.gaitJog[i] = trajectory.gaitJog[i + 1];
        trajectory.gaitCrouch[i] = trajectory.gaitCrouch[i + 1];
        trajectory.gaitJump[i] = trajectory.gaitJump[i + 1];
        trajectory.gaitBump[i] = trajectory.gaitBump[i + 1];
    }
}


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


// Main loop 
Script.update.connect(function (deltaTime) {
    update(deltaTime);
});

// Tidy up
Script.scriptEnding.connect(function () {
    print("PFNN: Closing down");
    //MyAvatar.clearJointsData();
    trajectory.clean_up();
});
