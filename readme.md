# PFNN for High Fidelity

This project is a work in progress - it doesn't work yet. 
The basic idea is to see if the PFNN can be adapted to run in HighFidelity using JavaScript. 
The C++ code is being translated into JavaScript. 
GLM-JS is being used for simpler translation of C++ glm functions.
Many thanks to HumbleTim and Fluffy Jenkins for their help so far. 

![HiFi PFNN so far](/pfnn-hifi.gif)

## Tasks complete
* Analysed PFNN data files: Xmean, Xstd, Ymean, Ystd (analysis in PFNN-io-parameter-definition.xls)
* Converted PFNN binary data into JSON files and loaded into HiFi JavaScript.
* Floor markers implemented to provide visual feedback whilst developing (see above)
* PFNN character armature and HiFi armature have a number of structural and naming differences. Quick and dirty re-targetting has been implemented but not yet tested.
* Initialisation functions implemented. Variable values verified by comparing with corresponding C++ output

## Current state 
I'm now hitting frame execution times of over 60mS, which indicates that JS may not be up to the task :-(

## Setting up environment
Ideally, the C++ project must first be compiled and run so it can be used as a reference. 
Then the JS project is set up for High Fidelity. There are too many PFNN data files to put on GitHub, so the 'pfnn-data' directory must be downloaded from here:
http://davedub.co.uk/downloads/hf/pfnn/pfnn-data.zip
Once downloaded, simply unzip into the same folder as 'ddAnimate.js'. Relative paths are used in the code, so it should work fine.

## Further information
Link 1: For an explanation of and some background on the PFNN, please see here:
http://theorangeduck.com/page/phase-functioned-neural-networks-character-control

Link 2: For the C++ source code for the demo see here:
https://github.com/sreyafrancis/PFNN

Link 3: For the GLM-JS library, see here:
https://github.com/humbletim/glm-js

Link 4: How the PFNN fits in to my larger project:
http://davedub.co.uk/davedub/wordpress/character-animation-for-virtual-reality/