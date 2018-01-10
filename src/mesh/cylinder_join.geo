SetFactory("OpenCASCADE");

// Cylinders

Macro Segment // (--)
  // It is assumed that all the vars are in the namespace
  cone = newv;
  Cylinder(cone) = {base_x, base_y, base_z, dir_x, dir_y, dir_z, rad};
  // Balls to make joints
  ball0 = newv;
  Sphere(ball0) = {base_x, base_y, base_z, rad};

  ball1 = newv;
  Sphere(ball1) = {base_x+dir_x, base_y+dir_y, base_z+dir_z, rad};
  
  // The jointed cone
  cball0() = BooleanDifference {Volume{ball0}; Delete; }{Volume{cone}; };
  cball1() = BooleanDifference {Volume{ball1}; Delete; }{Volume{cone}; };

  union() = BooleanUnion {Volume{cone}; Delete;}{Volume{cball0, cball1}; };
  Delete { Volume{cball0()}; }
  Delete { Volume{cball1()}; }

  segments[index] = union;
  index += 1;
  Return 

  
Macro ClosedSegment // [--)
  // It is assumed that all the vars are in the namespace
  cone = newv;
  Cylinder(cone) = {base_x, base_y, base_z, dir_x, dir_y, dir_z, rad};
  // End Ball to make joints
  ball1 = newv;
  Sphere(ball1) = {base_x+dir_x, base_y+dir_y, base_z+dir_z, rad};
  
  // The jointed cone
  cball1() = BooleanDifference {Volume{ball1}; Delete; }{Volume{cone};};

  union() = BooleanUnion {Volume{cball1}; Delete;}{Volume{cone}; Delete;};
  segments[index] = union;
  index += 1;
  Return 

Macro SegmentClosed // (--]
  // It is assumed that all the vars are in the namespace
  cone = newv;
  Cylinder(cone) = {base_x, base_y, base_z, dir_x, dir_y, dir_z, rad};
  // End Ball to make joints
  ball1 = newv;
  Sphere(ball1) = {base_x, base_y, base_z, rad};
  
  // The jointed cone
  cball1() = BooleanDifference {Volume{ball1}; Delete; }{Volume{cone}; };

  union() = BooleanUnion {Volume{cone}; Delete;}{Volume{cball1}; Delete;};
  
  segments[index] = union;
  index += 1;
  Return 
  
// ------------------------------------------------------------------

index = 0;
rad = 0.4;
// Goal
//        |
//        \/
//         |
//         u

base_x = 0;
base_y = 0;
base_z = -2;
dir_x = 0;
dir_y = 0;
dir_z = 3;
Call ClosedSegment;

base_x += dir_x;
base_y += dir_y;
base_z += dir_z;
dir_x = 1;
dir_y = 1;
dir_z = 1;
Call SegmentClosed;

dir_x = -2;
dir_y = -2;
dir_z = 2;
Call Segment;

base_x += dir_x;
base_y += dir_y;
base_z += dir_z;
dir_x = -1;
dir_y = 1;
dir_z = 1.5;
Call SegmentClosed;

nsegments = #segments[];
nsegments -= 1; // Call Increases
neuron = BooleanUnion {Volume{segments[0]}; Delete;}{Volume{segments[{1:nsegments}]}; Delete;};

// Embedding
// Surface of neauron will be tagged as 1
neuron_surface() = Unique(Abs(Boundary{ Volume{neuron()}; }));
Physical Surface(1) = {neuron_surface[]};
Physical Volume(1) = {neuron[]};

// // Probe
// probe = newv;
// Box(probe) = {1, 1, 0., 0.5, 0.5, 0.5};

// // Bounding box
// bbox = newv;
// Box(bbox) = {-4, -3, -2.5, 6, 6, 8};
// box_surface() = Unique(Abs(Boundary{ Volume{bbox}; }));
// Physical Surface(2) = {box_surface[]};

// // Make the hole
// bbox_probe() = BooleanDifference { Volume{bbox}; Delete; }{ Volume{probe}; Delete;};
// outside() = BooleanDifference { Volume{bbox_probe}; Delete; }{ Volume{neuron};};
// Physical Volume(2) = {outside[]};

// // Mesh size on neuron
// Field[1] = MathEval;
// Field[1].F = "0.2";

// Field[2] = Restrict;
// Field[2].IField = 1;
// Field[2].FacesList = {neuron_surface[]};

// // Mesh size everywhere else
// Field[3] = MathEval;
// Field[3].F = "0.5";

// Field[4] = Min;
// Field[4].FieldsList = {2, 3};
// Background Field = 4;
