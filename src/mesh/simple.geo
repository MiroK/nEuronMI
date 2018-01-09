//    |--|
//    |  |   
//    |  |     __
//    /---\   |__|
//   /     \
//   \     /
//    \---/
//

// NOTE: These are all assumed to be positive
Radius = 1;          // Some, spehere radius
radius = 0.5;        // radius of the cylinder
length = 2;          // length of the cylinder

dX = 0.3;            // Size of the probe box
dY = 0.4;
dZ = 0.3;

x_shift = 0.1;       // the probe lower left down corner
y_shift = 0.0;       // will be at [radius + x_shift,
z_shift = 0.2;       //             y_shift - dY/2
//                                  cog_z + z_shift]
dx = 0.2;            // Bounding box paddin
dy = 0.2;
dz = 0.2;

// ------------------------------------------------------------------
SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, Radius};

Cylinder(2) = {0, 0, Radius - Sqrt(Radius*Radius - radius*radius),
               0, 0, length, radius, 2*Pi};

neuron() = BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; };
// Surface of neauron will be tagged as 1
neuron_surface() = Unique(Abs(Boundary{ Volume{neuron()}; }));
Physical Surface(1) = {neuron_surface[]};
Physical Volume(1) = {neuron[]};

// Probe
probe = newv;
Box(probe) = {Radius + x_shift, y_shift-dY/2,
              0.5*(-Radius + Sqrt(Radius*Radius - radius*radius)+length)-z_shift,
              dX, dY, dZ};

// Bounding box
bbox = newv;
Box(bbox) = {-2, -2, -2, 4, 4, 5};
box_surface() = Unique(Abs(Boundary{ Volume{bbox}; }));
Physical Surface(2) = {box_surface[]};

// Make the hole
bbox_probe() = BooleanDifference { Volume{bbox}; Delete; }{ Volume{probe}; Delete;};
outside() = BooleanDifference { Volume{bbox_probe}; Delete; }{ Volume{neuron};};
Physical Volume(2) = {outside[]};

// Mesh size on neuron
Field[1] = MathEval;
Field[1].F = "0.3";

Field[2] = Restrict;
Field[2].IField = 1;
Field[2].FacesList = {neuron_surface[]};

// Mesh size everywhere else
Field[3] = MathEval;
Field[3].F = "0.5";

Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;
