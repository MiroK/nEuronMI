// A spherical soma with cylindrical probe

//    |--|       [|] 
//    |D |       [|]
//    |  |       [|]
//    /---\      [|]
//   / S   \ <---->    z = 0
//   \     /
//    \---/
//    |   |
//    | A |
//    |---|
// z
// ^
// |
// |___>x
//
// Neuron
// Soma is a sphere at (0, 0, 0) with a radius rad_soma (surface marked as 1)
// Dendrite is a cylinder with length_dend, rad_dend (surface marked as 3)
// Axon is a cylinder with length_axon, rad_axon (surface marked as 2)
//
// Bbox
// Based on the geometry of neuron a bounding box is defined which is
// [-rad_soma-dxn, rad_soma+dxp] x [-rad_soma-dy, rad_soma+dy] x 
// [bottom of axon - dz, top of dend + dz]
//
// Probe:
// The probe extends from the (z)top of the bbox to z = z_shift. It is 
// a cylinder with radius rad_probe whose centerline is at 
// x = rad_soma + x_shift + rad_probe and y = y_shift. The probe tip
// is marked as surface 41 while the side are marked as 40.
//
// NOTE:
// It is the user's responsibility to make sure that the probe fits 
// the domain and that other inputs are sane; e.g. soma radius is larger
// then axon/dend.

// Inputs: All but y_shift, z_shift are assumed to be positive
DefineConstant[
  rad_soma = {1, Min 0, Name "Radius of soma"}
  rad_dend = {0.4, Min 0, Name "Radius of dendrite"}
  length_dend = {2, Min 0, Name "Length of dendrite"}
  rad_axon = {0.3, Min 0, Name "Radius of axon"}
  length_axon = {4, Min 0, Name "Length of axon"}
  dxp = {2.5, Min 0, Name "Bbox x < dxp + rad_soma"}
  dxn = {0.5, Min 0, Name "Bbox x > -dxm - rad_soma "}
  dy = {0.2, Min 0, Name "Bbox at y extends from -rad_soma - dy"}
  dz = {0.2, Min 0, Name "Bbox padding in z-dir"}
  with_probe = {1, Choices{0,1}, Name "Use probe"}
  x_shift = {1.0, Min 0, Name "probe distance from soma at z = 0"}
  y_shift = {0.0, Name "y coord of probe centerline"}
  z_shift = {0.0, Name "z coord of probe tip"}
  rad_probe = {0.2, Min 0, Name "Probe radius"}
];

// ------------------------------------------------------------------

SetFactory("OpenCASCADE");

soma = newv;
Sphere(soma) = {0, 0, 0, rad_soma};

dend = newv;
base_dend = Sqrt(rad_soma*rad_soma-rad_dend*rad_dend);
Cylinder(dend) = {0, 0, base_dend,
                  0, 0, length_dend, rad_dend, 2*Pi}; // Extent from base

axon = newv;         
base_axon = Sqrt(rad_soma*rad_soma-rad_axon*rad_axon);
Cylinder(axon) = {0, 0, -base_axon,
                  0, 0, -length_axon, rad_axon, 2*Pi};

neuron() = BooleanUnion{ Volume{soma}; Delete;}{ Volume{dend, axon}; Delete;};
neuron_surface() = Unique(Abs(Boundary{ Volume{neuron()}; }));

Physical Volume(1) = {neuron[]};
// Bake the surface into pieces
// 0 1 2 3 4
// | - o - |
Physical Surface(3) = {neuron_surface[0], neuron_surface[1]}; // Dend
Physical Surface(1) = {neuron_surface[2]}; // Soma
Physical Surface(2) = {neuron_surface[3], neuron_surface[4]}; // Axon

// Bbox based on axon geometry
x0 = -rad_soma - dxn;
dx_box = rad_soma + dxp - x0;

y0 = -rad_soma - dy;
dy_box = 2*Abs(y0);

z0 = -base_axon - length_axon - dz;
z1 = base_dend + length_dend + dz;
dz_box = z1 - z0;

// Bounding box
bbox = newv;
Box(bbox) = {x0, y0, z0, dx_box, dy_box, dz_box};

If(with_probe)
  // Probe
  probe = newv;
  Cylinder(probe) = {rad_soma+x_shift+rad_probe, y_shift, z_shift,
                   0, 0, Abs(z_shift)+z1+1, // Longer to make the isection work
                   rad_probe};

  // Make the hole
  bbox_probe() = BooleanDifference { Volume{bbox}; Delete; }{ Volume{probe}; Delete;};
  outside() = BooleanDifference { Volume{bbox_probe}; Delete; }{ Volume{neuron};};
  
  // Probe surfaces
  Physical Surface(4) = {12};
  Physical Surface(41) = {13};
  probe_surface[] = {12, 13};
Else
  outside() = BooleanDifference { Volume{bbox}; Delete; }{ Volume{neuron};};
EndIf
Physical Volume(2) = {outside[]};  
  
// Mesh size on neuron
Field[1] = MathEval;
Field[1].F = "0.1";

Field[2] = Restrict;
Field[2].IField = 1;
Field[2].FacesList = {neuron_surface[]};

If(with_probe)
  // Mesh size on Probe
  Field[3] = MathEval;
  Field[3].F = "0.1";

  Field[4] = Restrict;
  Field[4].IField = 3;
  Field[4].FacesList = {probe_surface[]};
  
  // Mesh size everywhere else
  Field[5] = MathEval;
  Field[5].F = "0.25";
  
  Field[6] = Min;
  Field[6].FieldsList = {2, 4, 5};
  Background Field = 6;  
Else
  // Mesh size everywhere else
  Field[3] = MathEval;
  Field[3].F = "0.25";

  Field[4] = Min;
  Field[4].FieldsList = {2, 3};
  Background Field = 4;
EndIf
