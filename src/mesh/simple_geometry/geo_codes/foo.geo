SetFactory("OpenCASCADE");

probe_x = 0;
probe_y = 0;
probe_z = 0;

probe_top = 500;  // At least ....
probe_thick = 15;

contact_rad = 7.5;

with_pts = 1;

If(with_pts)
  volume_counter = 0;
  // Contacts
  For i In {0:11}
    circle = newv;
    cx  = probe_x - 0.5*probe_thick;
    cy = probe_y;
    cz = probe_z + 62 + i*25;
    Cylinder(circle) = {cx, cy, cz, probe_thick, 0, 0, contact_rad};
  
    all_volumes[volume_counter] = circle;
    volume_counter += 1;
  EndFor

  cz = 0;
  For i In {0:9}
    cx = probe_x - 0.5*probe_thick;
    cz = probe_z + 62 + Sqrt(22*22-18*18) +i*25;
  
    cy = probe_y + 18;
    circle = newv;
    Cylinder(circle) = {cx, cy, cz, probe_thick, 0, 0, contact_rad};
  
    all_volumes[volume_counter] = circle;
    volume_counter += 1;

    //--------
  
    cy = probe_y - 18;
    circle = newv;
    Cylinder(circle) = {cx, cy, cz, probe_thick, 0, 0, contact_rad};
  
    all_volumes[volume_counter] = circle;
    volume_counter += 1;
  EndFor
EndIf


// Bounding polygon
cz = probe_z + 62 + Sqrt(22*22-18*18) + 9*25;
p = newp;
points[] = {7, p, p+1, p+2, p+3, p+4, p+5, p+6};
Point(points[1]) = {probe_x - probe_thick/2, probe_y, probe_z};
Point(points[2]) = {probe_x - probe_thick/2, probe_y+31, probe_z+62};
Point(points[3]) = {probe_x - probe_thick/2, probe_y+57, probe_z+cz};
Point(points[4]) = {probe_x - probe_thick/2, probe_y+57, probe_z+probe_top};
Point(points[5]) = {probe_x - probe_thick/2, probe_y-57, probe_z+probe_top};
Point(points[6]) = {probe_x - probe_thick/2, probe_y-57, probe_z+cz};
Point(points[7]) = {probe_x - probe_thick/2, probe_y-31, probe_z+62};

l = newl;
lines[] = {7, l, l+1, l+2, l+3, l+4, l+5, l+6};
Line(lines[1]) = {points[1], points[2]};
Line(lines[2]) = {points[2], points[3]};
Line(lines[3]) = {points[3], points[4]};
Line(lines[4]) = {points[4], points[5]};
Line(lines[5]) = {points[5], points[6]};
Line(lines[6]) = {points[6], points[7]};
Line(lines[7]) = {points[7], points[1]};

// The polygon
polyg = news;
// {133, 134, 135, 129, 130, 131, 132};
Line Loop(polyg) = {lines[5], lines[6], lines[7], lines[1], lines[2], lines[3], lines[4]};
Plane Surface(polyg) = {polyg};
v[] = Extrude {probe_thick, 0, 0} { Surface{polyg}; };
probe_free = v[1];
   
If(with_pts)
  probe() = BooleanDifference {Volume{v[1]}; Delete; }{Volume{all_volumes[]}; };
Else
  probe = v[1];
EndIf