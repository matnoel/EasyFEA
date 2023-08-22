lc = 0.05;

a = 1;

R = 2*a/Sqrt(3);

r = R/Sqrt(2)/2;
phi = Pi/6;


Point(1) = {0.,R,0,lc};
Point(2) = {-0.866025*R,0.5*R,0,lc};
Point(3) = {-0.866025*R,-0.5*R,0,lc};
Point(4) = {0.,-1.*R,0,lc};
Point(5) = {0.866025*R,-0.5*R,0,lc};
Point(6) = {0.866025*R,0.5*R,0,lc};

Point(7) = {0.,(R-r),0,lc};
Point(8) = {-0.866025*(R-r),0.5*(R-r),0,lc};
Point(9) = {-0.866025*(R-r),-0.5*(R-r),0,lc};
Point(10) = {0.,-1.*(R-r),0,lc};
Point(11) = {0.866025*(R-r),-0.5*(R-r),0,lc};
Point(12) = {0.866025*(R-r),0.5*(R-r),0,lc};



Line (1) = {1,2};
Line (2) = {2,3};
Line (3) = {3,4};
Line (4) = {4,5};
Line (5) = {5,6};
Line (6) = {6,1};

Line (7) = {7,8};
Line (8) = {8,9};
Line (9) = {9,10};
Line (10) = {10,11};
Line (11) = {11,12};
Line (12) = {12,7};


Line Loop (1) = {1,2,3,4,5,6,-7,-8,-9,-10,-11,-12};
Line Loop (2) = {7,8,9,10,11,12};

Plane Surface(1) = {1};
Plane Surface(2) = {2};

Physical Surface(1)= {1}; 
Physical Surface(2)= {2}; 

// Periodic Line {1} = {4} ;
Periodic Line {2} = {5} ;
// Periodic Line {3} = {6} ;

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};







