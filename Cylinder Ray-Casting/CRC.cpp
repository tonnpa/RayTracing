//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2013-tol.          
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk. 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat. 
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni (printf is fajlmuvelet!)
// - new operatort hivni az onInitialization fA1ggvA©nyt kivA©ve, a lefoglalt adat korrekt felszabadA­tA!sa nA©lkA1l 
// - felesleges programsorokat a beadott programban hagyni
// - tovabbi kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan gl/glu/glut fuggvenyek hasznalhatok, amelyek
// 1. Az oran a feladatkiadasig elhangzottak ES (logikai AND muvelet)
// 2. Az alabbi listaban szerepelnek:  
// Rendering pass: glBegin, glVertex[2|3]f, glColor3f, glNormal3f, glTexCoord2f, glEnd, glDrawPixels
// Transzformaciok: glViewport, glMatrixMode, glLoadIdentity, glMultMatrixf, gluOrtho2D, 
// glTranslatef, glRotatef, glScalef, gluLookAt, gluPerspective, glPushMatrix, glPopMatrix,
// Illuminacio: glMaterialfv, glMaterialfv, glMaterialf, glLightfv
// Texturazas: glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexEnvi, 
// Pipeline vezerles: glShadeModel, glEnable/Disable a kovetkezokre:
// GL_LIGHTING, GL_NORMALIZE, GL_DEPTH_TEST, GL_CULL_FACE, GL_TEXTURE_2D, GL_BLEND, GL_LIGHT[0..7]
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nguyen Phan Anh
// Neptun : BQZUZ3
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy 
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem. 
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a 
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb 
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem, 
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.  
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat 
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__)                                                                                                                                                                                                            
#include <OpenGL/gl.h>                                                                                                                                                                                                            
#include <OpenGL/glu.h>                                                                                                                                                                                                           
#include <GLUT/glut.h>                                                                                                                                                                                                            
#else                                                                                                                                                                                                                             
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)                                                                                                                                                                       
#include <windows.h>                                                                                                                                                                                                              
#endif                                                                                                                                                                                                                            
#include <GL/gl.h>                                                                                                                                                                                                                
#include <GL/glu.h>                                                                                                                                                                                                               
#include <GL/glut.h>                                                                                                                                                                                                              
#endif          


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

#define MAX_NUM_OF_OBJ 10
#define EPSILON 0.001
#define T_MAX 1000000.0
#define PI 3.141592
#define MAX_DEPTH 5

struct Vector {
	float x, y, z;

	Vector() {
		x = y = z = 0;
	}
	Vector(float x0, float y0, float z0 = 0) {
		x = x0; y = y0; z = z0;
	}
	Vector operator*(float a) {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator/(double a){
		return Vector(x / a, y / a, z / a);
	}
	Vector operator+(const Vector& v) {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator-(const Vector& v) {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	float operator*(const Vector& v) { 	// dot product
		return (x * v.x + y * v.y + z * v.z);
	}
	Vector operator%(const Vector& v) { 	// cross product
		return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	float Length() { return sqrt(x * x + y * y + z * z); }

	Vector normalized(){
		float length = this->Length();
		return Vector(x /= length, y /= length, z /= length);
	}
};

typedef Vector Point;

struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0; g = g0; b = b0;
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator/(float a){
		return Color(r / a, g / a, b / a);
	}
	Color operator*(const Color& c) {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator/(const Color& c){
		return Color(r / c.r, g / c.g, b / c.b);
	}
	Color operator+(const Color& c) {
		return Color(r + c.r, g + c.g, b + c.b);
	}
	Color operator-(const Color& c){
		return Color(r - c.r, g - c.g, b - c.b);
	}
	void operator+=(const Color& c) {
		r += c.r; g += c.g; b += c.b;
	}
};

typedef Color Pixel;

struct Ray{
	Point origin;
	Vector direction;

	Ray(){}
	Ray(Point origin, Vector direction) : origin(origin), direction(direction){}
};

class Material{
	//type
	Color F0;	//Fresnel approximation constant
	Color n;	//refractive index
	Color k;	//extinction kappa
	Color kd;	//diffuse BRDF

	bool isReflective, isRefractive;
public:
	Material(Color n, Color kInput, bool reflective, bool refractive)
		:isReflective(reflective), isRefractive(refractive){
		//smooth surface
		if (reflective || refractive){
			this->k = kInput;
			this->n = n;
			F0 = ((n - Color(1, 1, 1))*(n - Color(1, 1, 1)) + k*k) / ((n + Color(1, 1, 1))*(n + Color(1, 1, 1)) + k*k);
			kd = Color(0, 0, 0);
		}
		//rough surface
		else{
			F0 = k = n = Color(0, 0, 0);
			kd = kInput;
		}
	}

	bool getReflectiveProperty(){
		return isReflective;
	}

	bool getDiffuseProperty(){
		if (isReflective || isRefractive){
			return false;
		}
		else
			return true;
	}

	//sets reflection direction to R
	Vector reflectionDirection(Vector& N, Vector& V){
		Vector R;
		double cosa = N*(-1.0)*V;
		R = V + N*cosa * 2;
		return R;
	}

	//returns reflected radiance for smooth surfaces
	Color Fresnel(Vector& N, Vector& V){
		double cosa = fabs(N*V);
		return F0 + (Color(1, 1, 1) - F0)*pow(1 - cosa, 5);
	}

	//returns reflected radiance for rough surfaces
	Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin){
		//all vectors are assumed to be normalized
		//x: hit point
		//L: from x to light source
		//N: surface normal
		//V: from x to eye
		//Lin: incoming radiance from ray

		double costheta = N*L;
		if (costheta < 0)
			return Color(0, 0, 0);
		Color Lref = Lin*kd*costheta;
		return Lref;
	}
	Color getDiffuseBRDF(){
		return kd;
	}
};

class Object;

struct Hit{
	Ray ray;	//the ray that intersected the object
	Object* iObject; //the object that intersected the ray - to get color properties of intersected object
	double t;	//hit parameter - to determine whether there was a hit
	Point x;	//hit point - to know the position from which a new ray might have to be shot
	Vector iNormal;	//surface normal - to know which direction the new ray goes

	Hit(Ray ray, Object* obj, double t, Point p, Vector v) :
		ray(ray), iObject(obj), t(t), x(p), iNormal(v){}
};

class Object{
protected:
	Material material;

public:
	Object(Material material) :material(material){}
	//calculates the parameter at which the ray is intersecting the object
	virtual Hit intersect(Ray&) = 0;
	//returns the normal vector of the surface at a given point
	virtual Vector surfaceNormal(Point&) = 0;
	//returns the smaller positive number
	double selectSmallerPositive(double t1, double t2){
		double t;
		if (t1 < EPSILON)
			t1 = -1.0;
		if (t2 < EPSILON)
			t2 = -1.0;
		if (t1 < 0 && t2 < 0){
			t = -1.0; //no hit point
		}
		else if (t1 > 0 && t2 < 0){
			t = t1; //hits at t1
		}
		else if (t1 < 0 && t2 > 0){
			t = t2; //hits at t2
		}
		else{
			t = fmin(t1, t2);
		}
		return t;
	}
	//returns the smallest positive solution, if exists
	double solveQuadraticEquation(double A, double B, double C){	
		//if (fabs(A) < EPSILON){
		//	if (fabs(B) < EPSILON)
		//		return -1;
		//	else
		//		return -C / B;
		//}
		double discriminant = B*B - 4 * A*C;
		if (discriminant < 0)
			return -1.0;
		//there are solutions
		else{
			double t1, t2;
			t1 = (-B + sqrt(discriminant)) / 2 / A;
			t2 = (-B - sqrt(discriminant)) / 2 / A;

			return selectSmallerPositive(t1, t2);
		}
	}
	virtual Color kd(Point&){
		return material.getDiffuseBRDF();
	}
	Color Fresnel(Vector& incomingDirection, Vector& surfaceNormal){
		return material.Fresnel(incomingDirection, surfaceNormal);
	}
	virtual Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin){
		return material.reflectedRadiance(L, N, V, Lin);
	}
	virtual bool isReflective(){
		return material.getReflectiveProperty();
	}
	virtual bool isDiffuse(Point& point){
		return material.getDiffuseProperty();
	}
	Ray reflect(Ray& ray, Point& hitPoint){
		Vector normal = surfaceNormal(hitPoint);
		Vector reflectionDirection = material.reflectionDirection(normal, ray.direction);
		return Ray(hitPoint, reflectionDirection);
	}
};

class Cylinder : public Object{
protected:
	Point referencePoint;	//the center of the bottom circle
	Vector standDirection;	//normalized vector of the axis line
	double height;
	double radius;

public:
	Cylinder(Material material, Point position, Vector direction, double height, double radius) :
		Object(material), referencePoint(position), standDirection(direction), height(height), radius(radius){}

	Hit sideHit(Ray& ray, double t){
		Vector r = ray.origin + ray.direction*t;
		if ((r - referencePoint)*standDirection < height && (r - referencePoint)*standDirection > 0){
			Vector normal = surfaceNormal(r);
			return Hit(ray, this, t, r, normal);
		}
		else
			return Hit(ray, NULL, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	Hit circleHit(Ray& ray, double tPlane, bool topCircleIntersected){
		Vector r = ray.origin + ray.direction*tPlane;
		Vector surfaceNormal;
		if (topCircleIntersected){
			if ((r - (referencePoint + standDirection*height)).Length() < radius){
				//surfaceNormal = a;
				Vector v = this->surfaceNormal(r);
				surfaceNormal = this->surfaceNormal(r);
				return Hit(ray, this, tPlane, r, surfaceNormal);
			}
		}
		else if ((r - referencePoint).Length() < radius){
				//surfaceNormal = a*(-1);
				surfaceNormal = this->surfaceNormal(r);
				return Hit(ray, this, tPlane, r, surfaceNormal);
		}
		return Hit(ray, NULL, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	Hit intersect(Ray& ray){
		/*
		C => -R^2-((eye-r0).a)^2+(eye-r0).(eye-r0)
		B => t (2 (eye-r0).v-2 (eye-r0).a v.a)
		A => t^2 (v.v-(v.a)^2)
		*/
		Vector a, eye, v, r0;
		a = standDirection;
		eye = ray.origin;
		v = ray.direction;
		r0 = referencePoint;
		double R = radius;

		//check for side
		double A, B, C;
		A = v*v - pow(v*a, 2);
		B = (eye - r0)*v * 2 - (eye - r0)*a *(v*a) * 2;
		C = -pow(R, 2) - pow((eye - r0)*a, 2) + (eye - r0)*(eye - r0);
		double tSide = solveQuadraticEquation(A, B, C);
		
		//check for plane
		double tPlane = -EPSILON;
		bool topCircleIntersected = false;
		if (fabs(v*a) > EPSILON){
			double t3, t4;
			//bottom
			t3 = -((eye - r0)*a) / (v*a);
			//top
			t4 = -((eye - (r0 + a*height))*a) / (v*a);

			tPlane = selectSmallerPositive(t3, t4);
			if (fabs(tPlane - t4) < EPSILON)
				topCircleIntersected = true;
		}

		if (tSide > 0 && tPlane > 0){
			if (tSide < tPlane){
				Hit result = sideHit(ray, tSide);
				if (result.t > EPSILON)
					return result;
			}
			else{
				Hit result = circleHit(ray, tPlane, topCircleIntersected);
				if (result.t > EPSILON)
					return result;
			}
		}
		//side
		if (tSide > EPSILON){
			Hit result = sideHit(ray, tSide);
			if (result.t > EPSILON)
				return result;
		}
		//circle
		if (tPlane > EPSILON){
			Hit result = circleHit(ray, tPlane, topCircleIntersected);
			if (result.t > EPSILON)
				return result;
		}
		
		return Hit(ray, NULL, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	virtual Vector surfaceNormal(Point& surfacePoint) = 0;
};

class ObjectCylinder : public Cylinder{
public:
	ObjectCylinder(Material material, Point position, Vector direction, double height, double radius) :
		Cylinder(material, position, direction, height, radius){}

	Vector surfaceNormal(Point& surfacePoint){
		Point r0 = referencePoint;
		Point r = surfacePoint;
		Vector a = standDirection;

		//bottom circle
		if ((r - r0).Length() < radius - EPSILON)
			return a*(-1);
		//top circle
		if ((r - (r0 + a*height)).Length() < radius - EPSILON)
			return a;

		Vector axisPoint = r0 + a*((r - r0)*a);
		Vector dir = r - axisPoint;
		return dir.normalized();
	}
};

class ContainerCylinder : public Cylinder{
	Material bottomMaterial;
public:
	ContainerCylinder(Material material, Material bottomMaterial, Point position, Vector direction, double height, double radius):
		Cylinder(material, position, direction, height, radius), bottomMaterial(bottomMaterial){}

	Color kd(Point& p){
		if ((p - referencePoint).Length() < radius - EPSILON)
			return bottomMaterial.getDiffuseBRDF();
		else
			return Object::kd(p);
	}

	bool isDiffuse(Point& point){
		Point r0 = referencePoint;
		Point r = point;
		Vector a = standDirection;

		//bottom circle
		if ((r - r0).Length() < radius - EPSILON)
			return true;
		else
			return false;
	}

	Vector surfaceNormal(Point& surfacePoint){
		Point r0 = referencePoint;
		Point r = surfacePoint;
		Vector a = standDirection;

		//bottom circle
		if ((r - r0).Length() < radius - EPSILON)
			return a;

		Vector axisPoint = r0 + a*((r - r0)*a);
		Vector dir = r - axisPoint;
		return dir.normalized()*(-1);
	}

	Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin){
		return bottomMaterial.reflectedRadiance(L, N, V, Lin);
	}
};

class Paraboloid : public Object{
	Point referencePoint;
	Vector normal;
	double focusDistance; //focusPoint = referencePoint + normal*focusDistance
	double length;
public:
	Paraboloid(Material material, Point position, Vector normal, double f, double length) :
		Object(material), referencePoint(position), normal(normal), focusDistance(f), length(length){}

	Hit intersect(Ray& ray){
		/*
		C => -(n.(eye - r0))^2 + (eye - f n - r0).(eye - f n - r0)
		B => t (-2 n.(eye - r0) n.v + 2 v.(eye - f n - r0))
		A => t^2 (-(n.v)^2 + v.v)
		*/
		Vector n, eye, v, r0;
		n = normal;
		eye = ray.origin;
		v = ray.direction;
		r0 = referencePoint;
		double f = focusDistance;

		//solve quadratic equation
		double A, B, C;
		A = v*v - pow((n*v),2);
		B = n*(eye - r0)*(n*v)*(-2) + v*(eye - n*f - r0) * 2;
		C = -pow(n*(eye - r0), 2) + (eye - n*f - r0)*(eye - n*f - r0);

		double t = solveQuadraticEquation(A, B, C);
		Vector r = ray.origin + ray.direction*t;

		if (t > EPSILON && 0 < n*(r-r0) && n*(r-r0) < f/2+length){
			Vector normal = surfaceNormal(r);
			return Hit(ray, this, t, r, normal);
		}
		else
			return Hit(ray, NULL, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	Vector surfaceNormal(Point& surfacePoint){
		Point r0 = referencePoint;
		Point r = surfacePoint;
		Vector n = normal;
		double f = focusDistance;

		return ((r - r0 - n*f) * 2 - n*2*(n*(r - r0))).normalized();
	}
};

class Light{
protected:
	Color intensity;

public:
	Light(Color intensity) :intensity(intensity){}

	virtual Point getPosition() = 0;

	virtual double getDistanceFromPoint(Point&) = 0;

	virtual Vector getDirectionFromPoint(Point&) = 0;

	virtual Color getRadianceTo(Point&) = 0;
};

class PositionalLight : public Light{
	Point position;

public:
	PositionalLight(Color intensity, Point position) : Light(intensity), position(position){}

	Point getPosition(){
		return position;
	}

	double getDistanceFromPoint(Point& point){
		return (point - position).Length();
	}
	Vector getDirectionFromPoint(Point& point){
		return (position - point).normalized();
	}
	Color getRadianceTo(Point& point){
		//the intensity is inversely proportional to the distance
		return intensity / pow((point-position).Length(),2);
	}
};

class DirectionalLight: public Light{
	double height; //this equals to the y parameter in the virtual world

public:
	DirectionalLight(Color intensity, double height) : Light(intensity), height(height){}

	Point getPosition(){
		return Point(0, height, 0);
	}

	double getDistanceFromPoint(Point& point){
		return fabs(height - point.y);
	}
	Vector getDirectionFromPoint(Point& point){
		return Vector(0, 1, 0);
	}
	Color getRadianceTo(Point& point){
		//constant intensity to everywhere
		return intensity;
	}
};

class Camera{
	Point lookAt;
	Point eye;
	Vector up;
	Vector right;

	double xMax; //width
	double yMax; //height
public:
	Camera(){}
	Camera(Point lookAt, Point eye, Vector up, Vector right, double width, double height) :
		lookAt(lookAt), eye(eye), up(up), right(right), xMax(width), yMax(height){}

	Ray getRayTo(double x, double y){ //returns a ray originating from the eye, leading into the pixel
		Vector direction = ((lookAt + right*(2 * x / xMax - 1 + 1 / xMax) + up*(2 * y / yMax - 1 + 1 / yMax))-eye).normalized();
		return Ray(eye, direction);
	}
};

const int screenWidth = 600;
const int screenHeight = 600;

Pixel image[screenWidth*screenHeight];

class Scene{
	Object* objectArray[MAX_NUM_OF_OBJ];
	int objectCount;
	Light* lightSources[MAX_NUM_OF_OBJ];
	int lightSourceCount;
	Camera camera;

	Color La;
	Color ka;
public:
	Scene() :objectCount(0), lightSourceCount(0){
		La = Color(0.1, 0.1, 0.1);
		ka = Color(0.1, 0.1, 0.1);
	}

	Hit intersectAll(Ray& ray){
		Object* iObject = NULL;
		Point x;
		Vector iNormal;
		double t = T_MAX;
		for (int i = 0; i<objectCount; ++i){
			Hit tnew = objectArray[i]->intersect(ray);
			if (tnew.t > 0 && tnew.t < t){
				t = tnew.t;
				iObject = objectArray[i];
				iNormal = tnew.iNormal;
			}
		}
		if (t < T_MAX){
			Point hitPoint = ray.origin + ray.direction*t;
			return Hit(ray, iObject, t, hitPoint, iNormal);
		}
		else
			return Hit(ray, NULL, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	Color directIllumination(Hit& hit){
		Color color = ka*La;
		Point x = hit.x;

		if (hit.iObject->isDiffuse(x)){
			Vector N = hit.iNormal;
			Ray rayToPoint = hit.ray;
			Ray shadowRay;

			for (int i = 0; i < lightSourceCount; ++i){
				shadowRay.origin = x;
				//shadowRay.direction = (lightSources[i]->getPosition() - x).normalized(); //getDirectionFromPoint
				shadowRay.direction = lightSources[i]->getDirectionFromPoint(x); //vector from point to light source
				Hit shadowHit = intersectAll(shadowRay);
				Point y = shadowHit.x;
				if (shadowHit.t < 0 || (x - y).Length() > lightSources[i]->getDistanceFromPoint(x)){ //distance of point and light source
					Vector V = rayToPoint.direction*(-1.0);
					Vector L = shadowRay.direction;
					Color Lin = lightSources[i]->getRadianceTo(x);
					color += hit.iObject->kd(x)*hit.iObject->reflectedRadiance(L, N, V, Lin);
				}
			}
		}
		return color;
	}

	Color trace(Ray rayToPixel, int depth=0){
		if (depth > MAX_DEPTH){
			return La;
		}
		Hit hit = intersectAll(rayToPixel);
		if (hit.t < 0)
			return La;

		Color color = directIllumination(hit);

		if (hit.iObject->isReflective()){
			Ray reflectedRay = hit.iObject->reflect(rayToPixel, hit.x);
			color += hit.iObject->Fresnel(rayToPixel.direction, hit.iNormal)*trace(reflectedRay, depth+1);
		}
		return color;
	}

	void render(){
		Ray rayToPixel;
		Color pixelRadiance;
		for (int y = 0; y < screenHeight; ++y){
			for (int x = 0; x < screenWidth; ++x){
				rayToPixel = camera.getRayTo(x, y);
				pixelRadiance = trace(rayToPixel);
				image[y*screenWidth + x] = pixelRadiance;
			}
		}
	}

	void addObject(Object* newObject){
		if (objectCount < MAX_NUM_OF_OBJ){
			objectArray[objectCount] = newObject;
			objectCount++;
		}
	}

	void addLightSource(Light* newLightSource){
		if (lightSourceCount < MAX_NUM_OF_OBJ){
			lightSources[lightSourceCount] = newLightSource;
			lightSourceCount++;
		}
	}

	void build(){
		Point lookAt = Point(0, 0, 0);
		Point eye = Point(0, 0, -1);
		Vector up = Vector(0, 1, 0);
		Vector right = Vector(1, 0, 0);

		//materials
		Material gold = Material(Color(0.17, 0.35, 1.5), Color(3.1, 2.7, 1.9), true, false);
		Material copper = Material(Color(3.6, 2.6, 2.3), Color(0.2, 1.1, 1.2), true, false);
		Material silver = Material(Color(4.1, 2.3, 3.1), Color(0.14, 0.16, 0.13), true, false);
		Material forest = Material(Color(0, 0, 0), Color(0, 1.2, .2), false, false);
		Material ground = Material(Color(0, 0, 0), Color(0.5, 0.01, 0.01), false, false);

		//view orientation
		camera = Camera(lookAt, eye, up, right, screenWidth, screenHeight);

		ObjectCylinder* cylinder = new ObjectCylinder(forest, Point(5, 0, 15), Vector(1, 1, -1).normalized(), 8, 1.5);
		ObjectCylinder* cylinder2 = new ObjectCylinder(silver, Point(0, -5, 15), Vector(0, 1, 1).normalized(), 8, 1.5);
		ObjectCylinder* cylinder3 = new ObjectCylinder(copper, Point(-5, 0, 15), Vector(-1, 1, -1).normalized(), 8, 1.5);

		Point paraboloidPosition = Point(0, 10, 15);
		Vector paraboloidDirection = Vector(-1, -5, 1).normalized();

		Point lightPosition = paraboloidPosition + paraboloidDirection*(2);
		Paraboloid* paraboloid = new Paraboloid(ground, paraboloidPosition, paraboloidDirection, 1, 7);

		ContainerCylinder* room = new ContainerCylinder(gold, ground, Point(0, -15, 0), Vector(0, 1, 0).normalized(), 100, 20);

		this->addObject(cylinder);
		this->addObject(cylinder2);
		this->addObject(cylinder3);

		this->addObject(paraboloid);
		
		this->addObject(room);

		//illumination
		DirectionalLight* ceiling = new DirectionalLight(Color(0.5, 1.5, 2.55), 40);
		PositionalLight* lightbulb = new PositionalLight(Color(2000, 2000, 2000), lightPosition);
		//PositionalLight* lightbulb = new PositionalLight(Color(1000, 1000, 1000), Point(0,0,0));

		this->addLightSource(ceiling);
		this->addLightSource(lightbulb);
	}
};

void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);
	Scene scene = Scene();
	scene.build();
	scene.render();
}

void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);		// torlesi szin beallitasa
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // kepernyo torles
	// Peldakent atmasoljuk a kepet a rasztertarba
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int x, int y) {}

void onKeyboardUp(unsigned char key, int x, int y) {}

void onMouse(int button, int state, int x, int y) {}

void onMouseMotion(int x, int y){}

void onIdle() {}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
	glutInit(&argc, argv); 				// GLUT inicializalasa
	glutInitWindowSize(600, 600);			// Alkalmazas ablak kezdeti merete 600x600 pixel 
	glutInitWindowPosition(100, 100);			// Az elozo alkalmazas ablakhoz kepest hol tunik fel
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);	// 8 bites R,G,B,A + dupla buffer + melyseg buffer

	glutCreateWindow("Grafika hazi feladat");		// Alkalmazas ablak megszuletik es megjelenik a kepernyon

	glMatrixMode(GL_MODELVIEW);				// A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);			// A PROJECTION transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();

	onInitialization();					// Az altalad irt inicializalast lefuttatjuk

	glutDisplayFunc(onDisplay);				// Esemenykezelok regisztralasa
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();					// Esemenykezelo hurok

	return 0;
}