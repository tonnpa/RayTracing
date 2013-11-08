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
	Ray(Point origin, Vector direction) : origin(origin){
		this->direction = direction.normalized();
	}
};

class Material{
	bool isDiffuse, isReflective, isRefractive;

public:
	Material(bool isDiffuse, bool reflective, bool refractive)
		:isDiffuse(isDiffuse), isReflective(reflective), isRefractive(refractive){}

	bool getDiffuseProperty(){
		return isDiffuse;
	}

	bool getReflectiveProperty(){
		return isReflective;
	}

	bool getRefractiveProperty(){
		return isRefractive;
	}

	virtual Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin) = 0;

	virtual Color getDiffuseBRDF(){
		return Color(0, 0, 0);
	}
};

class SmoothMaterial : public Material{
	Color n;	//refractive index
	Color k;	//extinction kappa
	Color F0;	//Fresnel approximation constant

public:
	SmoothMaterial(bool diffuse, bool reflect, bool refract, Color n, Color k) :
		Material(diffuse, reflect, refract), n(n), k(k){
		F0 = ((n - Color(1, 1, 1))*(n - Color(1, 1, 1)) + k*k) / ((n + Color(1, 1, 1))*(n + Color(1, 1, 1)) + k*k);
	}

	Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin){
		return Color(0, 0, 0);
	}

	//returns reflected radiance for smooth surfaces
	Color Fresnel(Vector& N, Vector& V){
		double cosa = fabs(N*V);
		return F0 + (Color(1, 1, 1) - F0)*pow(1 - cosa, 5);
	}

	//sets reflection direction to R
	Vector reflectionDirection(Vector& N, Vector& V){
		Vector R;
		double cosa = N*(-1.0)*V;
		R = V + N*cosa * 2;
		return R;
	}
};

class RoughMaterial : public Material{
	Color kd;	//diffuse BRDF

public:
	RoughMaterial(bool diffuse, bool reflect, bool refract, Color BRDF) :
		Material(diffuse, reflect, refract), kd(BRDF){}

	Color getDiffuseBRDF(){
		return kd;
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
};

class Object;
class Primitive;

struct Hit{
	Ray ray;	//the ray that intersected the object
	Object* iObject; //the object that intersected the ray - to get color properties of intersected object
	Primitive* iPrimitive; //the primitive that intersected the ray
	double t;	//hit parameter - to determine whether there was a hit
	Point x;	//hit point - to know the position from which a new ray might have to be shot
	Vector iNormal;	//surface normal - to know which direction the new ray goes

	Hit(Ray ray, Object* obj, Primitive* prim, double t, Point p, Vector v) :
		ray(ray), iObject(obj), iPrimitive(prim), t(t), x(p), iNormal(v){}
};

class Primitive{
	Material* material;
	
public:
	Primitive(Material* material) :material(material){}

	bool isDiffuse(){
		return material->getDiffuseProperty();
	}
	bool isReflective(){
		return material->getReflectiveProperty();
	}
	//calculates the parameter at which the ray is intersecting the object
	virtual Hit intersect(Ray&) = 0;
	//returns the normal vector of the surface at a given point
	virtual Vector surfaceNormal(Ray&, Point&) = 0;
	//tells whether a point is part of the primitive
	virtual bool containsPoint(Point& p) = 0;
	//reflected radiance of diffuse materials
	virtual Color reflectedRadiance(Vector& L, Vector& N, Vector& V, Color Lin){	
		return material->reflectedRadiance(L, N, V, Lin);
	}
	virtual Color getDiffuseBRDF(){
		return material->getDiffuseBRDF();
	}
};



class Circle : public Primitive{
	Point center;
	Vector normal;
	double radius;

public:
	Circle(Material* material, Point center, Vector normalDirection, double radius) :
		Primitive(material), center(center), radius(radius){
		normal = normalDirection.normalized();
	}
	Hit intersect(Ray& ray){
		double t = -1.0;
		Point hitPoint = Point(0, 0, 0);
		Vector iNormal = Vector(0, 0, 0);
		if (fabs(ray.direction*normal) > EPSILON){
			t = -((ray.origin - center)*normal) / (ray.direction*normal);
			hitPoint = ray.origin + ray.direction*t;
			if ((hitPoint - center).Length() < radius){
				iNormal = this->surfaceNormal(ray, hitPoint);
			}
			else
				t = -1.0;
		}
		return Hit(ray, NULL, this, t, hitPoint, iNormal);
	}

	Vector surfaceNormal(Ray& ray, Point& point){
		double cos = ray.direction*normal;
		return cos < 0 ? normal : normal*(-1);
	}

	bool containsPoint(Point& point){
		if ((point - center)*normal < EPSILON && (point - center).Length() < radius)
			return true;
		return false;
	}
};

class Object{
	Primitive* primitives[MAX_NUM_OF_OBJ];
	int primitiveCount;

public:
	Object() :primitiveCount(0){}

	void addPrimitive(Primitive* newPrimitive){
		if (primitiveCount < MAX_NUM_OF_OBJ){
			primitives[primitiveCount] = newPrimitive;
			primitiveCount++;
		}
	}

	bool isDiffuse(Point& point){
		bool found = false;
		for (int i = 0; i < primitiveCount; ++i){
			if (found = primitives[i]->containsPoint(point))
				return primitives[i]->isDiffuse();
		}
	}

	Hit intersect(Ray& ray){
		Primitive* primPtr = NULL; 
		Point hitPoint; Vector normal;
		double t = T_MAX;
		for (int i = 0; i < primitiveCount; ++i){
			Hit hit = primitives[i]->intersect(ray);
			if (hit.t > EPSILON){
				if (hit.t < t){
					primPtr = hit.iPrimitive;
					t = hit.t;
					hitPoint = hit.x;
					normal = hit.iNormal;
				}
			}
		}
		if (t == T_MAX){
			t = -1.0;
		}
		return Hit(ray, this, primPtr, t, hitPoint, normal);
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
		return intensity / pow((point - position).Length(), 2);
	}
};

class DirectionalLight : public Light{
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
		Vector direction = ((lookAt + right*(2 * x / xMax - 1 + 1 / xMax) + up*(2 * y / yMax - 1 + 1 / yMax)) - eye).normalized();
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
		Primitive* iPrim = NULL;
		Point x;
		Vector iNormal;
		double t = T_MAX;
		for (int i = 0; i<objectCount; ++i){
			Hit tnew = objectArray[i]->intersect(ray);
			if (tnew.t > 0 && tnew.t < t){
				t = tnew.t;
				iObject = objectArray[i];
				iPrim = tnew.iPrimitive;
				iNormal = tnew.iNormal;
			}
		}
		if (t < T_MAX){
			Point hitPoint = ray.origin + ray.direction*t;
			return Hit(ray, iObject, iPrim, t, hitPoint, iNormal);
		}
		else
			return Hit(ray, NULL, iPrim, -1.0, Point(0, 0, 0), Vector(0, 0, 0));
	}

	Color directIllumination(Hit& hit){
		Color color = ka*La;
		Point x = hit.x;

		if (hit.iPrimitive->isDiffuse()){
			Vector N = hit.iNormal;
			Ray rayToPoint = hit.ray;
			Ray shadowRay;

			for (int i = 0; i < lightSourceCount; ++i){
				shadowRay.origin = x;
				shadowRay.direction = lightSources[i]->getDirectionFromPoint(x); //vector from point to light source
				Hit shadowHit = intersectAll(shadowRay);
				Point y = shadowHit.x;
				if (shadowHit.t < 0 || (x - y).Length() > lightSources[i]->getDistanceFromPoint(x)){ //distance of point and light source
					Vector V = rayToPoint.direction*(-1.0);
					Vector L = shadowRay.direction;
					Color Lin = lightSources[i]->getRadianceTo(x);
					color += hit.iPrimitive->getDiffuseBRDF()*hit.iPrimitive->reflectedRadiance(L, N, V, Lin);
				}
			}
		}
		return color;
	}

	Color trace(Ray rayToPixel, int depth = 0){
		if (depth > MAX_DEPTH){
			return La;
		}
		Hit hit = intersectAll(rayToPixel);
		if (hit.t < 0)
			return La;

		Color color = directIllumination(hit);

		//if (hit.iObject->isReflective()){
		//	Ray reflectedRay = hit.iObject->reflect(rayToPixel, hit.x);
		//	color += hit.iObject->Fresnel(rayToPixel.direction, hit.iNormal)*trace(reflectedRay, depth + 1);
		//}
		return color;
	}

	void build(){
		Point lookAt = Point(0, 0, 0);
		Point eye = Point(0, 0, -1);
		Vector up = Vector(0, 1, 0);
		Vector right = Vector(1, 0, 0);
		//view orientation
		camera = Camera(lookAt, eye, up, right, screenWidth, screenHeight);

		//materials
		SmoothMaterial* gold = new SmoothMaterial(false, true, false, Color(0.17, 0.35, 1.5), Color(3.1, 2.7, 1.9));
		SmoothMaterial* silver = new SmoothMaterial(false, true, false, Color(4.1, 2.3, 3.1), Color(0.14, 0.16, 0.13));
		SmoothMaterial* copper = new SmoothMaterial(false, true, false, Color(3.6, 2.6, 2.3), Color(0.2, 1.1, 1.2));

		RoughMaterial* velvet = new RoughMaterial(true, false, false, Color(1, 0, 0));

		//objects
		Object* c = new Object();
		Circle* circle = new Circle(velvet, Point(0, 0, 10), Vector(0, 0, 1), 2);
		c->addPrimitive(circle);

		this->addObject(c);
		//illumination
		DirectionalLight* ceiling = new DirectionalLight(Color(0.5, 1.5, 2.55), 40);
		PositionalLight* lightbulb = new PositionalLight(Color(1000, 1000, 1000), Point(0, 0, 20));

		//this->addLightSource(ceiling);
		this->addLightSource(lightbulb);
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