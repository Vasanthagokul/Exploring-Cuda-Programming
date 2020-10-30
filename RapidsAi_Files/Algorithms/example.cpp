#include<bits/stdc++.h> // header file for all c++ libraries
using namespace std;   // stdout library for printing values 
double getSlope(vector<double> &x, vector<double> &y)
{
double sx = accumulate(x.begin(), x.end(), 0);
double sy = accumulate(y.begin(), y.end(), 0);
double sxx = inner_product(x.begin(), x.end(), x.begin(), 0);
double sxy = inner_product(x.begin(), x.end(), y.begin(), 0);
int n = static_cast<int>(x.size());
// (n*sxy — sx*sy)/(n*sxx — sx*sx)
double nor = n*sxy - sx*sy;
double denor = n*sxx - sx*sx;
if(denor!=0)
{
return nor/denor;
}
return numeric_limits<double>::max();
}
double getIntercept(vector<double> &x, vector<double> &y, double slope)
{
double sx = accumulate(x.begin(), x.end(), 0);
double sy = accumulate(y.begin(), y.end(), 0);
int n = static_cast<int>(x.size());
return (sy-slope*sx)/n;
}
// slope:a
// intercept:b
// derivative of slope: da
// derivative of intercept: db
double getCost(vector<double> &x, vector<double> &y, double a, double b, double &da, double &db)
{
int n = static_cast<int>(x.size());
double sx = accumulate(x.begin(), x.end(), 0);
double sy = accumulate(y.begin(), y.end(), 0);
double sxx = inner_product(x.begin(), x.end(), x.begin(), 0);
double sxy = inner_product(x.begin(), x.end(), y.begin(), 0);
double syy = inner_product(y.begin(), y.end(), y.begin(), 0);
double cost = syy - 2*a*sxy - 2*b*sy + a*a*sxx + 2*a*b*sx + n*b*b;
cost /= n;
da = 2*(-sxy + a*sxx + b*sx)/n;
db = 2*(-sy + a*sx + n*b)/n;
return cost;
}
void linearRegression(vector<double> &x, vector<double> &y, double slope = 1, double intercept = 0)
{
double lrate = 0.0002;
double threshold = 0.0001;
int iter = 0;
while(true)
{
double da = 0;
double db = 0;
double cost = getCost(x, y, slope, intercept, da, db);
if(iter%1000==0)
{
cout<<"Iter: "<<iter<< " cost = "<<cost<< " da = " << da << " db = "<<db<< endl;
}
iter++;
if(abs(da) < threshold && abs(db) < threshold)
{
cout<<"y = "<<slope<<" * x + "<<intercept<<endl;
break;
}
slope -= lrate* da;
intercept -= lrate * db;
}
}
int main() {
vector<double> x = { 71, 73, 64, 65, 61, 70, 65, 72, 63, 67, 64};
vector<double> y = {160, 183, 154, 168, 159, 180, 145, 210, 132, 168, 141};
// initialize with random two points
cout<< "Initialization with random 2 points"<<endl;
vector<double> xSub = { 71, 73};
vector<double> ySub = {160, 183};
double slopeSub =getSlope(xSub,ySub);
double interceptSub = getIntercept(xSub, ySub, slopeSub);
cout<<"y = "<<slopeSub<<" * x + "<<interceptSub<<endl;
linearRegression(x, y, slopeSub, interceptSub);
cout<< "Compare with ground truth"<<endl;
double slope = getSlope(x,y);
double intercept = getIntercept(x, y, slope);
cout<<"y = "<<slope<<" * x + "<<intercept<<endl;
return 0;
}